#!/usr/bin/env python3
"""GRP 质量探针：用「对局真实结果」衡量奖励模型，比 TB 的全排列 acc 更可信。

TB 的 acc/val 是对**每个局前缀**的全排列 argmax 命中（含大量开局不可预测的前缀），
数值低且难以跨架构比较。本脚本按月份抽样真实牌谱，**分位置**输出：

  hit@k    : 第 k 局前缀上预测顺位 == 真实顺位的比例（k=1/3/6/9/末）
  mae@k    : |E[rank] − 真实顺位| （越小越好）
  nll_last : 末局前缀上真实全排列的负对数似然（proper scoring，越小越好）
  rew      : delta_pt（训练用的逐步奖励）的 mean/std/absmax → 判断奖励尺度

分位置才能分辨好坏：第 1 局几乎不可预测，末局又几乎被分数定死；
真正决定奖励质量的是中段（k=3/6）的锐度。

用法：
    python tools/training/grp_quality_probe.py \
        --state mortal/checkpoints/grp_v2.pth --hidden 64 --layers 2
    # 换容量实验时只改 --state/--hidden/--layers，其余口径完全一致
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import random
import sys

os.environ.setdefault('MORTAL_CFG', '')

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4')
    ap.add_argument('--months', default='202609,202401,202608')
    ap.add_argument('--n', type=int, default=150, help='每个月抽多少局')
    ap.add_argument('--state', required=True, help='GRP checkpoint(.pth) 路径')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--pts', default='6,4,2,0')
    ap.add_argument('--seed', type=int, default=20261009)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'mortal'))
    from model import GRP  # noqa: E402
    from libriichi.dataset import Grp  # noqa: E402

    grp = GRP(hidden_size=args.hidden, num_layers=args.layers)
    state = torch.load(args.state, map_location='cpu', weights_only=True)
    grp.load_state_dict(state['model'])
    grp.eval()
    print(f'GRP: {args.state} (steps={state.get("steps")}, hidden={args.hidden}, '
          f'layers={args.layers}, best_val_loss={state.get("best_val_loss")})', flush=True)

    pts = torch.tensor([float(x) for x in args.pts.split(',')], dtype=torch.float64)

    rng = random.Random(args.seed)
    POS = (1, 3, 6, 9)          # 固定看这几个局的前缀（不足则跳过）；另加“末局”
    tot = {'n': 0, 'nll': 0.0}
    tot_hit = collections.defaultdict(float)
    tot_mae = collections.defaultdict(float)
    tot_cnt = collections.Counter()
    rew_all = []

    for month in args.months.split(','):
        files = sorted(glob.glob(f'{args.root}/{month}/*.json.gz'))
        if not files:
            print(f'跳过 {month}: 无文件')
            continue
        rng.shuffle(files)
        m_hit = collections.defaultdict(float)
        m_mae = collections.defaultdict(float)
        m_cnt = collections.Counter()
        m_nll = 0.0
        cnt = 0
        rews = []
        for f in files[:args.n]:
            data = Grp.load_gz_log_files([f])
            for g in data:
                feat = torch.as_tensor(g.take_feature(), dtype=torch.float64)
                rb = np.frombuffer(g.take_rank_by_player(), dtype=np.uint8)
                n_kyoku = feat.shape[0]
                if n_kyoku < 2:
                    continue
                seq = [feat[:i + 1] for i in range(n_kyoku)]
                with torch.inference_mode():
                    logits = grp(seq)                     # (B, 24)
                    matrix = grp.calc_matrix(logits)      # (B, 4玩家, 4顺位)
                true_rank = int(rb[0])                     # 玩家 0，0-based
                probs = matrix[:, 0, :].numpy()            # (B, 4顺位)
                for k in POS:
                    if k <= n_kyoku:
                        p = probs[k - 1]
                        m_hit[k] += float(np.argmax(p) == true_rank)
                        m_mae[k] += abs(float(p @ np.arange(1, 5)) - (true_rank + 1))
                        m_cnt[k] += 1
                p_last = probs[-1]
                m_hit['last'] += float(np.argmax(p_last) == true_rank)
                m_mae['last'] += abs(float(p_last @ np.arange(1, 5)) - (true_rank + 1))
                m_cnt['last'] += 1
                label = grp.get_label(torch.tensor(rb, dtype=torch.int64).unsqueeze(0))
                m_nll += float(-torch.log_softmax(logits[-1], dim=-1)[int(label[0])])
                # 逐步奖励 delta_pt（玩家 0），与 dataloader 完全同口径
                final = torch.zeros((1, 4), dtype=torch.float64)
                final[0, true_rank] = 1.0
                rank_prob = torch.cat((matrix[:, 0, :], final))
                exp_pts = rank_prob @ pts
                rews.append((exp_pts[1:] - exp_pts[:-1]).numpy())
                cnt += 1
        rews = np.concatenate(rews) if rews else np.array([0.0])
        keys = list(POS) + ['last']
        print(f'{month} 局数={cnt}  ' + '  '.join(
            f'hit@{k}={m_hit[k]/max(m_cnt[k],1):.3f}/mae={m_mae[k]/max(m_cnt[k],1):.3f}'
            for k in keys) + f'  nll_last={m_nll/max(cnt,1):.4f}  '
            f'rew mean={rews.mean():+.4f} std={rews.std():.4f} absmax={np.abs(rews).max():.3f}',
            flush=True)
        tot['n'] += cnt
        tot['nll'] += m_nll
        for k in keys:
            tot_hit[k] += m_hit[k]
            tot_mae[k] += m_mae[k]
            tot_cnt[k] += m_cnt[k]
        rew_all.append(rews)

    n = max(tot['n'], 1)
    rew_all = np.concatenate(rew_all)
    keys = list(POS) + ['last']
    print(f'合计 局数={n}  ' + '  '.join(
        f'hit@{k}={tot_hit[k]/max(tot_cnt[k],1):.4f}/mae={tot_mae[k]/max(tot_cnt[k],1):.4f}'
        for k in keys) + f'  nll_last={tot["nll"]/n:.4f}  '
        f'rew std={rew_all.std():.4f} absmax={np.abs(rew_all).max():.3f}', flush=True)


main()
