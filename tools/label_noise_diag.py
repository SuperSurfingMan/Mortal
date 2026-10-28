#!/usr/bin/env python3
"""标签噪声地板的可用替代诊断：残差错误是「近似平局」还是「系统性错误」。

精确重复匹配在麻将状态空间上不可行（实测 76 万决策 0 个精确重复、粗键 0.11% 覆盖）。
改测三个可观测、且直接决定「BC 还有没有空间」的量：

1. **top-k 准确率**（k=1/3/5）——如果 top1 0.82 而 top5 0.97，说明人类动作几乎总在
   模型的少数候选中，剩余错误更像「多个合理动作之间的选择」；
2. **错误处的 Q 差** `Q(top1) − Q(human)`——若绝大多数错误集中在极小差距（近似平局），
   则属于不可约歧义；若差距很大，说明模型在「看起来明确」的状态上判错（可学）；
3. **按 Q 差分层的准确率**——top1 与次优的差距越大，准确率应越高；若平坦则说明 Q 的
   排序信息弱。

用法（mortal/ 下，GPU 可选）：
    python ../tools/label_noise_diag.py --ckpt checkpoints/s6c/mortal.pth --n 40000
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

import config as C          # noqa: E402
import model as MM          # noqa: E402
import s6c_features         # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

BS = 512


def load_net(ckpt):
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    ver = st['config']['control'].get('version', 4)
    _, extra = s6c_features.resolve(st['config']['control'])
    net = MM.Brain(version=ver, extra_channels=extra, **st['config']['resnet']).eval()
    net.load_state_dict(st['mortal'])
    dqn = MM.DQN(version=ver).eval()
    dqn.load_state_dict(st['current_dqn'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(dev), dqn.to(dev), ver, extra, dev, int(st.get('steps', 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/s6c/mortal.pth')
    ap.add_argument('--n', type=int, default=40000, help='打牌决策样本数')
    ap.add_argument('--month', default='202609')
    ap.add_argument('--seed', type=int, default=20261028)
    a = ap.parse_args()
    files = sorted(glob.glob(f'../dataset/mjai/houou4/{a.month}/**/*.json.gz', recursive=True))
    random.Random(a.seed).shuffle(files)
    net, dqn, ver, extra, dev, steps = load_net(a.ckpt)
    print(f'[diag] ckpt steps={steps} extra={extra} n={a.n} month={a.month} dev={dev}', flush=True)

    ld = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None,
                        augmented=False, always_include_kan_select=True)
    Q, A, M = [], [], []
    got = 0
    for f in files:
        if got >= a.n:
            break
        try:
            chunks = list(ld.load_gz_log_files([f]))
        except Exception:
            continue
        for games in chunks:
            for game in games:
                try:
                    obs = np.stack([np.asarray(o, np.float32) for o in game.take_obs()])
                    acts = np.asarray(game.take_actions())
                except Exception:
                    continue
                m = acts < 34
                if not m.any():
                    continue
                o, y = obs[m], acts[m]
                own = np.rint(o[:, :4].sum(1)) > 0
                if extra:
                    o = s6c_features.attach(o, s6c_features.compute(o))
                qs = []
                with torch.no_grad():
                    for i in range(0, len(o), BS):
                        x = torch.as_tensor(o[i:i + BS]).to(dev)
                        phi = net(x)
                        mask = torch.zeros(phi.shape[0], MM.ACTION_SPACE, dtype=torch.bool, device=dev)
                        mask[:, :34] = torch.as_tensor(own[i:i + BS]).to(dev)
                        qs.append(dqn(phi, mask)[:, :34].float().cpu())
                q = torch.cat(qs).numpy()
                Q.append(q); A.append(y); M.append(own)
                got += len(y)
        if got and len(Q) % 20 == 0:
            print(f'  {got:,} 决策', flush=True)
    Q = np.concatenate(Q)[:a.n]; A = np.concatenate(A)[:a.n]
    n = len(A); ar = np.arange(n)
    order = np.argsort(-Q, axis=1)
    top1 = order[:, 0]
    acc1 = float((top1 == A).mean())
    rank_of_human = np.array([int(np.where(order[i] == A[i])[0][0]) + 1 for i in range(n)])
    acc3 = float((rank_of_human <= 3).mean()); acc5 = float((rank_of_human <= 5).mean())
    gap = Q[ar, top1] - Q[ar, A]                      # 错误处：top1 比人类高多少
    err = top1 != A
    gap_err = gap[err]
    q1 = Q[ar, top1]
    q2 = np.sort(Q, axis=1)[:, -2]
    margin = q1 - q2                                   # top1 与次优的差（全样本）
    print(f'\n[结果] n={n:,}')
    print(f'  top1 {acc1:.4f}   top3 {acc3:.4f}   top5 {acc5:.4f}')
    print(f'  错误处 Q(top1)−Q(human)：中位 {np.median(gap_err):+.4f}  p75 {np.percentile(gap_err,75):+.4f}  p90 {np.percentile(gap_err,90):+.4f}')
    for eps in (0.05, 0.1, 0.25, 0.5):
        print(f'  |差距| ≤ {eps:<4}: 占错误 {float((gap_err <= eps).mean()):.2%}')
    print('  按「top1−次优」分位分层的准确率：')
    qs = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
    for lo, hi, tag in ((-1e9, qs[0], 'Q1'), (qs[0], qs[1], 'Q2'), (qs[1], qs[2], 'Q3'),
                        (qs[2], qs[3], 'Q4'), (qs[3], 1e9, 'Q5')):
        m = (margin > lo) & (margin <= hi)
        print(f'    {tag}: n={int(m.sum()):>6d}  margin 中位 {np.median(margin[m]):.4f}  top1 {float((top1[m]==A[m]).mean()):.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
