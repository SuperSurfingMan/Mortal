#!/usr/bin/env python3
"""P1 分层诊断：按公开信息条件（立直/鸣牌/巡目）分层重算读牌模型的全部离线指标。

目的（对应「上界估计」计划 P1）：头部指标（如 tenpai AUC）会被"简单局"抬高，
必须把指标拆到难度分层上，才能判断真实读牌能力与剩余进步空间。同时给出
每个分层的**无信息基线**（均匀随机 hit@k、alloc 众数类、边际熵），用于回答
"模型是否已利用信息"。

分层维度（全部来自公开信息，与主判决同口径）：
  riichi ∈ {0,1}、called ∈ {0,1}、turn 桶（0-3/4-7/8-11/12-15/16+），
  以及组合：无立直无鸣牌（干净局）、其终盘子集、立直+终盘等。

每个分层输出：n、结构性听牌率、alloc CE/top-1/MAE、wait CE/hit@1/3/5、
tenpai AUC、向听准确率、头↔分解一致率、deal_pure（有效听牌且最安全牌概率>0）；
并给出 hit@1 与 AUC 的 95% bootstrap CI。

用法：
    python tools/read_model/strat_metrics.py \
        --state mortal/read_model_v2s_a2b/best.pth --samples 20000 \
        --out mortal/read_model_v2s_a2b/strat_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(HERE))          # 必须最后插入：避免 mortal/train.py 遮蔽本目录 train.py

import train as T                                      # noqa: E402
from compare_v1_v2 import batch_to_samples, load_net  # noqa: E402
from structure import MAX_SHANTEN_BUCKET, soft_shanten_from_counts  # noqa: E402

NUM_TILES = 34


# --------------------------------------------------------------------------- #
# 基础统计
# --------------------------------------------------------------------------- #
def rank_auc(y: np.ndarray, s: np.ndarray) -> float:
    """秩和 AUC（并列取平均秩）。"""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    if y.all() or (~y).all():
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int(y.sum())
    nneg = len(y) - npos
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def boot_ci(values: np.ndarray, n_boot: int = 200, seed: int = 20261018,
            stat='mean') -> tuple[float, float]:
    """对逐目标向量做 bootstrap 95% CI（stat='mean' 或 'auc'）。"""
    values = np.asarray(values)
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        v = values[idx]
        out.append(rank_auc(v[:, 0], v[:, 1]) if stat == 'auc' else float(v.mean()))
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def random_hit(wv: np.ndarray, k: int) -> float:
    """无信息基线：从 34 张里均匀无放回抽 k 张，至少一张命中待牌集的概率。"""
    out = []
    for w in wv:
        w = int(w)
        if w <= 0:
            continue
        out.append(1.0 - math.comb(NUM_TILES - w, k) / math.comb(NUM_TILES, k))
    return float(np.mean(out)) if out else float('nan')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='mortal/read_model_v2s_a2b/best.pth')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--root', default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609')
    ap.add_argument('--out', default=None)
    ap.add_argument('--batch-size', type=int, default=2048)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))
    batch = T.build_fixed_eval_set(files, args.samples)
    samples = batch_to_samples(batch)
    n = len(samples)
    net, cfg = load_net(args.state, device)
    print(f'[model] {args.state} ({cfg.get("model", "v1")}) | n={n}', flush=True)

    # ---- 前向收集 ----
    A, WP, SH, SC = [], [], [], []
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            ch = {k: v[i:i + args.batch_size].to(device) for k, v in batch.items()}
            wb = T.wait_bias_of(ch).to(device)
            if hasattr(net, 'timeline'):            # v2/v2s 需要时序 token
                o = net(ch['obs'], ch['unseen'], wb, ch['tl_tokens'], ch['tl_mask'])
            else:
                o = net(ch['obs'], ch['unseen'], wb)
            A.append(torch.softmax(o['alloc_logits'], -1).cpu().numpy())
            WP.append(o['wait_probs'].cpu().numpy())
            if 'struct_shanten_logits' in o:
                SH.append(o['struct_shanten_logits'].argmax(-1).cpu().numpy())
                SC.append(torch.softmax(o['struct_count_logits'], -1).cpu().numpy())
    alloc_p = np.concatenate(A)                    # [n,3,34,5]
    wait_p = np.concatenate(WP)                    # [n,3,35]

    # ---- 标签 ----
    alloc_t = batch['alloc'].numpy().astype(np.int64)          # [n,3,34]
    wait_t = batch['wait'].numpy()                             # [n,3,35]
    tenpai = batch['tenpai'].numpy() > 0.5
    riichi = batch['riichi'].numpy() > 0.5
    called = batch['called'].numpy() > 0.5
    turn = batch['turn'].numpy()
    own = batch['own_hand'].numpy().astype(bool)               # [n,34] 观察者手牌
    k_idx = np.arange(5)

    # ---- 逐目标指标（[n,3]）----
    p_true = np.take_along_axis(alloc_p, alloc_t[..., None], -1)[..., 0]
    alloc_ce = -np.log(np.clip(p_true, 1e-12, None)).mean(-1)
    alloc_top1 = (alloc_p.argmax(-1) == alloc_t).mean(-1)
    exp_cnt = (alloc_p * k_idx).sum(-1)
    alloc_mae = np.abs(exp_cnt - alloc_t).mean(-1)

    tile_p = wait_p[..., :NUM_TILES]
    supp = wait_t[..., :NUM_TILES] > 0
    eff = supp.any(-1)                                         # 有效荣和集非空
    wcnt = supp.sum(-1)
    ll_wait = -(wait_t * np.log(np.clip(wait_p, 1e-12, None))).sum(-1)
    topk = np.argsort(-tile_p, -1)
    hit1 = np.take_along_axis(supp, topk[..., :1], -1)[..., 0]
    hit3 = np.take_along_axis(supp, topk[..., :3], -1).any(-1)
    hit5 = np.take_along_axis(supp, topk[..., :5], -1).any(-1)

    score = 1.0 - wait_p[..., NUM_TILES]                       # P(非未听)
    sh_true = batch['shanten'].numpy()
    sh_ok = None
    consist = None
    if SH:
        sh_pred = np.concatenate(SH)
        sh_ok = (sh_pred == sh_true)
        len_div3 = (alloc_t.sum(-1) // 3).astype(np.float32)
        with torch.no_grad():
            soft = soft_shanten_from_counts(
                torch.from_numpy(np.concatenate(SC)), torch.from_numpy(len_div3)).numpy()
        consist = (sh_pred == np.clip(np.round(soft), 0, MAX_SHANTEN_BUCKET))

    # deal：观察者手牌中"模型认为最安全"的一张，是否落在目标有效荣和集
    own3 = np.broadcast_to(own[:, None, :], (n, 3, NUM_TILES))
    p_cand = np.where(own3, tile_p, np.inf)
    safest = p_cand.argmin(-1)
    p_safe = np.take_along_axis(tile_p, safest[..., None], -1)[..., 0]
    deal = np.take_along_axis(supp, safest[..., None], -1)[..., 0]
    deal_mask = eff & (p_safe > 0)

    # ---- 分层定义（[n,3] 布尔）----
    T3 = np.broadcast_to(turn[:, None], (n, 3))
    strata = {
        'all': np.ones((n, 3), bool),
        'riichi=1': riichi,
        'riichi=0': ~riichi,
        'called=1': called,
        'called=0': ~called,
        'riichi=0&called=0': ~riichi & ~called,
        'riichi=0&called=0&turn>=8': ~riichi & ~called & (T3 >= 8),
        'riichi=0&called=0&turn<=7': ~riichi & ~called & (T3 <= 7),
        'riichi=1&turn>=8': riichi & (T3 >= 8),
        'riichi=1&called=0': riichi & ~called,
        'riichi=0&called=1': ~riichi & called,
        'turn0-3': T3 <= 3,
        'turn4-7': (T3 >= 4) & (T3 <= 7),
        'turn8-11': (T3 >= 8) & (T3 <= 11),
        'turn12-15': (T3 >= 12) & (T3 <= 15),
        'turn16+': T3 >= 16,
    }

    report = {'state': args.state, 'model': cfg.get('model', 'v1'), 'n_samples': int(n),
              'n_targets': int(n * 3), 'strata': {}}
    print(f"\n{'分层':<26s} {'n目标':>7s} {'听牌%':>6s} {'alloc_ce':>9s} {'top1':>6s} "
          f"{'MAE':>6s} {'wait_ce':>8s} {'hit1':>6s} {'hit3':>6s} {'hit5':>6s} "
          f"{'AUC':>6s} {'向听':>6s} {'一致':>6s} {'点炮%':>6s}", flush=True)
    print('-' * 130, flush=True)
    for name, m in strata.items():
        if m.sum() == 0:
            continue
        ma = m & eff                                            # 有效听牌目标
        v = {
            'n_targets': int(m.sum()),
            'tenpai_rate': float(tenpai[m].mean()),
            'n_active': int(ma.sum()),
            'alloc_ce': float(alloc_ce[m].mean()),
            'alloc_top1': float(alloc_top1[m].mean()),
            'alloc_mae': float(alloc_mae[m].mean()),
            'wait_ce': float(ll_wait[ma].mean()) if ma.any() else float('nan'),
            'hit1': float(hit1[ma].mean()) if ma.any() else float('nan'),
            'hit3': float(hit3[ma].mean()) if ma.any() else float('nan'),
            'hit5': float(hit5[ma].mean()) if ma.any() else float('nan'),
            'tenpai_auc': rank_auc(tenpai[m].ravel(), score[m].ravel()),
            'n_pos': int(tenpai[m].sum()),
            'shanten_acc': float(sh_ok[m].mean()) if sh_ok is not None else float('nan'),
            'consist': float(consist[m].mean()) if consist is not None else float('nan'),
            'deal_pure': float(deal[deal_mask & m].mean()) if (deal_mask & m).any() else float('nan'),
            'n_deal': int((deal_mask & m).sum()),
        }
        # 无信息基线（仅依赖标签）
        if ma.any():
            v['random_hit1'] = random_hit(wcnt[ma], 1)
            v['random_hit3'] = random_hit(wcnt[ma], 3)
            v['random_hit5'] = random_hit(wcnt[ma], 5)
            v['mean_w'] = float(wcnt[ma].mean())
        cls = np.bincount(alloc_t[m].ravel(), minlength=5) / alloc_t[m].size
        v['alloc_majority'] = float(cls.max())
        v['alloc_marginal_entropy'] = float(-(cls * np.log(np.clip(cls, 1e-12, None))).sum())
        v['alloc_info_gain'] = v['alloc_marginal_entropy'] - v['alloc_ce']
        if ma.any():
            v['hit1_ci'] = boot_ci(hit1[ma].astype(float))
            v['tenpai_auc_ci'] = boot_ci(np.stack([tenpai[m].ravel(), score[m].ravel()], -1), stat='auc')
        report['strata'][name] = v
        print(f"{name:<26s} {v['n_targets']:>7d} {100*v['tenpai_rate']:>5.1f} "
              f"{v['alloc_ce']:>9.4f} {v['alloc_top1']:>6.4f} {v['alloc_mae']:>6.4f} "
              f"{v['wait_ce']:>8.4f} {v['hit1']:>6.4f} {v['hit3']:>6.4f} {v['hit5']:>6.4f} "
              f"{v['tenpai_auc']:>6.4f} {v['shanten_acc']:>6.4f} {v['consist']:>6.4f} "
              f"{100*v['deal_pure']:>5.2f}", flush=True)

    print(f"\n{'分层':<26s} {'hit1':>6s} {'(95%CI)':>14s} {'随机hit1':>8s} {'hit5':>6s} "
          f"{'随机hit5':>8s} {'AUC':>6s} {'(95%CI)':>14s} {'众数':>6s} {'信息增益':>8s}", flush=True)
    print('-' * 118, flush=True)
    for name, v in report['strata'].items():
        c1 = v.get('hit1_ci', (float('nan'), float('nan')))
        ca = v.get('tenpai_auc_ci', (float('nan'), float('nan')))
        print(f"{name:<26s} {v['hit1']:>6.4f} [{c1[0]:>6.4f},{c1[1]:>6.4f}] "
              f"{v.get('random_hit1', float('nan')):>8.4f} {v['hit5']:>6.4f} "
              f"{v.get('random_hit5', float('nan')):>8.4f} {v['tenpai_auc']:>6.4f} "
              f"[{ca[0]:>6.4f},{ca[1]:>6.4f}] {v['alloc_majority']:>6.4f} "
              f"{v['alloc_info_gain']:>8.4f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, 'w'), ensure_ascii=False, indent=1)
        print(f'\n[out] {args.out}', flush=True)


if __name__ == '__main__':
    main()
