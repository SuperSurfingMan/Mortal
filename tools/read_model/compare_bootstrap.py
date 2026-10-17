"""v1 vs v2 严格对决：同一固定评估集 + bootstrap 置信区间。

回答「差异是真的还是噪声」——wait 类指标在 ~2500 样本上方差极大
（60 文件子集曾给出与全集相反的结论），必须给 CI。

用法：
    python tools/read_model/compare_bootstrap.py --samples 20000 --n-boot 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D                                     # noqa: E402
import model as M                                       # noqa: E402
import model_v2 as M2                                   # noqa: E402
import train as T                                       # noqa: E402
from compare_v1_v2 import batch_to_samples, load_net, predict   # noqa: E402

NUM_TILES = 34


def auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=bool)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    s = y_score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1', default='mortal/read_model_v1/best.pth')
    ap.add_argument('--v2', default='mortal/read_model_v2/best.pth')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--root', default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import glob
    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))
    samples = batch_to_samples(T.build_fixed_eval_set(files, args.samples))
    n = len(samples)
    print(f'[data] {n} 个样本（固定评估集）', flush=True)

    net1, _ = load_net(args.v1, device)
    net2, _ = load_net(args.v2, device)
    p1 = predict(net1, samples, device)
    p2 = predict(net2, samples, device)

    # ---- 预计算每个样本的逐目标统计量（bootstrap 在这些向量上重采样）----
    H = np.stack([s['alloc'] for s in samples]).astype(float)
    U = np.stack([s['unseen'] for s in samples])
    WT = np.stack([s['wait'] for s in samples])
    TP = np.stack([s['tenpai'] for s in samples]) > 0.5
    RI = np.stack([s['riichi'] for s in samples]) > 0.5
    own = np.stack([s['own_hand'] for s in samples])
    held = H > 0

    def per_sample(pred):
        """返回每个样本的统计向量 [n,3] 或 [n,34]，供 bootstrap 重采样。"""
        E, WP = pred['E'], pred['WP']
        bidx = (np.arange(n)[:, None], np.arange(3)[None])
        # alloc top-1 命中（逐目标）
        top1 = E.argmax(-1)
        top1_hit = held[bidx + (top1,)]                             # [n,3]
        # wait hit@1（逐目标，仅听牌目标计）
        sup = WT[:, :, :NUM_TILES] > 0
        eff = sup.any(-1)
        tile_p = WP[:, :, :NUM_TILES]
        hit1 = np.take_along_axis(sup, np.argsort(-tile_p, -1)[..., :1], -1).squeeze(-1)
        hit5 = np.take_along_axis(sup, np.argsort(-tile_p, -1)[..., :5], -1).any(-1)
        # 听牌检测 AUC 的打分（逐目标）：P(非未听)
        score = 1.0 - WP[:, :, NUM_TILES]
        # 防守纯模型：最安全牌的点炮（逐目标，仅在模型概率>0 的听牌目标）
        OWN3 = np.broadcast_to(own[:, None, :], (n, 3, NUM_TILES))
        p_cand = np.where(OWN3, tile_p, np.inf)
        safest = p_cand.argmin(-1)
        p_safe = tile_p[bidx + (safest,)]
        deal_safe = sup[bidx + (safest,)]
        return {
            'top1_hit': top1_hit, 'hit1': hit1, 'hit5': hit5, 'eff': eff,
            'riichi': RI, 'score': score, 'tenpai': TP,
            'p_safe': p_safe, 'deal_safe': deal_safe, 'riichi_f': RI,
        }

    s1 = per_sample(p1)
    s2 = per_sample(p2)

    def agg(s, idx):
        """在样本子集 idx 上聚合各指标。"""
        out = {}
        out['alloc_top1'] = s['top1_hit'][idx].mean()
        eff = s['eff'][idx]
        out['wait_hit1'] = s['hit1'][idx][eff].mean() if eff.any() else np.nan
        out['wait_hit5'] = s['hit5'][idx][eff].mean() if eff.any() else np.nan
        ri = s['riichi'][idx] & eff
        out['riichi_hit1'] = s['hit1'][idx][ri].mean() if ri.any() else np.nan
        # AUC（展平所有目标）
        out['tenpai_auc'] = auc(s['tenpai'][idx].ravel(), s['score'][idx].ravel())
        # 防守纯模型
        pure = eff & (s['p_safe'][idx] > 0)
        out['deal_pure'] = s['deal_safe'][idx][pure].mean() if pure.any() else np.nan
        ri_pure = ri & (s['p_safe'][idx] > 0)
        out['riichi_deal_pure'] = s['deal_safe'][idx][ri_pure].mean() if ri_pure.any() else np.nan
        return out

    full1, full2 = agg(s1, np.ones(n, bool)), agg(s2, np.ones(n, bool))
    keys = list(full1.keys())

    # bootstrap：按样本重采样（整条样本的所有目标一起，保留样本内相关结构）
    rng = np.random.default_rng(20261017)
    boots = {k: [] for k in keys}
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        a1, a2 = agg(s1, idx), agg(s2, idx)
        for k in keys:
            boots[k].append(a2[k] - a1[k])

    print(f"\n{'指标':<22s} {'v1':>9s} {'v2':>9s} {'Δ':>9s} {'95% CI':>20s} {'判定':>8s}")
    print('-' * 74)
    for k in keys:
        d = np.array(boots[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = 'v2优' if lo > 0 else ('v1优' if hi < 0 else '不显著')
        print(f'{k:<22s} {full1[k]:>9.4f} {full2[k]:>9.4f} '
              f'{full2[k] - full1[k]:>+9.4f}   [{lo:+.4f},{hi:+.4f}] {sig:>8s}')


if __name__ == '__main__':
    main()
