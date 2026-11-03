#!/usr/bin/env python3
"""根因/可行性诊断 4：三家联合容量约束的价值。

`unseen[x]` 是「x 还有几张不在自家手里、不在公共区」——即**三家暗牌 + 牌山**总数。
因此对任意样本、任意牌 x，三家持有指示变量之和 ≤ unseen[x] 是硬约束。
A2c 的 alloc 头是逐目标独立输出，本阶段模型也是（reshape B*J 后互不交互），
两者都可能违反。

本脚本：
  1. 统计模型/A2c 的违反率（Σ_j p_j 超过 unseen 的平均超出量）；
  2. 做一次「按比例缩放到容量」的后处理投影，比较约束前后的 macro AUC；
  3. 只对 unseen=1（唯一持有者）单独看。

若投影能提升 AUC，说明「跨目标联合约束」是尚未被利用的真实信息；
若不能，说明这条约束已被现有模型隐式吸收。

用法：
  python tools/read_model/tile_presence/analyze_capacity.py \
      --cache-dir tools/read_model/tile_presence/.cache120 \
      --ckpt tools/read_model/tile_presence/results/run3_d224_120k/presence.pth \
      --out tools/read_model/tile_presence/results/run3_d224_120k/capacity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tokens as TK  # noqa: E402
import train_presence as TP  # noqa: E402
from model import TilePresenceNet  # noqa: E402


def project(p: np.ndarray, cap: np.ndarray, iters: int = 5) -> np.ndarray:
    """把三家概率按比例压到 Σ_j p_j ≤ cap（逐样本逐牌）。"""
    q = p.astype(np.float64).copy()
    for _ in range(iters):
        s = q.sum(1)                                   # [N,34]
        over = s > cap
        if not over.any():
            break
        scale = np.where(over, cap / np.maximum(s, 1e-9), 1.0)
        q = np.clip(q * scale[:, None, :], 0.0, 1.0)
    return q.astype(np.float32)


def macro_auc(y, p, m) -> float:
    vals = []
    for j in range(3):
        for t in range(TK.NUM_TILES):
            mm = m[:, j, t]
            if mm.sum() < 10:
                continue
            v = TP._rank_auc(y[mm, j, t], p[mm, j, t])
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = st['args']
    net = TilePresenceNet(d=a['d'], n_blocks=a['blocks'], heads=a['heads'],
                          dropout=a['dropout']).to(TP.DEV)
    net.load_state_dict(st['state_dict'])
    net.eval()
    ev_path = Path(args.cache_dir) / 'eval.npz'
    ev = np.load(ev_path)
    out = TP.predict(net, DataLoader(TP.PresDataset(ev_path),
                                     batch_size=args.batch, shuffle=False))
    y, p_m = out['y'], out['p']
    meta = TP._meta(ev_path)
    m = TP._masks(meta['unseen'], meta['turn'], meta['riichi'], meta['called'])['all']
    cap = ev['unseen'].astype(np.float64)               # [N,34]
    cap3 = np.broadcast_to(cap[:, None, :], y.shape)

    ref = np.load('/tmp/read_model_diag_cache_a2c/cache.npz', allow_pickle=True)
    p_a = (1.0 - ref['AP'][:y.shape[0], :, :, 0].astype(np.float32))

    res = {}
    for name, p in (('model', p_m), ('a2c', p_a)):
        s = p.sum(1)
        over = (s - cap).clip(min=0)
        res[name] = {
            'auc_raw': macro_auc(y, p, m),
            'auc_proj': macro_auc(y, project(p, cap), m),
            'violation_cells_frac': float(((s > cap + 1e-6) & (cap > 0)).mean()),
            'mean_overflow': float(over.mean()),
            'overflow_p95': float(np.percentile(over, 95)),
        }
        # unseen==1 子集
        m1 = m & (cap3 == 1)
        s1 = p.sum(1)
        res[name]['cap1_cells'] = int(m1.sum())
        res[name]['cap1_violation_frac'] = float((s1[cap == 1] > 1.0 + 1e-6).mean())
        res[name]['cap1_auc_raw'] = macro_auc(y, p, m1)
        res[name]['cap1_auc_proj'] = macro_auc(y, project(p, cap), m1)

    res['best_step'] = st.get('best_step')
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
