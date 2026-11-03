#!/usr/bin/env python3
"""根因诊断 3：手内排序（within-hand）AUC。

跨样本 AUC（逐 (target,tile)）衡量"不同局面下这张牌在不在"的区分度；
但下游防守更关心**同一手牌内哪几张更可能有**。这里对每个 (样本, 目标家)
在 34 张牌上算一次 AUC（informative 牌），再平均，比较模型与 A2c。

用法：
  python tools/read_model/tile_presence/analyze_within.py \
      --cache-dir tools/read_model/tile_presence/.cache120 \
      --ckpt tools/read_model/tile_presence/results/run3_d224_120k/presence.pth \
      --out tools/read_model/tile_presence/results/run3_d224_120k/within.json
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
import train_presence as TP  # noqa: E402
from model import TilePresenceNet  # noqa: E402


def within_auc(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> float:
    """y/p/m [N,34] 同一目标的 34 张牌；逐样本 AUC 后平均。"""
    vals = []
    for i in range(y.shape[0]):
        mm = m[i]
        yy = y[i][mm]
        if yy.min() == yy.max():
            continue
        pp = p[i][mm]
        _, inv, cnt = np.unique(pp, return_inverse=True, return_counts=True)
        cum = np.concatenate([[0.0], np.cumsum(cnt)])
        avg = (cum[:-1] + cum[1:] + 1.0) / 2.0
        r = avg[inv]
        npos = yy.sum()
        nneg = len(yy) - npos
        vals.append((r[yy > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(vals)), len(vals)


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
    ds = TP.PresDataset(ev_path)
    out = TP.predict(net, DataLoader(ds, batch_size=args.batch, shuffle=False))
    y, p_m = out['y'], out['p']
    meta = TP._meta(ev_path)
    m = (meta['unseen'] > 0)                     # [N,34]
    ref = np.load('/tmp/read_model_diag_cache_a2c/cache.npz', allow_pickle=True)
    p_a = (1.0 - ref['AP'][:y.shape[0], :, :, 0].astype(np.float32))

    res = {}
    for j in range(3):
        res[f'model_t{j}'] = within_auc(y[:, j], p_m[:, j], m)
        res[f'a2c_t{j}'] = within_auc(y[:, j], p_a[:, j], m)
    res['model_all'] = float(np.mean([res[f'model_t{j}'][0] for j in range(3)]))
    res['a2c_all'] = float(np.mean([res[f'a2c_t{j}'][0] for j in range(3)]))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
