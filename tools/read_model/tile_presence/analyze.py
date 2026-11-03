#!/usr/bin/env python3
"""诊断：逐牌 AUC、与 A2c 的互补性（混合/秩平均）。

用法：
  python tools/read_model/tile_presence/analyze.py \
      --cache-dir tools/read_model/tile_presence/.cache \
      --ckpt tools/read_model/tile_presence/results/presence.pth \
      --out tools/read_model/tile_presence/results/analyze.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--diag-cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = st['args']
    net = TilePresenceNet(d=a['d'], n_blocks=a['blocks'], heads=a['heads'],
                          dropout=a['dropout']).to(TP.DEV)
    net.load_state_dict(st['state_dict'])
    net.eval()
    print(f"[ckpt] step={st.get('best_step')} val_auc={st.get('best_val_auc'):.4f}",
          flush=True)

    ev_path = Path(args.cache_dir) / 'eval.npz'
    ds = TP.PresDataset(ev_path)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
    out = TP.predict(net, loader)
    y, p_m = out['y'], out['p']
    meta = TP._meta(ev_path)
    masks = TP._masks(meta['unseen'], meta['turn'], meta['riichi'], meta['called'])
    m = masks['all']

    ref = np.load(args.diag_cache, allow_pickle=True)
    p_a = (1.0 - ref['AP'][:y.shape[0], :, :, 0].astype(np.float32))
    assert np.array_equal(ref['alloc'][:y.shape[0]].astype(np.uint8),
                          np.load(ev_path)['alloc']), 'eval 顺序不一致'

    def macro(p):
        vals = []
        for j in range(3):
            for t in range(TK.NUM_TILES):
                mm = m[:, j, t]
                if mm.sum() < 10:
                    continue
                v = TP._rank_auc(y[mm, j, t], p[mm, j, t])
                if np.isfinite(v):
                    vals.append(v)
        return float(np.mean(vals)), len(vals)

    blends = {}
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        blends[f'w_a2c={w:.2f}'] = macro(w * p_a + (1 - w) * p_m)[0]

    # 秩平均（逐单元在样本维上转秩后混合，对尺度不敏感）
    def rank01(x):
        out = np.empty(x.shape, dtype=np.float32)
        for j in range(3):
            for t in range(TK.NUM_TILES):
                col = x[:, j, t]
                _, inv, cnt = np.unique(col, return_inverse=True,
                                        return_counts=True)
                cum = np.concatenate([[0.0], np.cumsum(cnt)])
                avg = (cum[:-1] + cum[1:] + 1.0) / 2.0
                out[:, j, t] = (avg[inv] / len(col)).astype(np.float32)
        return out

    r_m, r_a = rank01(p_m), rank01(p_a)
    for w in (0.25, 0.5, 0.75):
        blends[f'rank w_a2c={w:.2f}'] = macro(w * r_a + (1 - w) * r_m)[0]

    per_tile = {'model': [], 'a2c': []}
    for j in range(3):
        rm, ra = [], []
        for t in range(TK.NUM_TILES):
            mm = m[:, j, t]
            rm.append(TP._rank_auc(y[mm, j, t], p_m[mm, j, t]) if mm.sum() >= 10 else np.nan)
            ra.append(TP._rank_auc(y[mm, j, t], p_a[mm, j, t]) if mm.sum() >= 10 else np.nan)
        per_tile['model'].append(rm)
        per_tile['a2c'].append(ra)

    tm = np.nanmean(np.array(per_tile['model']), axis=0)
    ta = np.nanmean(np.array(per_tile['a2c']), axis=0)
    order = np.argsort(tm - ta)
    print('tiles worst vs A2c:', [(int(i), round(float(tm[i] - ta[i]), 4))
                                  for i in order[:8]], flush=True)
    print('tiles best vs A2c:', [(int(i), round(float(tm[i] - ta[i]), 4))
                                 for i in order[-8:]], flush=True)
    print('blends:', {k: round(v, 4) for k, v in blends.items()}, flush=True)

    res = {
        'ckpt': args.ckpt,
        'best_step': st.get('best_step'),
        'val_auc': st.get('best_val_auc'),
        'auc_model': macro(p_m)[0],
        'auc_a2c': macro(p_a)[0],
        'blends': blends,
        'per_tile_model': per_tile['model'],
        'per_tile_a2c': per_tile['a2c'],
        'per_tile_gap': (tm - ta).tolist(),
        'tenpai_aux_auc': TP._rank_auc(meta['tenpai'].reshape(-1),
                                       out['tp'].reshape(-1)),
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f'[out] {args.out}', flush=True)


if __name__ == '__main__':
    main()
