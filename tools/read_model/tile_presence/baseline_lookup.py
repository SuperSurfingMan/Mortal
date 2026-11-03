#!/usr/bin/env python3
"""根因诊断 1：公共统计查表基线（无训练）。

用 train 上的简单计数表预测 eval：
  key = (unseen, own_hand, min(tedashi,2), min(tsumogiri,2), in_river, is_decl)
逐 (target, tile) 统计 P(present | key)，加性平滑后直接当概率。
如果这个查表就能接近交叉注意力模型，说明信号几乎全在几个公开统计量里，
网络结构没有创造新信息。

用法：
  python tools/read_model/tile_presence/baseline_lookup.py \
      --cache-dir tools/read_model/tile_presence/.cache \
      --out tools/read_model/tile_presence/results/lookup.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tokens as TK  # noqa: E402
import train_presence as TP  # noqa: E402


def keys_of(z: dict, use_risk: bool) -> np.ndarray:
    T = z['tile_feat'].astype(np.float32)          # [n,3,34,FT]
    u = np.clip(np.rint(T[..., 12] * 4), 0, 4).astype(np.int64)
    own = (T[..., 13] > 0.5).astype(np.int64)
    td = np.clip(np.rint(T[..., 0] * 4), 0, 2).astype(np.int64)
    ts = np.clip(np.rint(T[..., 1] * 4), 0, 2).astype(np.int64)
    rv = (T[..., 10] > 0.5).astype(np.int64)
    dc = (T[..., 8] > 0.5).astype(np.int64)
    shape = u.shape
    key = ((((((u * 2 + own) * 3 + td) * 3 + ts) * 2 + rv) * 2 + dc))
    if use_risk:
        risk = np.clip(np.rint(T[..., 19] * 20), 0, 3).astype(np.int64)
        key = key * 4 + risk
    return key.reshape(-1), shape


def fit_apply(k_tr, y_tr, k_ev, n_key: int, alpha: float = 5.0) -> np.ndarray:
    pos = np.bincount(k_tr, weights=y_tr, minlength=n_key)
    cnt = np.bincount(k_tr, minlength=n_key)
    rate = (pos + alpha * 0.3) / (cnt + alpha)
    return rate[k_ev].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--diag-cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    args = ap.parse_args()

    tr = np.load(Path(args.cache_dir) / 'train.npz')
    ev = np.load(Path(args.cache_dir) / 'eval.npz')
    y_tr = tr['y'].astype(np.float32).reshape(-1)
    y_ev = ev['y'].astype(np.float32)
    un_ev = ev['unseen'].astype(np.float32)
    masks = TP._masks(un_ev, ev['turn'], ev['riichi'], ev['called'])
    m = masks['all']

    out = {}
    for use_risk, name in ((False, 'lookup'), (True, 'lookup+risk')):
        k_tr, _ = keys_of(tr, use_risk)
        k_ev, shape = keys_of(ev, use_risk)
        n_key = int(max(k_tr.max(), k_ev.max())) + 1
        p = fit_apply(k_tr, y_tr, k_ev, n_key).reshape(shape)
        vals = []
        for j in range(3):
            for t in range(TK.NUM_TILES):
                mm = m[:, j, t]
                if mm.sum() < 10:
                    continue
                v = TP._rank_auc(y_ev[mm, j, t], p[mm, j, t])
                if np.isfinite(v):
                    vals.append(v)
        out[name] = float(np.mean(vals))
        print(f'[{name}] all macro AUC {out[name]:.4f}  (keys={n_key})', flush=True)

    # unseen-only 参照
    T = ev['tile_feat'].astype(np.float32)
    u = np.clip(np.rint(T[..., 12] * 4), 0, 4).astype(np.int64)
    Tr = tr['tile_feat'].astype(np.float32)
    ur = np.clip(np.rint(Tr[..., 12] * 4), 0, 4).astype(np.int64)
    y_tr3 = tr['y'].astype(np.float32)
    pos = np.zeros(5)
    cnt = np.zeros(5)
    for k in range(5):
        sel = ur == k
        pos[k] = y_tr3[sel].sum()
        cnt[k] = sel.sum()
    rate = (pos + 1.5) / (cnt + 5.0)
    p_u = rate[u]
    vals = []
    for j in range(3):
        for t in range(TK.NUM_TILES):
            mm = m[:, j, t]
            if mm.sum() < 10:
                continue
            v = TP._rank_auc(y_ev[mm, j, t], p_u[mm, j, t])
            if np.isfinite(v):
                vals.append(v)
    out['unseen_only'] = float(np.mean(vals))
    print(f"[unseen_only] all macro AUC {out['unseen_only']:.4f}", flush=True)

    ref = np.load(args.diag_cache, allow_pickle=True)
    p_a = 1.0 - ref['AP'][: y_ev.shape[0], :, :, 0]
    vals = []
    for j in range(3):
        for t in range(TK.NUM_TILES):
            mm = m[:, j, t]
            if mm.sum() < 10:
                continue
            v = TP._rank_auc(y_ev[mm, j, t], p_a[mm, j, t])
            if np.isfinite(v):
                vals.append(v)
    out['a2c'] = float(np.mean(vals))
    print(f"[a2c] all macro AUC {out['a2c']:.4f}", flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
