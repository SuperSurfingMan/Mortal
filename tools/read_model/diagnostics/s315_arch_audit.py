"""A2b 核实性诊断（一次性，不入仓）：
1) wait CE 分解：-log P(t) 与条件 CE，量化「过标定是否美化了 wait_ce_active」
2) 标签熵 mean log|W|（可达下界参照）
3) alloc 行和（手牌数）违例的均值/分位数 + IPF 前后 MAE
4) 模型是否已知 hand_size：预测行和 vs 真 hand_size 的相关
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import (CKPT, DEV, EPS, OUT_DIR, SUIT, VAL_GLOB, boot_ci, build_eval,
                     col_ce, counts_tier, fmt_ci, forward, load_net, parse_tl,
                     suit_window)

import sys, glob, json
from pathlib import Path
import numpy as np
import torch


import train as T
from constraints import apply_constraints

device = 'cuda'
state = torch.load(str(CKPT),
                   map_location='cpu', weights_only=False)
cfg = state['config']
import model_v2 as M2
net = M2.build_model_v2s(conv_channels=cfg['conv_channels'],
                         num_blocks=cfg['num_blocks'],
                         with_aux=not cfg['no_aux'],
                         wait_head=cfg.get('wait_head', 'flat')).to(device).eval()
net.load_state_dict(state['model'])

files = sorted(glob.glob(VAL_GLOB,
                         recursive=True))
batch = T.build_fixed_eval_set(files, 20000)
n = batch['obs'].shape[0]
print(f'[data] n={n}', flush=True)

AP, WP = [], []
with torch.no_grad():
    for i in range(0, n, 2048):
        ch = {k: v[i:i + 2048].to(device) for k, v in batch.items()}
        wb = T.wait_bias_of(ch).to(device)
        o = net(ch['obs'], ch['unseen'], wb, ch['tl_tokens'], ch['tl_mask'])
        AP.append(o['alloc_probs'].cpu().numpy())
        WP.append(o['wait_probs'].cpu().numpy())
ap = np.concatenate(AP).astype(np.float64)     # [n,3,34,5]
wp = np.concatenate(WP).astype(np.float64)     # [n,3,35]

H = batch['alloc'].numpy().astype(np.int64)
WT = batch['wait'].numpy().astype(np.float64)
tenpai = batch['tenpai'].numpy() > 0.5
riichi = batch['riichi'].numpy() > 0.5
unseen = batch['unseen'].numpy()
sup = WT[:, :, :34] > 0
eff = sup.any(-1)
wcnt = sup.sum(-1)

# ---- 1) 标签熵（可达下界：完美识别待牌集时的 CE）----
m = eff
label_H = float(np.log(wcnt[m]).mean())
print(f'\n[wait] 有效目标 n={m.sum()}  mean|W|={wcnt[m].mean():.3f}  '
      f'标签熵 mean log|W| = {label_H:.4f} nats（完美识别待牌集的下界）')

# ---- 2) wait CE 分解 ----
p_not = wp[:, :, 34]
p_t = 1.0 - p_not
cond = wp[:, :, :34] / np.clip(p_t[..., None], 1e-12, None)
ll = -(WT * np.where(WT > 0, np.log(np.clip(wp, 1e-12, None)), 0)).sum(-1)
ll_cond = -(WT[:, :, :34] * np.where(WT[:, :, :34] > 0,
                                     np.log(np.clip(cond, 1e-12, None)), 0)).sum(-1)
neg_log_pt = -np.log(np.clip(p_t, 1e-12, None))
for name, sel in (('all-active', eff), ('tenpai&eff', tenpai & eff),
                  ('riichi', riichi & eff), ('non-riichi&eff', ~riichi & eff)):
    print(f'  {name:<16s} n={sel.sum():>6d}  CE={ll[sel].mean():.4f}  '
          f'= -logP(t) {neg_log_pt[sel].mean():.4f} + cond {ll_cond[sel].mean():.4f}  '
          f'| meanP(t)={p_t[sel].mean():.4f}')

# 标定
print(f'\n[calib] 全目标 mean P(t)={p_t.mean():.4f} vs 实际 tenpai 率={tenpai.mean():.4f} '
      f'({p_t.mean()/tenpai.mean():.2f}x)')
for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
    b = (p_t >= lo) & (p_t < hi)
    if b.sum():
        print(f'  P(t)∈[{lo:.1f},{hi:.1f}) n={b.sum():>6d}  实际听牌率={tenpai[b].mean():.4f}')
# 非听牌目标上的 P(t)
print(f'  非听牌目标 mean P(t)={p_t[~tenpai].mean():.4f} 中位={np.median(p_t[~tenpai]):.4f}')
print(f'  听牌目标   mean P(t)={p_t[tenpai].mean():.4f} 中位={np.median(p_t[tenpai]):.4f}')

# ---- 3) alloc 行和 / IPF ----
k = np.arange(5)
e = (ap * k).sum(-1)                       # [n,3,34]
hand = H.sum(-1)                           # [n,3] 真手牌数
pred_sum = e.sum(-1)
row_err = pred_sum - hand
print(f'\n[alloc] 真 hand_size 分布 {np.bincount(hand.ravel())[:15].tolist()}')
print(f'  预测行和 - 真行和: mean={row_err.mean():+.4f}  mean|.|={np.abs(row_err).mean():.4f}'
      f'  p50={np.percentile(np.abs(row_err),50):.3f} p90={np.percentile(np.abs(row_err),90):.3f}'
      f'  max={np.abs(row_err).max():.3f}')
print(f'  corr(pred_sum, hand)={np.corrcoef(pred_sum.ravel(), hand.ravel())[0,1]:.4f}')
print(f'  pred_sum 均值={pred_sum.mean():.3f} vs hand 均值={hand.mean():.3f}')
colv = np.maximum(e.sum(1) - np.maximum(unseen, 0), 0)
print(f'  列违例 mean={colv.mean():.5f} max={colv.max():.3f}')

mae0 = float(np.abs(e - H).mean())
out = apply_constraints(ap.astype(np.float32), unseen, hand.astype(np.float64))
e2 = out['exp_counts']
mae1 = float(np.abs(e2 - H).mean())
print(f'  MAE 原始={mae0:.4f} -> IPF后={mae1:.4f}  (Δ={mae1-mae0:+.4f})')
# 逐分层 IPF 收益
for name, sel in (('riichi', riichi), ('called', batch['called'].numpy() > 0.5), ('all', np.ones_like(tenpai))):
    a = np.abs(e - H)[sel].mean(); b = np.abs(e2 - H)[sel].mean()
    print(f'    {name:<8s} MAE {a:.4f} -> {b:.4f}')

# 只用行和缩放（不投影列）能拿多少
scale = hand[..., None] / np.clip(pred_sum[..., None], 1e-9, None)
e3 = e * scale
print(f'  仅行和缩放 MAE={np.abs(e3-H).mean():.4f}')

# ---- 4) exp_counts 作为「持有点估计」的质量 ----
p_true = ap[np.arange(n)[:, None, None], np.arange(3)[None, :, None],
          np.arange(34)[None, None, :], H]
ce = float(-np.log(np.clip(p_true, 1e-12, None)).mean())
print(f'\n[alloc] CE={ce:.4f}  top1(argmax_k==真k)={(ap.argmax(-1)==H).mean():.4f}  '
      f'众数基线={(np.bincount(H.ravel(),minlength=5).max()/H.size):.4f}')
held = H > 0
print(f'  argmax exp_count 命中持有率={held[np.arange(n)[:,None],np.arange(3)[None,:],e.argmax(-1)].mean():.4f}')
json.dump({'label_H': label_H, 'ce': ce, 'mae0': mae0, 'mae1': mae1},
          open(str(OUT_DIR / 'diag1.json'), 'w'), indent=1)
