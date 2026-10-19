"""wait_ce_active 的「听牌因子 / 条件分布」分解：A2b vs A3-rule。

问题：wait_ce_active = -Σ q log p，p 是 35 维 softmax（含 P(未听)）。若某模型的
P(听牌) 在真听牌目标上系统性过自信，其 wait_ce_active 会被「-log P(t) 项」白白
美化，跨模型比较就不成立。分解后可比的是条件项 cond_ce。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import (CKPT, DEV, EPS, OUT_DIR, SUIT, VAL_GLOB, boot_ci, build_eval,
                     col_ce, counts_tier, fmt_ci, forward, load_net, parse_tl,
                     suit_window)

import sys, glob
import numpy as np

import train as T
import rule_baseline as RB

files = sorted(glob.glob(VAL_GLOB,
                         recursive=True))
N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
b = T.build_fixed_eval_set(files, N_EVAL)
bn = {k: v.numpy() for k, v in b.items()}
rule = RB.rule_predict(bn)['wait_probs'].astype(np.float64)

# A2b 的 wait_probs 已在 probe 中前向过；此处直接重算太贵，改用 ref：从 npy 读不到，
# 因此只对 rule 做分解，A2b 侧用已测数字（-logP(t)=0.0037 / cond=2.8961）。
WT = bn['wait'].astype(np.float64)
ten = bn['tenpai'] > 0.5
ri = bn['riichi'] > 0.5
sup = WT[..., :34] > 0
eff = sup.any(-1)

for name, p in (('A3-rule', rule),):
    p_not = p[..., 34]
    p_t = 1 - p_not
    cond = p[..., :34] / np.clip(p_t[..., None], 1e-12, None)
    ll = -(WT * np.where(WT > 0, np.log(np.clip(p, 1e-12, None)), 0)).sum(-1)
    ll_c = -(WT[..., :34] * np.where(WT[..., :34] > 0,
                                     np.log(np.clip(cond, 1e-12, None)), 0)).sum(-1)
    nlt = -np.log(np.clip(p_t, 1e-12, None))
    for tag, sel in (('all-active', eff), ('riichi', ri & eff), ('non-riichi', ~ri & eff)):
        print(f'{name} {tag:<12s} n={sel.sum():>6d}  CE={ll[sel].mean():.4f} '
              f'= -logP(t) {nlt[sel].mean():.4f} + cond_ce {ll_c[sel].mean():.4f} '
              f'| meanP(t)={p_t[sel].mean():.4f}')
print('\nA2b 已测（diag1）：all-active CE=2.8999 = -logP(t) 0.0037 + cond 2.8961，'
      'meanP(t)=0.9963')
print('A2b 若改用标定后的 P(t)（sigmoid 口径，听牌目标 mean P(未听)=0.2757）：')
print(f'  -logP(t) 项将变为 {-np.log(1-0.2757):.4f} nats，'
      f'标定口径 wait_ce_active ≈ {2.8961 - np.log(1-0.2757):.4f}')
