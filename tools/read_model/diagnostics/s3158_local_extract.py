"""§3.15.8 Phase 1（贵，~2.5 min）：局部手牌信息探针的取数阶段。

产出 `CACHE`（默认 /tmp/read_model_diag_cache/cache.npz，约 120 MB，不入库），
内含 40000 个 202609 holdout 样本上的四套量：
  AP/EC   A2b 的 alloc_probs[n,3,34,5] 与 exp_counts[n,3,34]
  EC2     配对反事实（只对调 tl_tokens 里 m−1/m+1 两个 token，obs 不变）后的 exp_counts
  RAP/REC A3 规则法的对应量（顺序盲对照）
  CE/CPMF0 counts 精确档（超几何，行为盲对照地板）
  alloc/unseen/hand/turn/riichi/called/tenpai/hc/tg/ids/Jarr/Mids

用法：  python s3158_local_extract.py [n_eval]     # 默认 40000
后续：  s3158_local_report.py / _strata.py / _window.py 直接读 CACHE，无需 GPU。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import CACHE, build_eval, counts_tier, forward, load_net, parse_tl

import sys
import time

import numpy as np

import rule_baseline as RB

OUT = str(CACHE)
N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 40000

t0 = time.time()
batch, bn, n_files = build_eval(N_EVAL)
n = bn['obs'].shape[0]
print(f'[data] holdout 文件 {n_files} → n={n}  {time.time()-t0:.0f}s', flush=True)

net = load_net()
keys = ('obs', 'unseen', 'tl_tokens', 'tl_mask')
AP, EC = forward(net, batch, keys)
rule = RB.rule_predict(bn)
RAP = np.asarray(rule['alloc_probs'], np.float64)
REC = np.asarray(rule['exp_counts'], np.float64)
print(f'[model] A2b+A3 {time.time()-t0:.0f}s', flush=True)

unseen = np.asarray(bn['unseen'], np.float64)
hand = 13 - 3 * np.asarray(bn['meld_counts'], np.float64)
CE, CPMF = counts_tier(unseen, hand)
hc, tg = parse_tl(bn['tl_tokens'], bn['tl_mask'])

# 反事实：每样本取一个 (j, mid)，把 tl_tokens 里 m-1 / m+1 两个 token 对调
one = {}
for s in range(3):
    b = s * 9
    for mnum in range(2, 9):
        a_, mid, c_ = b + mnum - 2, b + mnum - 1, b + mnum
        for j in range(3):
            pa, pc = hc[:, j, a_], hc[:, j, c_]
            both = (pa < 99) & (pc < 99) & (unseen[:, mid] > 0)
            for i in np.where(both)[0]:
                if i not in one:
                    one[i] = (j, mid, int(pa[i]), int(pc[i]))
ids = np.array(sorted(one)) if one else np.zeros(0, np.int64)
print(f"[cf] 反事实样本 {len(ids)}  {time.time()-t0:.0f}s", flush=True)
tl2 = batch['tl_tokens'].clone()
Jarr = np.array([one[i][0] for i in ids], np.int64)
Mids = np.array([one[i][1] for i in ids], np.int64)
Parr = np.array([one[i][2] for i in ids], np.int64)
Qarr = np.array([one[i][3] for i in ids], np.int64)
if len(ids):
    tmp = tl2[ids, Jarr, Parr].clone()
    tl2[ids, Jarr, Parr] = tl2[ids, Jarr, Qarr]
    tl2[ids, Jarr, Qarr] = tmp
    _, EC2 = forward(net, batch, keys, tl_override=tl2)
else:
    EC2 = EC.copy()

np.savez_compressed(
    OUT, n=n,
    AP=AP.astype(np.float32), EC=EC.astype(np.float32), EC2=EC2.astype(np.float32),
    RAP=RAP.astype(np.float32), REC=REC.astype(np.float32),
    CE=CE.astype(np.float32), CPMF0=CPMF[..., 0].astype(np.float32),
    alloc=np.asarray(bn['alloc'], np.int8), unseen=unseen.astype(np.float32),
    hand=hand.astype(np.int8), turn=np.asarray(bn['turn'], np.float32),
    riichi=np.asarray(bn['riichi'], np.int8), called=np.asarray(bn['called'], np.int8),
    hc=hc, tg=tg, ids=ids, Jarr=Jarr, Mids=Mids,
    tenpai=np.asarray(bn['tenpai'], np.int8))
print(f'[done] → {OUT}  {time.time()-t0:.0f}s')
