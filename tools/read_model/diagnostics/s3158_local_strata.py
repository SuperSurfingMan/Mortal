"""Phase 2b：混杂控制 + 反事实逐样本期望回归。

⑤b 格内分层 ΔΔ 分别在全样本 / 非立直 / 立直 三层重算。
    顺序盲的 A3 在同层的 ΔΔ = 剩余混杂地板；A2b 超出 A3 的部分 = 真正的顺序利用。
A'b 反事实不再用「池化真值 ΔΔ」当期望（池化值≈0 使比例失真），改为逐样本
    期望 = ∓ΔΔ_cell，做 measured vs expected 的回归斜率与解释方差比。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import (CACHE, CKPT, DEV, EPS, OUT_DIR, SUIT, VAL_GLOB, boot_ci, build_eval,
                     col_ce, counts_tier, fmt_ci, forward, load_net, parse_tl,
                     suit_window)

import numpy as np, json

D = np.load(str(CACHE))
n = int(D['n'])
AP = D['AP'].astype(np.float64); EC = D['EC'].astype(np.float64)
EC2 = D['EC2'].astype(np.float64)
RAP = D['RAP'].astype(np.float64); REC = D['REC'].astype(np.float64)
CE = D['CE'].astype(np.float64)
alloc = D['alloc'].astype(np.float64); unseen = D['unseen'].astype(np.float64)
turn = D['turn'].astype(np.float64); riichi = D['riichi'].astype(np.float64)
hc = D['hc']; tg = D['tg']
SUIT = '万筒索'
RNG = np.random.default_rng(20261019)
B = 400
out = {}

resid_truth = alloc - CE
resid_a2b = EC - CE
resid_a3 = REC - CE

recs = {'desc': [], 'asc': []}
cell_of = {}                       # (s, mnum) -> mid
for s in range(3):
    b = s * 9
    for mnum in range(2, 9):
        a_, mid, c_ = b + mnum - 2, b + mnum - 1, b + mnum
        cell_of[(s, mnum)] = mid
        for j in range(3):
            pa, pc = hc[:, j, a_], hc[:, j, c_]
            av = unseen[:, mid] > 0
            hh, hcc = pa < 99, pc < 99
            for nm, sel in (('desc', hh & hcc & (pc < pa) & av),
                            ('asc', hh & hcc & (pa < pc) & av)):
                ii = np.where(sel)[0]
                if len(ii):
                    recs[nm].append((ii, j, mid, s, mnum))


def agg(rec, mask_fn=None):
    """mask_fn(ii, jj) -> bool 保留哪些 (样本, 目标家)"""
    I, J, M = [], [], []
    for ii, j, mid, s, mnum in rec:
        k = ii if mask_fn is None else ii[mask_fn(ii, j)]
        if len(k):
            I.append(k); J.append(np.full(len(k), j)); M.append(np.full(len(k), mid))
    if not I:
        return None
    ii = np.concatenate(I); jj = np.concatenate(J); mm = np.concatenate(M)
    q = (ii, jj, mm); q2 = (ii, jj)
    return dict(n=len(ii), turn=turn[ii].mean(), ii=ii, q=q, q2=q2,
                tr=resid_truth[q], b=resid_a2b[q], r=resid_a3[q],
                raw_tr=alloc[q], raw_cnt=CE[q])


NAT = {2: 'asc', 3: 'asc', 4: 'asc', 6: 'desc', 7: 'desc', 8: 'desc'}
STRATA = {
    '全样本': None,
    '目标家非立直': lambda ii, j: riichi[ii, j] < 0.5,
    '目标家已立直': lambda ii, j: riichi[ii, j] > 0.5,
}

print('=' * 88)
print('⑤b  格内分层 ΔΔ(非自然 − 自然)，按目标家立直状态分层')
print('     每格两组的舍牌集合完全相同 ⇒ 顺序盲预测器的格内 ΔΔ 只来自巡目/立直等组成差异，')
print('     即「混杂地板」。A2b 超出地板的部分才是真正用到了顺序。')
print('=' * 88)
for sname, fn in STRATA.items():
    cells = []
    for s in range(3):
        for mnum, nat in NAT.items():
            mid = cell_of[(s, mnum)]
            rd = [r for r in recs['desc'] if r[2] == mid]
            ra = [r for r in recs['asc'] if r[2] == mid]
            if not rd or not ra:
                continue
            x, y = agg(rd, fn), agg(ra, fn)
            if x is None or y is None or min(x['n'], y['n']) < 25:
                continue
            sgn = 1.0 if nat == 'asc' else -1.0
            cells.append(dict(s=s, m=mnum, nd=x['n'], na=y['n'], sgn=sgn,
                              td=x['turn'], ta=y['turn'], x=x, y=y,
                              t=sgn * (x['tr'].mean() - y['tr'].mean()),
                              b=sgn * (x['b'].mean() - y['b'].mean()),
                              r=sgn * (x['r'].mean() - y['r'].mean())))
    if not cells:
        print(f'\n【{sname}】格数不足'); continue
    w = np.array([min(c['nd'], c['na']) for c in cells], float); w /= w.sum()
    st = lambda k: float(np.sum([c[k] for c in cells] * w))
    tt, tb, tr = st('t'), st('b'), st('r')
    print(f'\n【{sname}】有效格 {len(cells)}，n外/n内 = '
          f'{sum(c["nd"] for c in cells)}/{sum(c["na"] for c in cells)}')
    print(f'   巡目平衡：外侧先 mean {np.mean([c["td"] for c in cells]):.2f} vs '
          f'内侧先 {np.mean([c["ta"] for c in cells]):.2f}')
    # bootstrap
    ub = [(np.unique(c['x']['ii']), np.unique(c['y']['ii'])) for c in cells]
    mx = [{u: np.where(c['x']['ii'] == u)[0] for u in ub[i][0]} for i, c in enumerate(cells)]
    my = [{u: np.where(c['y']['ii'] == u)[0] for u in ub[i][1]} for i, c in enumerate(cells)]
    bs = {'t': [], 'b': [], 'r': []}
    for _ in range(B):
        acc = {'t': 0.0, 'b': 0.0, 'r': 0.0}
        for i, c in enumerate(cells):
            X, Y = c['x'], c['y']
            i1 = np.concatenate([mx[i][u] for u in RNG.choice(ub[i][0], len(ub[i][0]))])
            i0 = np.concatenate([my[i][u] for u in RNG.choice(ub[i][1], len(ub[i][1]))])
            for k in bs:
                f = 'tr' if k == 't' else k
                acc[k] += w[i] * c['sgn'] * (X[f][i1].mean() - Y[f][i0].mean())
        for k in bs:
            bs[k].append(acc[k])
    for nm, k in (('真值', 't'), ('A2b ', 'b'), ('A3 顺序盲对照', 'r')):
        a = np.array(bs[k]); lo, hi = np.percentile(a, [2.5, 97.5])
        print(f'   {nm:<16} {a.mean():+.4f}  CI95[{lo:+.4f},{hi:+.4f}]  '
              f'P(>0)={(a>0).mean():.3f}')
    ex = tb - tr
    a = np.array(bs['b']) - np.array(bs['r'])
    lo, hi = np.percentile(a, [2.5, 97.5])
    print(f'   ── A2b 超出顺序盲地板 {ex:+.4f}  CI95[{lo:+.4f},{hi:+.4f}]  '
          f'P(>0)={(a>0).mean():.3f}   ← 真正由顺序贡献的部分')
    print(f'   ── 复现率：A2b {tb/tt*100:+.0f}%（地板 {tr/tt*100:+.0f}%）；'
          f'扣除地板后 {(tb-tr)/(tt-tr)*100:+.0f}%')
    out[sname] = dict(ncell=len(cells), truth=tt, a2b=tb, a3=tr,
                      excess=float(ex), excess_ci=[float(lo), float(hi)],
                      bs_truth=[float(np.mean(bs['t'])), float(np.percentile(bs['t'], 2.5)),
                                float(np.percentile(bs['t'], 97.5))],
                      bs_a2b=[float(np.mean(bs['b'])), float(np.percentile(bs['b'], 2.5)),
                              float(np.percentile(bs['b'], 97.5))],
                      bs_a3=[float(np.mean(bs['r'])), float(np.percentile(bs['r'], 2.5)),
                             float(np.percentile(bs['r'], 97.5))])

# ================= A'b：反事实逐样本期望回归 =================
print('\n' + '=' * 88)
print("A'b  配对反事实（obs 不变，只对调 tl_tokens 两 token）的逐样本期望回归")
print('=' * 88)
ids, Jarr, Mids = D['ids'], D['Jarr'], D['Mids']
q0 = (ids, Jarr, Mids)
pa = hc[ids, Jarr, Mids - 1]; pc = hc[ids, Jarr, Mids + 1]
is_desc = pc < pa
Sarr = Mids // 9; Marr = Mids % 9 + 1
# 逐格真值 ΔΔ(desc − asc)，非立直层（避免立直机制混入）
nr = lambda ii, j: riichi[ii, j] < 0.5
dd_cell = {}
for s in range(3):
    for mnum in range(2, 9):
        mid = cell_of[(s, mnum)]
        rd = [r for r in recs['desc'] if r[2] == mid]
        ra = [r for r in recs['asc'] if r[2] == mid]
        if not rd or not ra:
            continue
        x, y = agg(rd, nr), agg(ra, nr)
        if x is None or y is None or min(x['n'], y['n']) < 25:
            continue
        dd_cell[(s, mnum)] = x['tr'].mean() - y['tr'].mean()
print(f'  有 ΔΔ 估计的格：{len(dd_cell)}/21')
exp_d = np.array([(-1 if is_desc[k] else +1) *
                  dd_cell.get((Sarr[k], Marr[k]), np.nan) for k in range(len(ids))])
meas = EC2[q0] - EC[q0]
okm = ~np.isnan(exp_d)
e, m_ = exp_d[okm], meas[okm]
print(f'  可用反事实样本 {okm.sum()}（其中非立直层）')
if len(dd_cell) == 0 or okm.sum() < 50:
    # n_eval 太小 ⇒ 每格 (desc, asc) 不足 25，逐格 ΔΔ 全部被丢弃。
    # 此时回归无意义，直接跳过而不是吐 nan（默认 n_eval=40000 不会走到这里）。
    print('  ⚠️ 样本量不足：无格可估 ΔΔ，跳过反事实回归。请用默认 n_eval=40000 '
          '（或至少 ≥20000）重跑 s3158_local_extract.py。')
    out['Ap2'] = dict(n=int(okm.sum()), skipped='insufficient_cells',
                      ncell=len(dd_cell))
    with open(str(OUT_DIR / 'local_order_report2.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    raise SystemExit(0)
print(f'  expected Δ（若模型完美）: mean {e.mean():+.5f} sd {e.std():.5f}')
print(f'  measured Δ（A2b 实际）  : mean {m_.mean():+.5f} sd {m_.std():.5f}')
slope = float(np.dot(e - e.mean(), m_ - m_.mean()) / max(np.dot(e - e.mean(), e - e.mean()), 1e-12))
r2 = float(np.dot(e - e.mean(), m_ - m_.mean()) ** 2 /
           max(np.dot(e - e.mean(), e - e.mean()) * np.dot(m_ - m_.mean(), m_ - m_.mean()), 1e-12))
corr = float(np.corrcoef(e, m_)[0, 1])
print(f'  回归 measured = a + b·expected：b = {slope:+.4f}（b=1 为完美，b=0 为无响应）')
print(f'  Pearson r = {corr:+.4f}   r² = {r2:.5f}   '
      f'（顺序信号解释的方差比例 {r2*100:.3f}%）')
# 分组均值
for nm, sel in (('外侧先→内侧先', is_desc[okm]), ('内侧先→外侧先', ~is_desc[okm])):
    print(f'    {nm}(n={sel.sum()}): measured {m_[sel].mean():+.5f} '
          f'expected {e[sel].mean():+.5f}  比值 '
          f'{m_[sel].mean()/e[sel].mean()*100 if abs(e[sel].mean())>1e-9 else float("nan"):+.1f}%')
# 噪声底：对调两个「同花色但非 m±1」的 token 不应改变 exp_counts[m]
print(f'  量级参照：|measured| mean {np.abs(m_).mean():.5f}；'
      f'|expected| mean {np.abs(e).mean():.5f}')
out['Ap2'] = dict(n=int(okm.sum()), slope=slope, corr=corr, r2=r2,
                  exp_mean=float(e.mean()), meas_mean=float(m_.mean()),
                  exp_sd=float(e.std()), meas_sd=float(m_.std()),
                  dd_cell={f'{SUIT[s]}{m}': float(v) for (s, m), v in dd_cell.items()})

with open(str(OUT_DIR / 'local_order_report2.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=float)
print(f'\n→ {OUT_DIR / "local_order_report2.json"}')
