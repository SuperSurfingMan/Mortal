"""窗口版检验：用户说的是「7s 附近的索子张」，不是单张 8s。
对每格 (花色, m)，测「同色 m±2 窗口（5 张合计）」的 counts 残差 ΔΔ，
并另测「以最后被切的那张为中心 ±2」的窗口（更贴近"7s附近"）。
格内分层 + 按立直分层。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import (CACHE, CKPT, DEV, EPS, OUT_DIR, SUIT, VAL_GLOB, boot_ci, build_eval,
                     col_ce, counts_tier, fmt_ci, forward, load_net, parse_tl,
                     suit_window)

import numpy as np

D = np.load(str(CACHE))
n = int(D['n'])
EC = D['EC'].astype(np.float64); REC = D['REC'].astype(np.float64)
CE = D['CE'].astype(np.float64)
alloc = D['alloc'].astype(np.float64); unseen = D['unseen'].astype(np.float64)
riichi = D['riichi'].astype(np.float64); turn = D['turn'].astype(np.float64)
hc = D['hc']; tidx = np.arange(34); SUIT = '万筒索'
RNG = np.random.default_rng(20261019); B = 400

rt = alloc - CE; rm = EC - CE; rr = REC - CE


def win(src, half=2):
    w = np.zeros_like(src)
    for d in range(-half, half + 1):
        s = tidx - d
        ok = (s >= 0) & (s < 27) & (tidx < 27) & (s // 9 == tidx // 9)
        w[..., ok] += src[..., s[ok]]
    return w


WT, WM, WR = win(rt), win(rm), win(rr)

recs = {'desc': [], 'asc': []}
for s in range(3):
    b = s * 9
    for mnum in range(2, 9):
        a_, mid, c_ = b + mnum - 2, b + mnum - 1, b + mnum
        for j in range(3):
            pa, pc = hc[:, j, a_], hc[:, j, c_]
            av = unseen[:, mid] > 0
            for nm, sel in (('desc', (pa < 99) & (pc < 99) & (pc < pa) & av),
                            ('asc', (pa < 99) & (pc < 99) & (pa < pc) & av)):
                ii = np.where(sel)[0]
                if len(ii):
                    recs[nm].append((ii, j, mid))

NAT = {2: 'asc', 3: 'asc', 4: 'asc', 6: 'desc', 7: 'desc', 8: 'desc'}
STRATA = {'全样本': None, '非立直': lambda i, j: riichi[i, j] < 0.5,
          '已立直': lambda i, j: riichi[i, j] > 0.5}


def agg(rec, fn, src_t, src_m, src_r):
    I, J, M = [], [], []
    for ii, j, mid in rec:
        k = ii if fn is None else ii[fn(ii, j)]
        if len(k):
            I.append(k); J.append(np.full(len(k), j)); M.append(np.full(len(k), mid))
    if not I:
        return None
    ii = np.concatenate(I); jj = np.concatenate(J); mm = np.concatenate(M)
    q = (ii, jj, mm)
    return dict(n=len(ii), ii=ii, t=src_t[q], b=src_m[q], r=src_r[q], turn=turn[ii].mean())


print('=' * 92)
print('窗口版：同色 m±2（5 张合计）的 counts 残差，格内分层 ΔΔ(非自然 − 自然)')
print('  单张 m 的量级约 0.08~0.19 张；窗口是 5 张合计，量级应放大约 3~4 倍')
print('=' * 92)
for sname, fn in STRATA.items():
    cells = []
    for s in range(3):
        for mnum, nat in NAT.items():
            mid = s * 9 + mnum - 1
            rd = [r for r in recs['desc'] if r[2] == mid]
            ra = [r for r in recs['asc'] if r[2] == mid]
            if not rd or not ra:
                continue
            x, y = agg(rd, fn, WT, WM, WR), agg(ra, fn, WT, WM, WR)
            if x is None or y is None or min(x['n'], y['n']) < 25:
                continue
            sgn = 1.0 if nat == 'asc' else -1.0
            cells.append(dict(sgn=sgn, nd=x['n'], na=y['n'], x=x, y=y,
                              t=sgn * (x['t'].mean() - y['t'].mean()),
                              b=sgn * (x['b'].mean() - y['b'].mean()),
                              r=sgn * (x['r'].mean() - y['r'].mean())))
    if not cells:
        print(f'\n【{sname}】格数不足'); continue
    w = np.array([min(c['nd'], c['na']) for c in cells], float); w /= w.sum()
    st = lambda k: float(np.sum([c[k] for c in cells] * w))
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
                f = k
                acc[k] += w[i] * c['sgn'] * (X[f][i1].mean() - Y[f][i0].mean())
        for k in bs:
            bs[k].append(acc[k])
    print(f'\n【{sname}】有效格 {len(cells)}，n外/n内 = '
          f'{sum(c["nd"] for c in cells)}/{sum(c["na"] for c in cells)}')
    for nm, k in (('真值(窗口5张)', 't'), ('A2b        ', 'b'), ('A3 顺序盲地板', 'r')):
        a = np.array(bs[k]); lo, hi = np.percentile(a, [2.5, 97.5])
        print(f'   {nm:<14} {a.mean():+.4f}  CI95[{lo:+.4f},{hi:+.4f}]  P(>0)={(a>0).mean():.3f}')
    ex = np.array(bs['b']) - np.array(bs['r'])
    lo, hi = np.percentile(ex, [2.5, 97.5])
    tt, tb, trr = st('t'), st('b'), st('r')
    print(f'   A2b 超出地板 {ex.mean():+.4f} CI95[{lo:+.4f},{hi:+.4f}] P(>0)={(ex>0).mean():.3f}'
          f'  |  复现率 A2b {tb/tt*100:+.0f}%（地板 {trr/tt*100:+.0f}%）'
          f'，扣地板后 {(tb-trr)/(tt-trr)*100:+.0f}%')

# 单张 vs 窗口 对照（全样本层，逐格）
print('\n' + '=' * 92)
print('逐格对照：真值 ΔΔ(desc−asc)，单张 m vs 同色 ±2 窗口')
print('=' * 92)
print(f"  {'格':<6}{'n外/n内':>13}{'单张m':>10}{'窗口5张':>10}{'A2b单张':>10}{'A2b窗口':>10}")
for s in range(3):
    for mnum in range(2, 9):
        mid = s * 9 + mnum - 1
        rd = [r for r in recs['desc'] if r[2] == mid]
        ra = [r for r in recs['asc'] if r[2] == mid]
        if not rd or not ra:
            continue
        x1, y1 = agg(rd, None, rt, rm, rr), agg(ra, None, rt, rm, rr)
        x2, y2 = agg(rd, None, WT, WM, WR), agg(ra, None, WT, WM, WR)
        if min(x1['n'], y1['n']) < 25:
            continue
        print(f'  {SUIT[s]}{mnum:<5}{x1["n"]:>6}/{y1["n"]:<6}'
              f'{x1["t"].mean()-y1["t"].mean():>+10.4f}{x2["t"].mean()-y2["t"].mean():>+10.4f}'
              f'{x1["b"].mean()-y1["b"].mean():>+10.4f}{x2["b"].mean()-y2["b"].mean():>+10.4f}')
