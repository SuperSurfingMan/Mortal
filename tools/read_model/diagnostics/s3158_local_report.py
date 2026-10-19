"""Phase 2（便宜）：从 cache.npz 出全部报告。局部手牌信息视角。"""

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
CE = D['CE'].astype(np.float64); CPMF0 = D['CPMF0'].astype(np.float64)
alloc = D['alloc'].astype(np.float64); unseen = D['unseen'].astype(np.float64)
hand = D['hand'].astype(np.float64); turn = D['turn'].astype(np.float64)
riichi = D['riichi'].astype(np.float64); called = D['called'].astype(np.float64)
hc = D['hc']; tg = D['tg']; tidx = np.arange(34)
EPS = 1e-12
SUIT = '万筒索'
RNG = np.random.default_rng(20261019)
out = {'n': n}
print(f'n = {n}（202609 holdout，固定 seed 20261016）')
print(f'[check] Σalloc == 13−3×melds: {(alloc.sum(-1) == hand).mean()*100:.3f}% 一致')
live3 = np.broadcast_to((unseen > 0)[:, None, :], (n, 3, 34))

# 全列 CE（代表性自检）
def col_ce(P):
    return -np.take_along_axis(np.log(np.clip(P, EPS, None)),
                               D['alloc'][..., None].astype(np.int64), -1)[..., 0]
ce_a2b, ce_a3 = col_ce(AP), col_ce(RAP)
# counts 档 CE 需要完整 pmf，用 P0 递推重建
PMF = np.zeros((n, 3, 34, 5)); PMF[..., 0] = CPMF0
N = unseen.sum(-1)[:, None]
for j in range(3):
    h = hand[:, j][:, None]
    for k in range(1, 5):
        ratio = ((unseen - (k - 1)) / k) * ((h - (k - 1)) / np.maximum(N - unseen - h + k, EPS))
        ok = (h >= k) & (N - unseen >= h - k)
        PMF[:, j, :, k] = np.where(ok, PMF[:, j, :, k - 1] * np.maximum(ratio, 0), 0.0)
    PMF[:, j] /= np.maximum(PMF[:, j].sum(-1, keepdims=True), EPS)
ce_cnt = col_ce(PMF)
print(f'\n[CE] alloc CE（unseen>0 列）: counts {ce_cnt[live3].mean():.4f} | '
      f'A2b {ce_a2b[live3].mean():.4f} | A3 {ce_a3[live3].mean():.4f}')
print(f'     A2b 相对 counts 增益 {ce_cnt[live3].mean()-ce_a2b[live3].mean():+.4f} nats/列；'
      f'A3 {ce_cnt[live3].mean()-ce_a3[live3].mean():+.4f}')

# ================= PART C =================
print('\n' + '=' * 84)
print('PART C  alloc CE 增益落在哪些列：是否邻接（同色 ±1/±2）目标家的手切牌')
print('=' * 84)
hcm = hc < 99
adj = np.zeros((n, 3, 34), bool)
for d in (-2, -1, 1, 2):
    src = tidx - d
    ok = (src >= 0) & (src < 27) & (tidx < 27) & (src // 9 == tidx // 9)
    adj[..., ok] |= hcm[..., src[ok]]
out['C'] = {}
for nm, m_ in (('邻接手切', adj & live3), ('非邻接', ~adj & live3)):
    g_a2b = ce_cnt[m_].mean() - ce_a2b[m_].mean()
    g_a3 = ce_cnt[m_].mean() - ce_a3[m_].mean()
    print(f'  {nm:<8} n={m_.sum():>10,}  counts {ce_cnt[m_].mean():.4f}  A2b {ce_a2b[m_].mean():.4f}'
          f'  A3 {ce_a3[m_].mean():.4f} | 增益 A2b {g_a2b:+.4f}  A3 {g_a3:+.4f}  '
          f'A2b 超出规则 {g_a2b-g_a3:+.4f}')
    out['C'][nm] = dict(n=int(m_.sum()), counts=float(ce_cnt[m_].mean()),
                        a2b=float(ce_a2b[m_].mean()), a3=float(ce_a3[m_].mean()))
print('\n  逐列 counts→A2b 增益（nats）：')
g = ce_cnt - ce_a2b
gr = ce_cnt - ce_a3
for s in range(4):
    cols = np.arange(s * 9, s * 9 + 9) if s < 3 else np.arange(27, 34)
    print(f'   {SUIT[s] if s<3 else "字"}: ' + ' '.join(
        f'{g[:, :, c][live3[:, :, c]].mean():+.3f}' for c in cols))
print('  逐列 counts→A3 增益（nats）：')
for s in range(4):
    cols = np.arange(s * 9, s * 9 + 9) if s < 3 else np.arange(27, 34)
    print(f'   {SUIT[s] if s<3 else "字"}: ' + ' '.join(
        f'{gr[:, :, c][live3[:, :, c]].mean():+.3f}' for c in cols))

# ================= PART A =================
print('\n' + '=' * 84)
print('PART A  手切 (m+1)→(m-1)「外侧先」vs (m-1)→(m+1)「内侧先」：中间张 m')
print('=' * 84)
resid_truth = alloc - CE
resid_a2b = EC - CE
resid_a3 = REC - CE
# 同色 ±2 窗口（含 m）残差
win_t = np.zeros((n, 3, 34)); win_m = np.zeros((n, 3, 34))
for d in (-2, -1, 0, 1, 2):
    src = tidx - d
    ok = (src >= 0) & (src < 27) & (tidx < 27) & (src // 9 == tidx // 9)
    win_t[..., ok] += resid_truth[..., src[ok]]
    win_m[..., ok] += resid_a2b[..., src[ok]]

recs = {}
for s in range(3):
    b = s * 9
    for mnum in range(2, 9):
        a_, mid, c_ = b + mnum - 2, b + mnum - 1, b + mnum
        for j in range(3):
            pa, pc, ta, tc = hc[:, j, a_], hc[:, j, c_], tg[:, j, a_], tg[:, j, c_]
            av = unseen[:, mid] > 0
            hh, hcc = pa < 99, pc < 99
            for nm, sel in (('desc', hh & hcc & (pc < pa) & av),
                            ('asc', hh & hcc & (pa < pc) & av),
                            ('ctl', (((ta < 99) & hcc) | (hh & (tc < 99))) & av)):
                ii = np.where(sel)[0]
                if len(ii):
                    recs.setdefault(nm, []).append((ii, j, mid))

def agg(rec):
    ii = np.concatenate([r[0] for r in rec])
    jj = np.concatenate([np.full(len(r[0]), r[1]) for r in rec])
    mm = np.concatenate([np.full(len(r[0]), r[2]) for r in rec])
    q = (ii, jj, mm)
    q2 = (ii, jj)
    tr = alloc[q]
    return dict(n=len(ii), turn=turn[ii].mean(), riichi=riichi[q2].mean(),
                called=called[q2].mean(),
                truth=tr.mean(), truth1=(tr > 0).mean(), cnt=CE[q].mean(),
                cnt1=(1 - CPMF0[q]).mean(), a2b=EC[q].mean(),
                a2b1=(1 - AP[q + (0,)]).mean(), a3=REC[q].mean(),
                a31=(1 - RAP[q + (0,)]).mean(),
                r_truth=resid_truth[q].mean(), r_a2b=resid_a2b[q].mean(),
                r_a3=resid_a3[q].mean(), w_truth=win_t[q].mean(),
                w_a2b=win_m[q].mean(),
                _tr=tr, _cnt=CE[q], _a2b=EC[q], _a3=REC[q], _ii=ii)

AG = {k: agg(v) for k, v in recs.items()}
Ad, Aa = AG['desc'], AG['asc']
print(f"{'组':<13}{'n':>7}{'巡目':>7}{'立直%':>7}{'真值E[m]':>10}{'counts':>9}{'A2b':>8}"
      f"{'A3':>8}{'真残差':>9}{'A2b残差':>9}{'A3残差':>9}")
for nm, k in (('外侧先 9→7', 'desc'), ('内侧先 7→9', 'asc'), ('对照 含摸切', 'ctl')):
    A = AG[k]
    print(f'{nm:<13}{A["n"]:>7}{A["turn"]:>7.2f}{A["riichi"]*100:>7.1f}{A["truth"]:>10.4f}'
          f'{A["cnt"]:>9.4f}{A["a2b"]:>8.4f}{A["a3"]:>8.4f}{A["r_truth"]:>+9.4f}'
          f'{A["r_a2b"]:>+9.4f}{A["r_a3"]:>+9.4f}')
out['A'] = {k: {kk: float(vv) for kk, vv in v.items() if not kk.startswith('_')}
            for k, v in AG.items()}

print(f'\n  ①「两张开都切了」的集合级局部信号（vs 纯数牌 counts）：')
print(f'     真值残差 {Ad["r_truth"]:+.4f}  →  A2b 复现 {Ad["r_a2b"]:+.4f}'
      f'（{Ad["r_a2b"]/Ad["r_truth"]*100:.0f}%）  A3 {Ad["r_a3"]:+.4f}'
      f'（{Ad["r_a3"]/Ad["r_truth"]*100:.0f}%，方向{"同" if Ad["r_a3"]*Ad["r_truth"]>0 else "反"}）')
print(f'     同色 ±2 窗口（5 张合计）：真值 {Ad["w_truth"]:+.4f} → A2b {Ad["w_a2b"]:+.4f}'
      f'（{Ad["w_a2b"]/Ad["w_truth"]*100:.0f}%）')
print(f'\n  ② 顺序本身的增量 Δ(外侧先 − 内侧先)：')
print(f'     真值残差 {Ad["r_truth"]-Aa["r_truth"]:+.4f} | A2b残差 '
      f'{Ad["r_a2b"]-Aa["r_a2b"]:+.4f} | A3残差 {Ad["r_a3"]-Aa["r_a3"]:+.4f}')
print(f'     P(m≥1) 真值 {Ad["truth1"]-Aa["truth1"]:+.4f} '
      f'({Ad["truth1"]:.4f} vs {Aa["truth1"]:.4f}) | counts {Ad["cnt1"]-Aa["cnt1"]:+.4f}'
      f' | A2b {Ad["a2b1"]-Aa["a2b1"]:+.4f} | A3 {Ad["a31"]-Aa["a31"]:+.4f}')

B = 2000
def boot(A):
    u = np.unique(A['_ii']); return u, {x: np.where(A['_ii'] == x)[0] for x in u}
ud, md_ = boot(Ad); ua, ma_ = boot(Aa)
dt, dm, da = [], [], []
for _ in range(B):
    idd = np.concatenate([md_[x] for x in RNG.choice(ud, len(ud))])
    iaa = np.concatenate([ma_[x] for x in RNG.choice(ua, len(ua))])
    dt.append((Ad['_tr'][idd] - Ad['_cnt'][idd]).mean() - (Aa['_tr'][iaa] - Aa['_cnt'][iaa]).mean())
    dm.append((Ad['_a2b'][idd] - Ad['_cnt'][idd]).mean() - (Aa['_a2b'][iaa] - Aa['_cnt'][iaa]).mean())
    da.append((Ad['_a3'][idd] - Ad['_cnt'][idd]).mean() - (Aa['_a3'][iaa] - Aa['_cnt'][iaa]).mean())
print(f'\n     bootstrap {B}（按样本重抽）ΔΔ 残差：')
out['A_boot'] = {}
for nm, arr in (('真值', dt), ('A2b ', dm), ('A3  ', da)):
    a = np.array(arr)
    lo, hi = np.percentile(a, [2.5, 97.5])
    print(f'       {nm} {a.mean():+.4f}  CI95[{lo:+.4f},{hi:+.4f}]  '
          f'P(>0)={(a>0).mean():.3f} P(<0)={(a<0).mean():.3f}')
    out['A_boot'][nm.strip()] = [float(a.mean()), float(lo), float(hi)]

print('\n  ③ 按目标家是否立直分层：')
for nm, want in (('立直', True), ('非立直', False)):
    def filt(rec):
        o = []
        for i, j, m in rec:
            k = i[(riichi[i, j] > 0.5) == want]
            if len(k): o.append((k, j, m))
        return o
    rd, ra = filt(recs['desc']), filt(recs['asc'])
    if not rd or not ra:
        print(f'   {nm}: 样本不足'); continue
    x, y = agg(rd), agg(ra)
    print(f'   {nm}(n={x["n"]}/{y["n"]}): 真值ΔΔ {x["r_truth"]-y["r_truth"]:+.4f} | '
          f'A2bΔΔ {x["r_a2b"]-y["r_a2b"]:+.4f} | A3ΔΔ {x["r_a3"]-y["r_a3"]:+.4f} | '
          f'A2b 集合级复现率 {x["r_a2b"]/x["r_truth"]*100:.0f}%')

print('\n  ④ 按 (花色, 中间张) 细分 真值ΔΔ / A2bΔΔ [n外/n内]：')
for s in range(3):
    b = s * 9
    print(f'   {SUIT[s]}:')
    for mnum in range(2, 9):
        mid = b + mnum - 1
        rd = [r for r in recs['desc'] if r[2] == mid]
        ra = [r for r in recs['asc'] if r[2] == mid]
        if not rd or not ra: continue
        x, y = agg(rd), agg(ra)
        print(f'     m={mnum}: 真值 {x["r_truth"]-y["r_truth"]:+.4f}  '
              f'A2b {x["r_a2b"]-y["r_a2b"]:+.4f}  [{x["n"]}/{y["n"]}]  '
              f'集合级真值 {x["r_truth"]:+.3f} A2b {x["r_a2b"]:+.3f}')

# ============ PART A2：按「自然/非自然切序」合并（跨花色跨中间张） ============
print('\n  ⑤ 按「自然切序 vs 非自然切序」合并（m=5 两侧对称，剔除）')
print('     自然 = 先切靠边的那张（d(x)=min(x−1,9−x) 更小）；假设：非自然序 ⇒ 更可能持有 m')
NAT = {2: 'asc', 3: 'asc', 4: 'asc', 6: 'desc', 7: 'desc', 8: 'desc'}   # mnum -> 自然序
# 直接合并会被「m 组成差异」污染（A3 顺序盲却能"复现"58%）。改为格内分层：
# 每个 (花色, m) 格内算 ΔΔ(desc−asc)，乘以「非自然=desc ? +1 : −1」映射到
# (非自然 − 自然)，再按 min(n_desc,n_asc) 加权平均。这样每格的两组舍牌集合完全相同，
# 顺序盲的预测器在格内必然 ΔΔ≈0，混杂被消掉。
cells = []
print(f"\n     {'格':<8}{'n外/n内':>14}{'真值ΔΔ':>10}{'A2bΔΔ':>9}{'A3ΔΔ':>9}{'自然序':>7}")
for s in range(3):
    b = s * 9
    for mnum, nat in NAT.items():
        mid = b + mnum - 1
        rd = [r for r in recs['desc'] if r[2] == mid]
        ra = [r for r in recs['asc'] if r[2] == mid]
        if not rd or not ra:
            continue
        x, y = agg(rd), agg(ra)
        sgn = 1.0 if nat == 'asc' else -1.0        # 非自然=desc → desc−asc 即为所求
        cells.append(dict(s=s, m=mnum, nd=x['n'], na=y['n'], sgn=sgn,
                          t=sgn * (x['r_truth'] - y['r_truth']),
                          b=sgn * (x['r_a2b'] - y['r_a2b']),
                          r=sgn * (x['r_a3'] - y['r_a3']),
                          _x=x, _y=y))
        print(f'     {SUIT[s]}{mnum:<7}{x["n"]:>7}/{y["n"]:<6}'
              f'{x["r_truth"]-y["r_truth"]:>+10.4f}{x["r_a2b"]-y["r_a2b"]:>+9.4f}'
              f'{x["r_a3"]-y["r_a3"]:>+9.4f}{nat:>7}')

w = np.array([min(c['nd'], c['na']) for c in cells], float); w /= w.sum()
we = np.ones(len(cells)) / len(cells)
def strat(key, ww):
    return float(np.sum([c[key] for c in cells] * ww))
print(f'\n     分层加权 ΔΔ(非自然 − 自然)，{len(cells)} 格：')
print(f"       {'':<16}{'n 加权':>10}{'等权':>10}")
for nm_, k in (('真值', 't'), ('A2b', 'b'), ('A3（顺序盲对照）', 'r')):
    print(f'       {nm_:<16}{strat(k, w):>+10.4f}{strat(k, we):>+10.4f}')
tt, tb = strat('t', w), strat('b', w)
print(f'       A2b 复现率 {tb/tt*100:+.0f}%   A3 复现率 {strat("r", w)/tt*100:+.0f}%')

# bootstrap：按样本重抽，重算分层估计
ub = {i: (boot(c['_x']), boot(c['_y'])) for i, c in enumerate(cells)}
bs = {'t': [], 'b': [], 'r': []}
for _ in range(400):
    acc = {'t': 0.0, 'b': 0.0, 'r': 0.0}
    for i, c in enumerate(cells):
        (ux, mx), (uy, my) = ub[i]
        i1 = np.concatenate([mx[a] for a in RNG.choice(ux, len(ux))])
        i0 = np.concatenate([my[a] for a in RNG.choice(uy, len(uy))])
        X, Y = c['_x'], c['_y']
        for k, f in (('t', '_tr'), ('b', '_a2b'), ('r', '_a3')):
            d = (X[f][i1] - X['_cnt'][i1]).mean() - (Y[f][i0] - Y['_cnt'][i0]).mean()
            acc[k] += w[i] * c['sgn'] * d
    for k in bs:
        bs[k].append(acc[k])
print(f'       bootstrap 400：')
out['A2'] = {'cells': [{k: v for k, v in c.items() if not k.startswith('_')} for c in cells]}
for nm_, k in (('真值', 't'), ('A2b ', 'b'), ('A3  ', 'r')):
    a = np.array(bs[k]); lo, hi = np.percentile(a, [2.5, 97.5])
    print(f'         {nm_} {a.mean():+.4f}  CI95[{lo:+.4f},{hi:+.4f}]  P(>0)={(a>0).mean():.3f}')
    out['A2'][nm_.strip()] = [float(a.mean()), float(lo), float(hi)]

# 关键子组的 bootstrap CI
print('\n  ⑥ 关键子组 ΔΔ 残差的置信区间（bootstrap 2000）：')
def dd_boot(rd, ra, label):
    x, y = agg(rd), agg(ra)
    ux, mx = boot(x); uy, my = boot(y)
    v = []
    for _ in range(B):
        i1 = np.concatenate([mx[a] for a in RNG.choice(ux, len(ux))])
        i0 = np.concatenate([my[a] for a in RNG.choice(uy, len(uy))])
        v.append((x['_tr'][i1] - x['_cnt'][i1]).mean() - (y['_tr'][i0] - y['_cnt'][i0]).mean())
    v = np.array(v); lo, hi = np.percentile(v, [2.5, 97.5])
    print(f'     {label:<26} 真值ΔΔ {v.mean():+.4f} CI95[{lo:+.4f},{hi:+.4f}] '
          f'P(<0)={(v<0).mean():.3f}  [n={x["n"]}/{y["n"]}]')
    out.setdefault('A_ci', {})[label] = [float(v.mean()), float(lo), float(hi),
                                         x['n'], y['n']]
def cell(s, mnum, nm_):
    mid = s * 9 + mnum - 1
    return ([r for r in recs['desc'] if r[2] == mid],
            [r for r in recs['asc'] if r[2] == mid])
for s in range(3):
    rd, ra = cell(s, 8, '')
    if rd and ra:
        dd_boot(rd, ra, f'{SUIT[s]} m=8（用户的 8s 例）')
for nm, want in (('立直', True), ('非立直', False)):
    def filt(rec):
        o = []
        for i, j, m in rec:
            k = i[(riichi[i, j] > 0.5) == want]
            if len(k): o.append((k, j, m))
        return o
    rd, ra = filt(recs['desc']), filt(recs['asc'])
    if rd and ra:
        dd_boot(rd, ra, f'目标家{nm}')

# ================= PART A' =================
print('\n' + '=' * 84)
print("PART A'  配对反事实：同一样本，只把 tl_tokens 里这两个 token 对调（obs 不变）")
print('=' * 84)
ids, Jarr, Mids = D['ids'], D['Jarr'], D['Mids']
q0 = (ids, Jarr, Mids)
pa = hc[ids, Jarr, Mids - 1]        # 内侧张 (m-1) 的手切位置
pc = hc[ids, Jarr, Mids + 1]        # 外侧张 (m+1) 的手切位置
Grp = np.where(pc < pa, 'desc', 'asc')
d_ec = EC2[q0] - EC[q0]
noise = np.abs(EC2[ids].sum(-1) - EC[ids].sum(-1)).mean() / 102.0
dd = Ad['r_truth'] - Aa['r_truth']
print(f'  样本 {len(ids)}（外侧先 {(Grp=="desc").sum()} / 内侧先 {(Grp=="asc").sum()}）')
print(f'  Δexp_counts[m]（对调后 − 原）: mean {d_ec.mean():+.5f}  '
      f'mean|Δ| {np.abs(d_ec).mean():.5f}  sd {d_ec.std():.5f}')
print(f'  量级参照：exp_counts[m] mean {EC[q0].mean():.4f}；数值噪声底 ~{noise:.5f}/列')
print(f'  真值 ΔΔ = {dd:+.4f} ⇒ 若模型正确使用顺序，外侧先样本对调后应 '
      f'{-dd:+.4f}，内侧先样本应 {+dd:+.4f}')
for gname in ('desc', 'asc'):
    s_ = Grp == gname
    exp = -dd if gname == 'desc' else dd
    v = d_ec[s_]
    se = v.std() / max(np.sqrt(len(v)), 1)
    print(f'    {gname}(n={s_.sum()}): 实测 Δ {v.mean():+.5f} ± {se:.5f}   '
          f'期望 {exp:+.4f}   → 实现比例 '
          f'{v.mean()/exp*100 if abs(exp) > 1e-6 else float("nan"):+.1f}%')
out['Ap'] = dict(n=int(len(ids)), d_mean=float(d_ec.mean()),
                 d_absmean=float(np.abs(d_ec).mean()), d_sd=float(d_ec.std()),
                 base=float(EC[q0].mean()), noise=float(noise),
                 d_desc=float(d_ec[Grp == 'desc'].mean()),
                 d_asc=float(d_ec[Grp == 'asc'].mean()), truth_dd=float(dd))

# ================= PART D =================
print('\n' + '=' * 84)
print('PART D  打出 x 的方式（手切/摸切）对「x 的同色邻接列」推断的影响')
print('=' * 84)
out['D'] = {}
for kind, src in (('手切', hc), ('摸切', tg)):
    cut = src < 99
    out['D'][kind] = {}
    for d in (1, 2):
        mt = ma = mr = 0.0
        cnt = 0
        for sgn in (-1, 1):
            src_i = tidx - sgn * d
            ok = (src_i >= 0) & (src_i < 27) & (tidx < 27) & (src_i // 9 == tidx // 9)
            m_ = np.zeros((n, 3, 34), bool)
            m_[..., ok] = cut[..., src_i[ok]]
            m_ &= live3
            cnt += int(m_.sum())
            mt += np.where(m_, resid_truth, 0).sum()
            ma += np.where(m_, resid_a2b, 0).sum()
            mr += np.where(m_, resid_a3, 0).sum()
        vt, va, vr = mt / cnt, ma / cnt, mr / cnt
        print(f'  {kind} x → x±{d}（n={cnt:,}）: 真值残差 {vt:+.4f} | A2b {va:+.4f}'
              f'（{va/vt*100:.0f}%）| A3 {vr:+.4f}（{vr/vt*100:.0f}%）')
        out['D'][kind][f'pm{d}'] = dict(n=cnt, truth=float(vt), a2b=float(va), a3=float(vr))

# 手切 vs 摸切 的「差」是否被模型区分
t1 = out['D']['手切']['pm1']; g1 = out['D']['摸切']['pm1']
print(f'\n  手切 − 摸切（x±1）: 真值 {t1["truth"]-g1["truth"]:+.4f} | '
      f'A2b {t1["a2b"]-g1["a2b"]:+.4f} | A3 {t1["a3"]-g1["a3"]:+.4f}')

with open(str(OUT_DIR / 'local_order_report.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=float)
print(f'\n→ {OUT_DIR / "local_order_report.json"}')
