"""RM16（README §12.8）：顺序级/区域级常规诊断 —— RM12~RM15 的统一判据。

动机：现有常规指标（evaluate / strat_metrics / compare_a3）全是「单帧/逐家/
逐列边际」，把局部信息主导的集合级好成绩与顺序信息失败混在一起
（§12.2/§12.3：A2b 集合级复现 75~79%，但立直层顺序捕捉为零、区域级为负、
配对反事实斜率 b=0.0144 ≈ 时序流空转）。本脚本把三个一次性诊断
（diagnostics/s3158_local_strata.py / _window.py / _report.py 的 A' 部分）
沉淀为一条命令的常规工具，任何读牌 checkpoint 重训后都应跑一遍。

三项指标（口径与 MEMORY §3.15.8 / README §12 完全一致，同源 cache）：
  ① single_tile  格内分层 ΔΔ(非自然−自然)：每 (花色, m) 格内「先切靠边张/先切内侧张」
     两组的舍牌**集合**恒等。⚠️ A3 规则法**不是**顺序盲（rule_baseline.py:203,305
     消费 tl_tokens），故原样 A3 只能作「规则法地板」；cache 含 REC_OB 时另给
     **真顺序盲地板**（token 规范化排序），RM13/RM14 判据以后者为准，两者之差即
     A3 自身的顺序敏感度（README §11.5-⑩ / AUDIT_20261020 S1）。
     报 真值/模型/A3地板 + 超出地板的 bootstrap CI + 扣地板复现率 + 逐格明细。
  ② window_pm2   同①，但残差取同色 m±2 窗口 5 张合计（区域级，量级 3~4 倍）。
  ③ counterfactual 配对反事实：obs 不变、只对调 tl_tokens 里 (m−1, m+1) 两个
     token，Δexp_counts[m] 对逐样本期望（∓ΔΔ_cell，非立直层真值）回归，
     斜率 b（=1 完美利用顺序，=0 无响应）与 r²。

判据（历史 `verdicts` 保留 A2b 时代线，仅作对照；P1b 判决用 `p1b_verdicts`）：

  **P1b 主判据（2026-10-21 预注册，基线 A2c）**
  RM12  b ≥ 0.3 且 r² ≥ 0.2                 （A2c：b=0.0093, r²=1.28%）
  RM15  立直层 窗口超地板 CI95 下界 > 0       （A2c：−0.1134）

  **P1b 不退线（A2c 实测）**
  RM13  非立直层 单张扣地板复现率 ≥ 92.696%   （A2c 92.696%（显示 92.7%）；历史线 90% 仅存档）
  RM14  立直层   单张扣地板复现率 ≥ 66.2%     （A2c 66.2%；历史线 +30% 仅存档；
                                                +50 低于 A2c 66.2，作为新线无效）

  历史线（`verdicts`）：RM12 b≥0.3/r²≥0.2、RM13 ≥90%、RM14 ≥+30%、RM15 CI>0；
  A2b 时代口径，保留用于逐位回归，**不得**用作 P1b 判决。

用法：
    python tools/read_model/order_metrics.py                # 默认 A2b，cache 缺失/过期自动重建（需 GPU）
    python tools/read_model/order_metrics.py --state <ckpt> --rebuild
    python tools/read_model/order_metrics.py --out /path/to/order_metrics.json

产物：<ckpt 同目录>/order_metrics.json（默认）+ stdout 摘要。
cache 由 diagnostics/s3158_local_extract.py 生成（~2.5 min，GPU），
元数据 sidecar 记录 ckpt/n/schema，防止跨模型串 cache（坑位 35/42 教训）与
schema 过旧静默降级（2026-10-20 加）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE / 'diagnostics'

#: cache schema 版本。1 = 原始（AP/EC/EC2/RAP/REC/CE/…）；
#: 2 = 增 REC_OB（A3 真顺序盲变体，README §11.5-⑩）。版本不符即重建 cache。
CACHE_SCHEMA = 2

#: P1b 预注册判据（2026-10-21，基线 A2c；历史 `verdicts` 不动）
P1B_BASELINE = 'A2c@153k（2026-10-21 官方验收）'
P1B_BASELINE_VALUES = {
    'RM12': {'b': 0.009343552759278253, 'r2': 0.012849632174707222},
    'RM13': 92.69611640513835,
    'RM14': 66.21802819482939,
    'RM15': {'ci95_lower': -0.1133666826794949},
}
P1B_THRESHOLDS = {'RM12_b': 0.3, 'RM12_r2': 0.2, 'RM13': 92.696, 'RM14': 66.2, 'RM15': 0.0}


def _pct(d, k):
    v = d.get(k) if isinstance(d, dict) else None
    return v if isinstance(v, (int, float)) else None


def build_p1b_verdicts(single_ob, window, cf, single=None):
    """P1b 预注册判据（A2c 基线）：主判据 RM12/RM15，RM13/RM14 不退线。

    与 `verdicts`（A2b 时代 90%/+30% 历史线）解耦：本函数不改历史判决，
    只新增一套可机器判读的 P1b 判决。地板优先用真顺序盲 `single_ob`，
    缺失时退回规则法 `single`（与历史 verdicts 同口径，并在 rule 里标注）。
    """
    floor = single_ob if single_ob is not None else (single or {})
    floor_kind = ('order_blind' if single_ob is not None else 'rule_as_is')
    s_nr = floor.get('nonriichi', {})
    s_ri = floor.get('riichi', {})
    w_ri = window.get('riichi', {}) if isinstance(window, dict) else {}
    b = cf.get('slope_b') if isinstance(cf, dict) else None
    r2 = cf.get('r2') if isinstance(cf, dict) else None
    rm13 = _pct(s_nr, 'net_minus_floor_pct')
    rm14 = _pct(s_ri, 'net_minus_floor_pct')
    rm15 = (w_ri.get('excess_ci95') or [None])[0]
    return {
        'RM12': {
            'role': 'primary',
            'rule': 'P1b 主判据：反事实斜率 b>=0.3 且 r2>=0.2（时序流真的在用顺序）',
            'value': {'b': b, 'r2': r2},
            'passed': bool(b is not None and b >= P1B_THRESHOLDS['RM12_b']
                           and r2 is not None and r2 >= P1B_THRESHOLDS['RM12_r2']),
            'baseline_a2c': P1B_BASELINE_VALUES['RM12'],
        },
        'RM15': {
            'role': 'primary',
            'rule': 'P1b 主判据：立直层窗口超地板 CI95 下界 > 0',
            'value': rm15,
            'passed': bool(rm15 is not None and rm15 > P1B_THRESHOLDS['RM15']),
            'baseline_a2c': P1B_BASELINE_VALUES['RM15'],
        },
        'RM13': {
            'role': 'no_regression',
            'rule': (f'P1b 不退线：非立直层单张扣地板复现率 >= 92.696%'
                     f'（地板={floor_kind}；A2c 92.696%（显示 92.7%）；'
                     f'历史线 90% 仅存档；线取精确基线，避免 A2c 自身 FAIL）'),
            'value': rm13,
            'passed': bool(rm13 is not None and rm13 >= P1B_THRESHOLDS['RM13']),
            'baseline_a2c': P1B_BASELINE_VALUES['RM13'],
        },
        'RM14': {
            'role': 'no_regression',
            'rule': (f'P1b 不退线：立直层单张扣地板复现率 >= 66.2%'
                     f'（地板={floor_kind}；A2c 66.2%；历史线 +30% 仅存档；'
                     f'+50 低于 A2c 66.2，作为新线无效，不能沿用）'),
            'value': rm14,
            'passed': bool(rm14 is not None and rm14 >= P1B_THRESHOLDS['RM14']),
            'baseline_a2c': P1B_BASELINE_VALUES['RM14'],
            'note': 'RM14 +30% 历史线无区分力；+50 也低于 A2c 66.2，不能当新线。',
        },
        'meta': {
            'baseline': P1B_BASELINE,
            'primary': ['RM12', 'RM15'],
            'no_regression': ['RM13', 'RM14'],
            'historical_verdicts': 'verdicts（A2b 时代 90%/+30%，仅存档，不作 P1b 判决）',
            'thresholds': P1B_THRESHOLDS,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default=None,
                    help='checkpoint（默认与 diagnostics/_common 一致：RM_CKPT 或 A2b best）')
    ap.add_argument('--n', type=int, default=40000,
                    help='holdout 快照数（与 MEMORY §3.15.8 / README §12 同源）')
    ap.add_argument('--rebuild', action='store_true', help='强制重建 cache（需 GPU）')
    ap.add_argument('--boot', type=int, default=400, help='bootstrap 次数（诊断口径 400）')
    ap.add_argument('--out', default=None, help='JSON 输出（默认 <ckpt 目录>/order_metrics.json）')
    args = ap.parse_args()

    # RM_CKPT 必须在 import _common 之前设置（CKPT 在 import 期读取）
    if args.state:
        os.environ['RM_CKPT'] = str(Path(args.state).resolve())
    sys.path.insert(0, str(DIAG))
    from _common import CACHE, CKPT                      # noqa: E402

    import numpy as np

    # ---- Phase 1：cache 检查/重建（含 ckpt 身份 sidecar，防串 cache）----
    meta_path = CACHE.with_suffix('.meta.json')
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    need = args.rebuild
    if not CACHE.exists():
        need = True
    elif meta.get('ckpt') != str(CKPT) or int(meta.get('n', -1)) != args.n:
        print(f'[order] cache 身份不符（cache: {meta.get("ckpt")} n={meta.get("n")} | '
              f'请求: {CKPT} n={args.n}），重建', flush=True)
        need = True
    elif int(meta.get('schema', 1)) != CACHE_SCHEMA:
        # schema 2（2026-10-20）：新增 REC_OB = A3 真顺序盲变体（README §11.5-⑩）。
        # 旧 cache 身份相符但缺该字段 ⇒ 必须重建，否则「纯组成地板」退回原样 A3。
        print(f'[order] cache schema 过旧（{meta.get("schema", 1)} != {CACHE_SCHEMA}：'
              f'缺 REC_OB 真顺序盲地板），重建', flush=True)
        need = True
    elif Path(CKPT).exists() and CACHE.stat().st_mtime < Path(CKPT).stat().st_mtime:
        print('[order] cache 早于 ckpt（可能已重训），重建', flush=True)
        need = True
    if need:
        t0 = time.time()
        print(f'[order] Phase 1：{DIAG / "s3158_local_extract.py"} {args.n}'
              f'（RM_CKPT={CKPT}，需 GPU，~2.5 min）', flush=True)
        subprocess.run([sys.executable, str(DIAG / 's3158_local_extract.py'), str(args.n)],
                       check=True)
        json.dump({'ckpt': str(CKPT), 'n': args.n, 'schema': CACHE_SCHEMA,
                   'built_at': time.time()}, open(meta_path, 'w'))
        print(f'[order] Phase 1 完成 {time.time() - t0:.0f}s', flush=True)

    # ---- 装载 cache ----
    D = np.load(str(CACHE))
    n = int(D['n'])
    EC = D['EC'].astype(np.float64)          # A2b/被测模型 exp_counts
    EC2 = D['EC2'].astype(np.float64)        # 配对反事实后的 exp_counts
    REC = D['REC'].astype(np.float64)        # A3 规则 exp_counts（**规则法**地板，非顺序盲）
    REC_OB = (D['REC_OB'].astype(np.float64)  # A3 真顺序盲变体（schema>=2 才有）
              if 'REC_OB' in D.files else None)
    CE = D['CE'].astype(np.float64)          # counts 精确档 exp_counts
    alloc = D['alloc'].astype(np.float64)    # 真值 [n,3,34]
    unseen = D['unseen'].astype(np.float64)
    riichi = D['riichi'].astype(np.float64)
    turn = D['turn'].astype(np.float64)
    hc = D['hc']                             # 手切位置（99=未手切）
    tidx = np.arange(34)
    SUIT = '万筒索'
    B = args.boot
    RNG = np.random.default_rng(20261019)    # 与 s3158 系列同 seed

    rt = alloc - CE                          # 真值残差
    rm = EC - CE                             # 模型残差
    rr = REC - CE                            # A3 残差（规则法地板）
    rr_ob = (REC_OB - CE) if REC_OB is not None else None   # 真顺序盲地板

    # ---- 记录集：每 (花色, m, 目标家) 的「外侧先(desc)/内侧先(asc)」样本 ----
    recs = {'desc': [], 'asc': []}
    cell_of = {}
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

    def agg(rec, mask_fn, RT, RM, RR):
        I, J, M = [], [], []
        for ii, j, mid, s, mnum in rec:
            k = ii if mask_fn is None else ii[mask_fn(ii, j)]
            if len(k):
                I.append(k); J.append(np.full(len(k), j)); M.append(np.full(len(k), mid))
        if not I:
            return None
        ii = np.concatenate(I); jj = np.concatenate(J); mm = np.concatenate(M)
        q = (ii, jj, mm)
        return dict(n=len(ii), turn=float(turn[ii].mean()), ii=ii,
                    t=RT[q], b=RM[q], r=RR[q])

    NAT = {2: 'asc', 3: 'asc', 4: 'asc', 6: 'desc', 7: 'desc', 8: 'desc'}
    STRATA = {
        'all': None,
        'nonriichi': lambda ii, j: riichi[ii, j] < 0.5,
        'riichi': lambda ii, j: riichi[ii, j] > 0.5,
    }

    def cell_stratified(RT, RM, RR, tag):
        """格内分层 ΔΔ(非自然−自然) + 扣 A3 地板 + 按样本 bootstrap。"""
        out = {}
        for sname, fn in STRATA.items():
            cells = []
            for s in range(3):
                for mnum, nat in NAT.items():
                    mid = cell_of[(s, mnum)]
                    rd = [r for r in recs['desc'] if r[2] == mid]
                    ra = [r for r in recs['asc'] if r[2] == mid]
                    if not rd or not ra:
                        continue
                    x, y = agg(rd, fn, RT, RM, RR), agg(ra, fn, RT, RM, RR)
                    if x is None or y is None or min(x['n'], y['n']) < 25:
                        continue
                    sgn = 1.0 if nat == 'asc' else -1.0
                    cells.append(dict(cell=f'{SUIT[s]}{mnum}', nd=x['n'], na=y['n'],
                                      sgn=sgn, x=x, y=y,
                                      t=sgn * (x['t'].mean() - y['t'].mean()),
                                      b=sgn * (x['b'].mean() - y['b'].mean()),
                                      r=sgn * (x['r'].mean() - y['r'].mean())))
            if not cells:
                out[sname] = {'skipped': 'insufficient_cells'}
                continue
            w = np.array([min(c['nd'], c['na']) for c in cells], float); w /= w.sum()
            st = lambda k: float(np.sum([c[k] for c in cells] * w))   # noqa: E731
            tt, tb, tr = st('t'), st('b'), st('r')
            ub = [(np.unique(c['x']['ii']), np.unique(c['y']['ii'])) for c in cells]
            mx = [{u: np.where(c['x']['ii'] == u)[0] for u in ub[i][0]}
                  for i, c in enumerate(cells)]
            my = [{u: np.where(c['y']['ii'] == u)[0] for u in ub[i][1]}
                  for i, c in enumerate(cells)]
            bs = {'t': [], 'b': [], 'r': []}
            for _ in range(B):
                acc = {'t': 0.0, 'b': 0.0, 'r': 0.0}
                for i, c in enumerate(cells):
                    X, Y = c['x'], c['y']
                    i1 = np.concatenate([mx[i][u] for u in RNG.choice(ub[i][0], len(ub[i][0]))])
                    i0 = np.concatenate([my[i][u] for u in RNG.choice(ub[i][1], len(ub[i][1]))])
                    for k in bs:
                        acc[k] += w[i] * c['sgn'] * (X[k][i1].mean() - Y[k][i0].mean())
                for k in bs:
                    bs[k].append(acc[k])
            res = {'n_cells': len(cells),
                   'n_desc': int(sum(c['nd'] for c in cells)),
                   'n_asc': int(sum(c['na'] for c in cells)),
                   'truth_dd': tt, 'net_dd': tb, 'floor_dd': tr}
            for nm, k in (('truth', 't'), ('net', 'b'), ('floor', 'r')):
                a = np.array(bs[k]); lo, hi = np.percentile(a, [2.5, 97.5])
                res[f'bs_{nm}'] = [float(a.mean()), float(lo), float(hi)]
            ex = np.array(bs['b']) - np.array(bs['r'])
            lo, hi = np.percentile(ex, [2.5, 97.5])
            res['excess_over_floor'] = float(tb - tr)
            res['excess_ci95'] = [float(lo), float(hi)]
            res['excess_p_gt0'] = float((ex > 0).mean())
            res['capture_pct'] = float(tb / tt * 100) if abs(tt) > 1e-12 else None
            res['floor_pct'] = float(tr / tt * 100) if abs(tt) > 1e-12 else None
            res['net_minus_floor_pct'] = (float((tb - tr) / (tt - tr) * 100)
                                          if abs(tt - tr) > 1e-12 else None)
            # 逐格明细（2026-10-20，AUDIT S3/L9）：同时给**两种减数顺序**，杜绝
            # README §12.4 曾出现的「表头用 (非自然−自然)、逐格列表用 (desc−asc)，
            # 二者在 m∈{6,7,8} 反号」误读；per-cell 从此可事后裁决，不再只留聚合值。
            res['cells'] = [
                {'cell': c['cell'], 'n_desc': int(c['nd']), 'n_asc': int(c['na']),
                 'natural_order': ('asc(先切 m−1)' if c['sgn'] > 0 else 'desc(先切 m+1)'),
                 'weight': float(w[i]),
                 'dd_unnatural_minus_natural': {'truth': float(c['t']), 'net': float(c['b']),
                                                'floor': float(c['r'])},
                 'dd_desc_minus_asc': {'truth': float(c['t'] * c['sgn']),
                                       'net': float(c['b'] * c['sgn']),
                                       'floor': float(c['r'] * c['sgn'])}}
                for i, c in enumerate(cells)]
            out[sname] = res
            print(f'  [{tag}/{sname:<9s}] 格 {len(cells):>2d}  '
                  f'真值ΔΔ {tt:+.4f}  模型 {tb:+.4f}  地板 {tr:+.4f}  '
                  f'超地板 {tb - tr:+.4f} CI95[{lo:+.4f},{hi:+.4f}]  '
                  f'扣地板复现 {res["net_minus_floor_pct"]:+.0f}%', flush=True)
        return out

    def win_sum(src, half=2):
        """同色 m±half 窗口合计（区域级残差，_window 口径）。"""
        w = np.zeros_like(src)
        for d in range(-half, half + 1):
            s_ = tidx - d
            ok = (s_ >= 0) & (s_ < 27) & (tidx < 27) & (s_ // 9 == tidx // 9)
            w[..., ok] += src[..., s_[ok]]
        return w

    print(f'[order] n={n}（202609 holdout，seed 20261016）  ckpt={CKPT}', flush=True)
    print('\n① 单张 m 格内分层 ΔΔ(非自然−自然)，扣 A3 **规则法**地板'
          '（A3 消费 tl_tokens，并非顺序盲，§11.5-⑩）', flush=True)
    single = cell_stratified(rt, rm, rr, 'single')
    single_ob = None
    if rr_ob is not None:
        print('\n①b 同口径，但地板换成 A3 **真顺序盲**变体（token 规范化排序）'
              '——RM13/RM14 判据以此为准', flush=True)
        single_ob = cell_stratified(rt, rm, rr_ob, 'single-OB')
    else:
        print('\n①b 跳过：cache 无 REC_OB（schema<2）⇒ 判据退回规则法地板，'
              '会低估「超出地板」比例', flush=True)
    print('\n② ±2 窗口（5 张合计）同口径', flush=True)
    WT, WM, WR = win_sum(rt), win_sum(rm), win_sum(rr)
    window = cell_stratified(WT, WM, WR, 'window')

    # ---- ③ 配对反事实回归（_strata A'b 口径）----
    print('\n③ 配对反事实：对调 tl_tokens (m−1, m+1) 两 token，回归 Δexp_counts[m]',
          flush=True)
    cf = {'skipped': None}
    ids, Jarr, Mids = D['ids'], D['Jarr'], D['Mids']
    if len(ids) == 0:
        cf = {'skipped': 'no_counterfactual_samples'}
    else:
        q0 = (ids, Jarr, Mids)
        pa = hc[ids, Jarr, Mids - 1]
        pc = hc[ids, Jarr, Mids + 1]
        is_desc = pc < pa
        Sarr = Mids // 9
        Marr = Mids % 9 + 1
        nr = lambda ii, j: riichi[ii, j] < 0.5        # noqa: E731
        dd_cell = {}
        for s in range(3):
            for mnum in range(2, 9):
                mid = cell_of[(s, mnum)]
                rd = [r for r in recs['desc'] if r[2] == mid]
                ra = [r for r in recs['asc'] if r[2] == mid]
                if not rd or not ra:
                    continue
                x, y = agg(rd, nr, rt, rm, rr), agg(ra, nr, rt, rm, rr)
                if x is None or y is None or min(x['n'], y['n']) < 25:
                    continue
                dd_cell[(s, mnum)] = float(x['t'].mean() - y['t'].mean())
        exp_d = np.array([(-1 if is_desc[k] else +1) *
                          dd_cell.get((Sarr[k], Marr[k]), np.nan) for k in range(len(ids))])
        meas = EC2[q0] - EC[q0]
        okm = ~np.isnan(exp_d)
        if len(dd_cell) == 0 or okm.sum() < 50:
            cf = {'skipped': 'insufficient_cells', 'n_cells': len(dd_cell),
                  'n_pairs': int(okm.sum())}
        else:
            e, m_ = exp_d[okm], meas[okm]
            slope = float(np.dot(e - e.mean(), m_ - m_.mean())
                          / max(np.dot(e - e.mean(), e - e.mean()), 1e-12))
            r2 = float(np.dot(e - e.mean(), m_ - m_.mean()) ** 2
                       / max(np.dot(e - e.mean(), e - e.mean())
                             * np.dot(m_ - m_.mean(), m_ - m_.mean()), 1e-12))
            cf = {'n_pairs': int(okm.sum()), 'n_cells': len(dd_cell),
                  'slope_b': slope, 'r2': r2,
                  'exp_mean': float(e.mean()), 'meas_mean': float(m_.mean()),
                  'exp_sd': float(e.std()), 'meas_sd': float(m_.std()),
                  'meas_desc': float(m_[is_desc[okm]].mean()),
                  'meas_asc': float(m_[~is_desc[okm]].mean())}
            print(f'  样本 {cf["n_pairs"]}  斜率 b = {slope:+.4f}（1=完美, 0=无响应）  '
                  f'r² = {r2:.5f}', flush=True)

    # ---- 判据（RM12~RM15 验收，README §12.8）----
    def pct(d, k):
        v = d.get(k) if isinstance(d, dict) else None
        return v if isinstance(v, (int, float)) else None

    # RM13/RM14 的「扣地板复现率」以**真顺序盲地板**为准（cache 有 REC_OB 时）；
    # 否则退回规则法地板，并显式标注降级——A3 消费 tl_tokens（§11.5-⑩），用原样
    # A3 当「组成地板」会把一部分真实顺序响应算进地板，从而**低估**超出比例。
    floor_src = single_ob if single_ob is not None else single
    floor_kind = ('order_blind（A3 token 规范化排序 ⇒ 纯组成地板）'
                  if single_ob is not None else
                  'rule_as_is（A3 原样，非顺序盲 ⇒ 判据偏保守，见 §11.5-⑩）')
    s_nr, s_ri = floor_src.get('nonriichi', {}), floor_src.get('riichi', {})
    # 原样 A3（规则法）地板版复现率：与 OB 版并列留档（2026-10-21 修复：这两个变量
    # 此前从未定义，加 REC_OB 后拼 verdicts 必 NameError —— cache 已算完、报告却不落盘；
    # 这里用 single 即规则法地板作「原样」侧）。
    s_nr_asis = single.get('nonriichi', {})
    s_ri_asis = single.get('riichi', {})
    w_ri = window.get('riichi', {})
    # A3 自身的顺序敏感度 = 规则法地板 − 真顺序盲地板（逐层）
    a3_order_sens = ({k: {'floor_rule': pct(single.get(k, {}), 'floor_dd'),
                          'floor_order_blind': pct(single_ob.get(k, {}), 'floor_dd'),
                          'sensitivity': (pct(single.get(k, {}), 'floor_dd') or 0.0)
                          - (pct(single_ob.get(k, {}), 'floor_dd') or 0.0)}
                     for k in ('all', 'nonriichi', 'riichi')}
                     if single_ob is not None else None)
    verdicts = {
        'RM12': {'rule': '反事实斜率 b>=0.3 且 r2>=0.2（时序流真的在用顺序）',
                 'value': {'b': cf.get('slope_b'), 'r2': cf.get('r2')},
                 'passed': bool(cf.get('slope_b') is not None
                                and cf['slope_b'] >= 0.3 and cf.get('r2', 0) >= 0.2)},
        'RM13': {'rule': f'非立直层 单张扣地板复现率 >= 90%（地板={floor_kind}；'
                         f'A2b 现状 OB 81.9% / 原样 A3 67%）',
                 'value': pct(s_nr, 'net_minus_floor_pct'),
                 'value_rule_floor_as_is': pct(s_nr_asis, 'net_minus_floor_pct'),
                 'passed': bool((pct(s_nr, 'net_minus_floor_pct') or -1e9) >= 90.0)},
        'RM14': {'rule': f'立直层 单张扣地板复现率 >= +30%（地板={floor_kind}；'
                         f'A2b 现状 OB +31.0% / 原样 A3 −12%；⚠️ 阈值无区分力）',
                 'value': pct(s_ri, 'net_minus_floor_pct'),
                 'value_rule_floor_as_is': pct(s_ri_asis, 'net_minus_floor_pct'),
                 'passed': bool((pct(s_ri, 'net_minus_floor_pct') or -1e9) >= 30.0)},
        'RM15': {'rule': '立直层 窗口超地板 CI95 下界 > 0（A2b 现状显著为负）',
                 'value': (w_ri.get('excess_ci95') or [None])[0],
                 'passed': bool((w_ri.get('excess_ci95') or [-1e9])[0] > 0.0)},
    }
    # P1b 预注册判决（新增段；历史 verdicts 保持逐位不变）
    p1b_verdicts = build_p1b_verdicts(single_ob, window, cf, single)

    report = {
        'meta': {'ckpt': str(CKPT), 'n': n, 'boot': B, 'generated_at': time.time(),
                 'cache': str(CACHE), 'cache_schema': CACHE_SCHEMA,
                 'floor_kind': floor_kind,
                 'verdict_protocol': ('verdicts=历史（A2b 时代 90%/+30%，仅存档）；'
                                      'p1b_verdicts=A2c 基线（主 RM12/RM15，'
                                      '不退 RM13/RM14）'),
                 'p1b_baseline': P1B_BASELINE,
                 'dd_convention': ('聚合/表头 = (非自然−自然)；cells 里两种顺序都给'
                                   '（m∈{6,7,8} 二者反号，勿混，见 README §12.4）'),
                 'protocol': '202609 holdout + seed 20261016（与 s3158 系列同源）'},
        'single_tile': single,
        'single_tile_order_blind_floor': single_ob,
        'a3_order_sensitivity': a3_order_sens,
        'window_pm2': window,
        'counterfactual': cf,
        'verdicts': verdicts,
        'p1b_verdicts': p1b_verdicts,
    }
    out_path = Path(args.out) if args.out else Path(CKPT).parent / 'order_metrics.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)

    print('\n---- 历史判据（A2b 时代 90%/+30%，仅存档，不作 P1b 判决）----', flush=True)
    for k, v in verdicts.items():
        print(f'  {k}: {"PASS" if v["passed"] else "FAIL"}   {v["rule"]}   '
              f'value={v["value"]}', flush=True)
    print('\n---- P1b 判据（A2c 基线；主判据 RM12/RM15，RM13/RM14 不退线）----',
          flush=True)
    for k, v in p1b_verdicts.items():
        if k == 'meta':
            continue
        print(f'  {k} [{v["role"]}]: {"PASS" if v["passed"] else "FAIL"}   '
              f'{v["rule"]}   value={v["value"]}', flush=True)
    print(f'  baseline={p1b_verdicts["meta"]["baseline"]}；'
          f'RM14 +50 < A2c 66.2 ⇒ 不作为新线', flush=True)
    print(f'\n[out] {out_path}', flush=True)


if __name__ == '__main__':
    main()
