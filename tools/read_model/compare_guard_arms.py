#!/usr/bin/env python3
"""守卫臂配对比较：同种子块上 ``fixed − legacy`` 的逐 set 差与 bootstrap CI。

背景：2026-11-02 修正 risk 双乘 P(听) 后，需要判断修正版在**同一批牌山**上
相对旧口径是优、平还是劣。单臂的 ``avg_pt`` 与 ``run_meta`` 只有汇总值，
不能直接相减；本脚本按 set 配对（同 direction + 同 seed 的 4 局）后重采样。

用法::

    python tools/read_model/compare_guard_arms.py \
        --ref legacy=mortal/runs/guard_fix/legacy_l10 \
        --arm fixed=mortal/runs/guard_fix/fixed_l75 \
        --arm fixed_minus=mortal/runs/guard_fix/fixed_l60 \
        --workers 16 --n-boot 10000

每个 ``--arm name=dir1,dir2`` 可给多个根目录（例如 A/B 分开放），脚本会合并。
``--ref`` 是配对基准（通常是 legacy λ=10）。

判据（与计划一致）：
- 单臂 δ = (A − B)/2，bootstrap CI 下界 >0 才算有效；
- 配对 Δ = fixed − legacy，CI 下界 > −0.30 为非劣，>0 为优于。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'tools' / 'training'))
import agg_ladder as A  # noqa: E402

PTS = [90.0, 45.0, 0.0, -135.0]


def load_arm(roots, workers):
    """返回 {'A': {set_id: [pt...]}, 'B': {...}}。"""
    out = {'A': defaultdict(list), 'B': defaultdict(list)}
    for root in roots:
        root = str(root)
        for direction in ('A', 'B'):
            sets, n_files = A.collect_set_pts(root, direction, 'challenger', workers)
            if not sets:
                print(f'[warn] {root} {direction}: 没有牌谱（n_files={n_files}）')
                continue
            for k, v in sets.items():
                out[direction][k].extend(v)
    return out


def set_means(sets):
    return {k: statistics.fmean(v) for k, v in sets.items() if v}


def arm_delta(arm, n_boot, seed):
    """双向 δ=(A−B)/2 与 bootstrap CI。"""
    a = set_means(arm['A'])
    b = set_means(arm['B'])
    if not a or not b:
        return None
    delta = statistics.fmean(a.values()) - statistics.fmean(b.values())
    delta /= 2.0
    rng = random.Random(seed)
    ka, kb = list(a.values()), list(b.values())
    boot = []
    for _ in range(n_boot):
        ma = statistics.fmean([ka[rng.randrange(len(ka))] for _ in range(len(ka))])
        mb = statistics.fmean([kb[rng.randrange(len(kb))] for _ in range(len(kb))])
        boot.append((ma - mb) / 2.0)
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot)]
    return {
        'n_sets_A': len(ka), 'n_sets_B': len(kb),
        'A_pt': statistics.fmean(a.values()),
        'B_pt': statistics.fmean(b.values()),
        'delta': delta,
        'boot_se': statistics.pstdev(boot),
        'ci': [lo, hi],
    }


def paired_delta(ref, arm, n_boot, seed):
    """双臂同种子的 δ 差：Δδ = δ_arm − δ_ref = [(ΔA) − (ΔB)] / 2。

    注意：不能把 A/B 的逐 set 原始 pt 差直接平均——B 方向挑战者是“旧”侧，
    符号相反；直接平均会得到 (ΔA+ΔB)/2，不是强度差。这里按方向分别配对，
    再组合成 δ 差。
    """
    rA, xA = set_means(ref['A']), set_means(arm['A'])
    rB, xB = set_means(ref['B']), set_means(arm['B'])
    a_keys = sorted(set(rA) & set(xA))
    b_keys = sorted(set(rB) & set(xB))
    if not a_keys or not b_keys:
        return None
    dA = [xA[k] - rA[k] for k in a_keys]
    dB = [xB[k] - rB[k] for k in b_keys]
    mean_a = statistics.fmean(dA)
    mean_b = statistics.fmean(dB)
    mean = (mean_a - mean_b) / 2.0

    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sa = [dA[rng.randrange(len(dA))] for _ in range(len(dA))]
        sb = [dB[rng.randrange(len(dB))] for _ in range(len(dB))]
        boot.append((statistics.fmean(sa) - statistics.fmean(sb)) / 2.0)
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot)]
    return {
        'n_A': len(dA), 'n_B': len(dB), 'n_pairs': len(dA) + len(dB),
        'mean': mean,
        'boot_se': statistics.pstdev(boot),
        'ci': [lo, hi],
        'delta_A': mean_a,
        'delta_B': mean_b,
        'win_A': sum(1 for d in dA if d > 0) / len(dA),
        'win_B': sum(1 for d in dB if d < 0) / len(dB),
    }


def fmt_ci(ci):
    return f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='name=dir1,dir2（配对基准）')
    ap.add_argument('--arm', action='append', required=True,
                    help='name=dir1,dir2（可重复）')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    def parse(spec):
        name, _, dirs = spec.partition('=')
        return name, [x for x in dirs.split(',') if x]

    ref_name, ref_roots = parse(args.ref)
    print(f'[ref] {ref_name}: {ref_roots}')
    ref = load_arm(ref_roots, args.workers)
    ref_stat = arm_delta(ref, args.n_boot, args.seed)
    if ref_stat is None:
        print('[error] ref 没有可用牌谱')
        sys.exit(1)
    print(f"{'arm':<14s} {'nA':>6s} {'nB':>6s} {'A_pt':>8s} {'B_pt':>8s} "
          f"{'delta':>8s} {'95%CI':>18s}")
    print(f"{ref_name:<14s} {ref_stat['n_sets_A']:>6d} {ref_stat['n_sets_B']:>6d} "
          f"{ref_stat['A_pt']:>+8.3f} {ref_stat['B_pt']:>+8.3f} "
          f"{ref_stat['delta']:>+8.3f} {fmt_ci(ref_stat['ci']):>18s}")

    summary = {'ref': {'name': ref_name, 'stat': ref_stat}, 'arms': {}}
    print('\n==== 配对 Δ = arm − ref（同 direction+seed 的逐 set 差）====')
    for spec in args.arm:
        name, roots = parse(spec)
        arm = load_arm(roots, args.workers)
        stat = arm_delta(arm, args.n_boot, args.seed)
        pd = paired_delta(ref, arm, args.n_boot, args.seed)
        if stat is None or pd is None:
            print(f'[warn] {name}: 牌谱不足，跳过')
            continue
        print(f"{name:<14s} {stat['n_sets_A']:>6d} {stat['n_sets_B']:>6d} "
              f"{stat['A_pt']:>+8.3f} {stat['B_pt']:>+8.3f} "
              f"{stat['delta']:>+8.3f} {fmt_ci(stat['ci']):>18s}")
        print(f"  paired Δδ={pd['mean']:+.3f}  SE={pd['boot_se']:.3f}  "
              f"95%CI={fmt_ci(pd['ci'])}  nA/nB={pd['n_A']}/{pd['n_B']}  "
              f"ΔA={pd['delta_A']:+.3f} ΔB={pd['delta_B']:+.3f}  "
              f"winA={pd['win_A']:.1%} winB={pd['win_B']:.1%}")
        verdict = ('优于' if pd['ci'][0] > 0 else
                   ('非劣' if pd['ci'][0] > -0.30 else '劣于/不确定'))
        print(f'  判读：{verdict}（非劣界 −0.30；>0 为优于；Δδ=δ_arm−δ_ref）')
        summary['arms'][name] = {'roots': roots, 'stat': stat, 'paired': pd,
                                 'verdict': verdict}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                  encoding='utf-8')
        print(f'\n[out] {args.out}')


if __name__ == '__main__':
    main()
