#!/usr/bin/env python3
"""汇总双向 ladder 结果，给出带置信区间的强度差判读。

约定（与 ladder.py 一致）：每个半庄 1 个挑战者 + 3 个卫冕者，顺位点 [90,45,0,-135]
（四家和为 0，故零和）。A = 新当挑战者、旧当卫冕者；B = 反向。
    avg_pt_A ≈ +δ，avg_pt_B ≈ −δ   →   δ = (avg_pt_A − avg_pt_B) / 2
SE 用每次对战独立的保守估计（同牌山配对的真实方差更小，故结论偏保守）。
"""
from __future__ import annotations

import argparse
import ast
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mjai_score import challenger_pt  # noqa: E402  权威计分口径（与 stat.rs 对齐）

PTS = [90.0, 45.0, 0.0, -135.0]
RE_RANK = re.compile(r'raw rankings = (\[[^\]]+\])')
RE_SEED = re.compile(r'seed_start = (\d+)')


def load(dir_path: str, prefix: str):
    out = []
    for name in sorted(os.listdir(dir_path)):
        if not (name.startswith(prefix + '_') and name.endswith('.log')):
            continue
        txt = open(os.path.join(dir_path, name), encoding='utf-8', errors='ignore').read()
        m = RE_RANK.search(txt)
        s = RE_SEED.search(txt)
        if not m:
            print(f'  !! {name} 没有结果（可能未跑完）')
            continue
        counts = ast.literal_eval(m.group(1).replace('  ', ' ').strip())
        out.append((name, int(s.group(1)) if s else -1, [int(c) for c in counts]))
    return out


def stats(blocks):
    n = sum(sum(c) for _, _, c in blocks)
    total = sum(cnt * PTS[i] for _, _, c in blocks for i, cnt in enumerate(c))
    mean = total / n
    var = sum(cnt * (PTS[i] - mean) ** 2 for _, _, c in blocks for i, cnt in enumerate(c)) / (n - 1)
    return n, mean, var


# ---------------------------------------------------------------------------
# 可选的“按 set 的 block bootstrap”（评审建议）：把同一牌山的 4 局当一个块重采样，
# 不假设局与局独立；顺便给出组内相关的经验估计，判断解析 SE 是保守还是低估。
# 需要牌谱目录（A_*/B_* 每任务一个目录，文件名 <seed>_<key>_<seat>.json.gz）。
# ---------------------------------------------------------------------------

def _pt_job(job):
    path, name = job
    return challenger_pt(path, name)


def collect_set_pts(logs_root: str, prefix: str, chal_name: str, workers: int):
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for sub in sorted(os.listdir(logs_root)):
        d = os.path.join(logs_root, sub)
        if not (sub.startswith(prefix + '_') and os.path.isdir(d)):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.json.gz'):
                jobs.append((os.path.join(d, fn), chal_name))
    if not jobs:
        return None, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pts = list(ex.map(_pt_job, jobs, chunksize=64))
    sets = defaultdict(list)
    for (path, _), pt in zip(jobs, pts):
        if pt is not None:
            set_id = os.path.basename(path).split('_')[0]   # 同 seed → 同 set（同牌山四座位）
            sets[set_id].append(pt)
    return sets, len(jobs)


def bootstrap_report(A, B, logs_root, chal_name, n_boot, seed, workers):
    setsA, nA_files = collect_set_pts(logs_root, 'A', chal_name, workers)
    setsB, nB_files = collect_set_pts(logs_root, 'B', chal_name, workers)
    if not setsA or not setsB:
        print(f'  （bootstrap 跳过：牌谱不全 A={0 if not setsA else len(setsA)} sets / '
              f'B={0 if not setsB else len(setsB)} sets；先跑完两个方向再复核）')
        return

    def flat(sets):
        return [pt for v in sets.values() for pt in v]

    XA, XB = flat(setsA), flat(setsB)
    m = lambda xs: sum(xs) / len(xs)
    delta_hat = (m(XA) - m(XB)) / 2.0

    rng = random.Random(seed)
    kA, kB = list(setsA.values()), list(setsB.values())
    boot = []
    for _ in range(n_boot):
        sa = [kA[rng.randrange(len(kA))] for _ in range(len(kA))]
        sb = [kB[rng.randrange(len(kB))] for _ in range(len(kB))]
        ma = sum(sum(v) for v in sa) / sum(len(v) for v in sa)
        mb = sum(sum(v) for v in sb) / sum(len(v) for v in sb)
        boot.append((ma - mb) / 2.0)
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot)]
    se_boot = statistics.pstdev(boot)

    # 组内相关（ICC 粗估）：ρ = 1 − 组内方差/总方差；ρ>0 → 同 set 的 4 局正相关
    def icc(sets):
        xs = flat(sets)
        if len(xs) < 2:
            return float('nan')
        overall = statistics.pvariance(xs)
        within = statistics.fmean([statistics.pvariance(v) if len(v) > 1 else 0.0
                                   for v in sets.values()])
        return 1 - within / overall if overall > 0 else float('nan')

    print('\n---- block bootstrap（按 set 重采样，不假设局间独立）----')
    print(f'  规模：A {len(setsA):,} sets / {len(XA):,} 局；B {len(setsB):,} sets / {len(XB):,} 局'
          f'（牌谱文件 {nA_files:,}+{nB_files:,}）')
    print(f'  组内相关粗估 ρ：A {icc(setsA):+.3f}，B {icc(setsB):+.3f}'
          f'（>0 = 同牌山 4 局正相关）')
    print(f'  bootstrap δ = {delta_hat:+.3f} pt   SE={se_boot:.3f}   95%CI=[{lo:+.2f}, {hi:+.2f}]'
          f'（{n_boot} 次重采样）')
    verdict = '有效' if lo > 0 else ('变弱' if hi < 0 else '无结论')
    print(f'  按 bootstrap CI 判读：{verdict}（CI “跨 0”即无结论）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='logs/ladder_r2')
    ap.add_argument('--new', default='round2/mortal.pth')
    ap.add_argument('--old', default='mortal_step_000200000.pth')
    ap.add_argument('--logs-root', default='',
                    help='牌谱根目录（含 A_*/B_*）；给出后额外做按 set 的 block bootstrap')
    ap.add_argument('--challenger-name', default='challenger',
                    help='牌谱里挑战者的名字（ladder.py 用 challenger；test_play 用 mortal）')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--bseed', type=int, default=20261011)
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()

    A = load(args.dir, 'A')
    B = load(args.dir, 'B')
    if not A or not B:
        raise SystemExit(f'缺少 A/B 方向结果：A={len(A)} B={len(B)}')

    print(f'新 = {args.new}    旧 = {args.old}')
    print('\n各分块明细（挑战者侧 avg_pt；A=新当挑战者，B=旧当挑战者）')
    print(f'{"块":<14s} {"seed":>7s} {"sets":>6s} {"1st/2nd/3rd/4th":>22s} {"avg_pt":>9s}')
    for tag, blocks in (('A', A), ('B', B)):
        for name, seed, c in blocks:
            n = sum(c)
            pt = sum(ci * PTS[i] for i, ci in enumerate(c)) / n
            print(f'{name[:-4]:<14s} {seed:>7d} {n//4:>6d} {str(c):>22s} {pt:>+9.2f}')

    nA, mA, vA = stats(A)
    nB, mB, vB = stats(B)
    delta = (mA - mB) / 2.0
    se = math.sqrt(vA / nA + vB / nB) / 2.0
    t = delta / se if se > 0 else float('nan')
    print(f'\nA（新当挑战者, {nA} 半庄）  avg_pt = {mA:+.3f}  SE = {math.sqrt(vA/nA):.3f}')
    print(f'B（旧当挑战者, {nB} 半庄）  avg_pt = {mB:+.3f}  SE = {math.sqrt(vB/nB):.3f}')
    print(f'\n强度差 δ(新 − 旧) = {delta:+.3f} pt/半庄   SE(δ) = {se:.3f}   t = {t:+.2f}'
          f'   95%CI = [{delta - 1.96*se:+.2f}, {delta + 1.96*se:+.2f}]')
    n_sets = (nA + nB) // 4
    if abs(t) >= 2.0:
        print(f'判读：δ 在 2σ 外 → 第 2 轮【{"有效" if delta > 0 else "反而变弱"}】（{n_sets} sets）')
    else:
        # 要达 2σ 需 SE(δ) ≤ |δ|/2 → 样本量需再乘 (2·SE/|δ|)²
        factor = (2.0 * se / max(abs(delta), 1e-9)) ** 2
        need_sets = int(math.ceil(n_sets * factor))
        print(f'判读：|t|={abs(t):.2f} < 2 → 仍【无统计结论】（{n_sets} sets）')
        print(f'      若真实效应就是 {delta:+.2f} pt，要达 2σ 共需约 {need_sets:,} sets'
              f'（当前 {n_sets:,}，即再乘 {factor:.1f} 倍）')

    if args.logs_root:
        bootstrap_report(A, B, args.logs_root, args.challenger_name,
                         args.bootstrap, args.bseed, args.workers)


if __name__ == '__main__':
    main()
