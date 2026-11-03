#!/usr/bin/env python3
"""构建「局内 4 人强度」加权重表（数据质量加权探针，2026-11-02）。

思路：同一局里 4 个对手的长期强度不同，弱对手的坏习惯可能被 BC 模仿。
本工具按每个牌谱的 4 名玩家，用 `player_ratings_*_v3.csv` 的长期 avg_pt 取
**最弱一席（min）** 作为该局质量分，再按分位切档给出整数重复次数（1..max）。
训练侧（train.py `[dataset.strength_weight]`）只做文件级重复，不改损失代码，
也不影响 file_index 的 manifest；所有文件至少保留 1 次，不做过滤。

用法（仓库根目录）：
    python tools/training/build_strength_weights.py \
        --index mortal/indexes/reward_ablation_r4_v2.pth \
        --ratings dataset/index/player_ratings_2022_2026H1_v3.csv \
        --out mortal/indexes/strength_weights_v1.pth \
        --stat min --min-games 100 --cuts 0.5,0.75,0.9 --repeats 1,1,2,3 \
        --workers 16
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import torch

_RATINGS = {}
_STAT = 'min'
_MIN_RATED = 2


def load_ratings(path, min_games):
    """player_name -> avg_pt（只保留 games >= min_games）。"""
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                games = int(row['games'])
                if games < min_games:
                    continue
                out[row['player_name']] = float(row['avg_pt'])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def scan_one(path):
    """返回 (path, n_rated, stat_value_or_None)。只读首行 start_game.names。"""
    ratings, stat, min_rated = _RATINGS, _STAT, _MIN_RATED
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            names = json.loads(next(f)).get('names') or []
    except Exception:
        return path, 0, None
    vals = [ratings[n] for n in names if n in ratings]
    if len(vals) < min_rated:
        return path, len(vals), None
    if stat == 'min':
        v = min(vals)
    elif stat == 'max':
        v = max(vals)
    else:
        v = sum(vals) / len(vals)
    return path, len(vals), v


def _init_worker(ratings, stat, min_rated):
    global _RATINGS, _STAT, _MIN_RATED
    _RATINGS, _STAT, _MIN_RATED = ratings, stat, min_rated


def quantiles(values, qs):
    """线性插值分位点（避免依赖 numpy 版本差异）。"""
    xs = sorted(values)
    out = []
    for q in qs:
        if not xs:
            out.append(0.0)
            continue
        pos = (len(xs) - 1) * q
        lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
        out.append(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='mortal/indexes/reward_ablation_r4_v2.pth')
    ap.add_argument('--ratings', default='dataset/index/player_ratings_2022_2026H1_v3.csv')
    ap.add_argument('--out', default='mortal/indexes/strength_weights_v1.pth')
    ap.add_argument('--stat', choices=['min', 'mean', 'max'], default='min')
    ap.add_argument('--min-games', type=int, default=100,
                    help='玩家至少打过多少局才用于评分（降噪）')
    ap.add_argument('--min-rated', type=int, default=2,
                    help='一局至少有几名玩家有评分才参与分档')
    ap.add_argument('--cuts', default='0.5,0.75,0.9', help='质量分的分位切点')
    ap.add_argument('--repeats', default='1,1,2,3', help='各档重复次数（长度=切点+1）')
    ap.add_argument('--max-repeat', type=int, default=3)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='仅扫描前 N 个文件（调试用）')
    args = ap.parse_args()

    qs = [float(x) for x in args.cuts.split(',') if x.strip()]
    reps = [int(x) for x in args.repeats.split(',') if x.strip()]
    if len(reps) != len(qs) + 1:
        raise SystemExit(f'--repeats 长度应为 {len(qs) + 1}（当前 {len(reps)}）')
    reps = [max(1, min(r, args.max_repeat)) for r in reps]

    ratings = load_ratings(args.ratings, args.min_games)
    print(f'[weights] ratings: {len(ratings):,} 名玩家（games>={args.min_games}）')

    index = torch.load(args.index, weights_only=True)
    file_list = list(index['file_list'])
    if args.limit:
        file_list = file_list[:args.limit]
    print(f'[weights] files: {len(file_list):,}')

    tasks = file_list
    t0 = time.time()
    stats = [None] * len(file_list)
    n_rated = [0] * len(file_list)
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(ratings, args.stat, args.min_rated)) as ex:
        for i, (_, cnt, v) in enumerate(ex.map(scan_one, tasks, chunksize=512)):
            n_rated[i] = cnt
            stats[i] = v
    dt = time.time() - t0
    print(f'[weights] scan {len(file_list):,} files in {dt:.1f}s '
          f'({len(file_list) / max(dt, 1e-9):,.0f} files/s, workers={args.workers})')

    valid = [v for v in stats if v is not None]
    cuts = quantiles(valid, qs)
    coverage = {k: n_rated.count(k) for k in sorted(set(n_rated))}
    print(f'[weights] rated-per-game={coverage}  valid={len(valid):,}/{len(file_list):,}  '
          f'q{qs}={["%.3f" % c for c in cuts]}')

    repeats = []
    hist = {}
    for v in stats:
        if v is None:
            r = 1
        else:
            r = reps[bisect.bisect_right(cuts, v)]
        repeats.append(r)
        hist[r] = hist.get(r, 0) + 1
    mean_w = sum(repeats) / len(repeats)
    print(f'[weights] repeats={dict(sorted(hist.items()))}  mean={mean_w:.3f}  '
          f'expanded={sum(repeats):,}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({
        'file_list': file_list,
        'repeats': torch.tensor(repeats, dtype=torch.uint8),
        'stat': torch.tensor([float('nan') if v is None else v for v in stats],
                             dtype=torch.float32),
        'meta': {
            'stat': args.stat,
            'min_games': args.min_games,
            'min_rated': args.min_rated,
            'cuts': qs,
            'cut_values': cuts,
            'repeats': reps,
            'max_repeat': args.max_repeat,
            'index': args.index,
            'ratings': args.ratings,
            'n_files': len(file_list),
            'n_valid': len(valid),
            'mean_weight': mean_w,
        },
    }, args.out)
    print(f'[weights] 写入 {args.out}')


if __name__ == '__main__':
    main()
