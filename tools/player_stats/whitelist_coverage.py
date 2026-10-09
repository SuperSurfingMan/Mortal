#!/usr/bin/env python3
"""按不同 top-N 白名单统计其对训练池的覆盖度（命中文件数 / 白名单席位 / 等效样本与步数）。

player_names_files 的语义（libriichi/src/dataset/gameplay.rs::load_events）：
一张牌谱里「名字在白名单内的座位」才会产出训练样本，其余座位被丢弃 →
所以决定数据量的量是「席位总数」，而不是命中文件数。

用法（仓库根目录）：
    python tools/player_stats/whitelist_coverage.py \
        --globs 'dataset/mjai/houou4/202[2-5]??/*.json.gz' \
        --sizes 200,400,600,900,1200 --workers 28
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import gzip
import json
from concurrent.futures import ProcessPoolExecutor


def first_names(path):
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            o = json.loads(f.readline())
        return tuple(o.get('names') or [])
    except Exception:  # noqa: BLE000
        return ()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratings', default='dataset/index/player_ratings_2022_2026H1.csv')
    ap.add_argument('--globs', nargs='+', required=True)
    ap.add_argument('--min-games', type=int, default=400)
    ap.add_argument('--sizes', default='200,400,600,900,1200')
    ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--per-seat-samples', type=int, default=170,
                    help='单座位单半庄的决策样本数（实测 678 样本/文件 ÷ 4 家）')
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    rows = [(r['player_name'], int(r['games']), float(r['avg_pt']))
            for r in csv.DictReader(open(args.ratings, encoding='utf-8'))]
    rows.sort(key=lambda r: -r[2])
    elig = [r for r in rows if r[1] >= args.min_games]
    print(f'榜单人数 {len(rows):,} | 满足 min_games>={args.min_games} 的 {len(elig):,}', flush=True)

    files = []
    for pat in args.globs:
        files.extend(globmod.glob(pat, recursive=True))
    files.sort(reverse=True)
    print(f'训练池文件 {len(files):,}', flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        names = list(ex.map(first_names, files, chunksize=4000))
    named = sum(1 for n in names if n)
    print(f'有 names 的文件 {named:,}（无 names 的是自建 2026 部分，白名单模式下不产出样本）', flush=True)

    for s in args.sizes.split(','):
        k = int(s)
        sel = set(n for n, _, _ in elig[:k])
        hit = sum(1 for n in names if any(x in sel for x in n))
        seats = sum(sum(1 for x in n if x in sel) for n in names)
        samples = seats * args.per_seat_samples
        thr = elig[min(k, len(elig)) - 1][2] if elig else float('nan')
        print(f'top{k:>5}（avg_pt>={thr:+.1f}）: 命中文件 {hit:>7,} ({hit / max(len(names), 1):5.1%}) | '
              f'席位 {seats:>8,} | 样本≈{samples / 1e6:>5.0f}M | '
              f'1 epoch ≈ {samples / args.batch_size / 1000:>5.0f}k 步', flush=True)


main()
