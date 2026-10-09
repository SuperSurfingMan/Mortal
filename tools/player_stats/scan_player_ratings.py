#!/usr/bin/env python3
"""扫自有凤凰卓牌谱，按长期成绩自建玩家强度榜 → 供 dataset.player_names_files 用。

牌谱本身没有段位字段，但「同一批人反复打」可以自证强度：按最终得分排顺位，
累计每位玩家的 局数 / 平均顺位 / 平均顺位点（天凤 [90,45,0,-135]），
再按 avg_pt（等价于 avg_rank）排序取 top-N，输出名单文件。

用法（在仓库根目录）：
    python tools/player_stats/scan_player_ratings.py \
        --globs 'dataset/mjai/houou4/202[2-6]??/*.json.gz' \
        --out dataset/index/player_ratings.csv \
        --names-out dataset/index/top_players.txt --top 300 --min-games 300 \
        --workers 24 --progress-every 20000
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob as globmod
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

PTS = [90.0, 45.0, 0.0, -135.0]
TOTAL = 250000.0   # 南场结束前理论上界，仅用于无 end_game 时的兜底判断


def scan_one(path: str):
    """返回 ([(name, rank_index, pt), ...], 局数) 或 (None, 错误原因)。"""
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            text = f.read()
    except Exception as exc:  # noqa: BLE000
        return None, type(exc).__name__

    scores = [25000.0] * 4
    names = None
    kyoku = 0
    ended = False
    for line in text.splitlines():
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:  # noqa: BLE000
            return None, 'bad_json'
        t = o.get('type')
        if t == 'start_game':
            names = o.get('names')
        elif t == 'start_kyoku':
            kyoku += 1
            if kyoku == 1:
                s = o.get('scores')
                if isinstance(s, list) and len(s) == 4:
                    scores = [float(x) for x in s]
        elif t in ('hora', 'ryukyoku'):
            d = o.get('deltas')
            if isinstance(d, list) and len(d) == 4:
                for i in range(4):
                    scores[i] += float(d[i])
        elif t == 'end_game':
            ended = True

    if not names or len(names) != 4:
        # 注：自建 2026 数据（jansou 输出）的 start_game 只有 rules、**没有 names**，
        # 因此强度榜只能覆盖 HF 的 2009~2025 部分；这类文件归入 no_names 单独计数。
        return None, 'no_names'
    if not ended or kyoku == 0:
        return None, 'incomplete'

    order = sorted(range(4), key=lambda i: -scores[i])
    rows = []
    for rank, seat in enumerate(order):
        rows.append((str(names[seat]), rank, PTS[rank]))
    return rows, ''


def _map(path):
    return scan_one(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--globs', nargs='+', required=True)
    ap.add_argument('--out', default='dataset/index/player_ratings.csv')
    ap.add_argument('--names-out', default='')
    ap.add_argument('--top', type=int, default=300, help='写进 names-out 的人数')
    ap.add_argument('--min-games', type=int, default=300)
    ap.add_argument('--min-avg-pt', type=float, default=-1e9,
                    help='入选名单的最低平均顺位点（比如 8.0 只要明显长期正分的人）')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--progress-every', type=int, default=20000)
    ap.add_argument('--limit', type=int, default=0, help='只扫前 N 个文件（冒烟用）')
    args = ap.parse_args()

    files = []
    for pat in args.globs:
        files.extend(globmod.glob(pat, recursive=True))
    files.sort(reverse=True)
    if args.limit:
        files = files[:args.limit]
    print(f'待扫文件: {len(files):,} | workers={args.workers}', flush=True)

    games = collections.Counter()
    rank_sum = collections.Counter()
    pt_sum = collections.Counter()
    kyoku_sum = collections.Counter()
    errs = collections.Counter()

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows, err in ex.map(_map, files, chunksize=64):
            done += 1
            if rows is None:
                errs[err] += 1
                continue
            for name, rank, pt in rows:
                games[name] += 1
                rank_sum[name] += rank + 1
                pt_sum[name] += pt
            if done % args.progress_every == 0:
                el = time.time() - t0
                rate = done / el
                eta = (len(files) - done) / max(rate, 1e-9)
                print(f'  {done:,}/{len(files):,} files | {rate:.0f} files/s | '
                      f'玩家 {len(games):,} | 预计还需 {eta/3600:.2f} h', flush=True)

    el = time.time() - t0
    print(f'扫描完成: {done:,} 文件 / {el/3600:.2f} h | 失败 {sum(errs.values()):,} {dict(errs)}', flush=True)

    rows = []
    for name, n in games.items():
        rows.append((name, n, rank_sum[name] / n, pt_sum[name] / n))
    rows.sort(key=lambda r: (-r[3], r[2]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # 天凤昵称可以含逗号 → 必须用 csv 引号转义，否则下游解析错位
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['player_name', 'games', 'avg_rank', 'avg_pt'])
        for name, n, ar, ap_ in rows:
            w.writerow([name, n, f'{ar:.6f}', f'{ap_:.6f}'])
    print(f'已写 {args.out}（{len(rows):,} 位玩家）', flush = True)
    
    elig = [r for r in rows if r[1] >= args.min_games and r[3] >= args.min_avg_pt]
    print(f'满足 min_games>={args.min_games} 的玩家: {len(elig):,} 位；'
          f'top1 平均顺位点 {elig[0][3]:.2f}（{elig[0][0]}, {elig[0][1]} 局）' if elig else '无', flush=True)
    if args.names_out and elig:
        take = elig[:args.top]
        os.makedirs(os.path.dirname(os.path.abspath(args.names_out)), exist_ok=True)
        with open(args.names_out, 'w', encoding='utf-8') as f:
            for name, n, _, _ in take:
                f.write(f'{name}\n')
        print(f'已写名单 {args.names_out}（{len(take)} 人，阈值 avg_pt>={take[-1][3]:.2f}）', flush=True)


if __name__ == '__main__':
    main()
