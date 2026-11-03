#!/usr/bin/env python3
"""校验 mjai 树的 start_game.names 是否真的可用（P0 修复的验收工具，2026-10-12）。

回答四个问题（对应 NEXT_STEPS_ANALYSIS.md §2.1 的验收项）：

  1. **覆盖率**：多少文件带 / 不带 names（缺 names 的文件在白名单模式下产不出样本）；
  2. **席位量**：座位数、去重玩家数（决定数据量的是「白名单席位」而不是文件数）；
  3. **名字可用性（硬证据）**：抽样名字与「白名单 / 强度榜玩家池」的命中率。
     这条是必需的——本项目的名字坑很隐蔽：未 unquote 的百分号编码名结构完全合法、
     `validate_logs` 全绿，但命中率是 0%，会静默地把数据变废；
  4. **可疑名**：解码后仍含 `%XX` 的名字数（双重编码嫌疑）。

用法（仓库根目录 /mnt/data8/mortal/Mortal 下运行）：
    # 抽查 2 万个文件（秒级）
    python tools/tenhou_data/verify_names.py \
        --globs 'dataset/mjai/houou4_names/2026*/*.json.gz' \
        --whitelist dataset/index/top_players_400_v2.txt \
        --ratings dataset/index/player_ratings_2022_2026H1_v2.csv \
        --sample 20000 --workers 28

    # 全量（慢，13 万文件约数十秒）
    python tools/tenhou_data/verify_names.py --globs 'dataset/mjai/houou4_names/**/*.json.gz' --full

    # 对照旧树（应显示 0 个带 names）
    python tools/tenhou_data/verify_names.py --globs 'dataset/mjai/houou4/2026*/*.json.gz' --sample 2000
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob as globmod
import gzip
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor

_RE_PCT = re.compile(r'%[0-9A-Fa-f]{2}')


def read_first_line(path: str):
    """返回 (names, status)：names 为 4 元组或 None。"""
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            o = json.loads(f.readline())
    except Exception:  # noqa: BLE001
        return None, 'unreadable'
    if not isinstance(o, dict) or o.get('type') != 'start_game':
        return None, 'not_start_game'
    names = o.get('names')
    if not isinstance(names, list) or len(names) != 4 or not all(isinstance(x, str) for x in names):
        return None, 'no_names'
    if not all(names):
        return None, 'empty_names'
    return tuple(names), 'ok'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--globs', nargs='+', required=True)
    ap.add_argument('--whitelist', default='', help='白名单文件（每行一个名字），可省略')
    ap.add_argument('--ratings', default='', help='强度榜 CSV（取 player_name 列作玩家池），可省略')
    ap.add_argument('--sample', type=int, default=20000, help='抽样文件数（--full 时忽略）')
    ap.add_argument('--full', action='store_true', help='全量扫描（不抽样）')
    ap.add_argument('--seed', type=int, default=20261012)
    ap.add_argument('--workers', type=int, default=28)
    args = ap.parse_args()

    files = []
    for pat in args.globs:
        files.extend(globmod.glob(pat, recursive=True))
    files.sort()
    total = len(files)
    if not total:
        raise SystemExit(f'没匹配到任何文件: {args.globs}')
    if args.full or args.sample >= total:
        sample = files
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(files, args.sample)
        sample.sort()
    print(f'匹配文件 {total:,} | 本次扫描 {len(sample):,}'
          f'（{"全量" if len(sample) == total else f"随机抽样 seed={args.seed}"}）', flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(read_first_line, sample, chunksize=512))

    whitelist = set()
    if args.whitelist:
        whitelist = {l.strip() for l in open(args.whitelist, encoding='utf-8') if l.strip()}
    pool = set()
    if args.ratings:
        pool = {r['player_name'] for r in csv.DictReader(open(args.ratings, encoding='utf-8'))}

    status = collections.Counter()
    seats = 0
    distinct = set()
    susp = 0
    wl_seats = 0
    pool_seats = 0
    wl_files = 0
    for names, st in results:
        status[st] += 1
        if names is None:
            continue
        for n in names:
            seats += 1
            distinct.add(n)
            if _RE_PCT.search(n):
                susp += 1
            if whitelist and n in whitelist:
                wl_seats += 1
            if pool and n in pool:
                pool_seats += 1
        if whitelist and any(n in whitelist for n in names):
            wl_files += 1

    ok = status['ok']
    print(f'\n文件状态: ' + '  '.join(f'{k}={v:,}' for k, v in sorted(status.items())))
    print(f'  带 names 比例: {ok / len(sample):.1%}')
    print(f'座位/去重玩家: {seats:,} / {len(distinct):,}')
    if susp:
        print(f'  [警告] 含 %XX 的可疑名字: {susp:,}')
    if whitelist:
        print(f'白名单  : 人数 {len(whitelist)} | 命中座位 {wl_seats:,}/{seats:,} ({wl_seats / max(seats, 1):.2%})'
              f' | 命中文件 {wl_files:,}/{ok:,} ({wl_files / max(ok, 1):.2%})')
    if pool:
        print(f'榜单池  : 人数 {len(pool):,} | 命中座位 {pool_seats:,}/{seats:,}'
              f' ({pool_seats / max(seats, 1):.2%})')

    verdict = 'PASS' if ok and (not whitelist or wl_seats > 0) else 'FAIL'
    if not ok:
        verdict = 'FAIL（没有任何文件带可用 names）'
    elif whitelist and wl_seats == 0:
        verdict = 'FAIL（白名单命中 0 —— 名字可能仍是编码态或与训练口径不一致）'
    print(f'\n判读: {verdict}', flush=True)


if __name__ == '__main__':
    main()
