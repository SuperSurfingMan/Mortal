#!/usr/bin/env python3
"""按年份审计牌风/强度漂移（数据扩窗前置）：决定旧年份能否并入训练窗。

回答三个问题：
  1. 旧年份的牌风离 2022+ 有多远（立直/副露/和了/流局/和了点/局长）；
  2. 名单玩家（top400_v2）在旧年份的席位占比 —— 决定「按名单过滤」还剩多少数据；
  3. 旧年份的玩家池规模与重叠度（同一批人反复打才自证强度）。

口径与 `tools/mjai_score.py` 一致（每局 start_kyoku 重置分数、reach_accepted −1000）。

用法（仓库根目录）：
    python tools/data_audit/year_style_audit.py \
        --years 2019 2020 2021 2022 2025 --per-year 4000 --workers 24 \
        --names dataset/index/top_players_400_v2.txt \
        --out mortal/logs/data_audit/year_style.json
"""
from __future__ import annotations

import argparse
import collections
import glob as globmod
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mjai_score import final_scores, load_events  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLOB_TMPL = os.path.join(ROOT, 'dataset/mjai/houou4/{year}??/*.json.gz')

CALLS = ('chi', 'pon', 'daiminkan', 'ankan', 'kakan')


def audit_one(path: str):
    """返回一局的原始计数（由调用方按年聚合）。"""
    try:
        ev = load_events(path)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    out = collections.Counter()
    names = []
    for o in ev:
        t = o.get('type')
        if t == 'start_game':
            names = [n for n in (o.get('names') or []) if n]
            out['has_names'] += 1 if names else 0
        elif t == 'start_kyoku':
            out['kyoku'] += 1
        elif t == 'reach_accepted':
            out['riichi'] += 1
        elif t in CALLS:
            out['fuuro'] += 1
        elif t == 'dahai':
            out['dahai'] += 1
        elif t == 'hora':
            out['hora'] += 1
            # mjai 口径：自摸时 target == actor（无 tsumo 字段）
            if o.get('target') == o.get('actor'):
                out['tsumo'] += 1
            d = o.get('deltas')
            a = o.get('actor')
            if isinstance(d, list) and len(d) == 4 and isinstance(a, int):
                out['hora_pt_sum'] += int(d[a])   # 含供托/本场，跨年比较口径一致
                out['hora_pt_n'] += 1
        elif t == 'ryukyoku':
            out['ryukyoku'] += 1
    sc = final_scores(ev)
    out['spread'] = int(max(sc) - min(sc))
    return {'counts': dict(out), 'names': names, 'kyoku': int(out.get('kyoku', 1) or 1)}, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022, 2025])
    ap.add_argument('--per-year', type=int, default=4000, help='每年抽样文件数（0=全量，慎重）')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=20261024)
    ap.add_argument('--names', default='', help='名单文件（每行一个玩家名），用于席位占比')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    white = set()
    if args.names and os.path.exists(args.names):
        with open(args.names, encoding='utf-8') as f:
            white = {ln.strip() for ln in f if ln.strip()}
    print(f'名单: {args.names or "(无)"}  玩家数={len(white)}', flush=True)

    report = {}
    for year in args.years:
        files = sorted(globmod.glob(GLOB_TMPL.format(year=year)))
        if not files:
            print(f'{year}: 无文件', flush=True)
            continue
        n_total = len(files)
        rng = random.Random(args.seed + year)
        sample = files if args.per_year <= 0 else rng.sample(files, min(args.per_year, n_total))
        t0 = time.time()
        agg = collections.Counter()
        pool = set()
        white_seats = white_games = errs = 0
        chunk = 64
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i in range(0, len(sample), chunk):
                res = ex.map(audit_one, sample[i:i + chunk], chunksize=8)
                for r, err in res:
                    if r is None:
                        errs += 1
                        continue
                    agg.update(r['counts'])
                    pool.update(r['names'])
                    if white:
                        hit = sum(1 for n in r['names'] if n in white)
                        white_seats += hit
                        white_games += 1 if hit else 0
        n = len(sample) - errs
        seats = max(n * 4, 1)
        kyoku = max(agg['kyoku'], 1)
        row = {
            'files_total': n_total,
            'files_sampled': len(sample),
            'files_ok': n,
            'errors': errs,
            'games_per_file': round(agg['kyoku'] / max(n, 1), 2),
            'names_coverage': round(agg['has_names'] / max(n, 1), 4),
            'riichi_per_seat_game': round(agg['riichi'] / seats, 4),
            'fuuro_per_seat_game': round(agg['fuuro'] / seats, 4),
            'agari_per_seat_game': round(agg['hora'] / seats, 4),
            'tsumo_share': round(agg['tsumo'] / max(agg['hora'], 1), 4),
            'ryukyoku_per_kyoku': round(agg['ryukyoku'] / kyoku, 4),
            'avg_hora_pt': round(agg['hora_pt_sum'] / max(agg['hora_pt_n'], 1), 1),
            'dahai_per_kyoku': round(agg['dahai'] / kyoku, 2),
            'avg_final_spread': round(agg['spread'] / max(n, 1), 0),
            'unique_players': len(pool),
            'whitelist_seat_share': round(white_seats / seats, 4) if white else None,
            'whitelist_game_share': round(white_games / max(n, 1), 4) if white else None,
            'seconds': round(time.time() - t0, 1),
        }
        report[str(year)] = row
        print(f'{year}: ' + json.dumps(row, ensure_ascii=False), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'写出 {args.out}', flush=True)


if __name__ == '__main__':
    main()
