#!/usr/bin/env python3
"""统一行为画像口径：人类凤凰场 vs Mortal，用**同一套 Stat 定义**对比。

## 为什么需要

仓库里"副露率"出现过三个互不一致的数：

| 来源 | 值 |
|---|---|
| MEMORY §2.5（B/756k 行为对照） | 45.66% |
| MEMORY §3.15（年度审计，次数/半庄） | 5.277 次 |
| MEMORY §3.11（S6c 训练期画像） | 31.09% |

同一个指标差出 14~45 个百分点 ⇒ 任何"模型太保守/太激进"的判断都不可靠。

## 口径（唯一真源：`libriichi/src/stat.rs`）

```
round += 1                 # 每个 start_kyoku
agari  += 1                # hora 的 actor（自摸/荣和都算）
houjuu += 1                # hora 的 target，且 target != actor
riichi += 1                # 每个 reach 事件（一局最多 1 次）
fuuro  += 1（每局最多 1）  # 该局有 chi / pon / daiminkan（**不含 ankan/kakan**）
ryukyoku += 1              # 每局每人各 +1
率 = 分子 / round
```

⚠️ `fuuro_rate` 是**「有副露的局占比」**（≤1），不是「副露次数/局」。

## 用法

    python tools/behavior_profile.py --max-files 3000
    python tools/behavior_profile.py --mjai-glob 'dataset/mjai/houou4/2025*/**/*.json.gz'

产物：``mortal/logs/behavior_profile.json`` + stdout 对照表。
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Mortal 侧参照值（出处见注释；均为 riichi.Stat 口径）
MORTAL_REF = {
    'S6c/956k（test_play 末次，MEMORY §3.11）': {
        'agari': 0.2167, 'houjuu': 0.1227, 'fuuro': 0.3109, 'riichi': 0.1855,
    },
    'B/756k（ladder λ=0，runs/guard_fix/null_l0）': {
        'agari': 0.2192, 'houjuu': 0.1286, 'fuuro': 0.2826, 'riichi': 0.2005,
    },
}

KEYS = ('agari', 'houjuu', 'fuuro', 'riichi', 'ryukyoku')
LABEL = {'agari': '和了率', 'houjuu': '放铳率', 'fuuro': '副露局占比',
         'riichi': '立直率', 'ryukyoku': '流局率'}


def iter_events(path: Path):
    """读 mjai 牌谱。**格式是 JSONL**（每行一个事件），不是 JSON 数组。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def scan_log(path: Path):
    """按 stat.rs 的定义统计一局牌谱（4 人各自的计数）。"""
    n = [0] * 4                 # round
    cnt = {k: [0] * 4 for k in KEYS}
    fuuro_flag = [False] * 4
    started = 0

    def flush():
        for i in range(4):
            if fuuro_flag[i]:
                cnt['fuuro'][i] += 1
                fuuro_flag[i] = False

    try:
        for ev in iter_events(path):
            t = ev.get('type')
            if t == 'start_kyoku':
                flush()
                started += 1
                for i in range(4):
                    n[i] += 1
            elif t == 'reach':
                a = ev.get('actor')
                if isinstance(a, int) and 0 <= a < 4:
                    cnt['riichi'][a] += 1
            elif t in ('pon', 'chi', 'daiminkan'):
                a = ev.get('actor')
                if isinstance(a, int) and 0 <= a < 4:
                    fuuro_flag[a] = True
            elif t == 'hora':
                a = ev.get('actor')
                if isinstance(a, int) and 0 <= a < 4:
                    cnt['agari'][a] += 1
                    tg = ev.get('target', a)
                    if isinstance(tg, int) and 0 <= tg < 4 and tg != a:
                        cnt['houjuu'][tg] += 1
                flush()
            elif t == 'ryukyoku':
                for i in range(4):
                    cnt['ryukyoku'][i] += 1
                flush()
    except (json.JSONDecodeError, OSError, EOFError, UnicodeDecodeError):
        return None
    flush()
    if started == 0:
        return None
    return n, cnt


def aggregate(files):
    tot_round = 0
    tot = {k: 0 for k in KEYS}
    used = 0
    for p in files:
        r = scan_log(p)
        if r is None:
            continue
        n, cnt = r
        used += 1
        for i in range(4):
            tot_round += n[i]                 # 每人自己的局数（一个半庄约 8~12 局）
            for k in KEYS:
                tot[k] += cnt[k][i]
    if tot_round == 0:
        return None
    return {'files': used, 'round': tot_round,
            **{k: tot[k] / tot_round for k in KEYS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mjai-glob',
                    default='dataset/mjai/houou4/202[2-5]??/**/*.json.gz',
                    help='人类牌谱 glob（相对仓库根）；默认 = 训练池 2022-2025')
    ap.add_argument('--max-files', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=20261107)
    ap.add_argument('--out', default=str(REPO / 'mortal/logs/behavior_profile.json'))
    args = ap.parse_args()

    files = sorted(REPO.glob(args.mjai_glob))
    if not files:
        sys.exit(f'未找到牌谱：{args.mjai_glob}')
    rng = random.Random(args.seed)
    if len(files) > args.max_files:
        files = rng.sample(files, args.max_files)
    print(f'[data] {args.mjai_glob} → 抽样 {len(files)} 个文件', flush=True)

    human = aggregate(files)
    if human is None:
        sys.exit('解析失败：没有有效牌谱')
    print(f"[data] 有效 {human['files']} 文件 / {human['round']} 人·局\n", flush=True)

    rows = [('人类凤凰场（本脚本实测）', human)] + list(MORTAL_REF.items())
    print(f"{'口径':34s} " + ' '.join(f'{LABEL[k]:>10s}' for k in KEYS))
    print('-' * 100)
    for name, d in rows:
        cells = ' '.join(
            (f'{d[k]*100:9.2f}%' if isinstance(d.get(k), float) and d.get(k) == d.get(k)
             else f'{"—":>10s}')
            for k in KEYS)
        print(f'{name:34s} {cells}')
    print('-' * 100)
    base = human
    for k in KEYS:
        gaps = [(name, (d[k] - base[k]) * 100)
                for name, d in rows[1:]
                if isinstance(d.get(k), float) and d.get(k) == d.get(k)]
        if not gaps:
            print(f'{LABEL[k]:>10s} 差距 —（Mortal 侧无此指标）')
            continue
        txt = '  '.join(f'{n.split("（")[0]} {g:+.2f}pp' for n, g in gaps)
        flag = '⚠️ >3pp' if max(abs(g) for _, g in gaps) > 3 else '✓ <3pp'
        print(f'{LABEL[k]:>10s} 差距 {txt}  {flag}')

    res = {'meta': {'mjai_glob': args.mjai_glob, 'files': human['files'],
                    'round': human['round'], 'seed': args.seed,
                    'protocol': 'stat.rs 同口径：fuuro=有 chi/pon/daiminkan 的局占比，'
                                '不含 ankan/kakan；round=每人每局各 +1'},
           'human': human, 'mortal_ref': MORTAL_REF}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n[out] {args.out}')


if __name__ == '__main__':
    main()
