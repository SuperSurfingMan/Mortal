#!/usr/bin/env python3
"""防守天花板审计（零模型、纯标签）：理论上「防守做满」能把点炮率压到多少？

## 为什么需要这个数

A3（规则基线）与 §3.14（Bayes 上界）都说明「读牌精度已近饱和」。那下一步的唯一
方向就是「把读到的用上」（防守）。但在投入之前必须先回答一个更基本的问题：

    **就算读牌 100% 准确、并且完全照做，点炮率最低能到多少？**

如果答案是「已经接近实际点炮率」，那么任何读牌/注入/奖励工程都提不动点炮率（剩下的
部分是不可避的）；如果答案是「能压到一半」，那防守侧就有明确空间，值得投。

## 方法（只用手牌与待牌标签，不需要任何模型）

对每个「自家舍牌决策」样本（观察者自己的摸牌后）：
- `danger` = 三家**有效荣和集**的并（`wait` 标签 > 0，已剔除振听与不可表达）；
- `own`    = 观察者手牌（可打出的牌种）；
- `n_safe` = |own ∖ danger|（手上有几张安全的可选牌）；
- `human_deal`  = 人类实际舍牌是否落在 `danger`；
- `oracle_deal` = `n_safe == 0`（无安全牌可打 ⇒ 无论如何都点炮）。

于是：
- `人类点炮率` ≈ 实际点炮率（用于校验口径，应接近 12.5% 量级）；
- `oracle 点炮率` = **完美读牌 + 完全照做的下限**；
- `可避比例` = (人类点炮率 − oracle 点炮率) / 人类点炮率。

> 口径说明：`oracle` 忽略了「弃和要付出手牌价值」这一成本，所以它是**乐观下限**
> （真实最优防守不会达到，因为有时值得冒险推进）。但它给出的是**任何读牌驱动防守
> 的绝对上限收益**——如果这个数很小，防守侧就不值得投入。

用法：
    python tools/read_model/defense_ceiling.py --samples 6000
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(HERE))

import dataset as D                       # noqa: E402
import train as T                         # noqa: E402
import torch                              # noqa: E402
from bayes_bound import (build_eval_samples, kyoku_spans, load_events)  # noqa: E402

NUM_TILES = D.NUM_TILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',
                    default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609')
    ap.add_argument('--samples', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=20261016)
    args = ap.parse_args()

    import glob
    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))
    samples = build_eval_samples(files, args.samples, seed=args.seed)
    batch = T.collate(samples)
    ref = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in batch.items()}
    n = len(samples)

    ev_cache, sp_cache = {}, {}
    rows = []
    t0 = time.time()
    for i, s in enumerate(samples):
        f = s['_file']
        if f not in ev_cache:
            ev_cache[f] = load_events(f)
            sp_cache[f] = kyoku_spans(ev_cache[f])
        events, spans = ev_cache[f], sp_cache[f]
        ev = int(s['_event'])
        obs = int(s['_observer'])
        e = events[ev]
        # 只看「自家摸牌后要舍牌」的决策点
        if e.get('type') != 'tsumo' or e.get('actor') != obs:
            continue
        lo, hi = spans[int(s['_kyoku']) - 1]
        d_tile = None
        for k in range(ev + 1, hi):
            ek = events[k]
            if ek.get('type') == 'dahai' and ek.get('actor') == obs:
                d_tile = D.tile_id(ek['pai'])
                break
            if ek.get('type') in ('hora', 'ryukyoku', 'end_kyoku'):
                break
        if d_tile is None:            # 摸牌后直接和了 / 流局，没有舍牌决策
            continue
        own = ref['own_hand'][i].astype(bool)
        if not own[d_tile]:
            continue                  # 口径保护：实际舍牌必须在自己手里
        danger = ref['wait'][i][:, :NUM_TILES].max(0) > 0
        danger_ri = (ref['wait'][i][:, :NUM_TILES].max(0) > 0) & \
                    np.broadcast_to(ref['riichi'][i][:, None] > 0.5, (3, NUM_TILES)).max(0)
        n_safe = int((own & ~danger).sum())
        rows.append({
            'turn': int(s['turn']),
            'riichi_any': bool((ref['riichi'][i] > 0.5).any()),
            'riichi_only': bool((ref['riichi'][i] > 0.5).any()),
            'called_any': bool((ref['called'][i] > 0.5).any()),
            'n_safe': n_safe,
            'human_deal': bool(danger[d_tile]),
            'oracle_deal': n_safe == 0,
            'human_deal_ri': bool(danger_ri[d_tile]),
            'oracle_deal_ri': int((own & ~danger_ri).sum()) == 0,
            'n_danger_own': int((own & danger).sum()),
        })
        if (len(rows)) % 2000 == 0 and len(rows):
            print(f'[ceiling] {len(rows)} 个舍牌决策 | {time.time()-t0:.0f}s', flush=True)

    print(f'[ceiling] 共 {len(rows)} 个自家舍牌决策（{n} 样本中）', flush=True)
    if not rows:
        return

    def report(name, sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return
        hd = np.mean([x['human_deal'] for x in r])
        od = np.mean([x['oracle_deal'] for x in r])
        ns = np.mean([x['n_safe'] for x in r])
        nd = np.mean([x['n_danger_own'] for x in r])
        avoid = (hd - od) / hd if hd > 0 else float('nan')
        print(f'  {name:<22s} n={len(r):5d} 人类点炮 {100*hd:5.2f}%  '
              f'oracle 下限 {100*od:5.2f}%  可避 {100*avoid:5.1f}%  '
              f'手上安全牌 {ns:4.1f} 张 / 危险牌 {nd:4.1f} 张', flush=True)

    print('\n== 全部（含立直/非立直）', flush=True)
    report('all', lambda x: True)
    report('有立直家', lambda x: x['riichi_any'])
    report('无立直家', lambda x: not x['riichi_any'])
    report('有鸣牌家', lambda x: x['called_any'])
    report('turn<=3', lambda x: x['turn'] <= 3)
    report('turn4-7', lambda x: 4 <= x['turn'] <= 7)
    report('turn8-11', lambda x: 8 <= x['turn'] <= 11)
    report('turn12+', lambda x: x['turn'] >= 12)
    report('无安全牌可打', lambda x: x['n_safe'] == 0)
    report('有>=3张安全牌', lambda x: x['n_safe'] >= 3)

    print('\n== 仅按立直家的待牌判定（非立直家视为安全）', flush=True)
    hd = np.mean([x['human_deal_ri'] for x in rows])
    od = np.mean([x['oracle_deal_ri'] for x in rows])
    print(f'  all                    n={len(rows):5d} 人类 {100*hd:5.2f}%  '
          f'oracle {100*od:5.2f}%  可避 {100*(hd-od)/max(hd,1e-9):5.1f}%', flush=True)


if __name__ == '__main__':
    main()
