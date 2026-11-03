#!/usr/bin/env python3
"""A2c / P2c 按子集路由：收益测算（零推理）。

## 问题

cond 口径重算（`eval_cond_recheck.py`）显示两者分层互有胜负：

| 子集 | A2c | P2c | 谁好 |
|---|---:|---:|---|
| all/active_any | 0.3549 | **0.3670** | P2c |
| all/active_both | **0.5067** | 0.4893 | A2c |
| riichi/active_both | 0.6314 | **0.6483** | P2c |
| clean_t8/active_both | **0.4758** | 0.4390 | A2c |

「active_both 用 A2c」看似能白拿，但 active_both 在 active_any 里只占 3.4%
（cond 非空样本 5865/172942），路由收益会被稀释。

本脚本用缓存的概率直接算各种路由策略的全局收益，决定是否值得实现。

## 策略

| 名称 | 规则 |
|---|---|
| `p2c` | 全部用 P2c |
| `a2c` | 全部用 A2c |
| `both_a2c` | active_both → A2c，其余 → P2c |
| `nonri_both_a2c` | (非立直 且 active_both) → A2c，其余 → P2c |
| `oracle_cell` | 每个 (层,子集) 选更好的（**乐观上界，含选择偏差**）|

用法：
    python tools/read_model/local_shape/eval_router.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import local_shape as LS  # noqa: E402

LAYERS = ('all', 'riichi', 'clean_t8')
ACTIVES = ('all_centers', 'active_any', 'active_both')
K = 27


def load_probs(path: Path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_truth(cache: Path, n: int):
    z = np.load(cache, allow_pickle=True)
    alloc = np.asarray(z['alloc'])[:n].astype(np.int64)
    hc = np.asarray(z['hc'])[:n]
    riichi = np.asarray(z['riichi'])[:n].astype(bool)
    called = np.asarray(z['called'])[:n].astype(bool)
    turn = np.asarray(z['turn'])[:n].astype(np.int64)
    return alloc, hc, riichi, called, turn


def build_masks(n, hc, riichi, called, turn, centers):
    t8 = np.broadcast_to((turn >= 8)[:, None], (n, 3))
    layers = {
        'all': np.ones((n, 3), bool),
        'riichi': riichi,
        'clean_t8': (~riichi) & (~called) & t8,
    }
    any_ = np.stack([(hc[:, :, a] < 99) | (hc[:, :, c] < 99)
                     for a, _b, c in centers], -1)
    both = np.stack([(hc[:, :, a] < 99) & (hc[:, :, c] < 99)
                     for a, _b, c in centers], -1)
    actives = {'all_centers': np.ones_like(any_, bool),
               'active_any': any_, 'active_both': both}
    return layers, actives


def metrics(P, Y, mask):
    """cond 口径（EMPTY 置 0 重归一），返回 top1/top3/CE。"""
    if not mask.any():
        return None
    pp = P[mask].astype(np.float32).copy()
    yy = Y[mask]
    pp[:, 0] = 0.0
    s = pp.sum(1, keepdims=True)
    pp = np.where(s > 0, pp / np.maximum(s, 1e-12), 1.0 / K)
    order = np.argsort(-pp, 1)
    idx = np.arange(len(yy))
    return {
        'n': int(len(yy)),
        'top1': float((order[:, 0] == yy).mean()),
        'top3': float((order[:, :3] == yy[:, None]).any(1).mean()),
        'ce': float(-np.log(np.clip(pp[idx, yy], 1e-12, None)).mean()),
    }


def route(Pa, Pb, pick_a):
    """pick_a [n,3,21] bool → [n,3,21,27]。"""
    return np.where(pick_a[..., None], Pa, Pb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', default='/tmp/ls_cond_probs.npz')
    ap.add_argument('--cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    ap.add_argument('--out', default=str(HERE / 'results'))
    args = ap.parse_args()

    P = load_probs(Path(args.probs))
    n = P['a2c'].shape[0]
    centers = LS.centers_3()
    alloc, hc, riichi, called, turn = load_truth(Path(args.cache), n)
    layers, actives = build_masks(n, hc, riichi, called, turn, centers)
    Y = np.stack([LS.true_pattern(alloc, c) for c in centers], -1)   # [n,3,21]
    any_, both = actives['active_any'], actives['active_both']
    nonri = ~riichi[..., None]

    Pa, Pb = P['a2c'], P['p2c']
    strategies = {
        'p2c': Pb,
        'a2c': Pa,
        'both_a2c': route(Pa, Pb, both),
        'nonri_both_a2c': route(Pa, Pb, both & nonri),
        'riichi_both_p2c': route(Pb, Pa, ~riichi[..., None] & ~both),
    }

    print(f'[data] n={n}  active_both 占 active_any: '
          f'{float((both & any_).sum() / max(any_.sum(), 1)):.4f}', flush=True)

    table = {}
    for name, pr in strategies.items():
        table[name] = {}
        for layer in LAYERS:
            table[name][layer] = {}
            for act in ACTIVES:
                m = layers[layer][..., None] & actives[act] & (Y != 0)
                table[name][layer][act] = metrics(pr, Y, m)
        print(f'[eval] {name} done', flush=True)

    # oracle per cell（乐观上界）
    table['oracle_cell'] = {}
    for layer in LAYERS:
        table['oracle_cell'][layer] = {}
        for act in ACTIVES:
            a = table['a2c'][layer][act]
            b = table['p2c'][layer][act]
            table['oracle_cell'][layer][act] = (
                a if (a and b and a['top1'] >= b['top1']) else b)

    # 主判据：all/active_any 上各策略相对 p2c
    base = table['p2c']['all']['active_any']
    rows = []
    for name in ('a2c', 'both_a2c', 'nonri_both_a2c', 'riichi_both_p2c',
                 'oracle_cell'):
        r = table[name]['all']['active_any']
        rows.append((name, r['top1'] - base['top1'], r['ce'] - base['ce']))

    # 走 A2c 的样本比例（每策略）
    cov = {
        'both_a2c': float(both.mean()),
        'nonri_both_a2c': float((both & nonri).mean()),
    }

    print('\n== all/active_any（cond）相对 p2c ==')
    for name, dt, dc in rows:
        print(f'  {name:18s} Δtop1={dt:+.5f}  ΔCE={dc:+.5f}')
    print('\n== 各子集 top1 ==')
    for layer in LAYERS:
        for act in ACTIVES:
            vs = {k: (table[k][layer][act] or {}).get('top1') for k in
                  ('a2c', 'p2c', 'both_a2c', 'nonri_both_a2c')}
            s = '  '.join(f'{k}={v:.4f}' for k, v in vs.items() if v is not None)
            print(f'  {layer:9s}/{act:13s} {s}')

    best_simple = max(rows[1:4], key=lambda x: x[1])
    gain = best_simple[1]
    findings = [
        f"active_both 在 active_any（cond 非空）中占比 "
        f"{float((both & any_).sum() / max(any_.sum(), 1)):.4f}"
        f"（全窗口口径 {float(both.mean()):.4f}）。",
        f"全场景路由（both→A2c）在 all/active_any 上的收益 "
        f"Δtop1={rows[1][1]:+.5f}、ΔCE={rows[1][2]:+.5f}；"
        f"细分路由（非立直&both→A2c）Δtop1={rows[2][1]:+.5f}。",
        f"oracle 逐 cell 选择的上界 Δtop1={rows[4][1]:+.5f}（含选择偏差，仅参考）。",
        f"active_both 子集内部：A2c "
        f"{table['a2c']['all']['active_both']['top1']:.4f} vs P2c "
        f"{table['p2c']['all']['active_both']['top1']:.4f}"
        f"（差 {table['a2c']['all']['active_both']['top1'] - table['p2c']['all']['active_both']['top1']:+.4f}）。",
        f"riichi/active_both：P2c "
        f"{table['p2c']['riichi']['active_both']['top1']:.4f} vs A2c "
        f"{table['a2c']['riichi']['active_both']['top1']:.4f}"
        f"（差 {table['p2c']['riichi']['active_both']['top1'] - table['a2c']['riichi']['active_both']['top1']:+.4f}），"
        f"⇒ 「both→A2c」在立直子集上是负收益。",
    ]
    if gain < 0.001:
        decision = (f'最优简单路由的全局收益仅 {gain:+.5f} top1（<0.1pp）'
                    f'⇒ **不值得实现逐窗口路由**（需双模型推理，成本翻倍）。'
                    f'正确做法是**按场景选模型**：全场景用 P2c；'
                    f'若下游专门分析 active_both 局面，直接用 A2c 跑一遍即可。')
    else:
        decision = (f'最优简单路由 {best_simple[0]} 收益 {gain:+.5f} top1 '
                    f'⇒ 值得实现（需评估双模型推理成本）')

    res = {'meta': {'n': int(n), 'active_both_frac':
                    float((both & any_).sum() / max(any_.sum(), 1)),
                    'coverage': cov},
           'table': table, 'verdict': {'decision': decision,
                                       'findings': findings}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'router.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n[verdict] {decision}')
    for line in findings:
        print(' -', line)
    print(f'[out] {out}/router.json')


if __name__ == '__main__':
    main()
