"""第 0 步：局部形状标签是否依赖切牌顺序（纯 CPU，只读）。

问题
----
「根据某一家在某类牌周围切出的具体牌的顺序与类型，推断其第一次切出该类牌时
手上最有可能的局部牌型形状」——要让模型学这个，标签本身必须依赖顺序。
alloc（逐列暗牌计数）已证明不依赖顺序（输入消融 alloc_ce 0.6601→0.6605）。
本脚本检验：把标签换成**局部形状**（m±2 窗口 5 维计数向量）后，顺序是否
携带信息。

口径（与 order_metrics 对齐）
----------------------------
- 只统计**无鸣牌**的门清局；每个 (目标家 j, 花色 s, 中间张 m) 只取
  「m−1 与 m+1 都手切过」之后的**第一个观察者摸牌快照**，避免同一局重复。
- desc = m+1 先手切（hc[m+1] < hc[m−1]）；asc = m−1 先手切。
- 标签 = 该快照目标家当前手牌在 [m−2, m+2] 窗口的 5 维计数（越界补 0）。
- ΔΔ 一律按 order_metrics 的表头约定：**(非自然 − 自然)**；
  NAT = {2:'asc',3:'asc',4:'asc',6:'desc',7:'desc',8:'desc'}。
- 另报「中间张 m 的计数」ΔΔ、顺序→计数的互信息、随机打乱顺序的负对照。

用法
----
python3 tools/read_model/diagnostics/s316_local_shape_order.py --files 500
产物：results/local_shape_order.json（默认）
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GLOB = 'dataset/mjai/houou4/202609/**/*.json.gz'
SUIT = '万筒索'
NAT = {2: 'asc', 3: 'asc', 4: 'asc', 6: 'desc', 7: 'desc', 8: 'desc'}

PAI_TO_TID = {}
for _suit, _base in (('m', 0), ('p', 9), ('s', 18)):
    for _i in range(1, 10):
        PAI_TO_TID[f'{_i}{_suit}'] = _base + _i - 1
        if _i == 5:
            PAI_TO_TID[f'5{_suit}r'] = _base + 4
for _i, _p in enumerate('ESWNPFC'):
    PAI_TO_TID[_p] = 27 + _i


def tid_of(pai: str) -> int:
    return PAI_TO_TID[pai]


def iter_events(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def process_file(path: Path, cells, recorded, stats, mode='snapshot'):
    """回放一个文件；cells[(s,m)][order] 收集 5 维计数。

    mode='snapshot'：m−1/m+1 都手切后的第一个观察者摸牌快照（当前手牌）。
    mode='first'   ：第一次切出窗口内牌前的 14 张手牌；第二次切出时配对。
    """
    hands = None
    hc = None
    hpos = [0, 0, 0, 0]
    riichi = [False] * 4
    meld = False
    n_kyoku = 0
    first = {}
    second_done = set()

    def snapshot(observer):
        """观察者摸牌后：对每个目标家 j（≠观察者），记录首次满足 (m−1,m+1) 都手切的格。"""
        nonlocal cells, recorded
        if meld:
            return
        for j in range(4):
            if j == observer:
                continue
            for s in range(3):
                b = s * 9
                for m in NAT:
                    key = (j, s, m)
                    if key in recorded:
                        continue
                    a_, c_ = b + m - 2, b + m
                    pa, pc = hc[j][a_], hc[j][c_]
                    if pa >= 99 or pc >= 99 or pa == pc:
                        continue
                    order = 'desc' if pc < pa else 'asc'
                    win = []
                    for d in (-2, -1, 0, 1, 2):
                        t = b + m - 1 + d
                        win.append(hands[j][t] if 0 <= t < 34 and (t // 9) == s else 0)
                    cells[(s, m)][order].append(win)
                    recorded.add(key)
                    stats['n_samples'] += 1

    for ev in iter_events(path):
        t = ev['type']
        if t == 'start_kyoku':
            n_kyoku += 1
            hands = [[0] * 34 for _ in range(4)]
            hc = [[99] * 34 for _ in range(4)]
            hpos = [0, 0, 0, 0]
            riichi = [False] * 4
            meld = False
            recorded = set()
            first = {}
            second_done = set()
            for p, hand in enumerate(ev['tehais']):
                for pai in hand:
                    hands[p][tid_of(pai)] += 1
        elif hands is None:
            continue
        elif t == 'tsumo':
            a = ev['actor']
            hands[a][tid_of(ev['pai'])] += 1
            if mode == 'snapshot':
                snapshot(a)
        elif t == 'dahai':
            a = ev['actor']
            tid = tid_of(ev['pai'])
            hand_before = hands[a][:]
            hands[a][tid] -= 1
            tg = ev.get('tsumogiri', False)
            if not tg:
                if hc[a][tid] == 99:
                    hc[a][tid] = hpos[a]
                hpos[a] += 1
            if mode == 'first' and not meld:
                for s in range(3):
                    b = s * 9
                    for m in NAT:
                        mid = b + m - 1
                        if not (0 <= tid < 34 and (tid // 9) == s and abs(tid - mid) <= 2):
                            continue
                        key = (a, s, m)
                        win = [hand_before[b + m - 1 + d]
                               if 0 <= b + m - 1 + d < 34 and ((b + m - 1 + d) // 9) == s
                               else 0 for d in (-2, -1, 0, 1, 2)]
                        a_, c_ = b + m - 2, b + m
                        if key not in first:
                            # 只把 m−1/m+1 的第一次切出作为配对起点；m±2 不占用 key
                            if tid in (a_, c_):
                                first[key] = (win, tid, tg)
                        elif key not in second_done:
                            fwin, ftid, _ftg = first[key]
                            order = ('desc' if (ftid == c_ and tid == a_)
                                     else 'asc' if (ftid == a_ and tid == c_)
                                     else None)
                            if order:
                                cells[(s, m)][order].append(fwin)
                                stats['n_samples'] += 1
                                second_done.add(key)
        elif t == 'reach':
            riichi[ev['actor']] = True
        elif t in ('chi', 'pon', 'daiminkan', 'ankan', 'kakan'):
            meld = True
        elif t in ('hora', 'ryukyoku', 'end_kyoku'):
            # 局结束：清掉 recorded，下一局重新开始
            recorded = set()
    return n_kyoku


def entropy(counts):
    c = np.asarray(counts, dtype=np.float64)
    p = c / max(c.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def mi_mid(desc, asc, m_idx=2):
    """顺序(2 类) → 中间张计数(0..4) 的互信息（nats）。"""
    a = np.asarray(desc, dtype=np.int64)[:, m_idx]
    b = np.asarray(asc, dtype=np.int64)[:, m_idx]
    labels = np.concatenate([a, b])
    groups = np.concatenate([np.zeros(len(a), int), np.ones(len(b), int)])
    h_l = entropy(np.bincount(labels, minlength=5))
    # H(L|G)
    h_cond = 0.0
    n = len(labels)
    for g in (0, 1):
        sub = labels[groups == g]
        h_cond += (len(sub) / n) * entropy(np.bincount(sub, minlength=5))
    return max(0.0, h_l - h_cond), h_l, h_cond


def boot_delta(desc, asc, m_idx, sign, B=400, seed=20261021):
    rng = np.random.default_rng(seed)
    d = np.asarray(desc, dtype=np.float64)
    a = np.asarray(asc, dtype=np.float64)
    out = []
    for _ in range(B):
        di = rng.integers(0, len(d), len(d))
        ai = rng.integers(0, len(a), len(a))
        out.append(sign * (a[ai, m_idx].mean() - d[di, m_idx].mean()))
    out = np.asarray(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', type=int, default=500)
    ap.add_argument('--glob', default=DEFAULT_GLOB)
    ap.add_argument('--out', default=None)
    ap.add_argument('--boot', type=int, default=400)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--mode', choices=['snapshot', 'first'], default='snapshot')
    args = ap.parse_args()
    if args.out is None:
        suffix = '' if args.mode == 'snapshot' else '_first'
        args.out = str(Path(__file__).parent / 'results' / f'local_shape_order{suffix}.json')

    random.seed(args.seed)
    files = sorted(ROOT.glob(args.glob))
    random.shuffle(files)
    files = files[:args.files]
    print(f'[data] {len(files)} files', flush=True)

    cells = {(s, m): {'desc': [], 'asc': []} for s in range(3) for m in NAT}
    stats = {'n_samples': 0}
    n_kyoku = 0
    for i, f in enumerate(files):
        try:
            n_kyoku += process_file(f, cells, set(), stats, args.mode)
        except Exception as e:  # 单文件解析失败不影响整体
            print(f'[warn] {f}: {e}', flush=True)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(files)} files, kyoku={n_kyoku}, samples={stats["n_samples"]}', flush=True)

    # ---- 逐格统计 ----
    by_cell = []
    wsum = 0.0
    wdelta = 0.0
    for s in range(3):
        for m in NAT:
            d = cells[(s, m)]['desc']
            a = cells[(s, m)]['asc']
            if min(len(d), len(a)) < 25:
                continue
            sign = 1.0 if NAT[m] == 'desc' else -1.0
            dm = sign * (np.asarray(a)[:, 2].mean() - np.asarray(d)[:, 2].mean())
            dv = sign * (np.asarray(a).mean(0) - np.asarray(d).mean(0))
            mi, h, hc = mi_mid(d, a)
            lo, hi = boot_delta(d, a, 2, sign, args.boot, args.seed + s * 10 + m)
            by_cell.append(dict(
                cell=f'{SUIT[s]}{m}', s=s, m=m, nat=NAT[m],
                n_desc=len(d), n_asc=len(a),
                mean_desc=[float(x) for x in np.asarray(d).mean(0)],
                mean_asc=[float(x) for x in np.asarray(a).mean(0)],
                delta_m=float(dm), delta_vec=[float(x) for x in dv],
                ci95=[lo, hi], mi_nats=mi, h_label=h, h_cond=hc,
            ))
            w = min(len(d), len(a))
            wsum += w
            wdelta += w * dm

    if not by_cell:
        print('[error] no cells with enough samples', flush=True)
        sys.exit(2)

    overall_dm = wdelta / wsum
    # 汇总 bootstrap：每个格单独重抽，按 min(n) 加权
    rng = np.random.default_rng(args.seed)
    bs = []
    for _ in range(args.boot):
        acc = 0.0
        for c in by_cell:
            d = np.asarray(cells[(c['s'], c['m'])]['desc'])
            a = np.asarray(cells[(c['s'], c['m'])]['asc'])
            di = rng.integers(0, len(d), len(d))
            ai = rng.integers(0, len(a), len(a))
            sign = 1.0 if c['nat'] == 'desc' else -1.0
            acc += min(len(d), len(a)) * sign * (a[ai, 2].mean() - d[di, 2].mean())
        bs.append(acc / wsum)
    bs = np.asarray(bs)
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    # ---- 索8 专项 ----
    so8 = next((c for c in by_cell if c['cell'] == '索8'), None)
    # ---- 负对照：同格内随机打乱顺序 ----
    neg = []
    for c in by_cell:
        d = cells[(c['s'], c['m'])]['desc']
        a = cells[(c['s'], c['m'])]['asc']
        pooled = np.asarray(d + a)
        n_d = len(d)
        vals = []
        for _ in range(100):
            idx = np.random.permutation(len(pooled))
            vd = pooled[idx[:n_d], 2].mean()
            va = pooled[idx[n_d:], 2].mean()
            sign = 1.0 if c['nat'] == 'desc' else -1.0
            vals.append(sign * (va - vd))
        neg.append((len(d) + len(a), float(np.mean(vals))))
    neg = np.asarray(neg)
    neg_mean = float((neg[:, 0] * neg[:, 1]).sum() / neg[:, 0].sum())

    report = dict(
        meta=dict(
            files=len(files), kyoku=n_kyoku, samples=stats['n_samples'],
            glob=args.glob, seed=args.seed, boot=args.boot, mode=args.mode,
            protocol=('门清局；' + (
                '每个 (j,s,m) 取 m−1/m+1 都手切后的第一个观察者摸牌快照；'
                '标签=m±2 窗口 5 维当前手牌计数' if args.mode == 'snapshot' else
                '每个 (j,s,m) 取第一次切出窗口内牌前的 14 张手牌；'
                '第二次切出窗口内牌时按先 m−1/先 m+1 配对') + '；ΔΔ=非自然−自然'),
        ),
        overall=dict(
            n_cells=len(by_cell), n_desc=int(sum(c['n_desc'] for c in by_cell)),
            n_asc=int(sum(c['n_asc'] for c in by_cell)),
            delta_m=float(overall_dm), ci95=ci,
            mi_nats=float(sum(c['mi_nats'] * min(c['n_desc'], c['n_asc']) for c in by_cell) / wsum),
            negative_control=neg_mean,
        ),
        by_cell=by_cell,
        so8=so8,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print('\n=== 汇总 ===', flush=True)
    print(f'n_cells={len(by_cell)} n_desc={report["overall"]["n_desc"]} '
          f'n_asc={report["overall"]["n_asc"]}', flush=True)
    print(f'ΔΔ(非自然−自然, 中间张计数) = {overall_dm:+.4f}  CI95 [{ci[0]:+.4f},{ci[1]:+.4f}]', flush=True)
    print(f'顺序→中间张计数 互信息 = {report["overall"]["mi_nats"]:.5f} nats', flush=True)
    print(f'负对照（打乱顺序）= {neg_mean:+.5f}', flush=True)
    print('\n=== 逐格（只列 |ΔΔ| 或 MI 较大者）===', flush=True)
    for c in sorted(by_cell, key=lambda x: -abs(x['delta_m']))[:12]:
        print(f'{c["cell"]:>4s} nat={c["nat"]:<4s} n={c["n_desc"]}/{c["n_asc"]} '
              f'ΔΔ={c["delta_m"]:+.4f} CI[{c["ci95"][0]:+.3f},{c["ci95"][1]:+.3f}] '
              f'MI={c["mi_nats"]:.4f} vec={["%.2f" % x for x in c["delta_vec"]]}', flush=True)
    if so8:
        print('\n=== 索8 专项 ===', flush=True)
        print(f'n_desc(先9s)={so8["n_desc"]} n_asc(先7s)={so8["n_asc"]}', flush=True)
        print(f'mean_desc={["%.3f" % x for x in so8["mean_desc"]]}', flush=True)
        print(f'mean_asc ={["%.3f" % x for x in so8["mean_asc"]]}', flush=True)
        print(f'ΔΔ(先7s−先9s, 8s 计数)={so8["delta_m"]:+.4f} '
              f'CI[{so8["ci95"][0]:+.3f},{so8["ci95"][1]:+.3f}] MI={so8["mi_nats"]:.4f}', flush=True)
    print(f'\n[out] {args.out}', flush=True)


if __name__ == '__main__':
    main()
