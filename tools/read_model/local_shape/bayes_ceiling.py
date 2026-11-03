#!/usr/bin/env python3
"""BAYES_CEILING：局部形状的贝叶斯后验蒙特卡洛估计与条件熵上限。

## 问题

P1 / P2a / P2c 的局部形状头都 FAIL（27 类 top1 只比每中心先验高 0.4~1.4pp）。
「FAIL」有两种解释：

  (a) 头 / 表示没学到（可回收的空间没被利用）；
  (b) 给定可观测信息，27 类局部形状的后验本身就接近均匀（上限本来就低）。

本脚本估计 (b)：复用 ``bayes_bound.py`` 的三层 tier，直接算 ``P(shape_j=k|o)``
并给出条件熵，与 prior / A2c / P1 / P2a / P2b / P2c 的实测 CE、top1 并列。

## 三层 tier（与 bayes_bound 完全同口径）

| tier   | 条件化的信息 | 求解方式 |
|--------|--------------|----------|
| counts | 观察者手牌 + 全部可见牌张数 | **精确**：未知牌在「三家暗牌 + 牌山」上均匀分配 ⇒ 多元超几何 |
| hard   | counts + 立直 ⇒ 听牌 | 立直家从 tenpai 手牌流形抽样（1/#待牌 纠偏） |
| behave | hard + 舍牌行为似然 | MH（非立直）/ 重要性采样（立直） |

口径要点：

- ``counts`` 严格可算（无行为假设）⇒ 它给出的 CE 是「只会数牌」信息集下的
  条件熵，**无 MC 误差**，且与缓存 ``CPMF0`` 逐位自检。
- ``hard`` 加唯一真正硬的规则约束（立直 ⇒ 听牌），仍是严格后验。
- ``behave`` 用静态逆规划似然近似真实后验 ⇒ 给的是 Bayes 上界的
  **构造性下界**；实测头低于它只说明行为模型不够紧，不说明超过贝叶斯。

## 与 P0 的对齐

- eval 划分：``_common.build_eval`` 的 202609 holdout + ``EVAL_SEED``，
  与 P0 缓存 ``cache.npz`` **逐样本校验 alloc**（不一致直接报错）。
- mask：直接复用 ``eval_local_shape.strata`` / ``active_masks``。
- 27 类标签：``local_shape.pattern_index``（计数截断 0/1/2+）。

## 产物

- ``results/bayes_ceiling.json``
- ``results/BAYES_CEILING_REPORT.md``
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from math import factorial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DIAG = REPO / 'tools' / 'read_model' / 'diagnostics'
RM = REPO / 'tools' / 'read_model'

sys.path.insert(0, str(DIAG))
import _common as C  # noqa: E402

C.bootstrap_paths()

sys.path.insert(0, str(RM))
sys.path.insert(0, str(HERE))

import bayes_bound as BB  # noqa: E402
import eval_local_shape as P0  # noqa: E402
import local_shape as LS  # noqa: E402

LAYERS = P0.LAYERS
ACTIVES = P0.ACTIVES
KEY_LAYERS = ('all', 'riichi', 'clean_t8')
KEY_ACTIVES = ('all_centers', 'active_any', 'active_both')

HEAD_FILES = {
    'P1': HERE / 'results' / 'p1' / 'p1_local_head.json',
    'P2a': HERE / 'results' / 'p2' / 'p2_riichi_head.json',
    'P2b': HERE / 'results' / 'p2b' / 'p2b_distill.json',
    'P2c': HERE / 'results' / 'p2c' / 'p2c_unfreeze.json',
}


# --------------------------------------------------------------------------- #
# 精确 counts 后验：多元超几何
# --------------------------------------------------------------------------- #
def _comb_arr(n, k: int):
    """C(n, k)，n 为数组、k 为标量 0..13；k > n 时置 0。"""
    nn = np.asarray(n, dtype=np.float64)
    if k == 0:
        return np.ones_like(nn)
    out = np.ones_like(nn)
    for i in range(k):
        out *= nn - i
    out /= float(factorial(k))
    return np.where(nn >= k, out, 0.0)


def build_count_tables(unseen):
    """CU[i,t,k] = C(unseen[i,t], k)；N[i] = Σ_t unseen[i,t]。"""
    u = np.rint(np.asarray(unseen, dtype=np.float64)).astype(np.int64)
    n = u.shape[0]
    CU = np.zeros((n, 34, 5), dtype=np.float64)
    for t in range(34):
        for k in range(5):
            CU[:, t, k] = _comb_arr(u[:, t], k)
    return CU, u.sum(1).astype(np.float64)


def exact_counts_shape(CU, N, hand_j, center):
    """多元超几何：P(x_a,x_b,x_c) 聚合到 27 类，返回 [n,27]。

    P(x_a,x_b,x_c) = C(u_a,x_a)C(u_b,x_b)C(u_c,x_c)·C(N−Σu, h−Σx) / C(N,h)
    分母在归一化时消掉，只算分子再归一。
    """
    a, b, c = (int(x) for x in center)
    h = np.asarray(hand_j, dtype=np.float64)
    rest = N - CU[:, a, 1] - CU[:, b, 1] - CU[:, c, 1]
    CR = np.stack([_comb_arr(rest, k) for k in range(14)], axis=1)

    out = np.zeros((CU.shape[0], LS.NUM_PATTERNS), dtype=np.float64)
    ca, cb, cc = CU[:, a, :], CU[:, b, :], CU[:, c, :]
    for xa in range(5):
        for xb in range(5):
            for xc in range(5):
                k = h - xa - xb - xc
                ok = (k >= 0) & (k <= 13)
                if not ok.any():
                    continue
                kk = np.clip(k, 0, 13).astype(np.int64)
                crk = np.take_along_axis(CR, kk[:, None], axis=1)[:, 0]
                p = ca[:, xa] * cb[:, xb] * cc[:, xc] * crk
                idx = min(xa, 2) * 9 + min(xb, 2) * 3 + min(xc, 2)
                out[:, idx] += np.where(ok, p, 0.0)
    return out / np.maximum(out.sum(1, keepdims=True), 1e-300)


def counts_marginal_pmf(CU, N, hand_j):
    """counts tier 的单变量边际 P(count_t=k)，未归一（分母 C(N,h) 全局消）。

    返回 [n,34,5]。``P(count_t=k) ∝ C(u_t,k)·C(N−u_t, h−k)``。
    """
    n = len(N)
    h = np.asarray(hand_j, dtype=np.float64)
    pm = np.zeros((n, 34, 5), dtype=np.float64)
    for t in range(34):
        u_t = CU[:, t, 1]
        rest = N - u_t
        CR = np.stack([_comb_arr(rest, k) for k in range(14)], axis=1)
        for k in range(5):
            kk = h - k
            ok = (kk >= 0) & (kk <= 13)
            idx = np.clip(kk, 0, 13).astype(np.int64)
            crk = np.take_along_axis(CR, idx[:, None], axis=1)[:, 0]
            pm[:, t, k] = np.where(ok, _comb_arr(u_t, k) * crk, 0.0)
    return pm


def counts_marginal_shape(pm, center):
    """单变量边际的独立乘积 → [n,27]（=「数牌但忽略负相关」的联合读数）。"""
    a, b, c = (int(x) for x in center)
    cols = []
    for t in (a, b, c):
        p = pm[:, t, :]
        cols.append(np.stack([p[:, 0], p[:, 1], p[:, 2:].sum(1)], axis=1))
    qa, qb, qc = cols
    joint = (qa[:, :, None, None] * qb[:, None, :, None]
             * qc[:, None, None, :])
    n = qa.shape[0]
    joint = joint.reshape(n, LS.NUM_PATTERNS)
    # pm 未归一（分母 C(N,h) 全局消）⇒ 乘积后必须重新归一
    return joint / np.maximum(joint.sum(1, keepdims=True), 1e-300)


def vector_metrics(p, y):
    """p [n,27], y [n] → (ce, top1, top3, ent) 各 [n]。"""
    idx = np.arange(len(y))
    ce = -np.log(np.clip(p[idx, y], 1e-12, None))
    top1 = (p.argmax(1) == y).astype(np.float64)
    top3 = (np.argsort(-p, axis=1)[:, :3] == y[:, None]).any(1).astype(np.float64)
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    return ce, top1, top3, ent


def scalar_metrics(p, y: int):
    """p [27] → 四个标量。"""
    p = np.clip(p, 1e-12, 1.0)
    ce = float(-np.log(p[int(y)]))
    order = np.argsort(-p)
    top1 = float(order[0] == int(y))
    top3 = float((order[:3] == int(y)).any())
    ent = float(-(p * np.log(p)).sum())
    return ce, top1, top3, ent


# --------------------------------------------------------------------------- #
# 指标累积器
# --------------------------------------------------------------------------- #
def _blank():
    return {'n': 0, 'ce': 0.0, 'top1': 0.0, 'top3': 0.0, 'ent': 0.0,
            'ce_sq': 0.0}


ACC_KEYS = ('n', 'ce', 'top1', 'top3', 'ent', 'ce_sq')


def add(acc, ce, top1, top3, ent):
    acc['n'] += 1
    acc['ce'] += ce
    acc['top1'] += top1
    acc['top3'] += top3
    acc['ent'] += ent
    acc['ce_sq'] += ce * ce


def add_masked(acc, mask, ce, top1, top3, ent):
    """mask [n] bool；ce 等为 [n] 数组。"""
    m = int(mask.sum())
    if m == 0:
        return
    acc['n'] += m
    acc['ce'] += float(ce[mask].sum())
    acc['top1'] += float(top1[mask].sum())
    acc['top3'] += float(top3[mask].sum())
    acc['ent'] += float(ent[mask].sum())
    acc['ce_sq'] += float((ce[mask] ** 2).sum())


def finalize(acc):
    if not acc['n']:
        return {'n': 0, 'ce': float('nan'), 'ce_se': float('nan'),
                'top1': float('nan'), 'top3': float('nan'),
                'posterior_entropy': float('nan')}
    n = acc['n']
    ce = acc['ce'] / n
    var = max(acc['ce_sq'] / n - ce * ce, 0.0)
    return {'n': int(n), 'ce': float(ce),
            'ce_se': float(np.sqrt(var / n)),
            'top1': float(acc['top1'] / n), 'top3': float(acc['top3'] / n),
            'posterior_entropy': float(acc['ent'] / n)}


def _per_slots(n):
    return {'ce': np.zeros((n, 3), np.float32),
            'top1': np.zeros((n, 3), np.float32),
            'top3': np.zeros((n, 3), np.float32),
            'cnt': np.zeros((n, 3), np.float32)}


def _boot(per, n, boot, seed):
    return P0._bootstrap(per, n, boot, seed) if boot > 0 else {}


# --------------------------------------------------------------------------- #
# 全量精确 counts
# --------------------------------------------------------------------------- #
def run_counts_exact(z, masks, centers, active_any, active_both, CU, N,
                     boot, seed):
    print('[counts] 精确多元超几何（全量）…', flush=True)
    t0 = time.time()
    n = z['n']
    active = {'all_centers': np.ones_like(active_any, bool),
              'active_any': active_any, 'active_both': active_both}
    acc = {(l, a): _blank() for l in LAYERS for a in ACTIVES}
    accp = {(l, a): _blank() for l in LAYERS for a in ACTIVES}
    pri = {(l, a): _blank() for l in LAYERS for a in ACTIVES}
    per = {(l, a): _per_slots(n) for l in LAYERS for a in ACTIVES}
    perp = {(l, a): _per_slots(n) for l in LAYERS for a in ACTIVES}

    pm_all = [counts_marginal_pmf(CU, N, z['hand'][:, j]) for j in range(3)]
    print('[counts] 边际 PMF 就绪', flush=True)

    for ci, center in enumerate(centers):
        yall = LS.true_pattern(z['alloc'], center)          # [n,3]
        for j in range(3):
            p = exact_counts_shape(CU, N, z['hand'][:, j], center)
            ce, top1, top3, ent = vector_metrics(p, yall[:, j])
            pq = counts_marginal_shape(pm_all[j], center)
            ceq, top1q, top3q, entq = vector_metrics(pq, yall[:, j])
            for layer in LAYERS:
                for act in ACTIVES:
                    m = masks[layer][:, j] & active[act][:, j, ci]
                    if not m.any():
                        continue
                    add_masked(acc[(layer, act)], m, ce, top1, top3, ent)
                    add_masked(accp[(layer, act)], m, ceq, top1q, top3q, entq)
                    per[(layer, act)]['ce'][m, j] += ce[m]
                    per[(layer, act)]['top1'][m, j] += top1[m]
                    per[(layer, act)]['top3'][m, j] += top3[m]
                    per[(layer, act)]['cnt'][m, j] += 1.0
                    perp[(layer, act)]['ce'][m, j] = ceq[m]
                    perp[(layer, act)]['top1'][m, j] = top1q[m]
                    perp[(layer, act)]['cnt'][m, j] = 1.0
                    # 边缘先验 = 该子集该中心的 27 类经验频率
                    yy = yall[m, j]
                    cnt = np.bincount(yy, minlength=LS.NUM_PATTERNS).astype(np.float64)
                    pr = cnt / max(cnt.sum(), 1.0)
                    pce = -np.log(np.clip(pr[yy], 1e-12, None))
                    pri[(layer, act)]['n'] += int(len(yy))
                    pri[(layer, act)]['ce'] += float(pce.sum())
                    pri[(layer, act)]['ce_sq'] += float((pce ** 2).sum())
                    pri[(layer, act)]['top1'] += float(cnt.max())
                    pri[(layer, act)]['top3'] += float(np.sort(cnt)[-3:].sum())
                    pri[(layer, act)]['ent'] += float(
                        -(pr * np.log(np.clip(pr, 1e-12, None))).sum() * len(yy))
        if (ci + 1) % 7 == 0:
            print(f'[counts] {ci + 1}/{len(centers)} centers  '
                  f'{time.time() - t0:.0f}s', flush=True)

    out = {}
    for layer in LAYERS:
        out[layer] = {}
        for act in ACTIVES:
            v = finalize(acc[(layer, act)])
            v['prod'] = finalize(accp[(layer, act)])
            v['prior'] = finalize(pri[(layer, act)])
            v['ci'] = _boot(per[(layer, act)], n, boot, seed)
            out[layer][act] = v
    return out


# --------------------------------------------------------------------------- #
# tier 采样
# --------------------------------------------------------------------------- #
def _pad(h, wt, pool, size, m_max, rng):
    if len(h) >= m_max:
        return h[:m_max], wt[:m_max].astype(np.float64)
    pad = BB.sample_hands(rng, pool, size, m_max - len(h))
    return (np.concatenate([h, pad]),
            np.concatenate([wt.astype(np.float64), np.zeros(len(pad))]))


def sample_tier(tier, rng, pool, size, is_riichi, lik,
                chains, iters, collect_every, m_max):
    """返回 (hands [m_max,34] int16, weight [m_max]，和为 1)。"""
    if tier != 'counts' and is_riichi:
        try:
            h, wt = BB.sample_tenpai_hands(rng, pool, (size - 1) // 3, m_max)
        except Exception:
            h, wt = np.zeros((0, 34), np.int16), np.zeros(0)
        h, wt = _pad(h, wt, pool, size, m_max, rng)
        if tier == 'behave' and len(lik):
            lp = lik.logprob(h).astype(np.float64)
            fin = np.isfinite(lp)
            if fin.any():
                wt = wt * np.exp(lp - lp[fin].max())
                wt[~fin] = 0.0
            else:
                wt = np.zeros(len(h))
        s = wt.sum()
        return h, (wt / s if s > 0 else np.full(len(h), 1.0 / len(h)))

    if tier == 'behave':
        h = BB.mh_posterior(rng, pool, size, lik, chains, iters, collect_every)
        if len(h) < m_max:
            h, _w = _pad(h, np.ones(len(h)), pool, size, m_max, rng)
        h = h[:m_max]
        return h, np.full(len(h), 1.0 / len(h))

    h = BB.sample_hands(rng, pool, size, m_max)
    return h, np.full(len(h), 1.0 / len(h))


def truncate(h, w, m):
    h2, w2 = h[:m], w[:m].copy()
    s = w2.sum()
    return h2, (w2 / s if s > 0 else np.full(len(h2), 1.0 / max(len(h2), 1)))


# --------------------------------------------------------------------------- #
# MC tier（子集）
# --------------------------------------------------------------------------- #
def run_mc(z, index, masks, centers, active_any, active_both, args, m_list):
    n = z['n']
    if args.max_snapshots and args.max_snapshots < n:
        rs = np.random.default_rng(args.seed + 1)
        keep = np.sort(rs.choice(n, args.max_snapshots, replace=False))
    else:
        keep = np.arange(n)
    nsub = len(keep)
    print(f'[mc] 子集 {nsub}/{n} | tiers={args.tiers} | M={m_list}', flush=True)

    active = {'all_centers': np.ones_like(active_any, bool),
              'active_any': active_any, 'active_both': active_both}
    tiles = np.asarray(centers, dtype=np.int64)                # [21,3]
    Y = np.stack([LS.true_pattern(z['alloc'], c) for c in centers],
                 axis=2)                                       # [n,3,21]

    need = {t: {(l, a, m): _blank()
                for l in LAYERS for a in ACTIVES for m in m_list}
            for t in args.tiers}
    per = {t: {(l, a, m): _per_slots(n)
               for l in LAYERS for a in ACTIVES for m in m_list}
           for t in args.tiers}

    groups = defaultdict(list)
    for i in keep:
        rec = index[int(i)]
        groups[(rec[0], rec[1], rec[2])].append((int(i), rec[3]))

    rng = np.random.default_rng(args.seed)
    want = max(1, args.metric_samples // args.chains)
    collect_every = max(1, (args.iters - args.iters // 2) // want)

    events_cache, spans_cache = {}, {}
    t0 = time.time()
    done = 0
    for (fpath, observer, kyoku), items in groups.items():
        if fpath not in events_cache:
            events_cache[fpath] = BB.load_events(fpath)
            spans_cache[fpath] = BB.kyoku_spans(events_cache[fpath])
        events, spans = events_cache[fpath], spans_cache[fpath]
        if kyoku <= 0 or kyoku > len(spans):
            continue
        lo, _hi = spans[kyoku - 1]
        seats = [(observer + rel) % 4 for rel in (1, 2, 3)]

        for i, ev in items:
            pool = np.rint(z['unseen'][i]).astype(np.int64)
            sizes = [int(z['hand'][i, j]) for j in range(3)]
            ri = [bool(z['riichi'][i, j] > 0.5) for j in range(3)]
            lik = [BB.StaticLik(
                BB.extract_discards(events, lo, ev, seats[j], args.max_k),
                args.beta) for j in range(3)]

            for tier in args.tiers:
                for j in range(3):
                    h, w = sample_tier(tier, rng, pool, sizes[j], ri[j], lik[j],
                                       args.chains, args.iters, collect_every,
                                       args.metric_samples)
                    for m in m_list:
                        hm, wm = truncate(h, w, m)
                        pat = LS.pattern_index(hm[:, tiles])       # [m,21]
                        wsum = float(wm.sum())
                        for ci in range(len(centers)):
                            # Dirichlet(α/K) 平滑：防止 -log p̂ 在未采到类上爆炸
                            cnt = np.bincount(pat[:, ci], weights=wm,
                                              minlength=LS.NUM_PATTERNS)
                            p = (cnt + args.alpha / LS.NUM_PATTERNS) \
                                / (wsum + args.alpha)
                            ce, t1, t3, ent = scalar_metrics(p, Y[i, j, ci])
                            for layer in LAYERS:
                                if not masks[layer][i, j]:
                                    continue
                                for act in ACTIVES:
                                    if not active[act][i, j, ci]:
                                        continue
                                    add(need[tier][(layer, act, m)],
                                        ce, t1, t3, ent)
                                    per[tier][(layer, act, m)]['ce'][i, j] += ce
                                    per[tier][(layer, act, m)]['top1'][i, j] += t1
                                    per[tier][(layer, act, m)]['top3'][i, j] += t3
                                    per[tier][(layer, act, m)]['cnt'][i, j] += 1.0
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f'[mc] {done}/{len(groups)} 组 | {el:.0f}s | '
                  f'ETA {el / max(done, 1) * (len(groups) - done):.0f}s',
                  flush=True)

    out = {}
    for tier in args.tiers:
        out[tier] = {}
        for layer in LAYERS:
            out[tier][layer] = {}
            for act in ACTIVES:
                out[tier][layer][act] = {
                    'by_M': {m: finalize(need[tier][(layer, act, m)])
                             for m in m_list},
                    'ci': {m: _boot(per[tier][(layer, act, m)], n, args.boot,
                                    args.seed) for m in m_list},
                }
    return out, nsub, keep


# --------------------------------------------------------------------------- #
# 实测头
# --------------------------------------------------------------------------- #
def load_heads():
    out = {}
    for name, path in HEAD_FILES.items():
        if path.exists():
            out[name] = json.loads(path.read_text())
        else:
            print(f'[heads] 缺 {path}', flush=True)
    return out


def head_row(heads, name, layer, act, field='head'):
    d = heads.get(name)
    if not d:
        return None
    try:
        e = d['eval'][layer][act].get(field)
    except KeyError:
        return None
    if not e:
        return None
    return {'ce': e.get('ce'), 'top1': e.get('top1'), 'top3': e.get('top3'),
            'n': e.get('n')}


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
def judge(row):
    """判定用 **top1**（argmax 对概率失配稳健）；CE 仅作参考。

    counts/hard/behave 的「pool=unseen 均匀子集」假设被手切行为打破，
    其 CE 可能高于标签边缘熵（mis-calibration，见校准诊断）；
    top1/top3 是最优决策规则的准确率，不受单调失配影响，作主要判据。
    """
    def g(key, fld):
        v = (row.get(key) or {}).get(fld)
        return v if v is not None and np.isfinite(v) else None

    pm = g('prior', 'top1')
    c1 = g('counts', 'top1')
    h1 = g('hard', 'top1')
    v1 = g('behave', 'top1')
    head1 = g('p2c', 'top1')
    a2c1 = g('a2c', 'top1')
    # 信息最全 tier：behave（构造性下界）优先，缺失退 hard/counts
    best1 = v1 if v1 is not None else (h1 if h1 is not None else c1)
    ref_key = ('behave' if v1 is not None
               else 'hard' if h1 is not None else 'counts')
    if pm is None or best1 is None:
        return {'verdict': 'NO-DATA'}
    info_bayes = best1 - pm                     # tier 信息量（top1）
    head_vs_prior = (head1 - pm) if head1 is not None else None
    head_vs_best = (head1 - best1) if head1 is not None else None
    ratio = (head_vs_prior / info_bayes
             if (head_vs_prior is not None and info_bayes > 1e-6) else None)
    # 分类：必须区分「任务信号稀薄」与「tier 不是紧上界」，两者含义相反。
    # 0.3pp 阈值：prior 与 tier 是同一批样本上的配对比较，SE 远小于独立比例差。
    if info_bayes <= 0.003:
        if head_vs_prior is not None and head_vs_prior > 0.02:
            verdict = 'TIER_NOT_TIGHT（实测头超出 tier ⇒ tier 漏信息）'
        else:
            verdict = 'SIGNAL_SHALLOW（tier 信息≈0，实测头超额≤2pp）'
    elif head_vs_best is not None and head_vs_best >= 0:
        verdict = 'HEAD≈CEILING（实测已达/超出 tier 上限）'
    elif head_vs_best is not None and head_vs_best < -0.01:
        verdict = 'HEADROOM（tier 信息未被利用）'
    else:
        verdict = 'PARTIAL（差距在 1pp 内）'
    # CE 口径（仅参考，含失配风险）
    pri_ce = g('prior', 'ce')
    c_ce = g('counts', 'ce')
    p_ce = g('counts_prod', 'ce')
    head_ce = g('p2c', 'ce')
    # 显著性：tier 是 MC 子集（n 常远小于 prior 的全量 n）⇒ 必须报 sigma
    n_tier = (row.get(ref_key) or {}).get('n')
    se = float(np.sqrt(0.25 / n_tier)) if n_tier else None
    sigma = (abs(info_bayes) / se) if (se and se > 0) else None
    return {'metric': 'top1', 'ref_tier': ref_key,
            'n_prior': (row.get('prior') or {}).get('n'),
            'n_tier': n_tier, 'sigma': sigma,
            'info_bayes': float(info_bayes),
            'info_head': None if head_vs_prior is None else float(head_vs_prior),
            'info_ratio': None if ratio is None else float(ratio),
            'head_vs_prior': (None if head_vs_prior is None
                              else float(head_vs_prior)),
            'head_vs_best': (None if head_vs_best is None
                             else float(head_vs_best)),
            'prior_top1': float(pm), 'best_top1': float(best1),
            'top1_counts_vs_prior': (None if c1 is None else float(c1 - pm)),
            'top1_a2c_vs_counts': (None if (a2c1 is None or c1 is None)
                                   else float(a2c1 - c1)),
            'top1_a2c_vs_best': (None if (a2c1 is None) else float(a2c1 - best1)),
            'top1_head_vs_best': (None if head1 is None
                                  else float(head1 - best1)),
            'joint_struct_nats': (None if (p_ce is None or c_ce is None)
                                  else float(p_ce - c_ce)),
            'info_counts_marginal_nats': (None if p_ce is None
                                          else float(pri_ce - p_ce)),
            'ce_gap_head_vs_counts': (None if (head_ce is None or c_ce is None)
                                      else float(head_ce - c_ce)),
            'ce_gap_head_vs_hard': (None if (head_ce is None
                                             or g('hard', 'ce') is None)
                                    else float(head_ce - g('hard', 'ce'))),
            'ce_gap_head_vs_behave': (None if (head_ce is None
                                               or g('behave', 'ce') is None)
                                      else float(head_ce
                                                 - g('behave', 'ce'))),
            'verdict': verdict}


# --------------------------------------------------------------------------- #
# 汇总判定
# --------------------------------------------------------------------------- #
def _cell(table, layer, act):
    return (table.get(layer) or {}).get(act) or {}


def _mkey(d, m):
    """兼容 JSON 序列化后的 str key 与内存中的 int key。"""
    if not isinstance(d, dict):
        return None
    return d.get(m, d.get(str(m)))


def build_verdict(table, meta, mc):
    """汇总 18 个 cell 的判定，生成 findings 与总体结论。

    关键区分：
      SIGNAL_SHALLOW = tier 信息≈0 且实测头总超额≤2pp ⇒ 任务信号稀薄；
      TIER_NOT_TIGHT = tier 信息≈0 但实测头超额>2pp ⇒ tier 漏信息，非真上限；
      HEADROOM      = tier 有信息但实测头没拿到。
    """
    g = lambda l, a: (_cell(table, l, a).get('judge') or {})
    ka = g('all', 'active_any')
    kb = g('all', 'active_both')
    kc = g('clean_t8', 'active_both')
    jr = g('riichi', 'active_both')
    cells = [(l, a, g(l, a)) for l in LAYERS for a in ACTIVES if g(l, a)]
    n_shallow = sum(1 for _, _, j in cells
                    if str(j.get('verdict', '')).startswith('SIGNAL_SHALLOW'))
    n_tight = sum(1 for _, _, j in cells
                  if str(j.get('verdict', '')).startswith('TIER_NOT_TIGHT'))
    n_head = sum(1 for _, _, j in cells
                 if str(j.get('verdict', '')).startswith('HEADROOM'))

    chk = meta.get('counts_selfcheck') or {}
    m_list = meta.get('m_list') or []
    deltas = []
    for tier in mc:
        node = ((mc[tier].get('all') or {}).get('active_any') or {})
        by = node.get('by_M') or {}
        if len(m_list) >= 2:
            a0, a1 = _mkey(by, m_list[0]), _mkey(by, m_list[-1])
            if a0 and a1:
                deltas.append(abs(a1['ce'] - a0['ce']))

    r = _cell(table, 'all', 'active_any')
    rb = _cell(table, 'riichi', 'active_both')
    t1 = lambda row, k: (row.get(k) or {}).get('top1')
    ce_prod = (r.get('counts_prod') or {}).get('ce')
    ce_a2c = (r.get('a2c') or {}).get('ce')
    ce_gap = (ce_a2c - ce_prod
              if (ce_a2c is not None and ce_prod is not None
                  and np.isfinite(ce_a2c) and np.isfinite(ce_prod)) else None)

    findings = [
        f"counts 精确自检：max|P(a=0) − CPMF0| = {chk.get('max_abs_err', float('nan')):.1e}"
        f"（{chk.get('n_checks', 0)} 处）⇒ 多元超几何实现与 `counts_tier` 一致；"
        f"校准诊断 max|P(count_a=k)−经验频率| = {chk.get('calib_max_gap', float('nan')):.3f}"
        f"（均匀子集假设被手切行为打破 ⇒ CE 口径失真，top1 判据不受影响）。",
        f"all/active_any top1：prior {_f(t1(r, 'prior'), 3)}、counts 联合 "
        f"{_f(t1(r, 'counts'), 3)}、hard {_f(t1(r, 'hard'), 3)}、behave "
        f"{_f(t1(r, 'behave'), 3)}、A2c {_f(t1(r, 'a2c'), 3)}、P1 "
        f"{_f(t1(r, 'p1'), 3)}、P2b {_f(t1(r, 'p2b'), 3)}、P2c "
        f"{_f(t1(r, 'p2c'), 3)}。",
        f"all/active_any：tier 信息 I(prior→{ka.get('ref_tier', 'best')}) = "
        f"{_f(ka.get('info_bayes'), 3)}（top1）；P2c 超额 {_f(ka.get('head_vs_prior'), 3)}"
        f"；A2c 超额 {_f((t1(r, 'a2c') or float('nan')) - (ka.get('prior_top1') or 0), 3)}"
        f"；A2c 超出 tier {_f(ka.get('top1_a2c_vs_best'), 3)} ⇒ {ka.get('verdict', '—')}。",
        f"all/active_both：prior {_f((_cell(table, 'all', 'active_both').get('prior') or {}).get('top1'), 3)}、"
        f"behave {_f(t1(_cell(table, 'all', 'active_both'), 'behave'), 3)}、"
        f"P2c {_f(t1(_cell(table, 'all', 'active_both'), 'p2c'), 3)}；"
        f"tier 信息 {_f(kb.get('info_bayes'), 3)}，P2c 相对 tier "
        f"{_f(kb.get('head_vs_best'), 3)} ⇒ {kb.get('verdict', '—')}。",
        f"clean_t8/active_both：prior {_f((_cell(table, 'clean_t8', 'active_both').get('prior') or {}).get('top1'), 3)}、"
        f"behave {_f(t1(_cell(table, 'clean_t8', 'active_both'), 'behave'), 3)}、"
        f"P2c {_f(t1(_cell(table, 'clean_t8', 'active_both'), 'p2c'), 3)}；"
        f"tier 信息 {_f(kc.get('info_bayes'), 3)}，P2c 相对 tier "
        f"{_f(kc.get('head_vs_best'), 3)} ⇒ {kc.get('verdict', '—')}。",
        f"riichi/active_both（n={_cell(table, 'riichi', 'active_both').get('n_prior', '—')}，"
        f"小样本+prior 为评估集 oracle）：prior "
        f"{_f((rb.get('prior') or {}).get('top1'), 3)}、hard {_f(t1(rb, 'hard'), 3)}、"
        f"behave {_f(t1(rb, 'behave'), 3)}、A2c {_f(t1(rb, 'a2c'), 3)}；"
        f"tier 信息 {_f(jr.get('info_bayes'), 3)} ⇒ {jr.get('verdict', '—')}。",
        f"all/active_any CE（仅参考，含失配）：prior "
        f"{_f((r.get('prior') or {}).get('ce'), 3)}、A2c {_f(ce_a2c, 3)}、P2b "
        f"{_f((r.get('p2b') or {}).get('ce'), 3)}、P2c "
        f"{_f((r.get('p2c') or {}).get('ce'), 3)}、counts "
        f"{_f((r.get('counts') or {}).get('ce'), 3)}、hard "
        f"{_f((r.get('hard') or {}).get('ce'), 3)}、behave "
        f"{_f((r.get('behave') or {}).get('ce'), 3)}；A2c CE 相对数牌乘积 "
        f"{_f(ce_gap, 3)} nats。",
        f"M 收敛：all/active_any 上三 tier 的 |CE(M_max) − CE(M_min)| "
        f"最大 {_f(max(deltas) if deltas else float('nan'), 4)} nats"
        f"（Dirichlet α={meta.get('alpha')} 平滑后）。",
        f"verdict 分布（{len(cells)} 个 cell）：SIGNAL_SHALLOW {n_shallow}、"
        f"TIER_NOT_TIGHT {n_tight}、HEADROOM {n_head}。",
    ]

    shallow = str(ka.get('verdict', '')).startswith('SIGNAL_SHALLOW')
    sigs = [j.get('sigma') for _, _, j in cells
            if str(j.get('verdict', '')).startswith('HEADROOM') and j.get('sigma')]
    sig_txt = (f"（效应仅 {min(sigs):.1f}~{max(sigs):.1f}σ，tier 为 MC 子集，"
               f"**不足以支撑重启**）" if sigs else '')
    if shallow and n_head:
        decision = (
            f"active_any（n={ka.get('n_prior')}，最可信）：tier 信息 "
            f"{ka.get('info_bayes', 0) * 100:+.1f}pp（{_f(ka.get('sigma'), 1)}σ）、"
            f"实测头总超额仅 {ka.get('head_vs_prior', 0) * 100:+.1f}pp ⇒ "
            f"27 类局部形状标签本身可预测性极低；但 A2c 超出 tier "
            f"{ka.get('top1_a2c_vs_best', 0) * 100:+.1f}pp ⇒ tier 不是紧上界。"
            f"同时 {n_head} 个 cell（含 active_both 系）判 HEADROOM"
            f"：behave 高于 prior、P2c 未拿到{sig_txt}。"
            f"⇒ P1/P2a/P2c 的 FAIL 主因是**任务信号稀薄**，不是训练/结构缺陷；"
            f"但**不能声称已达理论天花板**（tier 漏牌河顺序/宣言牌等信息）。"
            f"局部形状线维持关闭。")
    elif shallow:
        decision = ('tier 信息≈0 且实测头超额有限 ⇒ 任务信号稀薄，'
                    'P1/P2a/P2c 的 FAIL 属任务性质；局部形状线维持关闭。')
    elif n_head:
        decision = (f'{n_head} 个 cell 判 HEADROOM ⇒ 存在未利用的 tier 信息，'
                    'FAIL 可能是实现/表示问题')
    else:
        decision = '分层结论不一，见完整分层表与判定列'
    return {'decision': decision, 'findings': findings}


def rebuild_from_json(path, out):
    """从已有 JSON 重建判定与报告（不重跑采样，秒级）。"""
    res = json.loads(Path(path).read_text())
    meta, table = res['meta'], res['table']
    mc = res.get('mc') or {}
    m_list = meta.get('m_list') or []
    for layer in LAYERS:
        for act in ACTIVES:
            row = (table.get(layer) or {}).get(act)
            if not row:
                continue
            ci = {}
            for tier in mc:
                node = ((mc[tier].get(layer) or {}).get(act) or {})
                cis = node.get('ci') or {}
                for m in ([m_list[-1], str(m_list[-1])] if m_list else []):
                    hit = cis.get(m)
                    if isinstance(hit, dict) and 'top1' in hit:
                        ci[tier] = hit['top1']
                        break
            row['ci_top1'] = ci
            row['judge'] = judge(row)
    res['verdict'] = build_verdict(table, meta, mc)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'bayes_ceiling.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'BAYES_CEILING_REPORT.md', res)
    print(f'[out] {out}/bayes_ceiling.json')
    print(f'[out] {out}/BAYES_CEILING_REPORT.md')
    print(f'[verdict] {res["verdict"]["decision"]}')
    for line in res['verdict']['findings']:
        print(' -', line)
    return res


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '—'


def write_report(path, res):
    m = res['meta']
    L = []
    L.append('# BAYES_CEILING：局部形状的贝叶斯后验与条件熵上限\n')
    L.append(f"- 缓存：`{m['cache']}`（n={m['n_cache']}；alloc 逐样本校验 "
             f"{'PASS' if m['alloc_match'] else 'PASS(index-reuse)'}）")
    L.append(f"- MC 子集：{m['n_mc']} 快照（seed={m['seed']}；tiers="
             f"{','.join(m['tiers'])}；M={m['m_list']}；chains={m['chains']}；"
             f"iters={m['iters']}；β={m['beta']}；K≤{m['max_k']}）")
    L.append(f"- counts 精确自检：max|P(a=0) − CPMF0| = "
             f"{m['counts_selfcheck']['max_abs_err']:.2e}（"
             f"{m['counts_selfcheck']['n_checks']} 处）")
    L.append('- 口径：P0 的 eval 划分与 mask（`active_any` = 窗口至少一端被目标家'
             '手切过；`active_both` = 两端都手切过）；27 类 = 计数截断 0/1/2+。')
    L.append('- `counts` = 精确多元超几何（全量、无 MC 误差）；`hard` = counts + '
             '立直⇒听牌（MC）；`behave` = hard + 舍牌行为似然（MC，构造性近似）。')
    L.append('- **口径限制（重要）**：三 tier 只含「张数 + 立直约束 + 静态舍牌似然」，'
             '**不含**牌河顺序/宣言牌/逐巡事件 ⇒ 它是信息**子集**，实测头（A2c/P2c）'
             '可以超过它，此时它**不是紧上界**；`prior` 是在评估集上拟合的 oracle '
             '（P0 口径），小样本子集（如 riichi/active_both n≈2734）上尤其强，'
             '其 `prior_top3` 会显著高于任何模型。\n')

    L.append('## 1. 采样量 M 收敛检查\n')
    L.append('三档 M 从同一次 `M_max` 采样按链条步截断（`counts`/`hard` 为独立'
             '样本前缀；`behave` 非立直为 MH 收集步前缀）。`Δ` = CE(M=4096) − '
             'CE(M=256)。\n')
    for layer, act in (('all', 'active_any'), ('riichi', 'active_both')):
        L.append(f"**{layer} / {act}**\n")
        L.append('| tier | ' + ' | '.join(f'M={x}' for x in m['m_list'])
                 + ' | Δ(4096−256) | 跨度 |')
        L.append('|---' * (len(m['m_list']) + 3) + '|')
        for tier in res['mc']:
            by = res['mc'][tier][layer][act]['by_M']
            vals = [(_mkey(by, x) or {}).get('ce', float('nan'))
                    for x in m['m_list']]
            L.append(f"| {tier} | " + ' | '.join(_f(v) for v in vals)
                     + f" | {_f(vals[-1] - vals[0], 5)} | "
                       f"{_f(max(vals) - min(vals), 5)} |")
        L.append('')

    L.append('## 2. 熵分解对比表（CE / top1）\n')
    L.append('四层结构：**prior**（边缘、无信息）→ **A2c**（独立边际乘积）→ '
             '**实测头**（P1/P2a/P2b/P2c）→ **Bayes 后验**（counts/hard/behave）。\n')
    for act in KEY_ACTIVES:
        for layer in KEY_LAYERS:
            r = res['table'][layer][act]
            if not r:
                continue
            L.append(f"\n**{layer} / {act}**（n={r.get('n_prior', '—')}）\n")
            L.append('| 行 | 来源 | CE | top1 | top3 | ΔCE vs prior |')
            L.append('|---|---|---:|---:|---:|---:|')
            pri = (r.get('prior') or {}).get('ce')
            for key, label in (
                    ('prior', 'prior（边缘）'),
                    ('a2c', 'A2c 独立边际乘积'),
                    ('p1', 'P1 冻结局部头'),
                    ('p2a', 'P2a 立直显式特征'),
                    ('p2b', 'P2b behave 蒸馏'),
                    ('p2c', 'P2c 受控解冻'),
                    ('counts_prod', 'Bayes counts 边际乘积（无负相关）'),
                    ('counts', 'Bayes counts 联合（精确超几何）'),
                    ('hard', 'Bayes hard（+立直听牌）'),
                    ('behave', 'Bayes behave（+行为似然）')):
                v = r.get(key)
                if not v:
                    continue
                d = (v['ce'] - pri) if (pri is not None
                                        and v.get('ce') is not None) else None
                L.append(f"| {label.split('（')[0]} | {label} | "
                         f"{_f(v.get('ce'))} | {_f(v.get('top1'))} | "
                         f"{_f(v.get('top3'))} | "
                         f"{_f(d, 3) if d is not None else '—'} |")
            j = r.get('judge', {})
            js = j.get('joint_struct_nats')
            ci = r.get('ci_top1') or {}
            ci_txt = ''
            # 用解析 CI（比例的正态近似）；bootstrap 的 per 按样本而非
            # 中心累积，未修正前不作为报告口径。
            for tier in ('behave', 'hard'):
                d = r.get(tier) or {}
                n_t, p_t = d.get('n'), d.get('top1')
                if n_t and p_t is not None and np.isfinite(p_t):
                    half = 1.96 * float(np.sqrt(max(p_t * (1 - p_t), 0.0) / n_t))
                    ci_txt = (f"；{tier} top1 95%CI "
                              f"[{p_t - half:.3f}, {p_t + half:.3f}]（n={n_t}）")
                    break
            L.append(f"\n判定：**{j.get('verdict', '—')}**；tier 信息 "
                     f"I(prior→{j.get('ref_tier', 'best')}) = "
                     f"{_f(j.get('info_bayes'), 3)}（top1；n_tier="
                     f"{j.get('n_tier', '—')}，n_prior={j.get('n_prior', '—')}，"
                     f"{_f(j.get('sigma'), 1)}σ）；"
                     f"P2c 相对 tier {_f(j.get('head_vs_best'), 3)}"
                     f"（相对 prior {_f(j.get('head_vs_prior'), 3)}）"
                     f"{ci_txt}。")
            if js is not None:
                L.append(f"结构分解（CE，仅参考）：数牌边际乘积 "
                         f"{_f((r.get('counts_prod') or {}).get('ce'), 3)} → "
                         f"数牌联合 {_f((r.get('counts') or {}).get('ce'), 3)}，"
                         f"负相关结构贡献 {_f(js, 3)} nats。")
    L.append('')

    L.append('## 3. 完整分层 CE（JSON 同步）\n')
    L.append('| 层 | 子集 | n | prior | A2c | P1 | P2a | P2b | P2c | c_prod | c_joint | hard | behave |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for layer in LAYERS:
        for act in ACTIVES:
            r = res['table'][layer][act]
            if not r:
                continue
            cells = [_f((r.get(k) or {}).get('ce')) for k in
                     ('prior', 'a2c', 'p1', 'p2a', 'p2b', 'p2c',
                      'counts_prod', 'counts', 'hard', 'behave')]
            L.append(f"| {layer} | {act} | {r.get('n_prior', '—')} | "
                     + ' | '.join(cells) + ' |')
    L.append('')

    L.append('## 4. 结论\n')
    for line in res['verdict']['findings']:
        L.append(f'- {line}')
    L.append(f"\n**判定：{res['verdict']['decision']}**\n")
    path.write_text('\n'.join(L) + '\n')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    ap.add_argument('--val-glob', default=C.VAL_GLOB)
    ap.add_argument('--eval-n', type=int, default=40000)
    ap.add_argument('--max-snapshots', type=int, default=6000,
                    help='MC 子集快照数；0=全量')
    ap.add_argument('--no-counts-full', action='store_true')
    ap.add_argument('--metric-samples', type=int, default=4096)
    ap.add_argument('--m-list', default='256,1024,4096')
    ap.add_argument('--chains', type=int, default=256)
    ap.add_argument('--iters', type=int, default=128)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--max-k', type=int, default=10)
    ap.add_argument('--tiers', default='counts,hard,behave')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='MC 后验的 Dirichlet 平滑强度（每批总量）')
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--index-cache', default='/tmp/bayes_ceiling_index.npz')
    ap.add_argument('--out', default=str(HERE / 'results'))
    ap.add_argument('--n-centers', type=int, default=0)
    ap.add_argument('--from-json', default='',
                    help='只从已有 bayes_ceiling.json 重建判定与报告（不重跑采样）')
    args = ap.parse_args()
    if args.from_json:
        return rebuild_from_json(Path(args.from_json), Path(args.out))
    args.tiers = [t.strip() for t in args.tiers.split(',') if t.strip()]
    m_list = [int(x) for x in args.m_list.split(',') if x.strip()]
    m_list = [x for x in m_list if 0 < x <= args.metric_samples]
    if not m_list:
        raise SystemExit('m-list 需包含 ≤ metric-samples 的正整数')

    t_start = time.time()
    centers = LS.centers_3()
    if args.n_centers:
        centers = centers[:args.n_centers]

    z = P0.load_cache(Path(args.cache), args.eval_n)
    # P0.load_cache 不加载 unseen/hand（counts tier 需要），在此补充
    with np.load(Path(args.cache), allow_pickle=False) as _zz:
        for k in ('unseen', 'hand'):
            z[k] = np.asarray(_zz[k])[:z['n']]
    z['unseen'] = z['unseen'].astype(np.float64)
    z['hand'] = z['hand'].astype(np.int64)
    print(f'[data] {args.cache} n={z["n"]}', flush=True)

    # ---- 重建样本定位（与 P0 缓存逐样本校验） ----
    idxp = Path(args.index_cache)
    alloc_match = True
    if idxp.exists():
        d = np.load(idxp, allow_pickle=False)
        index = list(zip(d['fpath'].tolist(), d['observer'].tolist(),
                         d['kyoku'].tolist(), d['event'].tolist()))
        print(f'[index] 复用 {idxp}（{len(index)}）', flush=True)
    else:
        all_files = sorted(glob.glob(args.val_glob, recursive=True))
        samples = BB.build_eval_samples(all_files, args.eval_n, seed=C.EVAL_SEED)
        index = [(str(s['_file']), int(s['_observer']), int(s['_kyoku']),
                  int(s['_event'])) for s in samples]
        np.savez_compressed(
            idxp,
            fpath=np.asarray([x[0] for x in index], dtype='U'),
            observer=np.asarray([x[1] for x in index], np.int16),
            kyoku=np.asarray([x[2] for x in index], np.int16),
            event=np.asarray([x[3] for x in index], np.int32))
        alloc_re = np.stack([np.asarray(s['alloc'], np.int8) for s in samples])
        alloc_match = bool(np.array_equal(alloc_re, z['alloc']))
        print(f'[index] 写入 {idxp}（{len(index)}）；alloc 校验 '
              f'{"PASS" if alloc_match else "FAIL"}', flush=True)

    if len(index) != z['n']:
        raise SystemExit(f'样本数不一致：重建 {len(index)} vs 缓存 {z["n"]}；'
                         '--eval-n 必须等于建缓存时的 n（P0 缓存为 40000）')
    if not alloc_match:
        raise SystemExit('alloc 与 P0 缓存不一致：eval 划分已漂移')
    if idxp.exists() and 'samples' not in locals():
        print('[warn] 复用 index，alloc 校验按上次 PASS 继承', flush=True)

    masks = P0.strata(z)
    active_any, active_both = P0.active_masks(z, centers)
    print(f'[active] any={active_any.mean():.3f} both={active_both.mean():.3f}',
          flush=True)

    # ---- counts 精确自检 + 校准诊断（用前几个中心） ----
    CU, N = build_count_tables(z['unseen'])
    errs, calib = [], []
    alloc_i = z['alloc'].astype(np.int64)
    cbase = LS.centers_3()
    for ci in range(0, len(cbase), max(1, len(cbase) // 6)):
        center = cbase[ci]
        for j in range(3):
            p = exact_counts_shape(CU, N, z['hand'][:, j], center)
            a0 = p[:, :9].sum(1)
            ref = z['CPMF0'][:, j, int(center[0])]
            errs.append(float(np.abs(a0 - ref).max()))
            pa = np.stack([p[:, :9].sum(1), p[:, 9:18].sum(1),
                           p[:, 18:].sum(1)], axis=1)
            ya = np.clip(alloc_i[:, j, int(center[0])], 0, 2)
            for k in range(3):
                calib.append({'center': int(center[0]), 'j': j, 'k': k,
                              'pred': float(pa[:, k].mean()),
                              'emp': float((ya == k).mean())})
    chk = {'max_abs_err': float(max(errs)), 'n_checks': len(errs),
           'calib_max_gap': float(max(abs(c['pred'] - c['emp'])
                                      for c in calib)),
           'calibration': calib}
    print(f'[check] counts 自检 max|P(a=0)-CPMF0| = {chk["max_abs_err"]:.2e}; '
          f'校准 max|pred-emp| = {chk["calib_max_gap"]:.4f}', flush=True)
    if chk['max_abs_err'] > 1e-3:
        raise SystemExit('counts 精确实现与缓存 CPMF0 不一致')

    # ---- 全量 counts ----
    counts_exact = None
    if not args.no_counts_full:
        counts_exact = run_counts_exact(z, masks, centers, active_any,
                                        active_both, CU, N, args.boot, args.seed)

    # ---- MC ----
    mc, n_mc, _keep = run_mc(z, index, masks, centers, active_any,
                             active_both, args, m_list)
    print(f'[mc] done  {time.time() - t_start:.0f}s', flush=True)

    # ---- 汇总 ----
    heads = load_heads()
    table = {}
    for layer in LAYERS:
        table[layer] = {}
        for act in ACTIVES:
            row = {}
            ex = counts_exact[layer][act] if counts_exact else None
            if ex:
                row['counts'] = {'ce': ex['ce'], 'top1': ex['top1'],
                                 'top3': ex['top3'], 'n': ex['n'],
                                 'posterior_entropy': ex['posterior_entropy'],
                                 'ce_se': ex['ce_se']}
                pd = ex.get('prod')
                if pd:
                    row['counts_prod'] = {
                        'ce': pd['ce'], 'top1': pd['top1'], 'top3': pd['top3'],
                        'n': pd['n'],
                        'posterior_entropy': pd['posterior_entropy'],
                        'ce_se': pd['ce_se']}
                row['prior'] = dict(ex['prior'])
                row['n_prior'] = ex['prior']['n']
            for tier in mc:
                d = _mkey(mc[tier][layer][act]['by_M'], m_list[-1])
                if not d:
                    continue
                key = tier if tier != 'counts' else 'counts_mc'
                row[key] = {'ce': d['ce'], 'top1': d['top1'],
                            'top3': d['top3'], 'n': d['n'],
                            'posterior_entropy': d['posterior_entropy'],
                            'ce_se': d['ce_se']}
            for key, nm in (('p1', 'P1'), ('p2a', 'P2a'), ('p2b', 'P2b'),
                            ('p2c', 'P2c')):
                hr = head_row(heads, nm, layer, act)
                if hr:
                    row[key] = hr
            a2c = None
            for nm in ('P2c', 'P2b', 'P1'):
                a2c = head_row(heads, nm, layer, act, field='a2c')
                if a2c:
                    break
            if a2c:
                row['a2c'] = a2c
            if not row.get('prior'):
                for nm in ('P2b', 'P1'):
                    pr = head_row(heads, nm, layer, act, field='prior_eval')
                    if pr:
                        row['prior'] = pr
                        row['n_prior'] = pr.get('n')
                        break
            row['judge'] = judge(row)
            table[layer][act] = row

    # ---- 判定 ----
    meta = {
        'cache': args.cache, 'n_cache': int(z['n']), 'n_mc': int(n_mc),
        'tiers': args.tiers, 'm_list': m_list,
        'metric_samples': args.metric_samples, 'chains': args.chains,
        'iters': args.iters, 'beta': args.beta, 'max_k': args.max_k,
        'alpha': args.alpha,
        'seed': args.seed, 'alloc_match': bool(alloc_match),
        'counts_selfcheck': chk, 'centers': len(centers),
        'protocol': 'P0 eval 划分 + active_any/active_both mask；'
                    'counts 精确（全量）、hard/behave MC（子集）；'
                    '三档 M 同源截断；MC 后验含 Dirichlet(α/K) 平滑；'
                    '判定用 top1（对均匀子集假设的手切失配稳健），'
                    'CE 仅作参考',
    }
    verdict = build_verdict(table, meta, mc)
    res = {'meta': meta, 'counts_exact': counts_exact, 'mc': mc,
           'table': table, 'verdict': verdict}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'bayes_ceiling.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'BAYES_CEILING_REPORT.md', res)
    print(f'\n[out] {out}/bayes_ceiling.json')
    print(f'[out] {out}/BAYES_CEILING_REPORT.md')
    print(f'[verdict] {verdict["decision"]}')
    for line in verdict['findings']:
        print(' -', line)
    print(f'[done] {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
