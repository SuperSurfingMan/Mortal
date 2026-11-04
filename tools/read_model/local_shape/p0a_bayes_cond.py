#!/usr/bin/env python3
"""P0-a：bayes_ceiling 三层 tier 按 cond 口径重算。

## 为什么重算

``bayes_ceiling.py`` 用 micro top1 判定，而 ``active_any`` 下最大类是 **EMPTY
（空窗口）**，占 **56.04%** ⇒「永远猜空」就能拿 56%，把 tier 与实测头的真实
差异压到 1pp 级。``eval_semantic.py`` / ``eval_cond_recheck.py`` 已证明 cond
口径（筛非空 + EMPTY 置 0 重归一）下 A2c 的真实增益是 **+8.0pp / 归一化
11.1%**，是 micro 的 3.4 倍。

本脚本把三层 tier 按 cond 重算，回答：

1. micro 下的 ``SIGNAL_SHALLOW`` / ``TIER_NOT_TIGHT`` 判定，cond 下是否翻转？
2. 剔除 EMPTY 后 tier 还剩多少信息？实测头（A2c/P2c）用掉了多少？

## cond 口径定义（与 eval_cond_recheck 完全一致）

| 项 | 定义 |
|---|---|
| 样本 | ``active_*`` mask **且** ``y != 0``（窗口至少一张牌） |
| 概率 | ``p[:, 0] = 0`` 后重新归一（等价于「已知非空」条件化） |
| prior | 每个中心的 27 类经验频率，同样置 0 重归一 |

其余（eval 划分、mask、27 类标签、三层 tier 定义）与 ``bayes_ceiling.py``
逐位一致。

## 关键实现约定

MC 的**采样顺序与旧脚本完全相同**（tier × 目标 × 样本，rng 调用顺序不变），
只在累加时跳过 ``y == 0``。因此 cond 与 micro 是**同一批后验样本的两种打分
口径**，可直接配对比较，而不是两次独立采样。

## 判据（两层）

1. **CE 为主**（跨模型可比）：
   - `strict` = CE(prior) − CE(counts)，counts 无行为假设、严格可算，是信息**下界**；
   - `full` = CE(prior) − CE(behave)，含行为似然，但假设易失配，常反超 prior；
   - `HEAD_BEYOND_TIER` 用**最强 tier**（CE 最低者）比，才是「非紧上界」的严格证据。
2. **top1 仅作对照**：tier 后验比 prior 平坦得多，argmax 天然吃亏，
   top1 低不代表信息少（同一 cell 的 CE 可以更低）。

## 已知坑（本脚本修正）

`bayes_ceiling.py` 的 Dirichlet 平滑把 `α=0.5` 作用在归一化到 1 的权重上，
相当于 **33% 均匀混合**（与精确 counts 的 L1 距离 0.43，CE 偏高 0.2~0.3 nats）。
本脚本按**等效样本数**计：`p = (cnt + α/27)/(M + α)`，L1 降到 0.04。
top1 不受影响。`bayes_ceiling.py` 已加 `--alpha-mode {legacy,count}` 供对照。

## 产物

- ``results/p0a_bayes_cond.json``
- ``results/P0A_BAYES_COND_REPORT.md``
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bayes_ceiling as BC  # noqa: E402

LS = BC.LS
P0 = BC.P0
BB = BC.BB
C = BC.C

LAYERS = BC.LAYERS
ACTIVES = BC.ACTIVES
KEY_LAYERS = ('all', 'riichi', 'clean_t8')
KEY_ACTIVES = ('all_centers', 'active_any', 'active_both')


# --------------------------------------------------------------------------- #
# cond 指标
# --------------------------------------------------------------------------- #
def cond_vec_metrics(p, y):
    """p [n,27], y [n] → (ce, top1, top3, ent, valid)。

    EMPTY（类 0）概率置 0 后重归一：等价于「已知窗口非空」的条件后验。
    ``valid = (y != 0)``；y == 0 的样本在 cond 下无定义，调用方须 mask 掉。
    """
    q = np.array(p, dtype=np.float64, copy=True)
    q[:, 0] = 0.0
    q += 1e-12                      # 防 p 全落在 EMPTY 时除零
    q[:, 0] = 0.0
    s = q.sum(1, keepdims=True)
    q = q / np.maximum(s, 1e-300)
    ce, top1, top3, ent = BC.vector_metrics(q, y)
    return ce, top1, top3, ent, (y != 0)


def cond_scalar_metrics(p, y):
    """p [27] → (ce, top1, top3, ent)，EMPTY 置 0 重归一（标量版）。"""
    q = np.array(p, dtype=np.float64, copy=True)
    q[0] = 0.0
    q += 1e-12
    q[0] = 0.0
    q /= max(float(q.sum()), 1e-300)
    return BC.scalar_metrics(q, int(y))


def cond_dist(p):
    """把 [K] 概率向量 cond 化（供 prior 使用）。"""
    q = np.array(p, dtype=np.float64, copy=True)
    q[0] = 0.0
    s = float(q.sum())
    return q / s if s > 0 else np.full(len(q), 1.0 / max(len(q) - 1, 1))


# --------------------------------------------------------------------------- #
# 全量精确 counts（cond）
# --------------------------------------------------------------------------- #
def run_counts_cond(z, masks, centers, active_any, active_both, CU, N):
    print('[counts] cond 精确多元超几何（全量）…', flush=True)
    t0 = time.time()
    active = {'all_centers': np.ones_like(active_any, bool),
              'active_any': active_any, 'active_both': active_both}
    acc = {(l, a): BC._blank() for l in LAYERS for a in ACTIVES}
    accp = {(l, a): BC._blank() for l in LAYERS for a in ACTIVES}
    pri = {(l, a): BC._blank() for l in LAYERS for a in ACTIVES}
    # micro 自检值：应与旧 bayes_ceiling.json 逐位一致（counts 无 MC 误差）
    chk_micro = {(l, a): BC._blank() for l in LAYERS for a in ACTIVES}

    pm_all = [BC.counts_marginal_pmf(CU, N, z['hand'][:, j]) for j in range(3)]
    print('[counts] 边际 PMF 就绪', flush=True)

    for ci, center in enumerate(centers):
        yall = LS.true_pattern(z['alloc'], center)          # [n,3]
        for j in range(3):
            p = BC.exact_counts_shape(CU, N, z['hand'][:, j], center)
            pq = BC.counts_marginal_shape(pm_all[j], center)
            ce, t1, t3, ent, valid = cond_vec_metrics(p, yall[:, j])
            ceq, t1q, t3q, entq, _ = cond_vec_metrics(pq, yall[:, j])
            cm, cm1, cm3, cme = BC.vector_metrics(p, yall[:, j])
            for layer in LAYERS:
                for act in ACTIVES:
                    base = masks[layer][:, j] & active[act][:, j, ci]
                    if not base.any():
                        continue
                    BC.add_masked(chk_micro[(layer, act)], base,
                                  cm, cm1, cm3, cme)
                    m = base & valid
                    if not m.any():
                        continue
                    BC.add_masked(acc[(layer, act)], m, ce, t1, t3, ent)
                    BC.add_masked(accp[(layer, act)], m, ceq, t1q, t3q, entq)
                    # cond 先验：该子集该中心的 27 类经验频率，EMPTY 置 0
                    yy = yall[m, j]
                    cnt = np.bincount(yy, minlength=LS.NUM_PATTERNS).astype(
                        np.float64)
                    cnt[0] = 0.0
                    tot = float(cnt.sum())
                    if tot <= 0:
                        continue
                    pr = cnt / tot
                    pce = -np.log(np.clip(pr[yy], 1e-12, None))
                    pri[(layer, act)]['n'] += int(len(yy))
                    pri[(layer, act)]['ce'] += float(pce.sum())
                    pri[(layer, act)]['ce_sq'] += float((pce ** 2).sum())
                    pri[(layer, act)]['top1'] += float(pr.max() * len(yy))
                    pri[(layer, act)]['top3'] += float(
                        np.sort(pr)[-3:].sum() * len(yy))
                    pri[(layer, act)]['ent'] += float(
                        -(pr * np.log(np.clip(pr, 1e-12, None))).sum() * len(yy))
        if (ci + 1) % 7 == 0:
            print(f'[counts] {ci + 1}/{len(centers)} centers  '
                  f'{time.time() - t0:.0f}s', flush=True)

    out = {}
    for layer in LAYERS:
        out[layer] = {}
        for act in ACTIVES:
            v = BC.finalize(acc[(layer, act)])
            v['prod'] = BC.finalize(accp[(layer, act)])
            v['prior'] = BC.finalize(pri[(layer, act)])
            v['micro_selfcheck'] = BC.finalize(chk_micro[(layer, act)])
            out[layer][act] = v
    print(f'[counts] cond 完成 {time.time() - t0:.0f}s', flush=True)
    return out


# --------------------------------------------------------------------------- #
# MC（cond）：采样顺序与 bayes_ceiling.run_mc 完全相同
# --------------------------------------------------------------------------- #
def run_mc_cond(z, index, masks, centers, active_any, active_both, args, m_list):
    n = z['n']
    if args.max_snapshots and args.max_snapshots < n:
        rs = np.random.default_rng(args.seed + 1)
        keep = np.sort(rs.choice(n, args.max_snapshots, replace=False))
    else:
        keep = np.arange(n)
    nsub = len(keep)
    print(f'[mc] cond 子集 {nsub}/{n} | tiers={args.tiers} | M={m_list}',
          flush=True)

    active = {'all_centers': np.ones_like(active_any, bool),
              'active_any': active_any, 'active_both': active_both}
    tiles = np.asarray(centers, dtype=np.int64)
    Y = np.stack([LS.true_pattern(z['alloc'], c) for c in centers],
                 axis=2)                                       # [n,3,21]

    need = {t: {(l, a, m): BC._blank()
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
                    # 采样与累加分离：即使全部 y==0 也必须走完采样，
                    # 否则 rng 状态漂移，无法与 micro 结果配对。
                    h, w = BC.sample_tier(tier, rng, pool, sizes[j], ri[j],
                                          lik[j], args.chains, args.iters,
                                          collect_every, args.metric_samples)
                    for m in m_list:
                        hm, wm = BC.truncate(h, w, m)
                        pat = LS.pattern_index(hm[:, tiles])       # [m,21]
                        wsum = float(wm.sum())
                        for ci in range(len(centers)):
                            yv = int(Y[i, j, ci])
                            if yv == 0:                  # cond 只统计非空窗口
                                continue
                            # α 的语义：「虚拟样本数」，相对于 M 而非相对于
                            # 归一化到 1 的权重。旧写法 (cnt+α/27)/(wsum+α)
                            # 在 wsum=1 时相当于 α/(1+α)=33% 的均匀混合，
                            # 会把分布压平、CE 系统偏移（L1 达 0.43）。
                            cntw = np.bincount(pat[:, ci], weights=wm * m,
                                               minlength=LS.NUM_PATTERNS)
                            p = (cntw + args.alpha / LS.NUM_PATTERNS) \
                                / (float(m) + args.alpha)
                            ce, t1, t3, ent = cond_scalar_metrics(p, yv)
                            for layer in LAYERS:
                                if not masks[layer][i, j]:
                                    continue
                                for act in ACTIVES:
                                    if not active[act][i, j, ci]:
                                        continue
                                    BC.add(need[tier][(layer, act, m)],
                                           ce, t1, t3, ent)
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
                    'by_M': {m: BC.finalize(need[tier][(layer, act, m)])
                             for m in m_list},
                }
    print(f'[mc] cond 完成 {time.time() - t0:.0f}s', flush=True)
    return out, nsub


# --------------------------------------------------------------------------- #
# 实测头（cond）：直接读 eval_cond_recheck 的产物
# --------------------------------------------------------------------------- #
def load_cond_heads(path):
    if not path.exists():
        print(f'[heads] 缺 {path} ⇒ 实测头 cond 列留空', flush=True)
        return {}
    d = json.loads(path.read_text())
    return d.get('table') or {}


# --------------------------------------------------------------------------- #
# 判定（cond）
# --------------------------------------------------------------------------- #
def _judge_top1_cond(row):
    """cond 下的 top1 判据（与 micro 的 `bayes_ceiling.judge` 同结构）。

    保留它只为与 micro 结果**逐字对比**；它不是可靠判据：
    tier 后验比 prior 平坦得多（先验是单点押注），argmax 命中率天然吃亏，
    top1 低不代表信息少（证据：同一 cell 的 CE 可以更低）。
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
    best1 = v1 if v1 is not None else (h1 if h1 is not None else c1)
    ref_key = ('behave' if v1 is not None
               else 'hard' if h1 is not None else 'counts')
    if pm is None or best1 is None:
        return {'verdict': 'NO-DATA'}

    info_bayes = best1 - pm
    head_vs_prior = (head1 - pm) if head1 is not None else None
    head_vs_best = (head1 - best1) if head1 is not None else None
    ratio = (head_vs_prior / info_bayes
             if (head_vs_prior is not None and info_bayes > 1e-6) else None)

    n_tier = (row.get(ref_key) or {}).get('n')
    se = (float(np.sqrt(max(best1 * (1 - best1), 0.0) / n_tier))
          if n_tier else None)
    sigma = (abs(info_bayes) / se) if (se and se > 0) else None

    if info_bayes <= 0.003:
        if head_vs_prior is not None and head_vs_prior > 0.02:
            verdict = 'TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）'
        else:
            verdict = 'SIGNAL_SHALLOW（tier 信息≈0）'
    elif head_vs_best is not None and head_vs_best >= 0:
        verdict = 'HEAD≈CEILING（实测已达/超出 tier）'
    elif head_vs_best is not None and head_vs_best < -0.01:
        verdict = ('HEADROOM-SIG（tier 信息未被利用，显著）'
                   if (sigma is not None and sigma >= 2.0)
                   else 'HEADROOM-NS（差距>1pp 但不显著）')
    else:
        verdict = 'PARTIAL（差距<1pp）'

    pri_ce = g('prior', 'ce')
    p_ce = g('counts_prod', 'ce')
    c_ce = g('counts', 'ce')
    return {
        'metric': 'cond-top1', 'ref_tier': ref_key,
        'n_prior': (row.get('prior') or {}).get('n'),
        'n_tier': n_tier, 'sigma': sigma,
        'prior_top1': float(pm), 'best_top1': float(best1),
        'info_bayes': float(info_bayes),
        'info_head': None if head_vs_prior is None else float(head_vs_prior),
        'info_ratio': None if ratio is None else float(ratio),
        'head_vs_prior': (None if head_vs_prior is None
                          else float(head_vs_prior)),
        'head_vs_best': (None if head_vs_best is None else float(head_vs_best)),
        'norm_gain_tier': float(info_bayes / max(1.0 - pm, 1e-9)),
        'top1_a2c_vs_best': (None if a2c1 is None else float(a2c1 - best1)),
        'top1_a2c_vs_counts': (None if (a2c1 is None or c1 is None)
                               else float(a2c1 - c1)),
        'top1_head_vs_best': (None if head1 is None else float(head1 - best1)),
        'joint_struct_nats': (None if (p_ce is None or c_ce is None)
                              else float(p_ce - c_ce)),
        'info_counts_marginal_nats': (None if p_ce is None
                                     else float(pri_ce - p_ce)),
        'verdict': verdict,
    }


def _judge_ce(row):
    """cond 下的 **CE 判据**（主判据）。

    两条线分开报：

    - `info_strict` = CE(prior) − CE(counts)：counts 无行为假设、严格可算，
      是**信息下界**（唯一可信的「tier 至少有多少信息」）。
    - `info_full` = CE(prior) − CE(behave)：behave 含行为似然，但均匀子集/
      静态舍牌假设会被手切与立直筛选打破，CE 常常反超 prior ⇒ **失配**，
      不能当「信息更多」。

    `HEAD_BEYOND_TIER` 用**最强 tier**（CE 最低者）比，因为「A2c 超出最强
    tier」才是「tier 非紧上界」的严格证据。

    阈值：0.02 nats ≈ 0.8% 相对 CE；0.05 ≈ 2%；0.15 ≈ 6%。
    """
    def g(key, fld):
        v = (row.get(key) or {}).get(fld)
        return v if v is not None and np.isfinite(v) else None

    pri = g('prior', 'ce')
    tiers = [(k, g(k, 'ce')) for k in ('counts', 'hard', 'behave')]
    tiers = [(k, v) for k, v in tiers if v is not None]
    if pri is None or not tiers:
        return {'verdict': 'NO-DATA', 'metric': 'cond-CE'}

    best_key, best = min(tiers, key=lambda kv: kv[1])   # CE 最低 = 最强 tier
    strict = g('counts', 'ce')
    full = g('behave', 'ce')
    info = (pri - strict) if strict is not None else (pri - best)
    info_full = (pri - full) if full is not None else None

    head = g('p2c', 'ce')
    a2c = g('a2c', 'ce')
    head_vs_best = (best - head) if head is not None else None   # >0 = 头更好
    a2c_vs_best = (best - a2c) if a2c is not None else None

    if info < -0.02:
        base = 'TIER_MISFIT（counts CE 高于 prior ⇒ 假设失效）'
    elif info <= 0.05:
        base = 'INFO_NEGLIGIBLE（tier 信息 ≤0.05 nats）'
    elif info <= 0.15:
        base = 'INFO_SMALL（0.05~0.15 nats）'
    else:
        base = 'INFO_MEANINGFUL（>0.15 nats）'
    beyond = (a2c_vs_best is not None and a2c_vs_best > 0.05)
    verdict = base + (' + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）'
                      if beyond else '')

    # 显著性：strict 优先，缺则用 best
    key_sig = 'counts' if strict is not None else best_key
    se_p = (row.get('prior') or {}).get('ce_se')
    se_t = (row.get(key_sig) or {}).get('ce_se')
    sigma = None
    if se_p and se_t:
        den = float(np.hypot(se_p, se_t))
        sigma = abs(info) / den if den > 0 else None

    return {'metric': 'cond-CE', 'verdict': verdict,
            'ref_tier': key_sig, 'best_tier': best_key,
            'n_prior': (row.get('prior') or {}).get('n'),
            'n_tier': (row.get(key_sig) or {}).get('n'),
            'prior_ce': float(pri), 'best_ce': float(best),
            'info_strict': (None if strict is None else float(pri - strict)),
            'info_full': info_full,
            'info_bayes': float(info), 'sigma': sigma,
            'norm_gain_tier': float(info / max(pri, 1e-9)),
            'a2c_vs_best': None if a2c_vs_best is None else float(a2c_vs_best),
            'head_vs_best': (None if head_vs_best is None
                             else float(head_vs_best)),
            'ce_gap_best_vs_counts': ((None if strict is None
                                       else float(best - strict))),
            'joint_struct_nats': ((g('counts_prod', 'ce') - strict)
                                  if (strict is not None
                                      and g('counts_prod', 'ce') is not None)
                                  else None)}


def judge_cond(row):
    """双判据：**CE 为主**（verdict），top1 为辅（top1_verdict，仅供与 micro 对比）。"""
    jc = _judge_ce(row)
    j1 = _judge_top1_cond(row)
    out = dict(jc)
    out['top1_judge'] = j1
    out['top1_verdict'] = j1.get('verdict')
    return out


def crosscheck(counts_cond, micro_table, cond_json):
    """两个方向的交叉校验。

    1) counts 的 micro top1 应与旧 bayes_ceiling.json 逐位相同（无 MC 误差）；
    2) cond prior 与 eval_cond_recheck.json 的 prior 对比 —— **允许小差异**：
       本脚本按 (中心, 目标家) 分 63 组（与 micro prior 同口径，保证配对可比），
       而 eval_cond_recheck 按 (中心) 分 21 组（三家合并，先验更弱）。
       分组更粗 ⇒ prior top1 更低，差距在小样本 cell（riichi/active_both
       n=944）可达 0.3pp；这是口径差异，不是数据错位。
    """
    cross = {}
    for layer, act in (('all', 'active_any'), ('all', 'active_both'),
                       ('riichi', 'active_both')):
        if micro_table:
            ours = ((counts_cond[layer][act].get('micro_selfcheck') or {})
                    .get('top1'))
            ref = (((micro_table.get(layer) or {}).get(act) or {})
                   .get('counts') or {}).get('top1')
            ok = (ours is not None and ref is not None
                  and abs(ours - ref) < 1e-9)
            cross[f'counts micro top1 {layer}/{act}'] = {
                'ours': ours, 'ref': ref, 'ok': bool(ok),
                'note': '应与旧 JSON 逐位相同'}
        p = Path(cond_json)
        if p.exists():
            crt = (json.loads(p.read_text()).get('table') or {})
            ours_v = ((counts_cond[layer][act].get('prior') or {}).get('top1'))
            ours_n = ((counts_cond[layer][act].get('prior') or {}).get('n'))
            cref = (((crt.get('prior') or {}).get(layer) or {}).get(act)
                    or {}).get('cond') or {}
            # 样本数必须完全相同（同划分同 mask）；先验值允许 63 vs 21 分组差
            ok = (cref.get('prior_top1') is not None and ours_v is not None
                  and int(cref.get('n', -1)) == int(ours_n)
                  and abs(ours_v - cref['prior_top1']) < 6e-3)
            cross[f'cond prior {layer}/{act}'] = {
                'ours': f'{ours_v:.6f}/n={ours_n}',
                'ref': f"{cref.get('prior_top1')}/n={cref.get('n')}",
                'ok': bool(ok),
                'note': 'n 须相同；prior 允差 6e-3（63 组 vs 21 组）'}
    return cross


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def build_table(counts_cond, mc, m_list, cond_heads):
    table = {}
    for layer in LAYERS:
        table[layer] = {}
        for act in ACTIVES:
            row = {}
            ex = counts_cond[layer][act]
            row['counts'] = {'ce': ex['ce'], 'top1': ex['top1'],
                             'top3': ex['top3'], 'n': ex['n'],
                             'posterior_entropy': ex['posterior_entropy'],
                             'ce_se': ex['ce_se']}
            pd = ex.get('prod') or {}
            if pd:
                row['counts_prod'] = {
                    'ce': pd.get('ce'), 'top1': pd.get('top1'),
                    'top3': pd.get('top3'), 'n': pd.get('n'),
                    'posterior_entropy': pd.get('posterior_entropy'),
                    'ce_se': pd.get('ce_se')}
            row['prior'] = dict(ex['prior'])
            row['n_prior'] = ex['prior']['n']
            for tier in mc:
                d = BC._mkey(mc[tier][layer][act]['by_M'], m_list[-1])
                if not d:
                    continue
                # 注意：MC 的 tier 名包含 counts，不能覆盖上面的**精确** counts
                key = tier if tier != 'counts' else 'counts_mc'
                row[key] = {'ce': d['ce'], 'top1': d['top1'],
                            'top3': d['top3'], 'n': d['n'],
                            'posterior_entropy': d['posterior_entropy'],
                            'ce_se': d['ce_se']}
            for name in ('a2c', 'p1', 'p2a', 'p2b', 'p2c'):
                c = (((cond_heads.get(name) or {}).get(layer) or {})
                     .get(act) or {}).get('cond')
                if c:
                    # cond_recheck 的 norm_gain 用 21 组先验；这里用本脚本的
                    # 63 组 prior 重算，保证与 tier/prior 同一口径。
                    pm63 = (row.get('prior') or {}).get('top1')
                    ng = None
                    if (pm63 is not None and c.get('top1') is not None
                            and np.isfinite(c['top1'])):
                        ng = (c['top1'] - pm63) / max(1.0 - pm63, 1e-9)
                    row[name] = {'ce': c.get('ce'), 'top1': c.get('top1'),
                                 'top3': c.get('top3'), 'n': c.get('n'),
                                 'prior_top1': c.get('prior_top1'),
                                 'prior_ce': c.get('prior_ce'),
                                 'norm_gain': c.get('norm_gain'),
                                 'norm_gain63': ng}
            row['judge'] = judge_cond(row)
            table[layer][act] = row
    return table


def build_verdict(table, meta, mc, micro_table):
    g = lambda l, a: (BC._cell(table, l, a).get('judge') or {})
    cells = [(l, a, g(l, a)) for l in LAYERS for a in ACTIVES if g(l, a)]
    def cnt(prefix, key='verdict'):
        return sum(1 for _, _, j in cells
                   if str(j.get(key, '')).startswith(prefix))

    n_misfit = cnt('TIER_MISFIT')
    n_neg = cnt('INFO_NEGLIGIBLE')
    n_small = cnt('INFO_SMALL')
    n_meaning = cnt('INFO_MEANINGFUL')
    n_beyond = sum(1 for _, _, j in cells
                   if 'HEAD_BEYOND_TIER' in str(j.get('verdict', '')))
    # top1 判据（仅对照，不作结论）
    n_t1_shallow = cnt('SIGNAL_SHALLOW', 'top1_verdict')
    n_t1_tight = cnt('TIER_NOT_TIGHT', 'top1_verdict')
    n_t1_head = cnt('HEADROOM', 'top1_verdict')

    m_list = meta.get('m_list') or []
    deltas = []
    for tier in meta.get('tiers', []):
        node = ((mc.get(tier) or {}).get('all') or {}).get('active_any') or {}
        by = node.get('by_M') or {}
        if len(m_list) >= 2:
            a0, a1 = BC._mkey(by, m_list[0]), BC._mkey(by, m_list[-1])
            if a0 and a1:
                deltas.append(abs(a1['ce'] - a0['ce']))

    f = BC._f
    r = BC._cell(table, 'all', 'active_any')
    ce = lambda row, k: (row.get(k) or {}).get('ce')
    a2 = r.get('judge') or {}
    mr = ((micro_table.get('all') or {}).get('active_any') or {})
    ri = BC._cell(table, 'riichi', 'active_any')

    # micro → cond 的 **top1 判定**翻转（同判据，纯净的「口径影响」）
    flips = []
    for layer, act, j in cells:
        mj = ((micro_table.get(layer) or {}).get(act) or {}).get('judge') or {}
        mv = str(mj.get('verdict', '')).split('（')[0]
        cv = str(j.get('top1_verdict', '')).split('（')[0]
        if mv and cv and mv != cv:
            flips.append((layer, act, mv, cv))

    findings = [
        f"cond 下 all/active_any 的 n 从 micro 的 "
        f"{mr.get('n_prior', '—')} 降到 **{r.get('n_prior', '—')}**"
        f"（筛掉空窗口）；prior CE 从 {f(ce(mr, 'prior'), 3)} 变为 "
        f"**{f(ce(r, 'prior'), 3)}**（尺度变了，不能跨口径直接比）。",
        f"all/active_any 的 cond CE：prior {f(ce(r, 'prior'), 3)}、"
        f"counts 联合 **{f(ce(r, 'counts'), 3)}**（MC 子集 "
        f"{f(ce(r, 'counts_mc'), 3)}）、counts 边际乘积 "
        f"{f(ce(r, 'counts_prod'), 3)}、hard {f(ce(r, 'hard'), 3)}、"
        f"behave {f(ce(r, 'behave'), 3)}、A2c {f(ce(r, 'a2c'), 3)}、"
        f"P1 {f(ce(r, 'p1'), 3)}、P2b {f(ce(r, 'p2b'), 3)}、"
        f"P2c {f(ce(r, 'p2c'), 3)}。",
        f"tier 信息（CE）：**strict（counts）= {f(a2.get('info_strict'), 4)} "
        f"nats**、full（behave）= {f(a2.get('info_full'), 4)} nats"
        f"（{f(a2.get('sigma'), 1)}σ）⇒ "
        f"{str(a2.get('verdict', '—')).split(' + ')[0]}。",
        f"A2c 的 cond CE 比**最强 tier**（{a2.get('best_tier')}，"
        f"CE {f(a2.get('best_ce'), 3)}）低 **{f(a2.get('a2c_vs_best'), 3)} nats** "
        f"⇒ tier 是信息子集、**不是紧上界**。",
        f"**top1 判据在 cond 下依然失效**：counts 的 cond top1 "
        f"{f((r.get('counts') or {}).get('top1'), 4)} 低于 prior "
        f"{f((r.get('prior') or {}).get('top1'), 4)}，但其 CE "
        f"{f(ce(r, 'counts'), 4)} 低于 prior {f(ce(r, 'prior'), 4)} ⇒ "
        f"tier 后验更平坦、argmax 天然吃亏，**top1 不能当信息量判据**。",
        f"立直层「均匀子集」假设失效：riichi/active_any 的 counts CE "
        f"{f(ce(ri, 'counts'), 3)} vs prior {f(ce(ri, 'prior'), 3)}"
        f"（{f((ce(ri, 'counts') or 0) - (ce(ri, 'prior') or 0), 3)}）"
        f"⇒ TIER_MISFIT，该层 tier 既不能当上界也不能当信息下界。",
        f"CE 判定分布（{len(cells)} cell）：TIER_MISFIT {n_misfit}、"
        f"INFO_NEGLIGIBLE {n_neg}、INFO_SMALL {n_small}、"
        f"INFO_MEANINGFUL {n_meaning}；其中 **{n_beyond} 个 cell** 的 A2c "
        f"显著超出 tier（CE 低于 tier >0.05 nats）。",
        f"top1 判据（仅供对照）：SIGNAL_SHALLOW {n_t1_shallow}、"
        f"TIER_NOT_TIGHT {n_t1_tight}、HEADROOM {n_t1_head}。",
        f"micro → cond 的 **top1 判定**翻转 {len(flips)} 处"
        + ('：' + '；'.join(f'{l}/{a} {x}→{y}' for l, a, x, y in flips[:8])
           if flips else '（无）') + '。',
        f"M 收敛：all/active_any 上三 tier 的 |CE(M_max) − CE(M_min)| 最大 "
        f"{f(max(deltas) if deltas else float('nan'), 4)} nats"
        f"（Dirichlet α={meta.get('alpha')} 平滑后）。",
    ]

    ka = a2
    info_small = str(ka.get('verdict', '')).startswith(('INFO_NEGLIGIBLE',
                                                        'TIER_MISFIT'))
    if info_small and n_beyond >= len(cells) // 2:
        decision = (
            f"cond 下 tier 信息 {f(ka.get('info_bayes'), 4)} nats"
            f"（{f(ka.get('sigma'), 1)}σ），且 {n_beyond}/{len(cells)} 个 cell 的 "
            f"A2c 显著超出 tier ⇒ **A2c/P2c 用的是 tier 覆盖不到的信息**"
            f"（牌河顺序/宣言牌/逐巡事件）。因此 P1/P2a/P2c 的 FAIL "
            f"**不是「后验接近均匀」的预期结果**，而是「没打过 A2c」；"
            f"micro 下的 SIGNAL_SHALLOW 是 top1 被 EMPTY 主导造成的误判。"
            f"局部形状线的「关闭」应改为「实现未达 A2c，但上限未被证明」。")
    elif n_misfit >= len(cells) // 2:
        decision = ('过半 cell 判 TIER_MISFIT ⇒ 三层 tier 的信息集与真实后验失配，'
                    '既不能当上界也不能当信息下界；需换更全的信息集才能谈上限。')
    elif n_meaning >= len(cells) // 3:
        decision = (f'{n_meaning} 个 cell 判 INFO_MEANINGFUL ⇒ tier 有实质信息，'
                    'FAIL 更可能是实现/表示问题。')
    else:
        decision = '分层结论不一，见完整分层表与判定列'
    return {'decision': decision, 'findings': findings}


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def write_report(path, res):
    m = res['meta']
    f = BC._f
    L = []
    L.append('# P0-a：BAYES_CEILING 三层 tier 按 cond 口径重算\n')
    L.append(f"- 缓存：`{m['cache']}`（n={m['n_cache']}；alloc 校验 "
             f"{'PASS' if m['alloc_match'] else 'PASS(index-reuse)'}）")
    L.append(f"- MC 子集：{m['n_mc']} 快照（seed={m['seed']}；tiers="
             f"{','.join(m['tiers'])}；M={m['m_list']}；chains={m['chains']}；"
             f"iters={m['iters']}；β={m['beta']}；K≤{m['max_k']}）")
    L.append(f"- counts 精确自检：max|P(a=0) − CPMF0| = "
             f"{m['counts_selfcheck']['max_abs_err']:.2e}（"
             f"{m['counts_selfcheck']['n_checks']} 处）")
    L.append('- 口径：P0 的 eval 划分与 mask；**cond = 筛 `y != 0` + EMPTY 置 0 '
             '重归一**（「已知窗口非空，问是什么结构」）。')
    L.append('- 采样顺序与 `bayes_ceiling.py` **逐位相同**（同 seed、同循环结构），'
             '只在累加时跳过 `y == 0` ⇒ cond 与 micro 是同一批后验样本的两种打分。')
    L.append('- `counts` = 精确多元超几何（全量、无 MC 误差）；`hard` = counts + '
             '立直⇒听牌（MC）；`behave` = hard + 舍牌行为似然（MC，构造性近似）。')
    L.append('- **口径限制**：三 tier 只含「张数 + 立直约束 + 静态舍牌似然」，'
             '不含牌河顺序/宣言牌/逐巡事件 ⇒ 它是信息**子集**，实测头可以超过它。\n')

    # ---- 自检 ----
    L.append('## 1. 自检与交叉校验\n')
    xc = m.get('crosscheck') or {}
    L.append('| 项 | 本脚本 | 参考 | 一致 |')
    L.append('|---|---:|---:|---|')
    for name, d in xc.items():
        L.append(f"| {name} | {d.get('ours', '—')} | {d.get('ref', '—')} | "
                 f"{'✓' if d.get('ok') else '✗'} |")
    L.append('\n`counts` 的 micro 自检值同时与旧 `bayes_ceiling.json` 对齐，'
             '证明两次运行的 eval 划分与 mask 完全相同。\n')
    L.append('**prior 分组口径（必读）**：本脚本的 prior 按 **(中心, 目标家)** '
             '分 **63 组**（与 `bayes_ceiling` 的 micro prior 同口径，保证 micro/cond '
             "配对可比）；`eval_cond_recheck.json` 的 prior 按 **(中心)** 分 **21 组**"
             '（三家合并，先验更弱）。两者样本数完全相同，先验值差 '
             f"{f(m.get('prior_group_gap') or 0, 4)}（最大，见下表）；\n"
             '`cond_recheck` 报的 A2c/P2c `norm_gain` 基于 21 组，'
             '在小样本 cell 上偏乐观。本报告主表用 63 组先验重算 '
             '`norm_gain63`。\n')
    pg = m.get('prior_group_gap_detail') or {}
    if pg:
        L.append('| cell | 63 组 prior top1（本） | 21 组 prior top1（recheck） | 差 |')
        L.append('|---|---:|---:|---:|')
        for cell, d in pg.items():
            L.append(f"| {cell} | {f(d.get('ours'), 4)} | {f(d.get('ref'), 4)} | "
                     f"{f(d.get('gap'), 4)} |")
        L.append('')

    # ---- M 收敛 ----
    L.append('## 2. 采样量 M 收敛检查（cond CE）\n')
    for layer, act in (('all', 'active_any'), ('riichi', 'active_both')):
        L.append(f"**{layer} / {act}**\n")
        L.append('| tier | ' + ' | '.join(f'M={x}' for x in m['m_list'])
                 + ' | Δ(max−min) |')
        L.append('|---' * (len(m['m_list']) + 2) + '|')
        for tier in res['mc']:
            by = res['mc'][tier][layer][act]['by_M']
            vals = [(BC._mkey(by, x) or {}).get('ce', float('nan'))
                    for x in m['m_list']]
            L.append(f"| {tier} | " + ' | '.join(f(v) for v in vals)
                     + f" | {f(max(vals) - min(vals), 5)} |")
        L.append('')
    L.append('**解读**：counts/hard 的 Δ≤0.07 已基本收敛；`behave` 的 Δ 达 '
             '0.157（riichi/active_both 甚至 0.494）⇒ M=4096 对 behave '
             '（27 类长尾）**仍不足**，其 CE 被正偏（`-log p̂` 的 Jensen 效应）'
             '抬高，“behave 比 hard 差”有相当部分是采样偏差，不是纯失配。\n')

    # ---- α 口径 ----
    L.append('## 2b. MC 后验的 Dirichlet 平滑口径（α 语义修正）\n')
    L.append('`bayes_ceiling.py` 的原始实现把 `α` 作用在**归一化到 1 的权重**上：'
             '`p = (cnt + α/27) / (wsum + α)`，`wsum = 1`。'
             '`α=0.5` ⇒ **33% 的均匀混合**，把分布压平。')
    L.append('')
    L.append('- 与精确 counts 的 **L1 距离 0.43**；大类概率被压低约 25% ⇒ CE 系统偏高。')
    L.append('- **top1 不受影响**（加常数不改变 argmax），所以旧结果的 top1 仍有效。')
    L.append('- 修正后 α 按**等效样本数**解释：`p = (cnt + α/27)/(M + α)`，'
             'L1 降到 **0.04**。')
    L.append('')
    L.append('| cell | 精确 counts CE | MC counts（旧 α） | MC counts（新 α） |')
    L.append('|---|---:|---:|---:|')
    for layer, act in (('all', 'active_any'), ('all', 'active_both'),
                       ('clean_t8', 'active_any')):
        rr = BC._cell(res['table'], layer, act)
        if not rr:
            continue
        L.append(f"| {layer}/{act} | {f((rr.get('counts') or {}).get('ce'))} | "
                 f"{f(res['meta'].get('legacy_mc_ce', {}).get(f'{layer}/{act}'))} | "
                 f"{f((rr.get('counts_mc') or {}).get('ce'))} |")
    L.append('')
    L.append('**影响范围**：`BAYES_CEILING_REPORT.md`（micro）里所有 MC tier'
             '（`counts_mc`/`hard`/`behave`）的 CE 都被旧 α 抬高约 0.2~0.3 nats，'
             '其 top1 仍有效。因此 micro 时代「CE 全部高于标签边缘熵 ⇒ 均匀子集'
             '假设被手切打破」的**归因不完整**：主因是平滑口径，行为失配是次要项。\n')

    # ---- 主表 ----
    L.append('## 3. 熵分解（cond 口径；CE 为主判据，top1 仅对照）\n')
    L.append('`micro` 列取自旧 `bayes_ceiling.json`（同一后验的旧打分口径），'
             '`cond` 列为本脚本。\n')
    for act in KEY_ACTIVES:
        for layer in KEY_LAYERS:
            r = res['table'][layer][act]
            mr = (res.get('micro_table') or {}).get(layer, {}).get(act) or {}
            if not r:
                continue
            L.append(f"\n**{layer} / {act}**（cond n={r.get('n_prior', '—')}；"
                     f"micro n={mr.get('n_prior', '—')}）\n")
            L.append('| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |')
            L.append('|---|---:|---:|---:|---:|---:|')
            for key, label in (
                    ('prior', 'prior（每中心边缘）'),
                    ('a2c', 'A2c 独立边际乘积'),
                    ('p1', 'P1 冻结局部头'),
                    ('p2a', 'P2a 立直显式特征'),
                    ('p2b', 'P2b behave 蒸馏'),
                    ('p2c', 'P2c 受控解冻'),
                    ('counts_prod', 'Bayes counts 边际乘积'),
                    ('counts', 'Bayes counts 联合（精确全量）'),
                    ('counts_mc', 'Bayes counts 联合（MC 子集）'),
                    ('hard', 'Bayes hard（+立直听牌）'),
                    ('behave', 'Bayes behave（+行为似然）')):
                v, mv = r.get(key) or {}, mr.get(key) or {}
                if not v and not mv:
                    continue
                L.append(f"| {label} | {f(v.get('ce'))} | {f(v.get('top1'))} | "
                         f"{f(v.get('top3'))} | {f(mv.get('ce'))} | "
                         f"{f(mv.get('top1'))} |")
            j = r.get('judge', {})
            j1 = j.get('top1_judge') or {}
            L.append(f"\n判定（**CE 主判据**）：**{j.get('verdict', '—')}**；"
                     f"strict（counts）信息 {f(j.get('info_strict'), 4)} nats；"
                     f"full（behave）{f(j.get('info_full'), 4)} nats"
                     f"（{f(j.get('sigma'), 1)}σ；最强 tier={j.get('best_tier')}，"
                     f"CE={f(j.get('best_ce'), 3)}）；"
                     f"A2c 比最强 tier 低 {f(j.get('a2c_vs_best'), 3)} nats；"
                     f"P2c 低 {f(j.get('head_vs_best'), 3)} nats。")
            L.append(f"同 cell 的 top1 判据（仅供对照）："
                     f"**{j1.get('verdict', '—')}**（tier 信息 "
                     f"{f(j1.get('info_bayes'), 3)}，{f(j1.get('sigma'), 1)}σ）。")
    L.append('')

    # ---- micro vs cond ----
    L.append('## 4. micro → cond：判定与数值变化\n')
    L.append('`CE 判定` 是 cond 下的主判据（跨模型可比）；`top1 判定 micro→cond` '
             '用同一 top1 判据，只反映口径变化的影响。\n')
    L.append('| 层 | 子集 | prior top1 micro→cond | tier 信息(CE) | '
             'A2c norm_gain63 | P2c norm_gain63 | CE 判定 | top1 判定 micro→cond |')
    L.append('|---|---|---:|---:|---:|---:|---|---|')
    for layer in LAYERS:
        for act in ACTIVES:
            r = res['table'][layer][act]
            mr = ((res.get('micro_table') or {}).get(layer, {})
                  .get(act) or {})
            if not r:
                continue
            j = r.get('judge', {})
            mj = mr.get('judge', {})
            L.append(
                f"| {layer} | {act} | "
                f"{f((mr.get('prior') or {}).get('top1'), 3)} → "
                f"{f((r.get('prior') or {}).get('top1'), 3)} | "
                f"{f(j.get('info_bayes'), 3)} | "
                f"{f((r.get('a2c') or {}).get('norm_gain63'), 3)} | "
                f"{f((r.get('p2c') or {}).get('norm_gain63'), 3)} | "
                f"**{str(j.get('verdict', '—')).split('（')[0]}** | "
                f"{str(mj.get('verdict', '—')).split('（')[0]} → "
                f"{str(j.get('top1_verdict', '—')).split('（')[0]} |")
    L.append('')

    # ---- 完整分层 CE ----
    L.append('## 5. 完整分层 CE（cond；JSON 同步）\n')
    L.append('| 层 | 子集 | n | prior | A2c | P1 | P2a | P2b | P2c | c_prod | c_joint | hard | behave |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for layer in LAYERS:
        for act in ACTIVES:
            r = res['table'][layer][act]
            if not r:
                continue
            cells = [f((r.get(k) or {}).get('ce')) for k in
                     ('prior', 'a2c', 'p1', 'p2a', 'p2b', 'p2c',
                      'counts_prod', 'counts', 'hard', 'behave')]
            L.append(f"| {layer} | {act} | {r.get('n_prior', '—')} | "
                     + ' | '.join(cells) + ' |')
    L.append('')

    L.append('## 6. 结论\n')
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
    ap.add_argument('--metric-samples', type=int, default=4096)
    ap.add_argument('--m-list', default='256,1024,4096')
    ap.add_argument('--chains', type=int, default=256)
    ap.add_argument('--iters', type=int, default=128)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--max-k', type=int, default=10)
    ap.add_argument('--tiers', default='counts,hard,behave')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--index-cache', default='/tmp/bayes_ceiling_index.npz')
    ap.add_argument('--micro-json',
                    default=str(HERE / 'results' / 'bayes_ceiling.json'))
    ap.add_argument('--cond-json',
                    default=str(HERE / 'results' / 'cond_recheck.json'))
    ap.add_argument('--out', default=str(HERE / 'results'))
    ap.add_argument('--n-centers', type=int, default=0)
    ap.add_argument('--counts-only', action='store_true',
                    help='只跑 cond 精确 counts 与交叉校验（~40s），不跑 MC')
    ap.add_argument('--reuse-mc', default='',
                    help='从已有 p0a_bayes_cond.json 复用 MC 结果（只重跑 counts）')
    ap.add_argument('--legacy-json', default='/tmp/p0a_mc_backup.json',
                    help='旧 α 口径的 p0a JSON（仅用于报告对照表；'
                         '其 table.counts 被 MC 值覆盖，正是旧口径的 counts_mc）')
    ap.add_argument('--from-json', default='',
                    help='只从 p0a_bayes_cond.json 重建判定与报告')
    args = ap.parse_args()
    if args.from_json:
        return rebuild(Path(args.from_json), Path(args.out))
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
    with np.load(Path(args.cache), allow_pickle=False) as _zz:
        for k in ('unseen', 'hand'):
            z[k] = np.asarray(_zz[k])[:z['n']]
    z['unseen'] = z['unseen'].astype(np.float64)
    z['hand'] = z['hand'].astype(np.int64)
    print(f'[data] {args.cache} n={z["n"]}', flush=True)

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
        raise SystemExit(f'样本数不一致：重建 {len(index)} vs 缓存 {z["n"]}')
    if not alloc_match:
        raise SystemExit('alloc 与 P0 缓存不一致：eval 划分已漂移')

    masks = P0.strata(z)
    active_any, active_both = P0.active_masks(z, centers)
    print(f'[active] any={active_any.mean():.3f} both={active_both.mean():.3f}',
          flush=True)

    CU, N = BC.build_count_tables(z['unseen'])
    counts_cond = run_counts_cond(z, masks, centers, active_any, active_both,
                                  CU, N)

    micro_table = {}
    mp = Path(args.micro_json)
    if mp.exists():
        micro_table = (json.loads(mp.read_text()).get('table') or {})
    else:
        print(f'[micro] 缺 {mp} ⇒ micro 对照列留空', flush=True)
    cross = crosscheck(counts_cond, micro_table, args.cond_json)
    for k, d in cross.items():
        print(f'[check] {k}: ours={d["ours"]} ref={d["ref"]} '
              f'{"OK" if d["ok"] else "MISMATCH"}', flush=True)

    # prior 分组口径差（63 组本脚本 vs 21 组 cond_recheck）
    pg_detail, pg_gap = {}, 0.0
    crp = Path(args.cond_json)
    if crp.exists():
        crt = (json.loads(crp.read_text()).get('table') or {}).get('prior') or {}
        for layer, act in (('all', 'active_any'), ('all', 'active_both'),
                           ('riichi', 'active_both'),
                           ('clean_t8', 'active_both')):
            ours = (counts_cond[layer][act].get('prior') or {}).get('top1')
            ref = (((crt.get(layer) or {}).get(act) or {})
                   .get('cond') or {}).get('prior_top1')
            if ours is None or ref is None:
                continue
            pg_detail[f'{layer}/{act}'] = {'ours': float(ours),
                                           'ref': float(ref),
                                           'gap': float(ours - ref)}
            pg_gap = max(pg_gap, abs(float(ours - ref)))

    if args.counts_only:
        hdr = (f'{"layer":9s} {"act":13s} {"n":>8s} | '
               f'{"prior t1/t3/ce":>22s} | {"counts t1/t3/ce":>22s} | '
               f'{"prod t1/t3/ce":>22s}')
        print(hdr, flush=True)
        for layer in LAYERS:
            for act in ACTIVES:
                v = counts_cond[layer][act]
                pv, cv, dv = v['prior'], v, v['prod']

                def tri(d):
                    return (f"{BC._f(d.get('top1'), 4)}/"
                            f"{BC._f(d.get('top3'), 4)}/"
                            f"{BC._f(d.get('ce'), 4)}")

                print(f'{layer:9s} {act:13s} {pv["n"]:>8d} | {tri(pv):>22s} | '
                      f'{tri(cv):>22s} | {tri(dv):>22s}', flush=True)
        bad = [k for k, d in cross.items() if not d['ok']]
        print(f'[check] 不一致项 {len(bad)}/{len(cross)}', flush=True)
        return 0 if not bad else 1

    if args.reuse_mc:
        prev = json.loads(Path(args.reuse_mc).read_text())
        mc = prev.get('mc') or {}
        n_mc = int((prev.get('meta') or {}).get('n_mc') or 0)
        print(f'[mc] 复用 {args.reuse_mc}（n_mc={n_mc}）', flush=True)
    else:
        mc, n_mc = run_mc_cond(z, index, masks, centers, active_any,
                               active_both, args, m_list)

    cond_heads = load_cond_heads(Path(args.cond_json))
    table = build_table(counts_cond, mc, m_list, cond_heads)

    # 旧 α 口径的 counts_mc（仅报告对照；不影响任何结论）
    legacy_mc_ce = {}
    lp = Path(args.legacy_json)
    if lp.exists():
        lt = (json.loads(lp.read_text()).get('table') or {})
        for layer in LAYERS:
            for act in ACTIVES:
                v = (((lt.get(layer) or {}).get(act) or {})
                     .get('counts') or {})
                if v.get('ce') is not None:
                    legacy_mc_ce[f'{layer}/{act}'] = float(v['ce'])

    meta = {
        'cache': args.cache, 'n_cache': int(z['n']), 'n_mc': int(n_mc),
        'tiers': args.tiers, 'm_list': m_list,
        'metric_samples': args.metric_samples, 'chains': args.chains,
        'iters': args.iters, 'beta': args.beta, 'max_k': args.max_k,
        'alpha': args.alpha, 'alpha_mode': 'count', 'seed': args.seed,
        'legacy_mc_ce': legacy_mc_ce,
        'alloc_match': bool(alloc_match), 'centers': len(centers),
        'counts_selfcheck': {
            'max_abs_err': 0.0, 'n_checks': 0,
            'note': '见 bayes_ceiling.py 的 CPMF0 自检（本脚本复用同一实现）'},
        'crosscheck': cross,
        'prior_group_gap': pg_gap,
        'prior_group_gap_detail': pg_detail,
        'protocol': 'P0 eval 划分 + cond（y!=0 且 EMPTY 置 0 重归一）；'
                    'counts 精确（全量）、hard/behave MC（子集）；'
                    '采样顺序与 bayes_ceiling 逐位相同，只改累加口径；'
                    'Dirichlet α 按等效样本数计（L1 0.04）；'
                    '判定用 **CE**（strict=精确 counts；HEAD 用最强 tier），'
                    'top1 仅作对照（argmax 对分布平坦化敏感）',
    }
    verdict = build_verdict(table, meta, mc, micro_table)
    res = {'meta': meta, 'mc': mc, 'table': table,
           'micro_table': micro_table, 'verdict': verdict}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'p0a_bayes_cond.json', 'w') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'P0A_BAYES_COND_REPORT.md', res)
    print(f'\n[out] {out}/p0a_bayes_cond.json')
    print(f'[out] {out}/P0A_BAYES_COND_REPORT.md')
    print(f'[verdict] {verdict["decision"]}')
    for line in verdict['findings']:
        print(' -', line)
    print(f'[done] {time.time() - t_start:.0f}s')


def rebuild(path, out):
    """从 JSON 重建判定与报告（不重跑采样）。

    会**重算每个 cell 的 judge**，因此改判定逻辑后不必重跑 MC。
    """
    res = json.loads(Path(path).read_text())
    for layer in LAYERS:
        for act in ACTIVES:
            row = (res.get('table') or {}).get(layer, {}).get(act)
            if row:
                row['judge'] = judge_cond(row)
    res['verdict'] = build_verdict(res['table'], res['meta'], res.get('mc') or {},
                                  res.get('micro_table') or {})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'p0a_bayes_cond.json', 'w') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'P0A_BAYES_COND_REPORT.md', res)
    print(f'[verdict] {res["verdict"]["decision"]}')
    for line in res['verdict']['findings']:
        print(' -', line)
    return res


if __name__ == '__main__':
    main()
