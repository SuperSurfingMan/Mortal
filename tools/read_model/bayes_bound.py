#!/usr/bin/env python3
"""Bayes 上界估计：从可观测信息里到底还能读出多少？（MCMC / 重要性采样）

## 问题

A2b 的分层诊断说「立直家 hit@1 0.112」「干净困难局 AUC 0.683」都远未饱和。但
「未饱和」有两种解释：**(a) 模型没学到；(b) 可观测信息本身只有这么多**。本脚本
估计 **Bayes 可达性能**，把两者分开。

## 三个信息集（tier）

| tier | 条件化的信息 | 求解方式 |
|---|---|---|
| `counts` | 观察者手牌 + 全部可见牌的张数 | ✅ 精确：未知牌在「三家暗牌（张数由副露定）+ 牌山」上均匀分配（洗牌⇒超几何） |
| `hard`   | `counts` + 立直⇒听牌 | ✅ 立直家从 tenpai 手牌流形抽样（1/#待牌 纠偏） |
| `behave` | `hard` + 舍牌行为似然 | MH（非立直）/ 重要性采样（立直） |

`counts` 是**严格可计算的后验**（无行为假设），其指标 = 「只会数牌的理论最优读牌
器」；`hard` 加上唯一真正硬的规则约束；`behave` 是对真实后验的构造性近似 ⇒
Bayes 上界的**构造性下界**（参考策略越接近人类越紧）。

## 行为似然（静态逆规划）

    L(H) = Π_k P(d_k | H + d_k),    P(舍 x | 手牌) ∝ exp(−β·value(x))

`value` = 对子/刻子 + 同色 ±1/±2 邻接（ukeire 的廉价代理）。语义：「玩家打掉这张
牌，说明它相对他手里的牌没用」——手牌越成形（tenpai 率越高），越能解释「打掉的
全是废牌」。

**两条被实测否掉的设计**（见 README/MEMORY 记录）：
1. 用「弃后向听」当主信号：14 张手牌的向听对几乎每张舍牌都不退化，无法区分
   「留住有用摸牌」与「打了无用摸牌」，且贵 34×。
2. 完整轨迹前向粒子滤波：手切要求「该牌当时在手里」，逐巡存活率 ~15%，
   连续 10 次手切后 ~1e-8，必然全灭；反向轨迹滤波虽不死，但「被留住的摸牌」
   隐变量随机抽样会打散 tenpai 手牌的组合（实测 tenpai 反而 −6 nats）。
   改用静态似然后，tenpai 手牌 vs 随机手牌的平均对数似然 **+8 nats（β=1）
   / +17 nats（β=3）**，方向正确。

## 采样器

- 非立直：`chains` 条链从先验出发做单张互换 MH；`value(H±e_x)` 用线性性增量更新
  （`value(H+e_t) = value(H) + Δ_t`），每次评估只需 K 个因素的 softmax。
- 立直：`hard` 用「完整和牌形 − 1 张」构造 tenpai 手牌并按 1/#待牌 纠偏；
  `behave` 在同一批样本上按似然加权（重要性采样）。

`beta` 建议用 **202608（非评估月）** 标定：使后验 P(听牌) 均值匹配经验听牌率
（`--calibrate`）。

## 自检

1. `--leak-test`：把非观察者配牌在它们之间随机重排，后验必须逐位不变。
2. `shanten_fast` 与 `structure.structure_labels` 交叉验证（后者与 Rust 零不一致）。
3. 张数守恒：136 = 观察者 13 + 宝牌 1 + 三家 39 + 牌山 83（构造保证）。

## 用法

    python tools/read_model/bayes_bound.py --samples 20000 --max-snapshots 2000 \
        --chains 256 --iters 120 --beta 1.0 --out-dir mortal/read_model_bayes
    python tools/read_model/bayes_bound.py --calibrate --root .../202608
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(HERE))          # 必须最后插入：避免 mortal/train.py 遮蔽本目录 train.py

import dataset as D                                     # noqa: E402
import shanten_fast as SF                               # noqa: E402
import train as T                                       # noqa: E402
import torch                                            # noqa: E402
from compare_v1_v2 import load_net                      # noqa: E402
from strat_metrics import boot_ci, random_hit, rank_auc  # noqa: E402

NUM_TILES = D.NUM_TILES
NUM_COUNTS = 5
NUM_WAIT_CLASSES = NUM_TILES + 1
NO_WAIT = NUM_TILES
TARGETS = 3

STRATA = ('all', 'riichi=1', 'riichi=0', 'called=1', 'called=0',
          'riichi=0&called=0', 'riichi=0&called=0&turn>=8', 'riichi=0&called=0&turn<=7',
          'riichi=1&turn>=8', 'riichi=1&called=0', 'riichi=0&called=1',
          'turn0-3', 'turn4-7', 'turn8-11', 'turn12-15', 'turn16+')


def strata_masks(riichi, called, turn):
    n = riichi.shape[0]
    T3 = np.broadcast_to(turn[:, None], (n, 3))
    return {
        'all': np.ones((n, 3), bool),
        'riichi=1': riichi,
        'riichi=0': ~riichi,
        'called=1': called,
        'called=0': ~called,
        'riichi=0&called=0': ~riichi & ~called,
        'riichi=0&called=0&turn>=8': ~riichi & ~called & (T3 >= 8),
        'riichi=0&called=0&turn<=7': ~riichi & ~called & (T3 <= 7),
        'riichi=1&turn>=8': riichi & (T3 >= 8),
        'riichi=1&called=0': riichi & ~called,
        'riichi=0&called=1': ~riichi & called,
        'turn0-3': T3 <= 3,
        'turn4-7': (T3 >= 4) & (T3 <= 7),
        'turn8-11': (T3 >= 8) & (T3 <= 11),
        'turn12-15': (T3 >= 12) & (T3 <= 15),
        'turn16+': T3 >= 16,
    }


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def build_eval_samples(files, n_samples: int, seed: int = 20261016):
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)
    collected, used = [], []
    for f in files:
        try:
            samples, _ = D.replay_file(f, augment=False)
        except Exception as exc:                       # pragma: no cover
            print(f'[eval] skip {f}: {exc}', flush=True)
            continue
        used.append(f)
        for s in samples:
            s['_file'] = f
        collected.extend(samples)
        if len(collected) >= n_samples:
            break
    if len(collected) > n_samples:
        idx = np.linspace(0, len(collected) - 1, n_samples).astype(int)
        collected = [collected[i] for i in idx]
    if not collected:
        raise RuntimeError('评估集为空')
    print(f'[eval] 固定评估集 {len(collected)} 样本（{len(used)} 个文件）', flush=True)
    return collected


def load_events(path: str) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def kyoku_spans(events) -> list[tuple[int, int]]:
    starts = [i for i, e in enumerate(events) if e.get('type') == 'start_kyoku']
    return [(s, starts[k + 1] if k + 1 < len(starts) else len(events))
            for k, s in enumerate(starts)]


# --------------------------------------------------------------------------- #
# 采样工具
# --------------------------------------------------------------------------- #
def _draw_one(rng, pool: np.ndarray) -> np.ndarray:
    tot = pool.sum(1, dtype=np.int64)
    cum = np.cumsum(pool, 1, dtype=np.int64)
    u = rng.integers(0, np.maximum(tot, 1))
    t = (cum <= u[:, None]).sum(1)
    t = np.clip(t, 0, NUM_TILES - 1)
    bad = pool[np.arange(pool.shape[0]), t] <= 0
    if bad.any():
        t = np.where(bad, pool.argmax(1), t)
    return t


def sample_hands(rng, pool, size: int, n: int) -> np.ndarray:
    """从 pool 无放回抽 size 张，n 份（联合均匀分配的单家边际）。"""
    wall = np.tile(np.asarray(pool, dtype=np.int16)[None, :], (n, 1))
    H = np.zeros((n, NUM_TILES), dtype=np.int16)
    rows = np.arange(n)
    for _ in range(int(size)):
        t = _draw_one(rng, wall)
        wall[rows, t] -= 1
        H[rows, t] += 1
    return H


def _set_matrix() -> np.ndarray:
    mats = []
    for s in range(3):
        for start in range(7):
            v = np.zeros(NUM_TILES, dtype=np.int64)
            v[s * 9 + start:s * 9 + start + 3] = 1
            mats.append(v)
    for t in range(NUM_TILES):
        v = np.zeros(NUM_TILES, dtype=np.int64)
        v[t] = 3
        mats.append(v)
    return np.stack(mats)                      # [55,34]


SETS = _set_matrix()


def sample_tenpai_hands(rng, pool, n_sets: int, n_samples: int):
    """`n_sets` 个面子 + 1 雀首的完整和牌形（14−3m 张）去掉一张 → tenpai 手牌。

    返回 (hands [N,34], weight 1/#待牌)。纠偏理由：每个 (手牌, 和牌张) 组合等概率
    被采到 ⇒ 手牌出现次数 ∝ 待牌数，需除以 #待牌 才是均匀 tenpai 手牌。
    """
    pool = np.asarray(pool, dtype=np.int64)
    out = np.zeros((n_samples, NUM_TILES), dtype=np.int16)
    k = tries = 0
    while k < n_samples and tries < n_samples * 500:
        tries += 1
        left = pool.copy()
        ok = True
        for _ in range(n_sets):
            fit = np.flatnonzero((SETS <= left[None, :]).all(1))
            if len(fit) == 0:
                ok = False
                break
            left -= SETS[fit[rng.integers(len(fit))]]
        if not ok:
            continue
        pair = np.flatnonzero(left >= 2)
        if len(pair) == 0:
            continue
        p = pair[rng.integers(len(pair))]
        left[p] -= 2
        full = pool - left
        present = np.flatnonzero(full > 0)
        hand = full.copy()
        hand[present[rng.integers(len(present))]] -= 1
        out[k] = hand
        k += 1
    out = out[:k]
    waits, _ = SF.waits_vec(out, int(out[0].sum()) // 3)
    return out, 1.0 / waits.sum(1).clip(min=1)


# --------------------------------------------------------------------------- #
# 手牌价值 / 静态行为似然
# --------------------------------------------------------------------------- #
def hand_value(h: np.ndarray) -> np.ndarray:
    """[N,34] → 每张牌的保留价值（对计数向量是**线性**的）。"""
    f = h.astype(np.float64)
    val = f - 1.0
    for s in range(3):
        b = s * 9
        sub = f[:, b:b + 9]
        for off, wgt in ((1, 1.0), (2, 0.5)):
            val[:, b:b + 9 - off] += wgt * sub[:, off:]
            val[:, b + off:b + 9] += wgt * sub[:, :-off]
    return val


DELTA = hand_value(np.eye(NUM_TILES))           # [34,34]：单张 e_t 的价值贡献


class StaticLik:
    """log L(H) = Σ_k log P(d_k | H + d_k)；线性性 ⇒ 只算一次 value(H)。"""

    def __init__(self, discards, beta: float):
        self.ds = [int(d) for d, _ in discards]
        self.beta = beta

    def __len__(self):
        return len(self.ds)

    def logprob_v(self, vH: np.ndarray, present: np.ndarray) -> np.ndarray:
        n = vH.shape[0]
        tot = np.zeros(n)
        for d in self.ds:
            v = vH + DELTA[d]
            lg = np.where(present | (np.arange(NUM_TILES) == d)[None, :],
                          -self.beta * v, -1e30)
            mx = lg.max(1)
            lse = mx + np.log(np.exp(lg - mx[:, None]).sum(1))
            tot += lg[:, d] - lse
        return tot

    def logprob(self, H: np.ndarray) -> np.ndarray:
        return self.logprob_v(hand_value(H), H > 0)


def extract_discards(events, lo: int, snap: int, seat: int, max_k: int):
    out = []
    for li in range(lo, snap + 1):
        e = events[li]
        if e.get('actor') == seat and e.get('type') == 'dahai':
            out.append((D.tile_id(e['pai']), bool(e.get('tsumogiri'))))
    return out[-max_k:]


def mh_posterior(rng, pool, size: int, lik: StaticLik, chains: int, iters: int,
                 collect_every: int = 8):
    """单张互换 MH；后半程每 collect_every 步收一次样本（等权）。"""
    pool = np.asarray(pool, dtype=np.int64)
    H = sample_hands(rng, pool, size, chains)
    rows = np.arange(chains)
    vH = hand_value(H)
    present = H > 0
    ll = lik.logprob_v(vH, present)
    kept = []
    for it in range(iters):
        out_w = H > 0
        u = rng.random(chains) * out_w.sum(1).clip(min=1)
        ix = (np.cumsum(out_w, 1) <= u[:, None]).sum(1).clip(0, NUM_TILES - 1)
        av = np.tile(pool[None, :], (chains, 1)) - H
        u2 = rng.random(chains) * av.sum(1).clip(min=1)
        iy = (np.cumsum(av, 1) <= u2[:, None]).sum(1).clip(0, NUM_TILES - 1)
        H2 = H.copy()
        H2[rows, ix] -= 1
        H2[rows, iy] += 1
        v2 = vH - DELTA[ix] + DELTA[iy]
        ll2 = lik.logprob_v(v2, H2 > 0)
        acc = np.log(rng.random(chains)) < (ll2 - ll)
        H = np.where(acc[:, None], H2, H)
        vH = np.where(acc[:, None], v2, vH)
        ll = np.where(acc, ll2, ll)
        if it >= iters // 2 and (it - iters // 2) % collect_every == 0:
            kept.append(H.copy())
    return np.concatenate(kept) if kept else H


# --------------------------------------------------------------------------- #
# 后验 → 预测概率
# --------------------------------------------------------------------------- #
def posterior_probs(hands, w, river, own_hand, unseen, meld_counts):
    """hands [N,3,34]；w [N] 或 [N,3]（逐目标独立权重）。"""
    if w.ndim == 1:
        w = np.tile(w[:, None], (1, TARGETS))
    expr = (np.rint(unseen) > 0) | own_hand
    alloc_p = np.zeros((TARGETS, NUM_TILES, NUM_COUNTS))
    wa_hit = np.zeros((TARGETS, NUM_WAIT_CLASSES))
    wa_ce = np.zeros((TARGETS, NUM_WAIT_CLASSES))
    sh_dist = np.zeros((TARGETS, 8))
    p_tenpai = np.zeros(TARGETS)
    for j in range(TARGETS):
        h = hands[:, j]
        wj = w[:, j]
        n13 = int(h.sum(1)[0])
        assert n13 == 13 - 3 * int(meld_counts[j]), (n13, meld_counts[j], j)
        ld = n13 // 3
        for t in range(NUM_TILES):
            alloc_p[j, t] = np.bincount(
                np.clip(h[:, t], 0, NUM_COUNTS - 1), weights=wj,
                minlength=NUM_COUNTS)[:NUM_COUNTS]
        sh = SF.calc_all_vec(h, ld)
        sh_dist[j] = np.bincount(np.clip(sh, 0, 7), weights=wj, minlength=8)[:8]
        p_tenpai[j] = float(wj[sh == 0].sum())
        b = np.zeros(NUM_TILES)
        s = np.zeros(NUM_TILES)
        tp = np.flatnonzero(sh == 0)
        if len(tp):
            cand = np.flatnonzero(SF.wait_candidates(h[tp]))
            waits, _ = SF.waits_vec(h[tp], ld, cand)
            eff = waits & (~river[j][cand]) & expr[cand]
            neff = eff.sum(1)
            hasw = neff > 0
            b[cand] = (wj[tp][:, None] * waits).sum(0)
            if hasw.any():
                ww = wj[tp][hasw] / neff[hasw]
                s[cand] = (ww[:, None] * eff[hasw]).sum(0)
            p_empty = float(wj[tp][~hasw].sum()) + float(wj[sh != 0].sum())
        else:
            p_empty = 1.0
        b_tot = b.sum() + p_empty
        wa_hit[j, :NUM_TILES] = b / max(b_tot, 1e-12)
        wa_hit[j, NO_WAIT] = p_empty / max(b_tot, 1e-12)
        wa_ce[j, :NUM_TILES] = s
        wa_ce[j, NO_WAIT] = p_empty
    return (alloc_p[None], wa_hit[None], wa_ce[None], sh_dist[None],
            p_tenpai[None])


# --------------------------------------------------------------------------- #
# 指标（与 strat_metrics 同口径）
# --------------------------------------------------------------------------- #
def metrics_per_target(alloc_p, wa_hit, wa_ce, sh_dist, p_tenpai, ref):
    alloc_t = ref['alloc'].astype(np.int64)
    wait_t = ref['wait']
    b_idx = (np.arange(len(alloc_t))[:, None, None],
             np.arange(TARGETS)[None, :, None],
             np.arange(NUM_TILES)[None, None, :])
    p_true = alloc_p[b_idx + (alloc_t,)]
    tile_p = wa_hit[..., :NUM_TILES]
    supp = wait_t[..., :NUM_TILES] > 0
    eff = supp.any(-1)
    topk = np.argsort(-tile_p, -1)
    own3 = np.broadcast_to(ref['own_hand'][:, None, :],
                           (len(alloc_t), TARGETS, NUM_TILES))
    p_cand = np.where(own3, tile_p, np.inf)
    safest = p_cand.argmin(-1)
    p_safe = np.take_along_axis(tile_p, safest[..., None], -1)[..., 0]
    return {
        'alloc_ce': -np.log(np.clip(p_true, 1e-12, None)).mean(-1),
        'alloc_top1': (alloc_p.argmax(-1) == alloc_t).mean(-1),
        'alloc_mae': np.abs((alloc_p * np.arange(NUM_COUNTS)).sum(-1) - alloc_t).mean(-1),
        'wait_ce': -(wait_t * np.log(np.clip(wa_ce, 1e-12, None))).sum(-1),
        'hit1': np.take_along_axis(supp, topk[..., :1], -1)[..., 0],
        'hit3': np.take_along_axis(supp, topk[..., :3], -1).any(-1),
        'hit5': np.take_along_axis(supp, topk[..., :5], -1).any(-1),
        'deal': np.take_along_axis(supp, safest[..., None], -1)[..., 0],
        'p_safe': p_safe, 'deal_mask': eff & (p_safe > 0),
        'eff': eff, 'wcnt': supp.sum(-1),
        'tenpai': ref['tenpai'] > 0.5, 'p_tenpai': p_tenpai,
        'sh_ok': (sh_dist.argmax(-1) == ref['shanten'].astype(np.int64)),
        'alloc_t': alloc_t,
    }


@torch.no_grad()
def model_probs(state_path, batch, device, bs=2048):
    net, cfg = load_net(state_path, device)
    n = batch['obs'].shape[0]
    A, W, SH = [], [], []
    for i in range(0, n, bs):
        ch = {k: v[i:i + bs].to(device) for k, v in batch.items()}
        wb = T.wait_bias_of(ch).to(device)
        if hasattr(net, 'timeline'):
            o = net(ch['obs'], ch['unseen'], wb, ch['tl_tokens'], ch['tl_mask'],
                    **net.tl_extra(ch))
        else:
            o = net(ch['obs'], ch['unseen'], wb)
        A.append(torch.softmax(o['alloc_logits'], -1).cpu().numpy())
        W.append(o['wait_probs'].cpu().numpy())
        if 'struct_shanten_logits' in o:
            SH.append(torch.softmax(o['struct_shanten_logits'], -1).cpu().numpy())
    alloc_p = np.concatenate(A).astype(np.float64)
    wait_p = np.concatenate(W).astype(np.float64)
    if SH:
        sh_dist = np.concatenate(SH).astype(np.float64)
    else:
        sh_dist = np.zeros(alloc_p.shape[:2] + (8,))
        sh_dist[..., 0] = 1.0 - wait_p[..., NO_WAIT]
        sh_dist[..., 1] = wait_p[..., NO_WAIT]
    # RM1（README §11.1）：forward 的 wait_probs 已是因子化读法，
    # wait_p[..., NO_WAIT] = sigmoid(l34) = P(未听)——BCE 直接标定量，
    # 与三档参考模型（本就按 P(听)·P(x|听) 构造）同口径。旧 softmax35
    # 读法曾低估 A2b 的 tenpai AUC（0.9461 vs 0.9685），§9 的
    # 「参考读牌器已达/超过 A2b」类结论必须以本口径重跑后再下。
    return alloc_p, wait_p, sh_dist, 1.0 - wait_p[..., NO_WAIT], cfg


# --------------------------------------------------------------------------- #
def _entropy_binary(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def aggregate(masks, mdict, ok=None):
    n = masks['all'].shape[0]
    ok = np.ones(n, bool) if ok is None else ok
    out = {}
    for name in STRATA:
        m = masks[name] & ok[:, None]
        if m.sum() == 0:
            continue
        ma = m & mdict['eff']
        v = {'n_targets': int(m.sum()),
             'tenpai_rate': float(mdict['tenpai'][m].mean()),
             'n_active': int(ma.sum())}
        for key in ('alloc_ce', 'alloc_top1', 'alloc_mae'):
            v[key] = float(mdict[key][m].mean())
        v['wait_ce'] = float(mdict['wait_ce'][ma].mean()) if ma.any() else float('nan')
        for key in ('hit1', 'hit3', 'hit5'):
            v[key] = float(mdict[key][ma].mean()) if ma.any() else float('nan')
        v['tenpai_auc'] = rank_auc(mdict['tenpai'][m].ravel(),
                                   mdict['p_tenpai'][m].ravel())
        v['shanten_acc'] = float(mdict['sh_ok'][m].mean())
        dm = m & mdict['deal_mask']
        v['deal_pure'] = float(mdict['deal'][dm].mean()) if dm.any() else float('nan')
        v['n_deal'] = int(dm.sum())
        v['deal_all'] = float(mdict['deal'][m & mdict['eff']].mean()) \
            if (m & mdict['eff']).any() else float('nan')
        if ma.any():
            v['random_hit1'] = random_hit(mdict['wcnt'][ma], 1)
            v['random_hit5'] = random_hit(mdict['wcnt'][ma], 5)
            v['hit1_ci'] = boot_ci(mdict['hit1'][ma].astype(float))
            v['tenpai_auc_ci'] = boot_ci(
                np.stack([mdict['tenpai'][m].ravel(),
                          mdict['p_tenpai'][m].ravel()], -1), stat='auc')
        cls = np.bincount(mdict['alloc_t'][m].ravel(), minlength=NUM_COUNTS) \
            / mdict['alloc_t'][m].size
        v['alloc_majority'] = float(cls.max())
        ent = float(-(cls * np.log(np.clip(cls, 1e-12, None))).sum())
        v['alloc_info_gain'] = ent - v['alloc_ce']
        p = mdict['p_tenpai'][m]
        h_emp = float(_entropy_binary(np.array([v['tenpai_rate']]))[0])
        v['mean_p_tenpai'] = float(p.mean())
        v['tenpai_h_cond'] = float(_entropy_binary(p).mean())
        v['tenpai_mi_nats'] = h_emp - v['tenpai_h_cond']
        out[name] = v
    return out


# --------------------------------------------------------------------------- #
# 逐快照的 tier 后验
# --------------------------------------------------------------------------- #
def build_tier(tier, rng, pool, sizes, ri, lik, chains, iters, n_metric,
               collect_every=8):
    H = np.zeros((n_metric, TARGETS, NUM_TILES), dtype=np.int16)
    W = np.zeros((n_metric, TARGETS))
    for j in range(TARGETS):
        if tier != 'counts' and ri[j]:
            h, wt = sample_tenpai_hands(rng, pool, (sizes[j] - 1) // 3, n_metric)
            if tier == 'behave' and len(lik[j]):
                lp = lik[j].logprob(h)
                wt = wt * np.exp(lp - lp.max())
            W[:, j] = wt / wt.sum()
        elif tier == 'behave':
            h = mh_posterior(rng, pool, sizes[j], lik[j], chains, iters,
                             collect_every)
            W[:, j] = 1.0 / len(h)
        else:
            h = sample_hands(rng, pool, sizes[j], n_metric)
            W[:, j] = 1.0 / len(h)
        H[:len(h), j] = h
        if len(h) < n_metric:                    # 理论上不会发生
            H[len(h):, j] = h[0]
            W[len(h):, j] = 0.0
    return H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',
                    default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--max-snapshots', type=int, default=0)
    ap.add_argument('--state', default='mortal/read_model_v2s_a2b/best.pth')
    ap.add_argument('--chains', type=int, default=256)
    ap.add_argument('--iters', type=int, default=120)
    ap.add_argument('--metric-samples', type=int, default=1024,
                    help='每个目标用于算指标的后验样本数（≈ chains × 收集次数）')
    ap.add_argument('--max-k', type=int, default=10)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--tiers', default='counts,hard,behave')
    ap.add_argument('--out-dir', default='mortal/read_model_bayes')
    ap.add_argument('--seed', type=int, default=20261018)
    ap.add_argument('--leak-test', action='store_true')
    ap.add_argument('--no-model', action='store_true')
    ap.add_argument('--calibrate', action='store_true',
                    help='只输出各 β 下的后验均值 P(听牌)，用于标定')
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(',') if t.strip()]
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    files = sorted(__import__('glob').glob(f'{args.root}/**/*.json.gz',
                                           recursive=True))
    samples = build_eval_samples(files, args.samples)
    batch = T.collate(samples)
    n = len(samples)
    if args.max_snapshots and n > args.max_snapshots:
        keep = np.sort(rng.choice(n, args.max_snapshots, replace=False))
        samples = [samples[i] for i in keep]
        batch = {k: (v[torch.from_numpy(keep)] if torch.is_tensor(v) else v[keep])
                 for k, v in batch.items()}
        n = len(samples)
        print(f'[eval] 抽稀到 {n} 个快照', flush=True)

    ref = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in batch.items()}
    riichi = ref['riichi'] > 0.5
    called = ref['called'] > 0.5
    turn = ref['turn'].astype(np.int64)
    masks = strata_masks(riichi, called, turn)
    want = max(1, args.metric_samples // args.chains)
    collect_every = max(1, (args.iters - args.iters // 2) // want)
    n_collect = len(range(args.iters // 2, args.iters, collect_every))
    n_metric = args.chains * max(n_collect, 1)
    print(f'[config] MCMC {args.chains} 链 × {n_collect} 采样 = {n_metric} 样本/目标, '
          f'β={args.beta}, K≤{args.max_k}', flush=True)

    # ---- 标定模式 ----
    if args.calibrate:
        return _calibrate(args, samples, ref, masks)

    results = {}
    if not args.no_model:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t0 = time.time()
        alloc_p, wait_p, sh_dist, p_ten, cfg = model_probs(args.state, batch, device)
        mdict = metrics_per_target(alloc_p, wait_p, wait_p, sh_dist, p_ten, ref)
        results['model'] = {'agg': aggregate(masks, mdict)}
        print(f'[model] {args.state} ({cfg.get("model")}) '
              f'{time.time()-t0:.1f}s', flush=True)

    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[(s['_file'], int(s['_observer']), int(s['_kyoku']))].append(
            (i, int(s['_event'])))
    print(f'[bayes] {len(groups)} 组 / {n} 快照 | tiers={tiers}', flush=True)
    if args.leak_test:
        run_leak_test(samples, groups, args)

    tier_mdict = {}
    for tier in tiers:
        tier_mdict[tier] = {k: np.full((n, TARGETS), np.nan)
                            for k in ('alloc_ce', 'alloc_top1', 'alloc_mae',
                                      'wait_ce', 'hit1', 'hit3', 'hit5', 'deal',
                                      'p_safe', 'wcnt', 'p_tenpai')}
        tier_mdict[tier]['deal_mask'] = np.zeros((n, TARGETS), bool)
        tier_mdict[tier]['eff'] = np.zeros((n, TARGETS), bool)
        tier_mdict[tier]['tenpai'] = ref['tenpai'] > 0.5
        tier_mdict[tier]['sh_ok'] = np.zeros((n, TARGETS), bool)
        tier_mdict[tier]['alloc_t'] = ref['alloc']
        tier_mdict[tier]['ok'] = np.zeros(n, bool)

    events_cache, spans_cache = {}, {}
    n_done = 0
    t0 = time.time()
    for (fpath, observer, kyoku), items in groups.items():
        if fpath not in events_cache:
            events_cache[fpath] = load_events(fpath)
            spans_cache[fpath] = kyoku_spans(events_cache[fpath])
        events, spans = events_cache[fpath], spans_cache[fpath]
        if kyoku <= 0 or kyoku > len(spans):
            continue
        lo, _hi = spans[kyoku - 1]
        seats = [(observer + rel) % 4 for rel in (1, 2, 3)]

        for i, ev in items:
            s = samples[i]
            one = {k: ref[k][i:i + 1] for k in ref}
            pool = np.rint(s['unseen']).astype(np.int64)
            sizes = [13 - 3 * int(m) for m in s['meld_counts']]
            ri = [bool(x) for x in (s['riichi'] > 0.5)]
            lik = [StaticLik(extract_discards(events, lo, ev, seats[j], args.max_k),
                             args.beta) for j in range(TARGETS)]
            for tier in tiers:
                hands, w = build_tier(tier, rng, pool, sizes, ri, lik,
                                      args.chains, args.iters, n_metric,
                                      collect_every)
                md = metrics_per_target(*posterior_probs(
                    hands, w, s['river'], s['own_hand'].astype(bool),
                    s['unseen'], s['meld_counts']), ref=one)
                _store(tier_mdict[tier], i, md)

        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t0
            print(f'[bayes] {n_done}/{len(groups)} 组 | {el:.0f}s | '
                  f'ETA {el / n_done * (len(groups) - n_done):.0f}s', flush=True)

    for tier in tiers:
        results[tier] = {'agg': aggregate(masks, tier_mdict[tier],
                                          tier_mdict[tier]['ok'])}

    tier_names = ([] if args.no_model else ['model']) + tiers
    lines = []
    for name in STRATA:
        if name not in results[tier_names[0]]['agg']:
            continue
        v0 = results[tier_names[0]]['agg'][name]
        lines.append(f"\n== {name}  (n={v0['n_targets']}, "
                     f"听牌率 {100*v0['tenpai_rate']:.1f}%, 有效 n={v0['n_active']})")
        lines.append(f"  {'tier':<8s} {'alloc_ce':>9s} {'top1':>7s} {'MAE':>7s} "
                     f"{'wait_ce':>8s} {'hit1':>6s} {'hit5':>6s} {'AUC':>6s} "
                     f"{'deal%':>6s} {'meanP(听)':>9s} {'MI':>6s} {'向听':>6s}")
        for tier in tier_names:
            v = results[tier]['agg'][name]
            lines.append(
                f"  {tier:<8s} {v['alloc_ce']:>9.4f} {v['alloc_top1']:>7.4f} "
                f"{v['alloc_mae']:>7.4f} {v['wait_ce']:>8.4f} {v['hit1']:>6.4f} "
                f"{v['hit5']:>6.4f} {v['tenpai_auc']:>6.4f} "
                f"{100*v['deal_pure']:>6.2f} {v['mean_p_tenpai']:>9.4f} "
                f"{v['tenpai_mi_nats']:>6.3f} {v['shanten_acc']:>6.4f}")
        for tier in tier_names:
            v = results[tier]['agg'][name]
            if 'hit1_ci' in v:
                lines.append(f"    {tier:<8s} hit1 CI [{v['hit1_ci'][0]:.4f},"
                             f"{v['hit1_ci'][1]:.4f}]  AUC CI "
                             f"[{v['tenpai_auc_ci'][0]:.3f},"
                             f"{v['tenpai_auc_ci'][1]:.3f}]")
    txt = '\n'.join(lines)
    print(txt, flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'report.json', 'w') as f:
        json.dump({'args': vars(args), 'n': n, 'n_groups': len(groups),
                   'results': results}, f, ensure_ascii=False, indent=1,
                  default=float)
    (out_dir / 'tables.txt').write_text(txt)
    print(f'\n[out] {out_dir}/report.json  ({time.time()-t_start:.0f}s)', flush=True)


def _calibrate(args, samples, ref, masks):
    """扫 β：报告后验均值 P(听牌) vs 经验听牌率（应在非评估月 202608 上做）。"""
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[(s['_file'], int(s['_observer']), int(s['_kyoku']))].append(
            (i, int(s['_event'])))
    rng = np.random.default_rng(args.seed)
    keys = list(groups.items())[:60]
    # 经验听牌率（同一子样本口径：非立直目标）
    emp = {'all': [0.0, 0], 'clean8': [0.0, 0]}
    for (fpath, observer, kyoku), items in keys:
        for i, ev in items[:1]:
            s = samples[i]
            for j in range(TARGETS):
                if s['riichi'][j] > 0.5:
                    continue
                emp['all'][0] += float(s['tenpai'][j])
                emp['all'][1] += 1
                if not s['called'][j] and s['turn'] >= 8:
                    emp['clean8'][0] += float(s['tenpai'][j])
                    emp['clean8'][1] += 1
    print('经验: all=%.4f (n=%d)  clean&turn>=8=%.4f (n=%d)' % (
        emp['all'][0] / max(emp['all'][1], 1), emp['all'][1],
        emp['clean8'][0] / max(emp['clean8'][1], 1), emp['clean8'][1]), flush=True)

    for beta in (0.25, 0.5, 0.75, 1.0, 1.5, 2.5):
        acc = {'all': [0.0, 0], 'clean8': [0.0, 0]}
        for (fpath, observer, kyoku), items in keys:
            events = load_events(fpath)
            spans = kyoku_spans(events)
            lo, _ = spans[kyoku - 1]
            seats = [(observer + rel) % 4 for rel in (1, 2, 3)]
            for i, ev in items[:1]:
                s = samples[i]
                pool = np.rint(s['unseen']).astype(np.int64)
                sizes = [13 - 3 * int(m) for m in s['meld_counts']]
                for j in range(TARGETS):
                    if s['riichi'][j] > 0.5:
                        continue
                    lik = StaticLik(extract_discards(events, lo, ev, seats[j],
                                                     args.max_k), beta)
                    h = mh_posterior(rng, pool, sizes[j], lik, 128, 100)
                    sh = SF.calc_all_vec(h, int(h.sum(1)[0]) // 3)
                    v = float((sh == 0).mean())
                    acc['all'][0] += v
                    acc['all'][1] += 1
                    if not s['called'][j] and s['turn'] >= 8:
                        acc['clean8'][0] += v
                        acc['clean8'][1] += 1
        print('β=%.2f  all=%.4f (n=%d)  clean&turn>=8=%.4f (n=%d)' % (
            beta, acc['all'][0] / max(acc['all'][1], 1), acc['all'][1],
            acc['clean8'][0] / max(acc['clean8'][1], 1), acc['clean8'][1]),
            flush=True)
    emp_all = float((ref['tenpai'] > 0.5).mean())
    m = masks['riichi=0&called=0&turn>=8']
    print(f'[全评估集] 经验听牌率 all={emp_all:.4f}  clean&turn>=8='
          f'{float((ref["tenpai"][m] > 0.5).mean()):.4f}', flush=True)
    return None


def _store(bucket, i, md):
    for k in ('alloc_ce', 'alloc_top1', 'alloc_mae', 'wait_ce', 'hit1', 'hit3',
              'hit5', 'deal', 'p_safe', 'wcnt', 'p_tenpai'):
        bucket[k][i] = md[k][0]
    bucket['deal_mask'][i] = md['deal_mask'][0]
    bucket['eff'][i] = md['eff'][0]
    bucket['sh_ok'][i] = md['sh_ok'][0]
    bucket['ok'][i] = True


# --------------------------------------------------------------------------- #
# 泄漏自检
# --------------------------------------------------------------------------- #
def _scramble_hidden(events, lo, hi, observer, seed):
    rng = random.Random(seed)
    sl = [dict(e) for e in events[lo:hi]]
    th = [list(h) for h in sl[0]['tehais']]
    pool = [t for i in range(4) if i != observer for t in th[i]]
    rng.shuffle(pool)
    k = 0
    for i in range(4):
        if i == observer:
            continue
        th[i] = pool[k:k + 13]
        k += 13
    sl[0]['tehais'] = th
    return sl


def run_leak_test(samples, groups, args):
    """重排非观察者配牌后，全部可观测信息（含舍牌/立直）不变 → 后验必须逐位不变。"""
    print('[leak] 重排非观察者配牌，后验必须逐位不变', flush=True)
    bad = checked = 0
    for (fpath, observer, kyoku), items in list(groups.items())[:12]:
        events = load_events(fpath)
        spans = kyoku_spans(events)
        lo, _ = spans[kyoku - 1]
        seats = [(observer + rel) % 4 for rel in (1, 2, 3)]
        i, ev = items[0]
        s = samples[i]
        pool = np.rint(s['unseen']).astype(np.int64)
        sizes = [13 - 3 * int(m) for m in s['meld_counts']]
        base = sample_hands(np.random.default_rng(11), pool, 13, 64)
        a = np.concatenate([StaticLik(extract_discards(events, lo, ev, seats[j],
                                                       args.max_k),
                                      args.beta).logprob(base) for j in range(3)])
        sl = _scramble_hidden(events, lo, ev + 1, observer, 99)
        b = np.concatenate([StaticLik(extract_discards(sl, 0, ev - lo, seats[j],
                                                       args.max_k),
                                      args.beta).logprob(base) for j in range(3)])
        checked += 1
        if not np.allclose(a, b):
            bad += 1
            print(f'[leak] 泄漏 {Path(fpath).name} obs={observer} kyoku={kyoku}',
                  flush=True)
    print(f'[leak] {checked} 个快照，泄漏 {bad} → '
          f'{"PASS" if bad == 0 else "FAIL"}', flush=True)
    if bad:
        raise SystemExit('信息泄漏自检失败')


if __name__ == '__main__':
    main()
