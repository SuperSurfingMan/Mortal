"""Phase 2 / P0：局部形状标签与指标（只读缓存，不训练）。

口径
----
- 中心牌：同花色 m=2..8（0-based 1..7），共 3 花色 × 7 = 21 个中心。
- 三张窗口：(m−1, m, m+1)，每张计数截断到 0/1/2 ⇒ 27 类局部形状。
  P0 用「计数形状」而不是「两面/嵌张」分解：后者依赖手牌分解歧义，
  先用无歧义、可复现的计数形状做基线；分解软标签留到 P1。
- 成对共持：同花色相邻 d=1、隔张 d=2；标签 = 两张都持有（计数>0）。
- 五张窗口：m−2..m+2，中心 m=3..7（0-based 2..6），共 15 个窗口；
  用于 MAE，不参与 27 类形状。

所有概率都来自缓存里的 alloc_probs[n,3,34,5]：
  独立边际乘积 P(a,b,c) = P_a(c_a)·P_b(c_b)·P_c(c_c)
这就是当前模型（A2c）在没有联合头时唯一能形成的「局部联合读数」。
"""
from __future__ import annotations

import numpy as np

NUM_TILES = 34
SUIT_BASES = (0, 9, 18)
NUM_PATTERNS = 27


def centers_3() -> np.ndarray:
    """21 个三张窗口中心：同花色 m=2..8（0-based 1..7）。"""
    out = []
    for base in SUIT_BASES:
        for m in range(1, 8):
            out.append((base + m - 1, base + m, base + m + 1))
    return np.asarray(out, dtype=np.int64)


def centers_5() -> np.ndarray:
    """15 个五张窗口中心：同花色 m=3..7（0-based 2..6）。"""
    out = []
    for base in SUIT_BASES:
        for m in range(2, 7):
            out.append(tuple(range(base + m - 2, base + m + 3)))
    return np.asarray(out, dtype=np.int64)


def pair_indices() -> np.ndarray:
    """42 个成对：3 花色 × 7 起点 × {d=1,d=2}，返回 (a, b, d)。"""
    out = []
    for base in SUIT_BASES:
        for i in range(1, 8):
            for d in (1, 2):
                out.append((base + i, base + i + d, d))
    return np.asarray(out, dtype=np.int64)


def tile_class_probs(alloc_probs: np.ndarray, tile: int) -> np.ndarray:
    """[n,3,34,5] → [n,3,3]：P(count=0), P(count=1), P(count>=2)。"""
    p = np.asarray(alloc_probs[:, :, int(tile), :5], dtype=np.float32)
    p = np.clip(p, 1e-9, 1.0)
    return np.stack([p[..., 0], p[..., 1], p[..., 2:].sum(-1)], axis=-1)


def pattern_index(counts: np.ndarray) -> np.ndarray:
    """[...,3] 计数（截断 0..2）→ 0..26 的 27 类索引。"""
    c = np.clip(np.asarray(counts), 0, 2).astype(np.int64)
    return c[..., 0] * 9 + c[..., 1] * 3 + c[..., 2]


def true_pattern(alloc: np.ndarray, center) -> np.ndarray:
    """[n,3,34] → [n,3] 的真值 27 类索引。"""
    a, b, c = (int(x) for x in center)
    return pattern_index(np.asarray(alloc)[:, :, [a, b, c]])


def joint_pattern_probs(alloc_probs: np.ndarray, center) -> np.ndarray:
    """[n,3,34,5] → [n*3,27]：三张窗口的独立边际乘积。"""
    a, b, c = (int(x) for x in center)
    pa = tile_class_probs(alloc_probs, a)
    pb = tile_class_probs(alloc_probs, b)
    pc = tile_class_probs(alloc_probs, c)
    joint = pa[..., :, None, None] * pb[..., None, :, None] * pc[..., None, None, :]
    n, j = alloc_probs.shape[0], alloc_probs.shape[1]
    return joint.reshape(n * j, NUM_PATTERNS)


def softmax_temperature(alloc_probs: np.ndarray, tile: int, temp: float) -> np.ndarray:
    """对单张 0/1/2+ 三类做温度缩放；temp=1 原样。"""
    q = tile_class_probs(alloc_probs, tile)
    if temp == 1.0:
        return q
    q = np.clip(q, 1e-9, 1.0) ** (1.0 / float(temp))
    return q / q.sum(-1, keepdims=True)


def joint_pattern_probs_t(alloc_probs: np.ndarray, center, temp: float) -> np.ndarray:
    a, b, c = (int(x) for x in center)
    pa = softmax_temperature(alloc_probs, a, temp)
    pb = softmax_temperature(alloc_probs, b, temp)
    pc = softmax_temperature(alloc_probs, c, temp)
    joint = pa[..., :, None, None] * pb[..., None, :, None] * pc[..., None, None, :]
    n, j = alloc_probs.shape[0], alloc_probs.shape[1]
    return joint.reshape(n * j, NUM_PATTERNS)


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """秩和 AUC；y 为 0/1，p 为概率。"""
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    pos = y > 0.5
    npos = int(pos.sum())
    nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float('nan')
    order = np.argsort(p, kind='mergesort')
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def pair_metrics(alloc: np.ndarray, alloc_probs: np.ndarray,
                 a: int, b: int, mask: np.ndarray) -> dict:
    """单对 (a,b) 的成对共持指标；mask [n,3]。"""
    y = ((alloc[:, :, int(a)] > 0) & (alloc[:, :, int(b)] > 0))[mask]
    p = ((1.0 - alloc_probs[:, :, int(a), 0]) *
         (1.0 - alloc_probs[:, :, int(b), 0]))[mask]
    return {
        'n': int(y.size),
        'base': float(y.mean()) if y.size else float('nan'),
        'auc': auc_score(y, p),
        'brier': brier_score(y, p),
        'logloss': log_loss(y, p),
    }


def shape_metrics(joint: np.ndarray, yidx: np.ndarray,
                  mask: np.ndarray) -> dict:
    """单中心 27 类形状指标；joint [n*3,27]，yidx [n*3]，mask [n,3]。"""
    sel = mask.reshape(-1)
    j = joint[sel]
    y = yidx[sel]
    if len(y) == 0:
        return {'n': 0}
    order = np.argsort(-j, axis=1)
    top1 = float((order[:, 0] == y).mean())
    top3 = float((order[:, :3] == y[:, None]).any(axis=1).mean())
    ce = float(np.mean(-np.log(np.clip(j[np.arange(len(y)), y], 1e-9, None))))
    cnt = np.bincount(y, minlength=NUM_PATTERNS).astype(np.float64)
    prior = cnt / max(cnt.sum(), 1.0)
    return {
        'n': int(len(y)),
        'top1': top1,
        'top3': top3,
        'ce': ce,
        'prior_major': float(prior.max()),
        'prior_top3': float(np.sort(prior)[-3:].sum()),
        'prior_entropy': float(-(prior * np.log(np.clip(prior, 1e-9, None))).sum()),
    }
