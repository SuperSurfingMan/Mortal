"""Guard-P2 局面特征：从 v4 obs 读取押し引き所需的三类状态量。

设计见 `memory/DESIGN_guard_statewise_20261027.md`。本模块只做**读取与权重**，
不碰 Q、不改 `defense_guard.py`（`mode='fixed'` 必须逐位不变）。

v4 obs 布局（本文件依赖的常量，均有 `tools/test_guard_state.py` 覆盖）：
- 0..2  自家手牌计数（非常量）
- 3..5  aka 在手的 one-hot
- 6    恒为零的手牌计数槽（count==4 不可能）
- 7..14 分数：4 家 × 2 行（`score/1e5`、`score/3e4` 截断），**整行常量**
- 15..18 名次 one-hot（本家名次，0-based）
- 19..22 局数 one-hot（kyoku）
- 835    tiles_seen（value = 已见张数/4）

⚠️ 分数按**绝对座次**排列（未旋转），本家分数靠「名次 + 排序」反推；并列时取排序后的
对应位置（并列概率很低，且只影响 w_score 的一个分档）。
"""
from __future__ import annotations

import numpy as np

SCORE_ROWS = 7            # 7..14
RANK_ROWS = 15            # 15..18
KYOKU_ROWS = 19           # 19..22
TILES_SEEN_ROW = 835

# 默认权重表（先验固定符号，只有强度可调；见设计文档 §1）
W_OPP = (1.0, 1.12, 1.24, 1.36)          # 立直家数 0/1/2/3
W_TURN = (0.88, 1.0, 1.18)               # 早 / 中 / 晚
W_LEAD = 1.20                            # 领先 ≥ LEAD_GAP
W_TRAIL = 0.75                           # 落后 ≥ LEAD_GAP 且晚巡
LEAD_GAP = 12000
PROGRESS_Q = (7.0, 13.8)                 # tiles_seen 总和的 p25 / p75（202609 实测）
CLAMP = (0.5, 2.0)


def _score_rows(o, first_row):
    """把「score/1e5」与「score/3e4」两行合成千点分数（3e4 行在 30000 处截断）。"""
    raw = o[:, first_row, :].max(1) * 100_000.0
    small = o[:, first_row + 1, :].max(1) * 30_000.0
    return np.where(raw < 29_999.0, small, np.minimum(raw, 100_000.0))


def features(obs: np.ndarray) -> dict:
    """(N,1012,34) 或 (1012,34) → 局面特征。

    ⚠️ 关键：v4 的分数是**已旋转到本家视角**的（实测 rank==0 ⟺ scores[:,0] 为最大 100%），
    即 index 0 = 本家，1..3 = 下家/对家/上家。
    """
    o = np.asarray(obs, dtype=np.float32)
    squeeze = o.ndim == 2
    if squeeze:
        o = o[None]
    my_score = _score_rows(o, SCORE_ROWS)
    others = np.stack([_score_rows(o, SCORE_ROWS + 2 * j) for j in (1, 2, 3)], axis=1)
    best_other = others.max(1)
    rank = o[:, RANK_ROWS:RANK_ROWS + 4].min(2).argmax(1)
    # 并列时用「对最高其他人」的口径，避免名次/索引在平分处不自洽（D 项本身是可选成分）
    lead = np.maximum(my_score - best_other, 0.0)
    deficit = np.maximum(best_other - my_score, 0.0)
    riichi_cnt = np.zeros(o.shape[0], np.float32)
    for base in (132, 327, 522):
        for i in range(24):
            riichi_cnt += (o[:, base + 8 * i + 7, :].sum(1) > 0.5).astype(np.float32)
    out = {
        'scores': np.concatenate([my_score[:, None], others], axis=1).astype(np.float32),
        'my_rank': rank.astype(np.int8),
        'my_score': my_score.astype(np.float32),
        'lead': lead.astype(np.float32),
        'deficit': deficit.astype(np.float32),
        'riichi_cnt': np.clip(riichi_cnt, 0, 3),
        'progress': o[:, TILES_SEEN_ROW].sum(1).astype(np.float32),
    }
    if squeeze:
        out = {k: v[0] for k, v in out.items()}
    return out


def turn_bucket(progress: np.ndarray) -> np.ndarray:
    lo, hi = PROGRESS_Q
    return np.where(progress <= lo, 0, np.where(progress <= hi, 1, 2)).astype(np.int8)


def weights(feat: dict, *, w_opp=W_OPP, w_turn=W_TURN, w_lead=W_LEAD,
            w_trail=W_TRAIL, lead_gap=LEAD_GAP, use_opp=True, use_turn=True,
            use_score=True, clamp=CLAMP) -> np.ndarray:
    """组合出 w_state（(N,) float32）。任何分项可单独关闭，便于逐项判定。"""
    cnt = np.asarray(feat['riichi_cnt']).astype(int)
    tb = turn_bucket(np.asarray(feat['progress']))
    w = np.ones_like(np.asarray(feat['my_score'], dtype=np.float32))
    if use_opp:
        w *= np.asarray(w_opp, np.float32)[np.clip(cnt, 0, 3)]
    if use_turn:
        w *= np.asarray(w_turn, np.float32)[tb]
    if use_score:
        lead = np.asarray(feat['lead'], np.float32)
        deficit = np.asarray(feat['deficit'], np.float32)
        w *= np.where(lead >= lead_gap, w_lead, 1.0)
        w *= np.where((deficit >= lead_gap) & (tb == 2), w_trail, 1.0)
    return np.clip(w, clamp[0], clamp[1]).astype(np.float32)
