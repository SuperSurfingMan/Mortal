"""读牌模型硬约束层（方案 §4.3 / §7）。

两部分：

A. ``compute_wait_bias`` —— H2 听牌头的加性对数偏置（训练/推理/Rust 同口径）：
   - 硬掩码（-inf）：该牌 4 张已全部可见**且不在观察者自己手里**——没人能打出它，
     也就不可能是荣和威胁。注意 unseen==0 不等于「不是听牌」：4 张全在「自家手牌 +
     公共区」时，观察者自己手里的副本依然能点炮，故 own_hand 中的牌不掩；
   - 硬掩码（-inf）：该牌在目标家自己本局的河里——按规则「自己的舍牌中有待牌
     即振听，无法荣和」，这对任意目标家都是硬事实（不分立直/非立直）。方案原文
     对非立直家用「软先验降权」，v1 改为硬掩码并与标签（有效荣和集）严格一致；
     软惩罚保留为 ``hard_river_mask=False`` 的消融旋钮；
   - 立直振听（荣和威胁归零）与抢槻门控：留 ``riichi_furiten`` 入参口子，
     v1 不自动检测（河牌硬掩码已覆盖：立直后打的牌必然不是冻结听牌，
     唯一例外是立直后摸到听牌却见逃的极罕见情形，由标签/掩码一致地按无可荣牌处理）。

B. ``apply_constraints`` —— 对 H1 期望张数做合法性投影：
   1. 列内守恒：Σ_j e[j,i] ≤ unseen_i（其余在牌山），越界列按比例缩放；
   2. 总手牌数：Σ_i e[j,i] == hand_sizes[j]（由公开信息精确推导，见下）；
   两者用迭代比例拟合（IPF，Sinkhorn 式交替行列缩放）联合求解。
   供给-容量可行性：Σ_j hand_j ≤ Σ_i unseen_i 恒成立（牌山容量充裕），故 IPF 收敛。

   hand_sizes 的推导（调用方职责，全部公开信息）：
       通用式（任意时点）：hand = 13 + tsumos - discards - meld_from_hand
       静止式（3n+1，本数据集全部快照均满足）：hand = 13 - 3*(chi+pon+daiminkan+ankan)
       注意加杠（kakan）对静止手牌数净贡献为 0（摸岭上牌补回），不计入。
"""
from __future__ import annotations

import numpy as np

NUM_TILES = 34
NUM_TARGETS = 3

#: 舍牌振听软先验的默认对数降权（e^-1 ≈ 0.37）；仅在 hard_river_mask=False 时生效
DEFAULT_FURITEN_SOFT_WEIGHT = 1.0


def compute_wait_bias(
    unseen: np.ndarray,                 # [B, 34] 或 [34]
    river: np.ndarray,                  # [B, 3, 34] bool 各家全部打牌
    post_riichi_river: np.ndarray | None = None,  # [B, 3, 34]（诊断用）
    riichi: np.ndarray | None = None,             # [B, 3]（诊断用）
    *,
    own_hand: np.ndarray | None = None,           # [B, 34] bool 观察者手持牌
    furiten_soft_weight: float = DEFAULT_FURITEN_SOFT_WEIGHT,
    hard_river_mask: bool = True,
) -> np.ndarray:
    """生成 H2 的加性对数偏置 [B, 3, 34]（-inf 表示硬掩码）。

    与 dataset 的 wait 标签（有效荣和集）严格对齐：
    hard_river_mask=True（默认）时，目标家本局河中的牌一律硬掩码——河里有待牌
    即振听、无法荣和，是硬规则。hard_river_mask=False 时退化为方案原文的软惩罚
    （仅用于消融对照，此时标签仍是有效荣和集，两者不再严格一致）。
    own_hand 给定时，「4 张全见」的牌若在观察者手中仍可表达（观察者能亲自打出），
    与标签口径一致；不给定则退回方案原文的 tiles_seen<4 口径。
    """
    unseen = np.asarray(unseen, dtype=np.float32)
    if unseen.ndim == 1:
        unseen = unseen[None, :]
    river = np.asarray(river, dtype=bool)
    B = unseen.shape[0]
    assert river.shape == (B, NUM_TARGETS, NUM_TILES)

    bias = np.zeros((B, NUM_TARGETS, NUM_TILES), dtype=np.float32)

    # 硬掩码 1：4 张全见且无人能打出（不在观察者手里）
    all_seen = (unseen <= 0)                                      # [B,34]
    if own_hand is not None:
        own_hand = np.asarray(own_hand, dtype=bool)
        assert own_hand.shape == (B, NUM_TILES)
        not_expressible = all_seen & ~own_hand
    else:
        not_expressible = all_seen
    bias = np.where(not_expressible[:, None, :], np.float32(-np.inf), bias)

    # 硬掩码 2（默认）：目标家本局河中的牌（舍牌振听，无法荣和）
    if hard_river_mask:
        bias = np.where(river, np.float32(-np.inf), bias)
    else:
        bias = np.where(river, bias - np.float32(furiten_soft_weight), bias)

    return bias


def expected_hand_size(
    n_chi: int, n_pon: int, n_daiminkan: int, n_ankan: int
) -> int:
    """静止态（3n+1）目标家暗牌总数。加杠净贡献 0，不计入。"""
    return 13 - 3 * (n_chi + n_pon + n_daiminkan + n_ankan)


def apply_constraints(
    alloc_probs: np.ndarray,            # [B, 3, 34, 5] 模型输出的计数分布
    unseen: np.ndarray,                 # [B, 34] 列容量上界
    hand_sizes: np.ndarray,             # [B, 3] 各家暗牌总数（公开信息推导）
    *,
    iters: int = 50,
) -> dict:
    """对期望张数做列守恒 + 总手牌数联合投影。

    返回 dict：
        exp_counts   [B,3,34] 投影后的期望张数（下游消费量）
        diag         每样本最大违例（投影前/后）、迭代是否收敛
    """
    alloc_probs = np.asarray(alloc_probs, dtype=np.float64)
    unseen = np.asarray(unseen, dtype=np.float64)
    hand_sizes = np.asarray(hand_sizes, dtype=np.float64)
    B = alloc_probs.shape[0]
    assert alloc_probs.shape == (B, NUM_TARGETS, NUM_TILES, 5)
    assert unseen.shape == (B, NUM_TILES)
    assert hand_sizes.shape == (B, NUM_TARGETS)

    k = np.arange(5, dtype=np.float64)
    e = (alloc_probs * k.reshape(1, 1, 1, -1)).sum(-1)          # [B,3,34]

    caps = np.maximum(unseen, 0.0)                              # [B,34]
    # 可行性检查：总供给不超过总容量（牌山必然充裕）
    supply_ok = hand_sizes.sum(-1) <= caps.sum(-1) + 1e-9

    col_viol_before = np.maximum(e.sum(1) - caps, 0).max(initial=0.0)
    row_err_before = np.abs(e.sum(-1) - hand_sizes).max(initial=0.0)

    for _ in range(iters):
        # 行缩放：每家总数 → hand_sizes
        row_sum = e.sum(-1, keepdims=True)                      # [B,3,1]
        scale = np.where(row_sum > 1e-12, hand_sizes[..., None] / np.maximum(row_sum, 1e-12), 1.0)
        e = e * scale
        # 列裁剪：三家合计 → caps
        col_sum = e.sum(1, keepdims=True)                       # [B,1,34]
        cscale = np.where(col_sum > caps[:, None, :],
                          caps[:, None, :] / np.maximum(col_sum, 1e-12), 1.0)
        e = e * cscale

    e = np.clip(e, 0.0, None)
    col_viol_after = np.maximum(e.sum(1) - caps, 0).max(initial=0.0)
    row_err_after = np.abs(e.sum(-1) - hand_sizes).max(initial=0.0)

    return {
        'exp_counts': e.astype(np.float32),
        'diag': {
            'col_viol_before': float(col_viol_before),
            'col_viol_after': float(col_viol_after),
            'row_err_before': float(row_err_before),
            'row_err_after': float(row_err_after),
            'supply_ok': bool(supply_ok.all()),
        },
    }


def renormalize_wait(
    wait_probs: np.ndarray,             # [B, 3, 35]
    *,
    riichi_furiten: np.ndarray | None = None,   # [B, 3] bool 立直振听（荣和威胁归零）
) -> np.ndarray:
    """听牌分布后处理：硬掩码已由 wait_bias 完成，这里做重归一化与振听归零。"""
    wait_probs = np.asarray(wait_probs, dtype=np.float64).copy()
    B = wait_probs.shape[0]
    assert wait_probs.shape == (B, NUM_TARGETS, NUM_TILES + 1)

    if riichi_furiten is not None:
        rf = np.asarray(riichi_furiten, dtype=bool)
        assert rf.shape == (B, NUM_TARGETS)
        # 荣和威胁归零：34 个牌类全部置 0，质量移到「未听」类（仅剩自摸威胁）
        wait_probs[rf, :NUM_TILES] = 0.0

    # 重归一化（掩码后可能整行除未听外全 0）
    s = wait_probs.sum(-1, keepdims=True)
    all_zero = s <= 0
    wait_probs = np.where(all_zero, 1.0 / (NUM_TILES + 1), wait_probs / np.where(s > 0, s, 1.0))
    return wait_probs.astype(np.float32)
