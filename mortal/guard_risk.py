"""守卫 risk 的单一真源：从读牌后验到逐牌危险度与 Q 惩罚。

**口径（2026-11-02 修正）**

``factorized_wait_probs`` 的输出形状是 ``(..., 35)``：

- ``wp[..., :34]`` = ``P(听) * P(待牌=t | 听)``，行和恒等于 ``1 - wp[...,34]``；
- ``wp[..., 34]`` = ``P(未听)``。

因此逐牌危险度必须直接对 34 个牌类求和：

    risk(t) = Σ_j wp[j, t]

**不能再乘一次 P(听)**。旧实现 ``p_ten * wp[..., :34]`` 实际算的是
``P(听)^2 * P(待牌=t|听)``，会系统性压低“听牌概率不高”的对手，并让 λ 的
标定尺度偏离文档公式。``legacy_risk_from_wait_probs`` 仅保留给回归测试与
2026-11-02 的 legacy 对照臂，生产路径不得调用。

本模块只依赖 torch，不加载读牌模型，避免 ``mortal/model.py`` 与
``tools/read_model/model.py`` 同名冲突（坑 69）。
"""
from __future__ import annotations

import torch

NUM_TILES = 34
N_DISCARD = 34
NUM_WAIT_CLASSES = NUM_TILES + 1


def risk_from_wait_probs(wait_probs: torch.Tensor) -> torch.Tensor:
    """``(B, 3, 35)`` 或 ``(3, 35)`` → ``(B, 34)`` / ``(34,)`` 的三家 union 危险度。

    ``wait_probs`` 必须是 ``model.factorized_wait_probs`` 的输出（或同语义的
    A3 规则法输出）：前 34 类已经是联合概率 ``P(听)·P(待牌=t|听)``。
    对三家求和后 clamp 到 ``[0, 1]``；多家同时听同一张时会饱和到 1。
    """
    r = wait_probs[..., :NUM_TILES].sum(dim=-2)
    return torch.nan_to_num(r, nan=0.0).clamp(0.0, 1.0)


def legacy_risk_from_wait_probs(wait_probs: torch.Tensor) -> torch.Tensor:
    """旧口径（双乘 P(听)），仅供回归/对照；生产路径禁止使用。

    与 ``risk_from_wait_probs`` 的关系：``legacy = p_ten * fixed``。
    ``tools/test_guard_risk.py`` 用合成案例锁死该关系。
    """
    p_ten = (1.0 - wait_probs[..., NUM_WAIT_CLASSES - 1]).clamp(0.0, 1.0)
    r = (p_ten.unsqueeze(-1) * wait_probs[..., :NUM_TILES]).sum(dim=-2)
    return torch.nan_to_num(r, nan=0.0).clamp(0.0, 1.0)


def apply_risk_penalty(q_out: torch.Tensor, risk: torch.Tensor,
                       lam: float, tau: float,
                       masks: torch.Tensor) -> torch.Tensor:
    """``Q' = Q - λ·max(risk-τ, 0)``，只改打牌动作 ``0..33``。

    - 返回新 tensor，不原地修改 ``q_out``；
    - 动作 34..45（立直/鸣牌/荣和/pass）保持不变；
    - ``masks`` 中为 False 的打牌动作不扣分；
    - ``q_out`` 动作维 <= 34 时直接返回原 tensor（与旧实现一致）。
    """
    if q_out.shape[-1] <= N_DISCARD:
        return q_out
    live = masks[..., :N_DISCARD]
    if not bool(live.any()):
        return q_out

    pen = (risk - float(tau)).clamp_min(0.0) * float(lam)
    q = q_out.clone()
    q[..., :N_DISCARD] -= pen.to(device=q.device, dtype=q.dtype) * live
    return q
