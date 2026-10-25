"""推理期 EV 重排：用 SP 前瞻（听牌率/和了率/EV）+ A2c 危险度重新排序打牌候选。

**动机**：数据侧三条路线（R3/R4/R5、D1、D2）全部 null/负，剩下的候选改进（+0.2~0.5）
低于 ladder 的分辨率；唯一被证明有效的机制是**推理期修正**（守卫 λ=10 → +0.78）。
本模块把守卫的「单纯风险惩罚」升级为**收益-风险同尺度比较**：

    score(t) = Q(t) + β · ev_norm(t) − λ · max(risk(t) − τ, 0)

- `ev_norm(t)`：SP 前瞻给出「打掉 t 之后」的期望价值，已按该状态的 max_ev 归一（0..1）；
  对 SP 未覆盖的候选（降向听打牌）填该状态覆盖候选的均值，避免系统性惩罚。
- `risk(t)`：A2c 后验 `Σ_j P(听_j)·P(待牌_j=t)`（与 defense_guard 同口径）。
- 只在打牌动作 0..33 上生效；立直/鸣牌/荣和/pass 不动。

**可选 margin（tie-break 模式）**：只允许在 `Q ≥ maxQ − margin` 的候选里重排——
Q 明显偏好某张时不动，避免用 SP 的「纯速度」信号覆盖 Q 的「牌值」判断。

**v4 obs 布局（本文件唯一依赖的常量，均有回归测试）**：
- 889/890（全行常量）：`max_ev` 的 1e5 / 3e4 归一化；
- 891..924：保向听的 required tiles（每张打牌一行）；925..958：降向听的；
- 959：`max_required_tiles` 那张打牌的 one-hot（960 未用）；
- 961..977 / 978..994 / 995..1011：SP 表，**每段 17 行 × 34 列**（行=剩余抽牌档位，
  列=打牌候选）；对每张候选取 `max over 行` 即「最好档位下的 P(听)/P(和)/EV」。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

NUM_TILES = 34
NATIVE_ROWS = 1012
SP_MAXEV_1E5 = 889
SP_MAXEV_3E4 = 890
SP_DISCARD_CAND = 874     # 874..878：discard_candidates_aka / keep_shanten / next_shanten / 无条 件听牌 / 立直中
SP_REQ_KEEP = 891          # 891..924
SP_REQ_DOWN = 925          # 925..958
SP_BEST_DISCARD = 959
SP_TABLE = 961             # 961..977 tenpai / 978..994 win / 995..1011 ev
SP_N_TURNS = 17


def sp_features(obs: np.ndarray) -> dict:
    """从 v4 obs 抽 SP 前瞻特征（(N,1012,34) 或 (1012,34)）。

    返回 dict：
        ev_norm  (N,34) float32  SP 期望价值 / max_ev（0..1，未覆盖处为 0）
        p_win    (N,34) float32  和了概率（未覆盖处为 0）
        p_tenpai (N,34) float32  听牌概率（未覆盖处为 0）
        max_ev   (N,)   float32  该状态的 max_ev（点）
        covered  (N,34) bool     该牌是否有 SP 前瞻
    """
    obs = np.asarray(obs, dtype=np.float32)
    squeeze = obs.ndim == 2
    o = obs[None] if squeeze else obs
    n = o.shape[0]
    ev_rows = o[:, SP_TABLE + 2 * SP_N_TURNS:SP_TABLE + 3 * SP_N_TURNS, :]
    win_rows = o[:, SP_TABLE + SP_N_TURNS:SP_TABLE + 2 * SP_N_TURNS, :]
    ten_rows = o[:, SP_TABLE:SP_TABLE + SP_N_TURNS, :]
    # ⚠️ can_discard=false（对手舍牌后的鸣牌/立直决策）时 SP 表是**整行广播**的
    # （encode_sp_table 的 else 分支 arr.fill），此时列不代表候选牌 → 必须用候选行
    # 874 过滤，否则全员 "covered" 且与手牌不一致。
    cand = o[:, SP_DISCARD_CAND, :] > 0.5
    ev = np.where(cand, ev_rows.max(1), 0.0)
    win = np.where(cand, win_rows.max(1), 0.0)
    ten = np.where(cand, ten_rows.max(1), 0.0)
    covered = cand & ((ev > 0) | (win > 0) | (ten > 0))
    max_ev = o[:, SP_MAXEV_3E4].max(1) * 30_000.0
    out = {
        'ev_norm': ev.astype(np.float32),
        'p_win': win.astype(np.float32),
        'p_tenpai': ten.astype(np.float32),
        'max_ev': max_ev.astype(np.float32),
        'covered': covered,
    }
    if squeeze:
        out = {k: v[0] for k, v in out.items()}
    return out


def ev_neutral(feats: dict, own: np.ndarray) -> np.ndarray:
    """把未覆盖候选的 ev_norm 填成「该状态覆盖候选的均值」（逐样本）。"""
    ev = np.array(feats['ev_norm'], dtype=np.float32, copy=True)
    cov = np.asarray(feats['covered'], dtype=bool) & np.asarray(own, dtype=bool)
    single = ev.ndim == 1
    e2, c2 = (ev[None], cov[None]) if single else (ev, cov)
    for i in range(e2.shape[0]):
        idx = np.where(c2[i])[0]
        if idx.size:
            e2[i][~c2[i]] = float(e2[i][idx].mean())
    return e2[0] if single else e2


def rerank_scores(q: np.ndarray, obs: np.ndarray, risk: np.ndarray, *,
                  beta: float = 0.3, lam: float = 10.0, tau: float = 0.0,
                  margin: Optional[float] = None, own: Optional[np.ndarray] = None) -> dict:
    """计算重排后的打牌选择。

    q (N,34) 打牌 Q；obs (N,1012,34)；risk (N,34) 危险度；own (N,34) 自家手牌掩码。
    margin 非 None 时只在 Q ≥ maxQ − margin 的候选里重排。
    """
    q = np.asarray(q, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    feats = sp_features(obs)
    ev = ev_neutral(feats, own) if own is not None else feats['ev_norm'].astype(np.float64)
    ev = np.asarray(ev, dtype=np.float64)
    score = q + beta * ev - lam * np.maximum(risk - tau, 0.0)
    if own is not None:
        score = np.where(own, score, -np.inf)
    if margin is not None:
        qm = np.where(own, q, -np.inf) if own is not None else q
        best = qm.max(-1, keepdims=True)
        allowed = qm >= (best - margin)
        score = np.where(allowed, score, -np.inf)
    pick = score.argmax(-1)
    base = (np.where(own, q, -np.inf) if own is not None else q).argmax(-1)
    return {
        'pick': pick,
        'base_pick': base,
        'changed': pick != base,
        'ev_chosen': ev[np.arange(len(pick)), pick],
        'ev_base': ev[np.arange(len(pick)), base],
        'risk_chosen': risk[np.arange(len(pick)), pick],
        'risk_base': risk[np.arange(len(pick)), base],
        'covered': feats['covered'],
        'max_ev': feats['max_ev'],
    }


class EVRerank:
    """推理期钩子：`__call__(q_out, obs_tensor, masks)`（与 DefenseGuard 同接口）。

    `risk_fn(obs_tensor) -> (N,34) tensor` 由调用方提供（复用 A2c 读牌栈），避免本模块
    依赖读牌代码；`CACHE_RISK=True` 时与 DefenseGuard 共用一次前向（见 engine 接线）。
    """

    def __init__(self, risk_fn, *, beta: float = 0.3, lam: float = 10.0, tau: float = 0.0,
                 margin: Optional[float] = None):
        self.risk_fn = risk_fn
        self.beta = float(beta)
        self.lam = float(lam)
        self.tau = float(tau)
        self.margin = None if margin is None else float(margin)
        self.n_calls = 0
        self.n_changed = 0

    def __call__(self, q_out, obs, masks, own=None):
        live = masks[:, :NUM_TILES]
        if not bool(live.any()):
            return q_out
        q = q_out[:, :NUM_TILES].float().cpu().numpy()
        risk = np.asarray(self.risk_fn(obs), dtype=np.float64)
        obs_np = obs.detach().float().cpu().numpy()
        own_np = (live.cpu().numpy() if own is None else np.asarray(own, dtype=bool))
        out = rerank_scores(q, obs_np, risk, beta=self.beta, lam=self.lam,
                            tau=self.tau, margin=self.margin, own=own_np)
        self.n_calls += 1
        self.n_changed += int(out['changed'].sum())
        import torch
        q_new = q_out.clone()
        pick = torch.as_tensor(out['pick'], device=q_out.device)
        # 选中的候选必须本来就在合法集合里（own ⊆ masks[:, :34]）
        q_new[:, :NUM_TILES] = torch.full_like(q_new[:, :NUM_TILES], -1e9)
        q_new[torch.arange(q_new.shape[0], device=q_out.device), pick] = 1e9
        return q_new
