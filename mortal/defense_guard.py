"""推理期防守守卫：用读牌后验修正主模型 Q（G3 连续形式）。

**为什么是 G3（2026-10-22，按「可达上界」比较）**

- **G1**（非筋固定罚）是 ``risk`` 的**二值离散近似**：同一「非筋」内 1p 与 5m
  的危险度差异被抹平，且只覆盖立直家，漏掉默听/副露听牌；
- **G2**（硬 mask）在手上无现物/筋时退化，且强行放弃高价值手牌（听牌+宝牌
  也不许攻），上界被「必须进攻的局面」封住；
- **G3** 用连续 ``risk(t)`` 覆盖全部对手（含默听/副露），是「Q 减去期望放铳
  损失」的贝叶斯形式，上界严格包含 G1/G2。

**λ 不随牌值自适应**：主模型 Q 已含手牌价值（打危险牌换来的进攻收益就在 Q
里），再乘「牌值调制」等于把同一项算两遍。λ 固定 = 一次放铳的期望顺位损失
（pt），由 ladder 扫描确定。

    Q'(t) = Q(t) - λ · max(risk(t) - τ, 0),  t ∈ [0, 34) 打牌动作
    risk(t) = Σ_j P(听_j) · P(待牌_j = t)          （A2c 后验，三家 union）

``τ``（risk_floor）是 2026-10-22 λ=20 双向 ladder 后的修正：线性版把低 risk
区（0.001~0.01，噪声大、无区分度）也按比例扣分，导致「houjuu −21%、agari
−7.4%、avg_pt −0.40（CI[−1.69,+0.89]）」——用和了换放铳，净不划算。τ 把
惩罚集中到真正危险的牌上（只拦 risk 明显高于背景的牌），保留进攻。

只改打牌动作（34..45 的立直/鸣牌/荣和/pass 不动）：守卫要回答的是「打哪张
更安全」，不是「要不要和了」。
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

NUM_TILES = 34
N_DISCARD = 34
TILES_SEEN_ROW = 835                 # v4：tiles_seen = count/4
RIVER_BASES = (132, 327, 522)        # 下家/对家/上家
CH_KAWA = 8
N_KAWA_SLOTS = 24                    # take(6) + rev().take(18)
OFF_SUTEHAI = 3
OFF_RIICHI = 7

READ_DIR = Path(__file__).resolve().parents[1] / 'tools' / 'read_model'
DEFAULT_CKPT = Path(__file__).resolve().parent / 'read_model_v2s_a2c' / 'best.pth'

_STACK = None


def _load_read_stack():
    """加载读牌 model/model_v2/structure/constraints（处理同名 ``model.py``）。

    在 ``mortal/`` 进程里 ``sys.modules['model']`` 已是主模型；读牌 ``model_v2``
    需要 ``import model as M`` 指向读牌版，故临时替换、加载完立即恢复（坑 69）。
    """
    global _STACK
    if _STACK is not None:
        return _STACK
    saved = {k: sys.modules.get(k) for k in
             ('model', 'model_v2', 'structure', 'constraints')}
    sys.path.insert(0, str(READ_DIR))

    def _load(name, path):
        spec = _ilu.spec_from_file_location(name, str(path))
        mod = _ilu.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        _load('model', READ_DIR / 'model.py')
        _load('structure', READ_DIR / 'structure.py')
        M2 = _load('model_v2', READ_DIR / 'model_v2.py')
        cons = _load('constraints', READ_DIR / 'constraints.py')
    finally:
        sys.path.remove(str(READ_DIR))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    _STACK = (M2, cons)
    return _STACK


def obs_aux(obs: np.ndarray):
    """numpy 版反解（离线诊断用；推理主路径走 ``obs_aux_torch``）。

    与 ``tools/read_model/precompute_danger.py`` 同口径（逐位实测：own/unseen/
    river 100%、riichi 99.83%）。
    """
    obs = np.asarray(obs, np.float32)
    if obs.ndim == 2:
        obs = obs[None]
    B = obs.shape[0]
    own = obs[:, 0:4, :].any(1) > 0.5
    unseen = 4.0 - np.round(obs[:, TILES_SEEN_ROW, :] * 4.0)
    river = np.zeros((B, 3, NUM_TILES), bool)
    riichi = np.zeros((B, 3), np.float32)
    for j, base in enumerate(RIVER_BASES):
        for i in range(N_KAWA_SLOTS):
            row = base + CH_KAWA * i
            has = obs[:, row + OFF_SUTEHAI, :].sum(1) > 0.5
            tid = obs[:, row + OFF_SUTEHAI, :].argmax(1)
            river[np.where(has)[0], j, tid[has]] = True
            riichi[:, j] = np.maximum(
                riichi[:, j],
                (obs[:, row + OFF_RIICHI, :].sum(1) > 0.5).astype(np.float32))
    return own, unseen, river, riichi


def obs_aux_torch(obs: torch.Tensor):
    """``obs_aux`` 的 torch 版（与主模型同设备，零拷贝；推理主路径）。

    把 72 次 Python 循环换成一次 gather：``(B,24,34)`` 的 sutehai 通道直接
    ``argmax`` 得牌号、``sum`` 得「有牌」掩码，再 scatter 进 river。
    """
    own = obs[:, 0:4, :].any(1) > 0.5
    unseen = 4.0 - torch.round(obs[:, TILES_SEEN_ROW, :] * 4.0)
    B, dev = obs.shape[0], obs.device
    river = torch.zeros(B, 3, NUM_TILES, dtype=torch.bool, device=dev)
    riichi = torch.zeros(B, 3, dtype=torch.float32, device=dev)
    rows = torch.arange(N_KAWA_SLOTS, device=dev) * CH_KAWA
    for j, base in enumerate(RIVER_BASES):
        sute = obs[:, base + rows + OFF_SUTEHAI, :]          # (B,24,34) 每槽 one-hot
        river[:, j] = sute.any(1)                            # (B,34) 该家河中牌集合
        riichi[:, j] = (obs[:, base + rows + OFF_RIICHI, :]
                        .sum((1, 2)) > 0.5).float()
    return own, unseen, river, riichi


def wait_bias_torch(unseen: torch.Tensor, river: torch.Tensor,
                    own: torch.Tensor) -> torch.Tensor:
    """H2 加性对数偏置（与 ``constraints.compute_wait_bias`` 默认口径一致）。

    两条硬掩码：① 4 张全见且不在自家手里（没人能打出）；② 目标家河中已有
    该牌（舍牌振听，无法荣和）。``post_riichi_river``/``riichi`` 在原实现里
    只是诊断入参，不影响结果，故不传。
    """
    not_expr = (unseen <= 0) & ~own
    bias = torch.zeros(unseen.shape[0], 3, NUM_TILES,
                       dtype=torch.float32, device=unseen.device)
    bias = bias.masked_fill(not_expr[:, None, :], float('-inf'))
    bias = bias.masked_fill(river, float('-inf'))
    return bias


class DefenseGuard:
    """``Q' = Q - λ·risk``；``__call__(q_out, obs, masks) -> q_out``。

    ``obs``/``masks`` 与 ``MortalEngine._react_batch`` 同源：obs ``(B,1012,34)``
    torch tensor（device 上），masks ``(B,46)`` bool。返回新的 Q，不改原 tensor。
    """

    def __init__(self, lambda_: float = 1.0, risk_floor: float = 0.0,
                 ckpt: Optional[str] = None,
                 device: Optional[torch.device] = None):
        M2, cons = _load_read_stack()
        self._cons = cons
        self.lam = float(lambda_)
        self.risk_floor = float(risk_floor)
        self.device = torch.device(device or 'cpu')

        path = Path(ckpt) if ckpt else DEFAULT_CKPT
        st = torch.load(path, map_location='cpu', weights_only=False)
        cfg = st['config']
        net = M2.build_model_v2s(
            conv_channels=cfg['conv_channels'],
            num_blocks=cfg['num_blocks'],
            with_aux=not cfg['no_aux'],
            wait_head=cfg.get('wait_head', 'flat'),
            tl_mode=cfg.get('tl_mode', 'pooled'),
            next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
            tile_order=cfg.get('tile_order', False),
        )
        net.load_state_dict(st['model'])
        self.net = net.to(self.device).eval()
        self.ckpt = str(path)
        self.n_calls = 0
        self.n_skip = 0

    @torch.inference_mode()
    def __call__(self, q_out: torch.Tensor, obs: torch.Tensor,
                 masks: torch.Tensor) -> torch.Tensor:
        if q_out.shape[-1] <= N_DISCARD:
            return q_out
        live = masks[:, :N_DISCARD]
        if not bool(live.any()):
            self.n_skip += 1
            return q_out

        o = obs if obs.device == self.device else obs.to(self.device)
        o = o.float()
        own, unseen, river, _riichi = obs_aux_torch(o)
        w = wait_bias_torch(unseen, river, own)

        B = o.shape[0]
        tl = torch.zeros(B, 3, 26, 43, device=self.device)
        tlm = torch.zeros(B, 3, 26, dtype=torch.bool, device=self.device)
        out = self.net(o, unseen, w, tl, tlm)

        wp = out['wait_probs']                                   # (B,3,35)
        p_tenpai = (1.0 - wp[..., NUM_TILES]).clamp(0.0, 1.0)     # (B,3)
        risk = (p_tenpai[..., None] * wp[..., :NUM_TILES]).sum(1)  # (B,34)
        risk = torch.nan_to_num(risk, nan=0.0).clamp(0.0, 1.0)

        self.n_calls += 1
        q = q_out.clone()
        pen = (risk - self.risk_floor).clamp_min(0.0) * self.lam
        q[:, :N_DISCARD] -= pen.to(q.device).to(q.dtype) * live
        return q
