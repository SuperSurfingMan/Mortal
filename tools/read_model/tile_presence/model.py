#!/usr/bin/env python3
"""逐牌存在性模型：牌 token × 动作 token 的双流交叉注意力。

结构
----
- 牌流：34 个 tile token = `tile_id` 嵌入 + 逐牌公开履历 MLP。
- 动作流：目标家的公开动作 token（打牌/吃/碰/杠/立直）= `tile_id` 嵌入
  + 动作内容 MLP（**手切/摸切**、动作类型、巡目、立直后、被鸣、安全性）。
- context token：巡目/牌山/立直/副露/手牌数/防守倾向/自家手牌分布 + **听牌概率**。
- 每个 block：牌流自注意力 → 动作流自注意力 → 牌流对动作流交叉注意力 →
  动作流对牌流交叉注意力 → 各自 FFN（pre-LN，残差）。
- 输出：34 个 logit（该牌是否在目标家手里）。

听牌概率
--------
`tp_mlp` 先用「动作特征均值池化 + context」预测 P(tenpai)，作为辅助任务训练；
推理时 `sigmoid(tp_logit).detach()` 既拼进 context，也加到牌 token 上，
让"他有多可能已经听牌"直接调制每张牌的判断（不泄漏标签，因为这是模型自己的预测）。
"""
from __future__ import annotations

import torch
from torch import nn

import tokens as TK


def _kpm(valid: torch.Tensor) -> torch.Tensor:
    """bool 有效位 → key_padding_mask；全 padding 的行放开第 0 位避免 NaN。"""
    kpm = ~valid
    all_pad = kpm.all(dim=1)
    if bool(all_pad.any()):
        kpm = kpm.clone()
        kpm[all_pad, 0] = False
    return kpm


class CrossBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, ff_mult: int = 4):
        super().__init__()
        self.ln_t1 = nn.LayerNorm(d)
        self.self_t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_e1 = nn.LayerNorm(d)
        self.self_e = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_t2 = nn.LayerNorm(d)
        self.cross_te = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_e2 = nn.LayerNorm(d)
        self.cross_et = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_t3 = nn.LayerNorm(d)
        self.ff_t = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.ln_e3 = nn.LayerNorm(d)
        self.ff_e = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor, e: torch.Tensor,
                e_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ln_t1(t)
        t = t + self.drop(self.self_t(x, x, x, need_weights=False)[0])

        x = self.ln_e1(e)
        e = e + self.drop(self.self_e(x, x, x, key_padding_mask=_kpm(e_mask),
                                      need_weights=False)[0])

        e_kv = self.ln_e2(e)
        t = t + self.drop(self.cross_te(self.ln_t2(t), e_kv, e_kv,
                                        key_padding_mask=_kpm(e_mask),
                                        need_weights=False)[0])
        e = e + self.drop(self.cross_et(e_kv, self.ln_t2(t), self.ln_t2(t),
                                        need_weights=False)[0])

        t = t + self.drop(self.ff_t(self.ln_t3(t)))
        e = e + self.drop(self.ff_e(self.ln_e3(e)))
        return t, e


class TilePresenceNet(nn.Module):
    def __init__(self, d: int = 192, n_blocks: int = 2, heads: int = 8,
                 dropout: float = 0.1, ft: int = TK.FT, fe: int = TK.FE,
                 fc: int = TK.FC, n_tiles: int = TK.NUM_TILES):
        super().__init__()
        self.n_tiles = n_tiles
        self.tile_tok = nn.Embedding(n_tiles, d)
        self.tile_mlp = nn.Sequential(nn.Linear(ft, d), nn.GELU(), nn.Linear(d, d))
        self.tile_ln = nn.LayerNorm(d)

        self.ev_tok = nn.Embedding(n_tiles, d)
        self.cons_tok = nn.Embedding(n_tiles + 1, d)      # 0 = 无被鸣牌
        self.ev_mlp = nn.Sequential(nn.Linear(fe, d), nn.GELU(), nn.Linear(d, d))
        self.ev_ln = nn.LayerNorm(d)

        self.ctx_mlp = nn.Sequential(nn.Linear(fc + 1, d), nn.GELU(), nn.Linear(d, d))
        self.ctx_ln = nn.LayerNorm(d)

        self.tp_mlp = nn.Sequential(nn.LayerNorm(fc + fe), nn.Linear(fc + fe, d),
                                    nn.GELU(), nn.Linear(d, 1))
        self.tp_gate = nn.Linear(1, d)

        self.tgt = nn.Embedding(3, d)
        self.blocks = nn.ModuleList(
            [CrossBlock(d, heads, dropout) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, tile_feat: torch.Tensor, ev_tile: torch.Tensor,
                ev_feat: torch.Tensor, ev_mask: torch.Tensor,
                ctx: torch.Tensor, cons: torch.Tensor) -> dict:
        tf = tile_feat.float()
        ef = ev_feat.float()
        cf = ctx.float()
        b, j, qt, _ = tf.shape

        w = ev_mask.unsqueeze(-1).float()
        pooled = (ef * w).sum(2) / w.sum(2).clamp(min=1.0)
        tp_logit = self.tp_mlp(torch.cat([cf, pooled], -1)).squeeze(-1)   # [B,J]
        tp = torch.sigmoid(tp_logit).detach()

        t = self.tile_ln(self.tile_mlp(tf) + self.tile_tok.weight[None, None])
        e = self.ev_ln(self.ev_mlp(ef) + self.ev_tok(ev_tile.long())
                       + self.cons_tok(cons.long() + 1).sum(3))
        c = self.ctx_ln(self.ctx_mlp(torch.cat([cf, tp[..., None]], -1)))

        e = torch.cat([c.unsqueeze(2), e], 2)
        em = torch.cat([torch.ones_like(ev_mask[:, :, :1]), ev_mask], 2)

        tgt = self.tgt.weight[None]
        t = t + tgt[:, :, None, :] + self.tp_gate(tp[..., None])[:, :, None, :]
        e = e + tgt[:, :, None, :]

        ke = e.shape[2]
        t = t.reshape(b * j, qt, -1)
        e = e.reshape(b * j, ke, -1)
        em = em.reshape(b * j, ke)
        for blk in self.blocks:
            t, e = blk(t, e, em)
        logits = self.out(t).squeeze(-1).reshape(b, j, qt)
        return {'logits': logits, 'tenpai_logit': tp_logit,
                'tenpai_prob': torch.sigmoid(tp_logit)}
