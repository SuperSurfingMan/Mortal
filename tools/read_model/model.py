"""读牌模型网络结构（监督学习路线方案 §4）。

架构（v1，相对方案的微调见 README「与方案的差异」）：

    输入 x ∈ [B, 1012, 34]（观察者 v4 obs，直接复用主管线 obs_repr，零新特征通道）

    骨干：Conv1d(1012→96, k=3) + BN + ReLU
          12 × ResBlock(96)  （Conv-BN-ReLU-Conv-BN + 残差 + ReLU，无通道注意力）
          → h ∈ [B, 96, 34]；沿 34 列 mean pooling → g ∈ [B, 96]

    H1 逐列计数分配头（每目标家一个，3 个）：
        h.transpose → [B, 34, 96] 过共享 MLP(96→128→5)（按列应用）
        logits 用 unseen 掩码（k > unseen_i 置 -inf）后 softmax → P(h_{j,i}=k)
        期望计数 e_j = Σ_k k·P(k) ∈ [B, 34]

    H2 听牌头（每目标家一个）：
        输入 cat([g, e_j]) ∈ [B, 130] → MLP(128→35)
        logits 加 wait_bias（调用方预计算的加性对数偏置：硬掩码 -inf /
        舍牌振听软惩罚 / 0）后 softmax → P(wait=x)，x∈{0..33, 未听}

    H3 辅助头（每目标家一个，仅表征塑形，权重 0.2）：
        输入 cat([g, e_j]) → MLP(128) 后四个输出：
        下一打 35 类、向听 8 桶、立直 logit、鸣牌 logit

前向签名（与 ONNX 导出一致）：
    forward(obs, unseen, wait_bias) -> dict(
        alloc_logits [B,3,34,5]（已掩码）、alloc_probs、exp_counts [B,3,34]、
        wait_logits [B,3,35]（已加 bias）、wait_probs（RM1 因子化读法：
            p34 = sigmoid(l34) = P(未听)，p_x = P(听)·softmax34(l_x)，
            见 factorized_wait_probs）、
        next_logits [B,3,35]、shanten_logits [B,3,8]、riichi_logit [B,3]、called_logit [B,3])
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
from typing import Dict

from libriichi.consts import obs_shape

NUM_TILES = 34
NUM_COUNTS = 5                          # k = 0..4
NUM_WAIT_CLASSES = NUM_TILES + 1        # 34 种听牌 + 未听
NO_DISCARD = NUM_TILES                  # 下一打：本局无后续打牌
NUM_SHANTEN_BUCKETS = 8                 # 0..6 + (>=6)
NUM_TARGETS = 3                         # 相对座位 1/2/3


def factorized_wait_probs(wait_logits: torch.Tensor) -> torch.Tensor:
    """RM1（README §11.1）：35 维 wait 分布的因子化读法（行和恒为 1）。

        P(未听) = sigmoid(l34)          —— L_wait_tenpai BCE 直接标定的量；
        P(待牌 x|听) = softmax34(l[:34])；
        p_x = P(听)·P(x|听)，p_34 = P(未听)。

    旧 softmax35 读法（P(未听) = 1 − softmax(l)[34]）把 34 个牌列 logit 在
    85.4% 非听牌目标上无约束的配分函数噪声（sd 1.94）混进 P(听牌)：
    tenpai AUC 0.9461 → 0.9685，标定比值 6.26× → 0.998
    （diagnostics/s315_wait_calib.py、s315_strat_auc.py，2026-10-19 复核）。
    两种读法的条件分布恒等（都是 softmax34(l[:34])），故 hit@k / 最安全牌等
    排序类指标不变；CE 类与 P(听) 类指标以本读法为准。A3 规则基线本就是
    因子化结构（wait_probs[34] = 1 − tile_mass），本读法使跨模型比较同口径。
    """
    p_not = torch.sigmoid(wait_logits[..., NUM_TILES:NUM_TILES + 1])   # P(未听)
    cond = torch.softmax(wait_logits[..., :NUM_TILES], dim=-1)         # P(待|听)
    cond = torch.nan_to_num(cond, nan=0.0)          # 34 列全掩码的退化行
    return torch.cat([(1.0 - p_not) * cond, p_not], dim=-1)


class ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + 残差 + ReLU（方案 §4.1 的块结构，不带动 channel attention）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actv(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.actv(out + x)


class ReadBackbone(nn.Module):
    """单帧 1D ResNet：输出逐列特征 h [B, C, 34] 与全局池化 g [B, C]。

    pool='mean'（默认，方案伪代码口径）：g = h.mean(34)。
    pool='attention'（方案 §4.1 备选）：g = Σ_i softmax_i(w·h[:,:,i])·h[:,:,i]，
    让全局摘要保留「哪些列重要」的信息——听牌头需要逐列推理（对手牌河结构），
    mean 池化会抹掉这一信息；两种口径可 A/B（train.py --pool）。
    """

    def __init__(self, in_channels: int, conv_channels: int = 96, num_blocks: int = 12,
                 pool: str = 'mean'):
        super().__init__()
        assert pool in ('mean', 'attention'), pool
        self.pool = pool
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(conv_channels) for _ in range(num_blocks)])
        if pool == 'attention':
            self.attn_query = nn.Linear(conv_channels, 1)

    def forward(self, x: Tensor) -> tuple:
        h = self.blocks(self.stem(x))
        if self.pool == 'mean':
            g = h.mean(dim=-1)
        else:
            score = self.attn_query(h.transpose(1, 2)).squeeze(-1)   # [B,34]
            w = score.softmax(dim=-1)
            g = (h * w.unsqueeze(1)).sum(dim=-1)                    # [B,C]
        return h, g


class AllocHead(nn.Module):
    """H1：按列共享的 MLP(96→128→5)，输出逐牌种计数 logits（未掩码）。"""

    def __init__(self, in_features: int = 96, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, NUM_COUNTS),
        )

    def forward(self, h: Tensor) -> Tensor:
        # h: [B, C, 34] → [B, 34, C] → [B, 34, 5]
        return self.net(h.transpose(1, 2))


class WaitHead(nn.Module):
    """H2：cat([g, e_j]) → MLP(130→128→35)。"""

    def __init__(self, in_features: int = 96 + NUM_TILES, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, NUM_WAIT_CLASSES),
        )

    def forward(self, g: Tensor, e_h: Tensor) -> Tensor:
        return self.net(torch.cat([g, e_h], dim=-1))


class AuxHead(nn.Module):
    """H3：cat([g, e_j]) → 公共 trunk → 下一打/向听/立直/鸣牌 四个输出。"""

    def __init__(self, in_features: int = 96 + NUM_TILES, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
        )
        self.next_discard = nn.Linear(hidden, NUM_WAIT_CLASSES)  # 34 牌 + 无后续
        self.shanten = nn.Linear(hidden, NUM_SHANTEN_BUCKETS)
        self.riichi = nn.Linear(hidden, 1)
        self.called = nn.Linear(hidden, 1)

    def forward(self, g: Tensor, e_h: Tensor) -> dict:
        z = self.trunk(torch.cat([g, e_h], dim=-1))
        return {
            'next_discard': self.next_discard(z),
            'shanten': self.shanten(z),
            'riichi': self.riichi(z),
            'called': self.called(z),
        }


class ReadNet(nn.Module):
    """三家暗牌/听牌预测网络（共享骨干，各自 H1/H2/H3 头）。"""

    def __init__(
        self,
        version: int = 4,
        conv_channels: int = 96,
        num_blocks: int = 12,
        with_aux: bool = True,
        pool: str = 'mean',
    ):
        super().__init__()
        self.version = version
        self.with_aux = with_aux
        self.pool = pool
        in_channels = obs_shape(version)[0]

        self.backbone = ReadBackbone(in_channels, conv_channels, num_blocks, pool=pool)
        self.alloc_heads = nn.ModuleList([AllocHead(conv_channels) for _ in range(NUM_TARGETS)])
        self.wait_heads = nn.ModuleList(
            [WaitHead(conv_channels + NUM_TILES) for _ in range(NUM_TARGETS)]
        )
        if with_aux:
            self.aux_heads = nn.ModuleList(
                [AuxHead(conv_channels + NUM_TILES) for _ in range(NUM_TARGETS)]
            )

        # k = 0..4 的计数向量，用于期望张数
        self.register_buffer('counts', torch.arange(NUM_COUNTS, dtype=torch.float32))

    def forward(
        self,
        obs: Tensor,                 # [B, 1012, 34]
        unseen: Tensor,              # [B, 34]，4 - tiles_seen（H1 容量掩码）
        wait_bias: Tensor,           # [B, 3, 34] 加性对数偏置（-inf=硬掩码）
    ) -> Dict[str, Tensor]:
        h, g = self.backbone(obs)    # h [B,96,34], g [B,96]

        # H1 容量掩码：k > unseen_i 的 logits 置 -inf（广播到 [B,34,5]）
        k_idx = self.counts.view(1, 1, NUM_COUNTS)
        alloc_mask = k_idx > unseen.unsqueeze(-1).to(torch.float32)  # [B,34,5]

        alloc_logits_list, alloc_probs_list, exp_counts_list = [], [], []
        wait_logits_list, wait_probs_list = [], []
        aux_out = {k: [] for k in ('next_discard', 'shanten', 'riichi', 'called')}

        for j in range(NUM_TARGETS):
            logits = self.alloc_heads[j](h)                       # [B,34,5]
            logits = logits.masked_fill(alloc_mask, float('-inf'))
            probs = logits.softmax(dim=-1)
            e_h = (probs * self.counts.view(1, 1, -1)).sum(-1)    # [B,34] 期望张数
            alloc_logits_list.append(logits)
            alloc_probs_list.append(probs)
            exp_counts_list.append(e_h)

            w_logits = self.wait_heads[j](g, e_h)                 # [B,35]
            # wait_bias 只作用于 34 个牌类；第 35 类（未听）无偏置
            bias = torch.cat(
                [wait_bias[:, j, :], torch.zeros_like(wait_bias[:, j, :1])], dim=-1
            )
            w_logits = w_logits + bias
            w_probs = factorized_wait_probs(w_logits)      # RM1 因子化读法
            wait_logits_list.append(w_logits)
            wait_probs_list.append(w_probs)

            if self.with_aux:
                a = self.aux_heads[j](g, e_h)
                for key in aux_out:
                    aux_out[key].append(a[key])

        out = {
            'alloc_logits': torch.stack(alloc_logits_list, dim=1),   # [B,3,34,5]
            'alloc_probs': torch.stack(alloc_probs_list, dim=1),
            'exp_counts': torch.stack(exp_counts_list, dim=1),      # [B,3,34]
            'wait_logits': torch.stack(wait_logits_list, dim=1),    # [B,3,35]
            'wait_probs': torch.stack(wait_probs_list, dim=1),
        }
        if self.with_aux:
            out['next_logits'] = torch.stack(aux_out['next_discard'], dim=1)
            out['shanten_logits'] = torch.stack(aux_out['shanten'], dim=1)
            out['riichi_logit'] = torch.stack(aux_out['riichi'], dim=1).squeeze(-1)
            out['called_logit'] = torch.stack(aux_out['called'], dim=1).squeeze(-1)
        return out


def build_model(version: int = 4, conv_channels: int = 96, num_blocks: int = 12,
                with_aux: bool = True, pool: str = 'mean') -> ReadNet:
    return ReadNet(version=version, conv_channels=conv_channels,
                   num_blocks=num_blocks, with_aux=with_aux, pool=pool)
