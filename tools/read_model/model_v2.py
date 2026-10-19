"""读牌模型 v2：空间流 + 时序流 + 结构估计流（显式向听/搭子构造）。

相对 v1（单帧 1D ResNet）的设计动机与对应关系：

1. **手切时序流（v1 的最大盲区）**：v4 obs 里其实有序码放着每家的牌河（含摸切标志），
   但那些信息躺在**通道维**上，沿列卷积的 ResNet 无法按时间推理——"第 3 巡手切了 5m"
   与"第 12 巡摸切了 5m"在通道里只是两个不同的行，Conv1d(3) 的感受野跨不过去。
   v2 显式抽出每目标家的**公开动作序列**（打牌+摸切标志 / 立直宣言 / 鸣牌，
   共 43 维 token，一局内最多 26 个），用小 Transformer + attention pooling 编码
   为时序嵌入 e_T。关键归纳偏置：摸切打出的牌就是当回摸到的牌（模型可直接用），
   手切则摸到的牌仍在手里（需推断）。

2. **结构估计流（显式 向听数/搭子构造）**：新增结构头，先显式估计
   - 7 个搭子构造计数的软分布（顺子/刻子/对子/两面/嵌张/边张/孤立，标签取
     "全部最小向听分解"上的边际直方图，保留真实歧义）；
   - 3 类形制（normal/七对/国士）；
   - 向听数（8 桶）。
   结构估计作为**信念瓶颈**注入 H1/H2（soft，无泄漏）：先判断"3 个顺子 + 一个两面
   搭 + 一个雀头 = 听牌"，再分配具体牌与预测待牌——匹配人类读牌的粗到细路径。
   标签由 ``structure.py`` 的分解枚举器从真实手牌算出（与 Rust calc_all 对齐）。

3. 空间流与 v1 完全一致（同 backbone、同 H1/H2/H3 输出语义），保证 v1/v2 可比。

输出与 v1 一致（alloc_probs / exp_counts / wait_probs / 辅助头；wait_probs 为
RM1 因子化读法，见 ``model.factorized_wait_probs``），另加
``struct_counts`` / ``struct_form`` / ``struct_shanten`` 供诊断与消融。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from libriichi.consts import obs_shape

import model as M
from structure import COUNT_BINS, NUM_FORMS, NUM_STRUCT_COUNTS

NUM_TILES = 34
NUM_WAIT_CLASSES = NUM_TILES + 1
NO_DISCARD = NUM_TILES
MAX_SHANTEN_BUCKET = 7


class _ManualAttention(nn.Module):
    """手写多头自注意力（全标准算子，ONNX 可导出）。

    不依赖 ``nn.TransformerEncoderLayer``——其 PyTorch 快速路径会融合成
    ``aten::_transformer_encoder_layer_fwd``，ONNX opset 17 不支持导出。
    行为与 norm_first 的 TransformerEncoderLayer 等价。
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: [N, L, D]；pad_mask: [N, L] True=忽略
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.nhead, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                     # [3, N, h, L, dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        att = att.masked_fill(pad_mask[:, None, None, :], float('-inf'))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(N, L, D)
        return self.proj(out)


class _ManualEncoderLayer(nn.Module):
    """pre-norm Transformer 编码层：x + Attn(LN(x))，x + FFN(LN(x))。"""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = _ManualAttention(d_model, nhead)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x), pad_mask)
        x = x + self.ffn(self.n2(x))
        return x


class TimelineEncoder(nn.Module):
    """目标家公开动作序列 -> 时序嵌入。

    输入 tokens [B, J, L, F] + mask [B, J, L]（True=有效）。
    2 层 pre-norm Transformer（d=64, 4 heads）+ attention pooling。
    空序列（如一局开始时的快照）输出全零嵌入。
    """

    def __init__(self, d_in: int, d_model: int = 64, nhead: int = 4,
                 nlayers: int = 2, max_len: int = 32, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_ManualEncoderLayer(d_model, nhead) for _ in range(nlayers)])
        self.attn_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, J, L, _ = tokens.shape
        if L > self.max_len:                        # 防御：超出预置位置编码长度
            raise ValueError(f'timeline length {L} > max_len {self.max_len}')
        # pos: [1, max_len, d] -> [1, L, d]，广播到 [B, J, L, d]
        h = self.proj(tokens) + self.pos[:, :L, :]
        h = h.reshape(B * J, L, -1)
        m = mask.reshape(B * J, L)
        # 全 padding 行会让 softmax 出 NaN：强制首位有效，输出时再把空行清零。
        empty = ~m.any(dim=-1, keepdim=True)
        m_safe = m | empty
        for layer in self.layers:
            h = layer(h, ~m_safe)
        # attention pooling（只在有效位上进行）
        score = (h * self.attn_query).sum(-1)                       # [B*J, L]
        score = score.masked_fill(~m, float('-inf'))
        w = torch.softmax(score, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        e = (h * w.unsqueeze(-1)).sum(dim=1)                        # [B*J, d]
        e = e * empty.logical_not()
        return e.reshape(B, J, -1)


class GroupedLinear(nn.Module):
    """逐目标家（相对座位）独立的线性层：权重 [J, in, out]，一次 einsum 完成。

    三家共享骨干但各有 H1/H2/H3 头——obs 的牌河/副露通道本就按相对座位排布，
    头的特化有据可依（与 v1 一致）。
    """

    def __init__(self, n_targets: int, d_in: int, d_out: int):
        super().__init__()
        self.n_targets = n_targets
        self.weight = nn.Parameter(torch.empty(n_targets, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_targets, d_out))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        bound = 1.0 / (d_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, J, ..., d_in] -> [B, J, ..., d_out]
        # 显式按维度数写 einsum（3-D 时空省略号语义有坑，4-D 时空省略号可用，
        # 统一显式写最稳）。
        if x.dim() == 3:                            # [B, J, d]
            return torch.einsum('bjd,jde->bje', x, self.weight) + self.bias
        if x.dim() == 4:                            # [B, J, N, d]
            return torch.einsum('bjnd,jde->bjne', x, self.weight) + self.bias[:, None, :]
        raise ValueError(f'unsupported input dim {x.dim()}')


class StructureHead(nn.Module):
    """结构估计头：7 个搭子构造计数分布 + 3 类形制 + 向听数（8 桶）。"""

    def __init__(self, d_in: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, 128))
        # 7 个结构计数 x 5 个 bin = 35 个 logit，再重塑回 [B,3,7,5]
        self.count_head = GroupedLinear(3, 128, NUM_STRUCT_COUNTS * COUNT_BINS)
        self.form_head = GroupedLinear(3, 128, NUM_FORMS)
        self.shanten_head = GroupedLinear(3, 128, MAX_SHANTEN_BUCKET + 1)

    def forward(self, z: torch.Tensor) -> dict:
        """z: [B, 3, d_in]（全局空间特征 g 广播 + 时序嵌入 e_T）。"""
        h = self.trunk(z)
        return {
            'count_logits': self.count_head(h).unflatten(
                -1, (NUM_STRUCT_COUNTS, COUNT_BINS)),      # [B,3,7,5]
            'form_logits': self.form_head(h),              # [B,3,3]
            'shanten_logits': self.shanten_head(h),        # [B,3,8]
        }


class ReadNetV2(nn.Module):
    """v2 前向：obs/unseen/wait_bias + tl_tokens/tl_mask -> 与 v1 一致的输出。"""

    def __init__(self, version: int = 4, conv_channels: int = 96, num_blocks: int = 12,
                 with_aux: bool = True, tl_features: int = 43, tl_len: int = 26,
                 d_time: int = 64, struct_shanten_belief: bool = False):
        super().__init__()
        self.version = version
        self.with_aux = with_aux
        self.tl_features = tl_features
        self.struct_shanten_belief = struct_shanten_belief
        in_channels = obs_shape(version)[0]

        # 空间流（与 v1 完全同构，保证可比）
        self.backbone = M.ReadBackbone(in_channels, conv_channels, num_blocks)
        self.C = conv_channels

        # 时序流（三家共享权重：token 语义与座位无关）
        self.timeline = TimelineEncoder(tl_features, d_model=d_time,
                                        max_len=max(tl_len, 32))

        # 结构流
        s_in = conv_channels + d_time
        self.struct = StructureHead(s_in)
        # 结构瓶颈维度：counts(35) + form(3) [+ v2s: 向听分布(8)]
        self.S = (NUM_STRUCT_COUNTS * COUNT_BINS + NUM_FORMS
                  + (MAX_SHANTEN_BUCKET + 1 if struct_shanten_belief else 0))

        # H1：逐列计数分配头（逐目标家）——输入 = 列特征 h + 时序嵌入 + 结构瓶颈
        h1_in = conv_channels + d_time + self.S
        self.h1 = GroupedLinear(3, h1_in, 128)
        self.h1_out = GroupedLinear(3, 128, 5)

        # H2：听牌头——输入 = 全局 g + 时序 + 结构瓶颈 + H1 期望张数
        h2_in = conv_channels + d_time + self.S + NUM_TILES
        self.h2 = GroupedLinear(3, h2_in, 128)
        self.h2_out = GroupedLinear(3, 128, NUM_WAIT_CLASSES)

        # H3：辅助头（下一打/向听/立直/鸣牌）——输入 = 全局 g + 时序 + 结构瓶颈
        self.h3 = GroupedLinear(3, s_in + self.S, 128)
        self.next_head = GroupedLinear(3, 128, NUM_WAIT_CLASSES)
        self.shanten_head = GroupedLinear(3, 128, MAX_SHANTEN_BUCKET + 1)
        self.riichi_head = GroupedLinear(3, 128, 1)
        self.called_head = GroupedLinear(3, 128, 1)

    def forward(self, obs, unseen, wait_bias, tl_tokens, tl_mask):
        B = obs.shape[0]
        h, g = self.backbone(obs)                             # [B,C,34], [B,C]
        e_T = self.timeline(tl_tokens, tl_mask)               # [B,3,dT]

        # ---- 结构估计（粗到细的第一步）----
        z = torch.cat([g[:, None, :].expand(B, 3, self.C), e_T], dim=-1)
        st = self.struct(z)
        s_counts = F.softmax(st['count_logits'], dim=-1).flatten(-2)    # [B,3,35]
        s_form = F.softmax(st['form_logits'], dim=-1)                   # [B,3,3]
        if self.struct_shanten_belief:
            # v2s：向听分布也进入瓶颈（H1/H2 直接消费显式向听信念）
            s_belief = torch.cat(
                [s_counts, s_form, F.softmax(st['shanten_logits'], dim=-1)], dim=-1)  # [B,3,46]
        else:
            s_belief = torch.cat([s_counts, s_form], dim=-1)            # [B,3,38]

        # ---- H1：逐列计数分配 ----
        # h: [B,C,34] -> [B,3,34,C]（三家共享列特征，头各自特化）
        h_j = h.unsqueeze(1).expand(B, 3, self.C, NUM_TILES).transpose(-1, -2)
        e_T_b = e_T[:, :, None, :].expand(B, 3, NUM_TILES, e_T.shape[-1])
        s_b = s_belief[:, :, None, :].expand(B, 3, NUM_TILES, self.S)
        x1 = torch.cat([h_j, e_T_b, s_b], dim=-1)                       # [B,3,34,*]
        x1 = F.gelu(self.h1(x1))
        alloc_logits = self.h1_out(x1)                                  # [B,3,34,5]

        # unseen 掩码：k > unseen 的位置置 -inf（列守恒上界）
        k_idx = torch.arange(5, device=obs.device, dtype=unseen.dtype)
        cap = unseen[:, None, :, None]                                  # [B,1,34,1]
        alloc_logits = alloc_logits.masked_fill(k_idx > cap, float('-inf'))
        alloc_probs = F.softmax(alloc_logits, dim=-1)
        alloc_probs = torch.nan_to_num(alloc_probs, nan=0.0)
        exp_counts = (alloc_probs * k_idx).sum(-1)                      # [B,3,34]

        # ---- H2：听牌头 ----
        x2 = torch.cat([z, s_belief, exp_counts], dim=-1)               # [B,3,*]
        x2 = F.gelu(self.h2(x2))
        wait_logits = self.h2_out(x2)
        # wait_bias 是逐牌 [B,3,34] 的加性偏置；第 35 类「未听」不受 tile 级
        # 掩码影响，补 0。
        wait_logits = wait_logits + F.pad(wait_bias, (0, 1))            # [B,3,35]
        # RM1（README §11.1）因子化读法：p34 = sigmoid(l34) = P(未听)，即
        # L_wait_tenpai BCE 直接标定的量；p_x = P(听)·softmax34(l_x)。
        # 旧 softmax35 读法把 34 个牌列 logit 的配分函数噪声混进 P(听)
        # （tenpai AUC 0.9461 → 0.9685，标定比值 6.26× → 0.998）。
        wait_probs = M.factorized_wait_probs(wait_logits)

        out = {
            'alloc_logits': alloc_logits,
            'alloc_probs': alloc_probs,
            'exp_counts': exp_counts,
            'wait_logits': wait_logits,
            'wait_probs': wait_probs,
            'struct_count_logits': st['count_logits'],
            'struct_form_logits': st['form_logits'],
            'struct_shanten_logits': st['shanten_logits'],
        }

        if self.with_aux:
            x3 = torch.cat([z, s_belief], dim=-1)
            x3 = F.gelu(self.h3(x3))
            out['next_logits'] = self.next_head(x3)
            out['shanten_logits'] = self.shanten_head(x3)
            out['riichi_logit'] = self.riichi_head(x3).squeeze(-1)
            out['called_logit'] = self.called_head(x3).squeeze(-1)
        return out


def build_model_v2(version: int = 4, conv_channels: int = 96, num_blocks: int = 12,
                   with_aux: bool = True, tl_features: int = 43,
                   tl_len: int = 26) -> ReadNetV2:
    return ReadNetV2(version=version, conv_channels=conv_channels,
                     num_blocks=num_blocks, with_aux=with_aux,
                     tl_features=tl_features, tl_len=tl_len)


def build_model_v2s(version: int = 4, conv_channels: int = 96, num_blocks: int = 12,
                    with_aux: bool = True, tl_features: int = 43,
                    tl_len: int = 26) -> ReadNetV2:
    """v2s = v2 + 向听分布注入结构瓶颈（结构头专项，见模块 docstring）。"""
    return ReadNetV2(version=version, conv_channels=conv_channels,
                     num_blocks=num_blocks, with_aux=with_aux,
                     tl_features=tl_features, tl_len=tl_len,
                     struct_shanten_belief=True)
