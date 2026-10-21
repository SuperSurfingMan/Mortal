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

**RM12（P1b，2026-10-21）**：``tl_mode='tile'`` 把时序流从 mean-pool 的
``[B,3,dT]`` 改为逐牌列 ``[B,3,dT,34]``——每个 token 在自身牌列 ±1/±2 散射
（chi/pon 的 consumed 也散射），配逐 token 真实巡目（dataset 的 tl_turn）与
``recency=exp(-age)`` 加权；H1 与 tile H2 头消费 ``e_col``，结构头/H3/tenpai
仍消费 pooled ``e_T``。``tl_mode='pooled'`` 是 A2b/A2c 旧路径，参数集与
数值逐位不变（旧 ckpt 直接加载）。
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

    ``tile_indexed``（RM12）：额外输出逐牌列时序图 ``[B,J,d,34]``。除 attention
    pooling 的全局 e_T 外，把逐 token 表示按牌列 ±1/±2 散射（chi/pon 的 consumed
    也散射），并拼上「recency 加权均值 + 邻域计数」，使 H1/H2 能直接看到
    「哪一列、第几巡、先切还是后切」。旧 ckpt 走 ``tile_indexed=False``，参数集
    与数值路径完全不变。

    ``order_features``（RM15/显式 order）：在 tile 图上再加**显式顺序特征**——
    每列 last-token 的 turn/age/embedding、邻域平均 turn、以及 ``turn[m−1]−
    turn[m+1]`` 的局部先后差。加权均值/recency 仍是软聚合；这些通道直接给
    「左邻先切还是右邻先切」一个可线性读取的坐标。旧 tile ckpt 走 False，
    ``tile_proj`` 形状不变。
    """

    def __init__(self, d_in: int, d_model: int = 64, nhead: int = 4,
                 nlayers: int = 2, max_len: int = 32, dropout: float = 0.0,
                 tile_indexed: bool = False, order_features: bool = False):
        super().__init__()
        assert not order_features or tile_indexed
        self.max_len = max_len
        self.d_model = d_model
        self.tile_indexed = tile_indexed
        self.order_features = bool(order_features)
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_ManualEncoderLayer(d_model, nhead) for _ in range(nlayers)])
        self.attn_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        if tile_indexed:
            # 逐 token 时间/上下文进入 Transformer（旧路径不加，保证兼容）
            self.turn_proj = nn.Linear(1, d_model)
            self.age_proj = nn.Linear(1, d_model)
            self.flag_proj = nn.Linear(2, d_model)
            self.tile_mlp = nn.Sequential(
                nn.Linear(d_model + 4, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            tile_in = ((3 * d_model + 5) if order_features else (2 * d_model + 1))
            self.tile_proj = nn.Linear(tile_in, d_model)
            self.tile_norm = nn.LayerNorm(d_model)
            # 散射权重：本列 1.0、±1 0.6、±2 0.3；consumed 本列 0.8、±1 0.3。
            # 同花色限制在 _scatter_weights 里做（9m 不散射到 1p）。
            self._offset_weights = ((0, 1.0), (-1, 0.6), (1, 0.6), (-2, 0.3), (2, 0.3))
            self._c_offset_weights = ((0, 0.8), (-1, 0.3), (1, 0.3))

    def _encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor,
                       turn=None, age=None, flags=None):
        """token 序列 -> (上下文表示 h [N,L,d], mask [N,L], empty [N,1])。"""
        B, J, L, _ = tokens.shape
        if L > self.max_len:                        # 防御：超出预置位置编码长度
            raise ValueError(f'timeline length {L} > max_len {self.max_len}')
        # pos: [1, max_len, d] -> [1, L, d]，广播到 [B, J, L, d]
        h = self.proj(tokens) + self.pos[:, :L, :]
        if self.tile_indexed:                        # RM12：真实巡目/衰减进 token
            h = (h + self.turn_proj(turn[..., None])
                 + self.age_proj(age[..., None]) + self.flag_proj(flags))
        h = h.reshape(B * J, L, -1)
        m = mask.reshape(B * J, L)
        # 全 padding 行会让 softmax 出 NaN：强制首位有效，输出时再把空行清零。
        empty = ~m.any(dim=-1, keepdim=True)
        m_safe = m | empty
        for layer in self.layers:
            h = layer(h, ~m_safe)
        return h, m, empty

    def _pool(self, h: torch.Tensor, m: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
        """attention pooling（只在有效位上进行；空序列输出零向量）。"""
        score = (h * self.attn_query).sum(-1)                       # [N, L]
        score = score.masked_fill(~m, float('-inf'))
        w = torch.softmax(score, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        e = (h * w.unsqueeze(-1)).sum(dim=1)                        # [N, d]
        return e * empty.logical_not()

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """旧路径（pooled e_T [B,J,d]），逐位保持 A2b/A2c 口径。"""
        B, J, L, _ = tokens.shape
        h, m, empty = self._encode_tokens(tokens, mask)
        return self._pool(h, m, empty).reshape(B, J, -1)

    def _scatter_weights(self, idx: torch.Tensor, mask: torch.Tensor,
                         consumed: torch.Tensor) -> torch.Tensor:
        """token 牌列 -> 逐列散射权重 [N,L,34]（同花色 ±2，字牌只本列）。"""
        N, L = idx.shape
        S = idx.new_zeros((N, L, NUM_TILES), dtype=torch.float32)
        valid = mask & (idx >= 0)
        for d, w in self._offset_weights:
            col = idx + d
            num = (idx >= 0) & (idx < 27)
            same = num & (col >= 0) & (col < 27) & ((col // 9) == (idx // 9))
            # 字牌只散射本列；用 OR 而不是 torch.where(Tensor, Tensor, Tensor)，
            # 避免 onnxruntime 的 Where 类型实现缺失（RM12 ONNX 导出必需）。
            honor_ok = (valid & (idx >= 27)) if d == 0 else torch.zeros_like(valid)
            ok = same | honor_ok
            S.scatter_add_(-1, col.clamp(0, NUM_TILES - 1).unsqueeze(-1),
                           (ok.to(S.dtype) * w).unsqueeze(-1))
        for c in range(consumed.shape[-1]):
            cidx = consumed[..., c]
            cvalid = mask & (cidx >= 0)
            for d, w in self._c_offset_weights:
                col = cidx + d
                num = (cidx >= 0) & (cidx < 27)
                same = num & (col >= 0) & (col < 27) & ((col // 9) == (cidx // 9))
                honor_ok = (cvalid & (cidx >= 27)) if d == 0 else torch.zeros_like(cvalid)
                ok = same | honor_ok
                S.scatter_add_(-1, col.clamp(0, NUM_TILES - 1).unsqueeze(-1),
                               (ok.to(S.dtype) * w).unsqueeze(-1))
        return S

    def encode_tile(self, tokens: torch.Tensor, mask: torch.Tensor,
                    turn: torch.Tensor, age: torch.Tensor,
                    consumed: torch.Tensor, flags: torch.Tensor):
        """RM12：返回 (pooled e_T [B,J,d], tile-indexed e [B,J,d,34])。"""
        B, J, L, _ = tokens.shape
        h, m, empty = self._encode_tokens(tokens, mask, turn, age, flags)
        e = self._pool(h, m, empty)

        idx = tokens[..., :NUM_TILES].argmax(-1).reshape(B * J, L)
        has = (tokens[..., :NUM_TILES].sum(-1) > 0.5).reshape(B * J, L)
        # reach token 无牌：置 -1（用算术而不是 torch.where，避免 ONNX Where）
        idx = idx * has.to(idx.dtype) - (~has).to(idx.dtype)
        v = self.tile_mlp(torch.cat([
            h, turn.reshape(B * J, L, 1), age.reshape(B * J, L, 1),
            flags.reshape(B * J, L, 2)], dim=-1))          # [N,L,d]
        S = self._scatter_weights(idx, m, consumed.reshape(B * J, L, -1))
        den = S.sum(dim=-2)                                 # [N,34]
        e_mean = (torch.einsum('nld,nlt->ndt', v, S)
                  / den.clamp(min=1.0).unsqueeze(1))
        # recency：exp(-age) 加权均值（age 归一化到 18 巡），让「先切/后切」可分
        w_rec = torch.exp(-age.reshape(B * J, L)) * m.to(v.dtype)
        S_rec = S * w_rec.unsqueeze(-1)
        den_rec = S_rec.sum(dim=-2)
        e_rec = (torch.einsum('nld,nlt->ndt', v, S_rec)
                 / den_rec.clamp(min=1.0).unsqueeze(1))
        cnt = torch.log1p(den).unsqueeze(1)                 # [N,1,34]
        if self.order_features:
            # 显式顺序：每列 last-token 的 turn/age/embedding + 邻域平均 turn
            # + turn[m−1]−turn[m+1] 的局部先后差。30× 温度让“最新 token”
            # 近似硬选（相邻巡目差 1/18≈0.055 ⇒ 权重比≈5），但保持可导。
            turn_flat = turn.reshape(B * J, L)
            age_flat = age.reshape(B * J, L)
            S_bin = (S > 0).to(v.dtype)                     # [N,L,34]
            has_col = (den > 0).to(v.dtype)                 # [N,34]
            score = turn_flat.unsqueeze(-1) * torch.ones_like(S) * 30.0
            score = score.masked_fill(S_bin == 0, -1e9)     # [N,L,34]
            w_last = torch.softmax(score, dim=1) * has_col.unsqueeze(1)
            last_turn = (w_last * turn_flat.unsqueeze(-1)).sum(1)   # [N,34]
            last_age = (w_last * age_flat.unsqueeze(-1)).sum(1)     # [N,34]
            last_v = torch.einsum('nlt,nld->ntd', w_last, v).transpose(1, 2)
            turn_mean = ((S * turn_flat.unsqueeze(-1)).sum(1)
                         / den.clamp(min=1.0))              # [N,34]
            # 同花色左右邻（m−1 / m+1）的 last-turn 差；边界/字牌/缺列置 0
            tm = last_turn
            left = F.pad(tm, (1, 0))[..., :NUM_TILES]
            right = F.pad(tm, (0, 1))[..., 1:]
            lhas = F.pad(has_col, (1, 0))[..., :NUM_TILES]
            rhas = F.pad(has_col, (0, 1))[..., 1:]
            midx = torch.arange(NUM_TILES, device=tm.device)
            num = (midx < 27).to(tm.dtype)
            lok = ((midx % 9) != 0).to(tm.dtype) * num
            rok = ((midx % 9) != 8).to(tm.dtype) * num
            valid = lhas * rhas * lok * rok
            turn_diff = (left - right) * valid              # [N,34]
            raw = torch.cat([
                e_mean, e_rec, last_v, cnt,
                turn_mean.unsqueeze(1), last_turn.unsqueeze(1),
                turn_diff.unsqueeze(1), last_age.unsqueeze(1)], dim=1)
            raw = raw.permute(0, 2, 1)                      # [N,34,3d+5]
        else:
            raw = torch.cat([e_mean, e_rec, cnt], dim=1).permute(0, 2, 1)
        e_tile = self.tile_proj(raw).permute(0, 2, 1)       # [N,d,34]
        e_tile = self.tile_norm(e_tile.transpose(1, 2)).transpose(1, 2)
        e_tile = e_tile * empty.logical_not().view(-1, 1, 1)
        return e.reshape(B, J, -1), e_tile.reshape(B, J, self.d_model, NUM_TILES)


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
    """v2 前向：obs/unseen/wait_bias + tl_tokens/tl_mask -> 与 v1 一致的输出。

    ``wait_head``（H2 听牌头参数化，RM4/RM5，README §11.1/§11.3）：
    ``flat`` = 旧 tile-agnostic 头（34 个 logit 经同一 128 维瓶颈）；
    ``tile`` = 逐牌列打分头（与 H1 同构，每列独立打分 + tenpai 标量头）。
    输出接口（含 ONNX）两种模式完全一致。
    """

    def __init__(self, version: int = 4, conv_channels: int = 96, num_blocks: int = 12,
                 with_aux: bool = True, tl_features: int = 43, tl_len: int = 26,
                 d_time: int = 64, struct_shanten_belief: bool = False,
                 wait_head: str = 'flat', tl_mode: str = 'pooled',
                 next_tedashi: bool = False, tile_order: bool = False):
        super().__init__()
        self.version = version
        self.with_aux = with_aux
        self.tl_features = tl_features
        self.struct_shanten_belief = struct_shanten_belief
        assert wait_head in ('flat', 'tile'), wait_head
        self.wait_head = wait_head
        assert tl_mode in ('pooled', 'tile'), tl_mode
        assert not tile_order or tl_mode == 'tile', tile_order
        self.tl_mode = tl_mode
        # RM15/显式 order：tile 图上加 last-turn/turn-diff/last-embedding 通道。
        # 旧 tile ckpt（config 无 tile_order）默认 False ⇒ tile_proj 形状不变。
        self.tile_order = bool(tile_order and tl_mode == 'tile')
        # RM14：next-tedashi 辅助头只在「tile 时序流 + aux」下存在；旧 tile
        # ckpt（config 无 next_tedashi_weight）默认 False ⇒ 参数集不变、严格加载。
        self.next_tedashi = bool(next_tedashi and with_aux and tl_mode == 'tile')
        in_channels = obs_shape(version)[0]

        # 空间流（与 v1 完全同构，保证可比）
        self.backbone = M.ReadBackbone(in_channels, conv_channels, num_blocks)
        self.C = conv_channels

        # 时序流（三家共享权重：token 语义与座位无关）
        # tl_mode='tile'（RM12）：额外输出逐牌列时序图 [B,3,dT,34]；
        # 'pooled'（A2b/A2c 旧 ckpt）：只输出全局 e_T [B,3,dT]，参数集不变。
        self.timeline = TimelineEncoder(tl_features, d_model=d_time,
                                        max_len=max(tl_len, 32),
                                        tile_indexed=(tl_mode == 'tile'),
                                        order_features=self.tile_order)

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

        # H2：听牌头（两种参数化，RM4/RM5，README §11.1/§11.3）
        if wait_head == 'tile':
            # RM5 逐牌列打分头（与 H1 同构）：每列输入
            # [h_x(C), e_T, s_belief(S), exp_counts_x, unseen_x, wait_bias_x]，
            # 逐列独立打分。证据（§11.3 冻结骨干探针）：旧 flat 头 tile-agnostic
            # （34 个 logit 只经一个 128 维瓶颈交互），只加逐列 h_x 即
            # cond_ce −0.152、hit@1 +35% 相对、立直 hit@1 +34% 相对；且探针
            # 1 epoch 就超过 A2b 端到端 43k 步。tenpai 标量走独立小头
            # （P(未听)=sigmoid(l34)，与 L_wait_tenpai BCE 口径一致=RM4）。
            self.h2 = GroupedLinear(3, conv_channels + d_time + self.S + 3, 128)
            self.h2_out = GroupedLinear(3, 128, 1)            # 每列一个 cond logit
            self.tenpai_head = GroupedLinear(3, s_in + self.S, 1)
        else:
            self.h2 = GroupedLinear(3, conv_channels + d_time + self.S + NUM_TILES, 128)
            self.h2_out = GroupedLinear(3, 128, NUM_WAIT_CLASSES)

        # H3：辅助头（下一打/向听/立直/鸣牌）——输入 = 全局 g + 时序 + 结构瓶颈
        self.h3 = GroupedLinear(3, s_in + self.S, 128)
        self.next_head = GroupedLinear(3, 128, NUM_WAIT_CLASSES)
        self.shanten_head = GroupedLinear(3, 128, MAX_SHANTEN_BUCKET + 1)
        self.riichi_head = GroupedLinear(3, 128, 1)
        self.called_head = GroupedLinear(3, 128, 1)
        if self.next_tedashi:
            # RM14：H1 逐列特征直接预测「目标家下一张手切牌」的 34 类 logit，
            # 无手切标量（第 35 类）由 H3 的 x3 提供——直接给 tile 时序通路压力。
            self.next_tedashi_head = GroupedLinear(3, 128, 1)
            self.next_tedashi_none = GroupedLinear(3, 128, 1)

    def tl_extra(self, batch):
        """RM12 侧车字段（pooled 模式返回空 dict，旧 ckpt/旧调用零改动）。"""
        if self.tl_mode != 'tile':
            return {}
        return {k: batch[k] for k in ('tl_turn', 'tl_age', 'tl_consumed', 'tl_flags')}

    def forward(self, obs, unseen, wait_bias, tl_tokens, tl_mask,
                tl_turn=None, tl_age=None, tl_consumed=None, tl_flags=None):
        B = obs.shape[0]
        h, g = self.backbone(obs)                             # [B,C,34], [B,C]
        if self.tl_mode == 'tile':
            # RM12：逐牌列时序图。e_T 仍保留给结构头/H3/tenpai（全局），
            # e_col 进 H1/H2——替换原来把 e_T 在 34 列上复制的广播。
            if tl_turn is None or tl_age is None or tl_consumed is None or tl_flags is None:
                raise ValueError('tl_mode="tile" 需要 tl_turn/tl_age/tl_consumed/'
                                 'tl_flags（RM12；dataset._emit 已提供）')
            e_T, e_tile = self.timeline.encode_tile(
                tl_tokens, tl_mask, tl_turn, tl_age, tl_consumed, tl_flags)
            e_col = e_tile.transpose(-1, -2)                  # [B,3,34,dT]
        else:
            e_T = self.timeline(tl_tokens, tl_mask)           # [B,3,dT]
            e_col = e_T[:, :, None, :].expand(B, 3, NUM_TILES, e_T.shape[-1])

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
        s_b = s_belief[:, :, None, :].expand(B, 3, NUM_TILES, self.S)
        x1 = torch.cat([h_j, e_col, s_b], dim=-1)                       # [B,3,34,*]
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
        if self.wait_head == 'tile':
            # RM5 逐列打分：列特征 = h_x + 全局（e_T/s_belief 广播）+ 列特定三项。
            # wait_bias 双通路：作输入特征（tanh 压到 [-1,0]——constraints 的
            # 值域是 {0, -inf}（硬掩码）/ {0, -1}（软振听），tanh 后三档可辨
            # 且有界，防 -inf 经 Linear 正权重侧放大成 inf/NaN）+ 直接加在
            # cond logit 上（保证硬掩码列概率严格为 0）。
            ec_j = exp_counts[..., None]                                # [B,3,34,1]
            un_j = unseen[:, None, :, None].expand(B, 3, NUM_TILES, 1)  # [B,3,34,1]
            bias_j = torch.tanh(wait_bias)[..., None]                   # [B,3,34,1]
            x2 = torch.cat([h_j, e_col, s_b, ec_j, un_j, bias_j], dim=-1)
            x2 = F.gelu(self.h2(x2))
            cond_logits = self.h2_out(x2).squeeze(-1) + wait_bias       # [B,3,34]
            tenpai_logit = self.tenpai_head(
                torch.cat([z, s_belief], dim=-1)).squeeze(-1)           # [B,3]
            wait_logits = torch.cat([cond_logits, tenpai_logit[..., None]], dim=-1)
        else:
            x2 = torch.cat([z, s_belief, exp_counts], dim=-1)           # [B,3,*]
            x2 = F.gelu(self.h2(x2))
            wait_logits = self.h2_out(x2)
            # wait_bias 是逐牌 [B,3,34] 的加性偏置；第 35 类「未听」不受 tile 级
            # 掩码影响，补 0。
            wait_logits = wait_logits + F.pad(wait_bias, (0, 1))        # [B,3,35]
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
            if self.next_tedashi:
                # [B,3,34] 逐列 + [B,3] 无手切 → [B,3,35]
                out['next_tedashi_logits'] = torch.cat([
                    self.next_tedashi_head(x1).squeeze(-1),   # [B,3,34]
                    self.next_tedashi_none(x3)], dim=-1)       # [B,3,1] → [B,3,35]
        return out


def build_model_v2(version: int = 4, conv_channels: int = 96, num_blocks: int = 12,
                   with_aux: bool = True, tl_features: int = 43,
                   tl_len: int = 26, wait_head: str = 'flat',
                   tl_mode: str = 'pooled', next_tedashi: bool = False,
                   tile_order: bool = False) -> ReadNetV2:
    return ReadNetV2(version=version, conv_channels=conv_channels,
                     num_blocks=num_blocks, with_aux=with_aux,
                     tl_features=tl_features, tl_len=tl_len,
                     wait_head=wait_head, tl_mode=tl_mode,
                     next_tedashi=next_tedashi, tile_order=tile_order)


def build_model_v2s(version: int = 4, conv_channels: int = 96, num_blocks: int = 12,
                    with_aux: bool = True, tl_features: int = 43, tl_len: int = 26,
                    struct_shanten_belief: bool = True,
                    wait_head: str = 'flat', tl_mode: str = 'pooled',
                    next_tedashi: bool = False,
                    tile_order: bool = False) -> ReadNetV2:
    """v2s = v2 + 向听分布注入结构瓶颈（结构头专项，见模块 docstring）。

    ``wait_head``：``flat``（**本 builder 的**默认）= 旧 H2（tile-agnostic，A2b
    及之前所有 ckpt 的架构）；``tile`` = RM5 逐牌列打分头（A2c 起）。旧 ckpt 加载
    必须显式/缺省用 ``flat``——两者 state_dict 键形状不兼容。

    ⚠️ 「默认」有二义（README §11.5 审计 M5，2026-10-20 注）：训练入口
    ``train.py --wait-head`` 的默认**已翻为 ``tile``**（A2c 起），故新训练不走本
    函数的默认值；本处保留 ``flat`` 只为旧 ckpt 兼容。评测/诊断脚本一律
    ``cfg.get('wait_head', 'flat')`` 从 ckpt 的 config 读取 ⇒ 旧 ckpt 自动 flat、
    新 ckpt 自动 tile（`load_state_dict` 严格模式，形状不符会响亮失败而非静默）。
    """
    return ReadNetV2(version=version, conv_channels=conv_channels,
                     num_blocks=num_blocks, with_aux=with_aux,
                     tl_features=tl_features, tl_len=tl_len,
                     struct_shanten_belief=struct_shanten_belief,
                     wait_head=wait_head, tl_mode=tl_mode,
                     next_tedashi=next_tedashi, tile_order=tile_order)
