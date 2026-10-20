"""A3 规则法基线（纯 numpy，零学习）——读牌模型的「vs 人类规则」增量参照。

> 对应读牌模型方案 §8.1 第 2 条与 MEMORY §3.12 下一步：「A3 规则基线（现物/筋/壁
> + 立直枚举 + 见逃排除）」。目的：给神经读牌模型（A2b，定为注入基线）提供一个
> **不学习、只靠经典人类读牌启发式**的参照系——A2b 的「增量」= A2b 指标 − 本基线
> 指标，回答「学到的东西比人类规则多多少」。

**输出与神经网络同构**（与 model.py / model_v2.py 的 alloc_probs / wait_probs 逐位
可比，可直接喂 evaluate.py 与 compare_bootstrap 口径的指标）：

    alloc_probs  [n, 3, 34, 5]   各家暗牌逐列计数分布 P(h_{j,i}=k)（k=0..4）
    wait_probs   [n, 3, 35]      听牌/放铳威胁分布（第 35 类 = 未听）
    exp_counts   [n, 3, 34]      期望张数（alloc_probs 的期望，对照/诊断用）

**输入全部是公开信息**（与注入路径将来从 Rust 状态可取的量一致，不含任何标签）：

    unseen        [n, 34]     4 - tiles_seen（观察者手牌 + 公共区之后的未见张数）
    river         [n, 3, 34]  各家本局全部舍牌（振听/现物/筋的硬软规则来源）
    riichi        [n, 3]      各家是否已立直 accepted
    own_hand      [n, 34]     观察者手牌（决定「可表达」：4 张全见时自家能点炮）
    turn          [n]         观察者 at_turn（巡目）
    tiles_left    [n]         牌山剩余
    called        [n, 3]      各家是否已鸣牌
    meld_counts   [n, 3]      各家吃/碰/大明杠/暗杠数（公开副露）→ 手牌数 = 13 - 3m
    tl_tokens     [n, 3, L, 43]  各家公开动作时序（v2 时序流同款 token：牌 one-hot
                               + 摸切标志 + 动作类别 + 归一化巡目）
    tl_mask       [n, 3, L]   有效 token 位置

> ⚠️ **A3 不是「顺序盲」预测器**（2026-10-20 审计，README §11.5-⑩）：`tl_tokens`
> 经 `early_discard_mask`（前 3 次舍牌）与 `timeline_features`（末动作、连续摸切尾长、
> 舍牌数/幺九占比）进入 alloc 与听牌打分，故 A3 对切牌**顺序**有真实响应。
> 把它当作 §12.4 的「组成混杂地板」是高估了地板的纯净度；需要真顺序盲参照时用
> `rule_predict(batch, order_blind=True)`（token 规范化排序，破坏顺序保留多重集）。
    dora          [n, 34] 或 None  当前宝牌**指示牌**计数（dataset.dora_indicators；
                                 可选，缺失时退化为无宝牌先验）

**规则集**（权重全部为常数先验，出处见 `calibrate_a3.py` 在非评估月 202508 上的
标定测量——「经典人类规则 × 数据证伪/证实」；评估月 202609 是严格 holdout。
「软」= 乘性权重，「硬」= 支撑集剪除）：

 硬规则（人类规则的确定性部分）：
  1. 现物/振听（见逃排除）：目标家本局河中的牌 → 荣和概率 0。自家舍牌即振听，
     对任意目标家都是硬事实（不分立直与否；立直宣言不允许振听，故立直后河牌
     同样排除——「立直后摸到听牌却见逃」的极罕见情形与标签口径一致按无可荣牌处理）。
  2. 不可表达：4 张全可见且不在观察者手中 → 无人能打出 → 概率 0（与 constraints.
     compute_wait_bias / dataset 的 wait 标签同口径）。
  3. 立直枚举：立直家的可能听牌集 = 全部满足「牌池 ∪ {t} 中存在含 t 的完整和牌形」
     的 t（normal 4 面子+雀首 / 七对 / 国士 三条路径全查，面子数 = 4-副露数）。
     牌池 = unseen 多重集（目标家暗牌 ⊆ 牌池是硬事实），这是信息论上最紧的硬
     支撑集；真实听牌必在其中（回归测试用暴力重算守门）。
  4. 手牌数守恒：alloc 的行期望 = 13 - 3*副露数（公开信息精确推导）。

 软规则（乘性危险/倾向权重；括号为 202508 标定值，评估月不参与标定）：
  5. 筋（suji）：目标家**前 3 次舍牌**的 ±3 → ×0.32；立直家 → ×0.14（经典理论
     最强处）。晚切 ±3 无效应（条件于听牌后实测归 1；其表观危险全是巡目混杂）。
  6. 价值先验：宝牌 ×1.60、宝牌 ±1 ×1.15、中张两端 ×0.85；**字牌强巡目依赖**——
     初盘（turn≤3）字牌单骑常见 ×1.05，中盘 ×0.30，后盘 ×0.20（「字牌危险」
     只成立于初盘；中后盘字牌等待极少，是本文档最有反直觉色彩的证伪）。
  7. 壁（kabe）：未见 1 张 ×0.75、2 张 ×0.85、3 张 ×0.90（条件于听牌后支持
     较弱，取温和值——「最后一张在牌山」对**持有率**很强（×0.57），对**等待**
     只是弱效应）。
  8. 邻接先验：舍牌同色 ±1/±2 ×0.90（条件于听牌后弱负相关；alloc 侧早切邻牌
     ×0.45 是强效应——切张说明周围形状已散）。
  9. 听牌先验（非立直）：分段线性巡目先验（实测听牌率-巡目饱和曲线）+
     鸣牌 ×1.10/个（副露家常接近听牌）+ 连续摸切尾长 ×0.35（终盘摸切 tell）+
     全简单色 flow ×0.80 + 字牌迟迟不打 ×0.35。立直家 → P(听) = 1。
 10. alloc 倾向：字牌 ×0.45、中张两端 ×0.65、宝牌 ×1.15、已舍 ×0.38、
     早切邻牌 ×0.45、未见张数梯度 ×0.57/0.82/0.96/1.0；转为逐列二项分布
     （与 evaluate.binomial_baseline 同式；flat=True 时与其逐位相等）。

实现要点：除立直枚举（逐 (样本, 目标家) 的递归存在性检验）外全部 numpy 向量化；
枚举用「首个非零牌型 + 分支限界」的经典和牌形分解搜索，带节点预算（预算耗尽时
保守返回 True——支撑集保持真实听牌的超集，不制造结构性错判）。
"""
from __future__ import annotations

from math import comb

import numpy as np

NUM_TILES = 34
NUM_COUNTS = 5                       # k = 0..4
NUM_WAIT_CLASSES = NUM_TILES + 1     # 34 种听牌 + 未听
NUM_TARGETS = 3
NOT_TENPAI = NUM_TILES               # wait_probs 的「未听」类索引

#: 幺九牌（字牌 + 中张两端 1/9）
YAOCHUU = np.array([t for t in range(NUM_TILES)
                    if t >= 27 or t % 9 in (0, 8)], dtype=np.int64)
YAOCHUU_SET = set(YAOCHUU.tolist())
YAOCHUU_MASK_F = np.zeros(NUM_TILES, dtype=np.float64)
YAOCHUU_MASK_F[YAOCHUU] = 1.0
HONOR_MASK = np.zeros(NUM_TILES, dtype=bool)
HONOR_MASK[27:] = True
TERMINAL_MASK = np.zeros(NUM_TILES, dtype=bool)
for _t in range(27):
    if _t % 9 in (0, 8):
        TERMINAL_MASK[_t] = True

#: tl_tokens 的动作类别布局（与 dataset.TL_ACTION_KINDS 一致）
TL_ACTION_KINDS = ('dahai', 'reach', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan')
TL_IDX_DAHAI = TL_ACTION_KINDS.index('dahai')
TL_FEATURES = NUM_TILES + 1 + len(TL_ACTION_KINDS) + 1   # 43

# --------------------------------------------------------------------------- #
# 软规则权重（先验常数；出处见 calibrate_a3.py 在 202508 的标定测量，评估月 202609
# 是严格 holdout。调整权重需重跑标定并在 README 记录，属「改规则」而非调参）
# --------------------------------------------------------------------------- #
# alloc 倾向（持有率 per-tile 比率，相对中张=1.0；标定值见括号）
W_HONOR_ALLOC = 0.45       # 字牌持有率仅为中张的 45%（中张才是搭子素材）(0.449)
W_TERMINAL_ALLOC = 0.65    # 中张两端 65%（早期保留一部分，仍低于中张）(0.653)
W_DORA_ALLOC = 1.15        # 宝牌保留 (1.142)
W_RIVER_ALLOC = 0.38       # 已舍过的牌更不可能还在手里 (0.383)
W_ADJ_EARLY_ALLOC = 0.45   # 早切邻牌持有率仅 45%（切张说明周围形状已散）(0.442)
W_U1_ALLOC = 0.57          # 未见 1 张（3 张已见）多在牌山 (0.570)
W_U2_ALLOC = 0.82          # 未见 2 张部分在牌山 (0.820)
W_U3_ALLOC = 0.96          # 未见 3 张轻微壁效应 (0.963)
# wait 危险度（有效等待 per-tile 比率，条件于听牌；相对中张=1.0）
W_TERMINAL_WAIT = 0.85     # 1/9 单骑/边张等待较少 (0.854)
# 字牌：强巡目依赖——初盘字牌单骑常见（turn<=3 时与中牌相当），中后盘急降
W_HONOR_WAIT_EARLY = 1.05  # turn<=3 (≈1.05)
W_HONOR_WAIT_MID = 0.30    # 4<=turn<=7 (0.29)
W_HONOR_WAIT_LATE = 0.20   # turn>=8 (0.28→0.07 取中)
W_DORA_WAIT = 1.60         # 宝牌等待常见 (1.613)
W_DORA_ADJ_WAIT = 1.15     # 宝牌同色 ±1（留宝延申到邻张）(1.123)
W_SUJI = 0.32              # 单筋（仅早切前 3 次舍牌有效）(0.321)
W_SUJI_RICHI = 0.14        # 立直家的早切筋（经典理论最强处）(0.138)
W_ADJ_WAIT = 0.90          # 舍牌 ±1/±2：条件于听牌后弱负相关 (0.89~0.91)
W_KABE_U1 = 0.75           # 壁：未见 1 张略安全（条件化后支持较弱，取温和值）
W_KABE_U2 = 0.85
W_KABE_U3 = 0.90
# 听牌逻辑回归（非立直；log-odds 项，出处为条件于巡目桶的测量）
# 巡目先验为分段线性（测量到的听牌率-巡目曲线饱和形状）；整条曲线相对原始测量
# 下移 0.8 logit——tell 特征（鸣牌/摸切尾）与巡目正相关，直接相加会重复计数
# （标定实测：仅巡目先验已匹配基准率，加特征后均值过热 ~1.7 倍）。
TENPAI_TURN_KNOTS = ((0.0, -6.4), (3.0, -4.6), (6.0, -2.96),
                     (10.0, -1.74), (15.0, -1.42), (18.0, -1.40))
TENPAI_W_CALLED = 1.10         # 鸣牌（副露家常接近听牌；log-odds 1.0~1.4）(1.1)
TENPAI_W_CALLED_EXTRA = 0.30   # 第 2/3 个副露的附加
TENPAI_W_TG_TAIL = 0.35        # 连续摸切尾长（cap 3；终盘摸切=做牌推进 tell）
TENPAI_W_ALL_SIMPLES = 0.80    # 全简单色 flow（一路打 Simple 的做牌推进相）
TENPAI_W_HONOR_KEPT = 0.35     # 字牌迟迟不打（<=2 类）→ 持字对危险

#: 立直枚举的节点预算（耗尽则保守判「可能」——保持真实听牌的超集）
ENUM_NODE_BUDGET = 30000


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def dora_successor(tids: np.ndarray) -> np.ndarray:
    """宝牌指示牌 tid → 实际宝牌 tid（9m→1m 回绕；字牌 E→S→W→N→E、P→F→C→P）。"""
    tids = np.asarray(tids, dtype=np.int64)
    out = tids.copy()
    suit = tids // 9
    num = tids % 9
    wrap_suit = (suit < 3) & (num == 8)
    out = np.where(wrap_suit, tids - 8, tids + 1)
    # 字牌循环：27→28→29→30→27；31→32→33→31
    e_cycle = (tids == 30)
    out = np.where(e_cycle, 27, out)
    p_cycle = (tids == 33)
    out = np.where(p_cycle, 31, out)
    return out


def indicators_to_dora(indicator_counts: np.ndarray) -> np.ndarray:
    """宝牌指示牌计数 [n,34] → 实际宝牌计数 [n,34]（同一被指示牌可多次出现）。"""
    indicator_counts = np.asarray(indicator_counts, dtype=np.int64)
    if indicator_counts.ndim == 1:
        indicator_counts = indicator_counts[None, :]
    n = indicator_counts.shape[0]
    out = np.zeros((n, NUM_TILES), dtype=np.int64)
    for tid in range(NUM_TILES):
        c = indicator_counts[:, tid]
        if c.any():
            out[:, int(dora_successor(np.array([tid]))[0])] += c
    return out


def _shift_within_suit(mask: np.ndarray, offset: int) -> np.ndarray:
    """把 [..., 34] 的布尔掩码在同色内平移 offset 列（0<=j+offset<=8）。"""
    out = np.zeros_like(mask)
    for s in range(3):
        base = s * 9
        for j in range(9):
            jj = j + offset
            if 0 <= jj <= 8:
                out[..., base + jj] |= mask[..., base + j]
    return out


def _dilate_suit(mask: np.ndarray, offsets=(-2, -1, 1, 2)) -> np.ndarray:
    """同色 ±offsets 膨胀（邻接先验用）。"""
    out = np.zeros_like(mask)
    for off in offsets:
        out |= _shift_within_suit(mask, off)
    return out


_COMB_TABLE = np.array([[float(comb(u, k)) if k <= u else 0.0
                         for k in range(NUM_COUNTS)] for u in range(NUM_COUNTS)],
                       dtype=np.float64)                          # [5,5] C(u,k)


# --------------------------------------------------------------------------- #
# alloc：倾向加权二项（与 evaluate.binomial_baseline 同式，w≡1 时逐位相等）
# --------------------------------------------------------------------------- #
def canonicalize_tl_order(tl_tokens: np.ndarray, tl_mask: np.ndarray):
    """把每 (样本, 目标家) 的有效 token 按 (牌, 动作, 摸切) 稳定排序。

    **破坏顺序、保留多重集** ⇒ A3 的全部时序派生量（`early_discard_mask` 的前 3 次
    舍牌、`timeline_features` 的末动作/连续摸切尾长等）都退化为「舍牌多重集」的函数，
    即构造出一个**真顺序盲**的 A3 变体。用途：README §11.5-⑩ / AUDIT_20261020 S1——
    A3 原样版消费 `tl_tokens`，其格内 ΔΔ 含真实顺序响应，不能当「组成混杂地板」；
    本变体给出的才是纯组成地板，两者之差 = A3 自身的顺序敏感度。

    排序键相同的 token 在「牌/动作/摸切」三项上完全相同（第 43 维归一化巡目是
    快照常数，见 dataset.py:290 的已知缺陷），故稳定排序的并列次序不影响任何下游量。
    无效（padding）位置键设为 +inf 排到末尾，与左对齐的 `tl_mask` 保持一致。
    """
    tl_tokens = np.asarray(tl_tokens, dtype=np.float64).copy()
    tl_mask = np.asarray(tl_mask, dtype=bool)
    tile_oh = tl_tokens[..., :NUM_TILES]
    act = tl_tokens[..., NUM_TILES + 1:NUM_TILES + 1 + len(TL_ACTION_KINDS)]
    tg = tl_tokens[..., NUM_TILES]
    # reach 事件 tid=-1 ⇒ one-hot 全 0，用 NUM_TILES 作为其牌种编号
    tile_id = np.where(tile_oh.any(-1), tile_oh.argmax(-1), NUM_TILES)
    act_id = np.where(act.any(-1), act.argmax(-1), len(TL_ACTION_KINDS))
    key = (tile_id * 16 + act_id * 2 + (tg > 0.5)).astype(np.float64)
    key = np.where(tl_mask, key, np.inf)
    order = np.argsort(key, axis=-1, kind='stable')
    return np.take_along_axis(tl_tokens, order[..., None], axis=-2), tl_mask


def early_discard_mask(tl_tokens: np.ndarray, tl_mask: np.ndarray,
                       k: int = 3) -> np.ndarray:
    """各目标家前 k 次舍牌的牌种集合 [n,3,34]（早切 = 配牌形状的直接信息）。

    筋牌理论只对**早切**成立（标定实测：早切筋 ×0.32，晚切筋 ×1.0）；
    晚切 ±3 的「危险」完全是听牌基率随巡目上升的混杂（条件于听牌后归 1）。
    """
    tl_tokens = np.asarray(tl_tokens, dtype=np.float64)
    tl_mask = np.asarray(tl_mask, dtype=bool)
    n = tl_tokens.shape[0]
    dahai = tl_tokens[..., 35] > 0.5
    tile_oh = tl_tokens[..., :NUM_TILES]
    early = np.zeros((n, NUM_TARGETS, NUM_TILES), dtype=bool)
    L = tl_mask.shape[-1]
    for s in range(n):
        for j in range(NUM_TARGETS):
            c = 0
            for p in range(L):
                if tl_mask[s, j, p] and dahai[s, j, p]:
                    early[s, j, int(np.argmax(tile_oh[s, j, p]))] = True
                    c += 1
                    if c >= k:
                        break
    return early


def rule_alloc_probs(
    unseen: np.ndarray,          # [n, 34]
    river: np.ndarray,           # [n, 3, 34] bool
    hand_sizes: np.ndarray,      # [n, 3]
    tiles_left: np.ndarray,      # [n]
    dora: np.ndarray | None = None,   # [n, 34] 宝牌计数
    early_river: np.ndarray | None = None,   # [n,3,34] 前 3 次舍牌掩码
    *,
    flat: bool = False,          # True → 权重全 1（= 均匀二项基线，测试不动点用）
) -> np.ndarray:
    """规则法暗牌计数分布 [n, 3, 34, 5]。

    P(h_{j,i}=k) = C(u_i, k) p_{j,i}^k (1-p_{j,i})^(u_i-k) / Z，
    p_{j,i} = hand_j·w_{j,i} / (Σ_q hand_q·w_{q,i} + tiles_left)，
    w_{j,i} 为规则倾向权重（见模块头规则 10；各因子**乘性复合**，顺序无关）。
    分母含义：牌种 i 的未见副本在三家手牌（按各自倾向加权）与牌山间的软分配；
    w≡1 时逐位退化为 evaluate.binomial_baseline。行期望 Σ_i E[h_{j,i}] 不强制
    等于 hand_j（与均匀基线同性质的软约束；硬约束由 constraints.apply_constraints
    承担，模型侧同样如此）。
    """
    unseen = np.asarray(unseen, dtype=np.float64)
    river = np.asarray(river, dtype=bool)
    hand_sizes = np.asarray(hand_sizes, dtype=np.float64)
    tiles_left = np.asarray(tiles_left, dtype=np.float64)
    n = unseen.shape[0]
    u = np.clip(np.round(unseen), 0, 4).astype(np.int64)          # [n,34]
    u3 = np.broadcast_to(u[:, None, :], (n, NUM_TARGETS, NUM_TILES))

    if flat:
        w = np.ones((n, NUM_TARGETS, NUM_TILES), dtype=np.float64)
    else:
        w = np.ones((n, NUM_TARGETS, NUM_TILES), dtype=np.float64)
        honor = np.broadcast_to(HONOR_MASK, (NUM_TARGETS, NUM_TILES))
        term = np.broadcast_to(TERMINAL_MASK & ~HONOR_MASK, (NUM_TARGETS, NUM_TILES))
        w = w * np.where(honor[None], W_HONOR_ALLOC, 1.0)
        w = w * np.where(term[None], W_TERMINAL_ALLOC, 1.0)
        if dora is not None:
            dora = np.asarray(dora, dtype=np.float64)
            w = w * np.where(dora[:, None, :] > 0, W_DORA_ALLOC, 1.0)
        w = w * np.where(river, W_RIVER_ALLOC, 1.0)
        if early_river is not None:
            early_river = np.asarray(early_river, dtype=bool)
            adj_e = _dilate_suit(early_river) & ~river
        else:
            adj_e = np.zeros_like(river)
        w = w * np.where(adj_e, W_ADJ_EARLY_ALLOC, 1.0)
        # 未见张数梯度（壁：3 张已见的最后 1 张多在牌山）
        w = w * np.select(
            [u3 == 1, u3 == 2, u3 == 3],
            [W_U1_ALLOC, W_U2_ALLOC, W_U3_ALLOC], default=1.0)
    w = np.where(u3 <= 0, 0.0, w)                                  # 未见 0 → 不可能持有

    # 分母：Σ_q hand_q·w_{q,i}（三家按各自倾向）+ 牌山（对牌种 i 的软分配）
    denom = (hand_sizes[..., None] * w).sum(1) + tiles_left[:, None]   # [n,34]
    p = np.clip(hand_sizes[..., None] * w / np.maximum(denom[:, None, :], 1e-9),
                0.0, 1.0)                                         # [n,3,34]

    k = np.arange(NUM_COUNTS, dtype=np.float64)                   # [5]
    cu = _COMB_TABLE[u3]                                          # [n,3,34,5]
    expo = (u3[..., None] - k).clip(min=0)                        # 防 0*inf=NaN
    probs = cu * np.power(p[..., None], k) * np.power(1.0 - p[..., None], expo)
    probs = np.where(k[None, None, None, :] <= u3[..., None], probs, 0.0)
    probs = probs / probs.sum(-1, keepdims=True).clip(min=1e-12)
    return probs.astype(np.float32)


def expected_counts(alloc_probs: np.ndarray) -> np.ndarray:
    """alloc_probs [n,3,34,5] → 期望张数 [n,3,34]。"""
    alloc_probs = np.asarray(alloc_probs, dtype=np.float64)
    k = np.arange(NUM_COUNTS, dtype=np.float64)
    return (alloc_probs * k.reshape((1, 1, 1, -1))).sum(-1)


# --------------------------------------------------------------------------- #
# 公开行为时序特征（立直家之外，全部来自 tl_tokens：摸切/手切/巡目是明信息）
# --------------------------------------------------------------------------- #
def timeline_features(tl_tokens: np.ndarray, tl_mask: np.ndarray) -> dict:
    """从 v2 时序 token 提取听牌先验特征。返回 {name: [n,3]}。"""
    tl_tokens = np.asarray(tl_tokens, dtype=np.float64)
    tl_mask = np.asarray(tl_mask, dtype=bool)
    assert tl_tokens.shape[-1] == TL_FEATURES, tl_tokens.shape
    tile_oh = tl_tokens[..., :NUM_TILES]                    # [n,3,L,34]
    tg_flag = tl_tokens[..., NUM_TILES]                     # [n,3,L] 摸切标志
    act = tl_tokens[..., NUM_TILES + 1:NUM_TILES + 1 + len(TL_ACTION_KINDS)]
    dahai = act[..., TL_IDX_DAHAI]                          # [n,3,L]
    tg_disc = tg_flag * dahai                               # 只计打牌的摸切

    D = (tile_oh * dahai[..., None]).sum(-2)                # [n,3,34] 舍牌计数
    n_disc = dahai.sum(-1)                                  # [n,3]
    yaochuu_frac = D[..., YAOCHUU].sum(-1) / np.maximum(n_disc, 1.0)
    honor_types = (D[..., 27:] > 0).sum(-1)                 # [n,3]

    # 末个有效 token 的摸切（非打牌 token 的 tg_flag 恒 0）
    L = tl_mask.shape[-1]
    has_tok = tl_mask.any(-1)                               # [n,3]
    last_idx = L - 1 - np.argmax(tl_mask[..., ::-1], axis=-1)
    last_idx = np.clip(last_idx, 0, L - 1)
    last_tg = np.take_along_axis(tg_disc, last_idx[..., None], axis=-1)[..., 0]
    last_tg = np.where(has_tok, last_tg, 0.0)

    # 连续摸切尾长：把有效 token 右对齐后，自尾端反向 cumprod 连续 True 的长度。
    # （token 左对齐存放，直接反转会把 padding 带到 cumprod 头部导致恒 0——先右对齐）
    lens = tl_mask.sum(-1)                                       # [n,3]
    pos = np.arange(L)
    idx = pos[None, None, :] - (L - lens)[..., None]             # 右对齐后的原索引
    valid = idx >= 0
    ra = np.take_along_axis(tg_disc, np.clip(idx, 0, L - 1), axis=-1) * valid
    tg_tail = np.cumprod(ra[..., ::-1], axis=-1).sum(-1)

    return {
        'n_disc': n_disc,
        'yaochuu_frac': yaochuu_frac,
        'honor_types_discarded': honor_types.astype(np.float64),
        'last_tg': last_tg,
        'tg_tail': tg_tail.astype(np.float64),
        'all_simples': ((n_disc >= 4) & (yaochuu_frac <= 0)).astype(np.float64),
        'discard_counts': D,
    }


def _turn_prior(turn: np.ndarray) -> np.ndarray:
    """听牌先验的巡目项：分段线性插值 TENPAI_TURN_KNOTS（标定测量的饱和曲线）。"""
    turn = np.asarray(turn, dtype=np.float64)
    xs = np.array([k[0] for k in TENPAI_TURN_KNOTS])
    ys = np.array([k[1] for k in TENPAI_TURN_KNOTS])
    return np.interp(turn, xs, ys)


def rule_tenpai_prob(
    feat: dict,
    riichi: np.ndarray,            # [n, 3]
    turn: np.ndarray,              # [n]
    called: np.ndarray,            # [n, 3]
    meld_counts: np.ndarray | None = None,   # [n, 3] 副露数（called 的细化）
) -> np.ndarray:
    """规则法听牌概率 [n,3]（分段线性巡目先验 + 经典 tell；立直家硬置 1）。"""
    riichi = np.asarray(riichi, dtype=np.float64)
    called = np.asarray(called, dtype=np.float64)
    turn = np.asarray(turn, dtype=np.float64)
    t = _turn_prior(turn)[:, None]                          # [n,1]
    z = t + TENPAI_W_CALLED * called + TENPAI_W_TG_TAIL * np.minimum(feat['tg_tail'], 3.0)
    if meld_counts is not None:
        z = z + TENPAI_W_CALLED_EXTRA * np.maximum(
            np.asarray(meld_counts, dtype=np.float64) - 1.0, 0.0)
    z = z + TENPAI_W_ALL_SIMPLES * feat['all_simples']
    z = z + TENPAI_W_HONOR_KEPT * ((feat['honor_types_discarded'] <= 2)
                                   & (feat['n_disc'] >= 4))
    p = 1.0 / (1.0 + np.exp(-z))
    return np.where(riichi > 0.5, 1.0, p)


# --------------------------------------------------------------------------- #
# wait：规则危险度（软权重）+ 硬支撑集
# --------------------------------------------------------------------------- #
def rule_wait_weights(
    unseen: np.ndarray,            # [n, 34]
    river: np.ndarray,             # [n, 3, 34]
    own_hand: np.ndarray,          # [n, 34]
    dora: np.ndarray | None = None,
    turn: np.ndarray | None = None,   # [n] 巡目（字牌危险度的巡目分段）
    early_river: np.ndarray | None = None,   # [n,3,34] 前 3 次舍牌（筋只对早切成立）
    riichi: np.ndarray | None = None,        # [n,3] 立直标志（立直家的筋更强）
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (weights [n,3,34], support [n,3,34] bool)。

    support = 可表达（未见>0 或自家有）& 非目标家河牌（现物/振听硬规则）。
    weights 为乘性复合的危险度（见模块头规则 5-8；顺序无关），support 外不保证为 0
    （调用方负责掩掉）。
    """
    unseen = np.asarray(unseen, dtype=np.float64)
    river = np.asarray(river, dtype=bool)
    own_hand = np.asarray(own_hand, dtype=bool)
    n = unseen.shape[0]
    u = np.clip(np.round(unseen), 0, 4).astype(np.int64)
    u3 = np.broadcast_to(u[:, None, :], (n, NUM_TARGETS, NUM_TILES))

    expressible = (u > 0) | own_hand                          # [n,34]
    support = expressible[:, None, :] & ~river                # [n,3,34]

    w = np.ones((n, NUM_TARGETS, NUM_TILES), dtype=np.float64)
    honor = np.broadcast_to(HONOR_MASK, (NUM_TARGETS, NUM_TILES))
    term = np.broadcast_to(TERMINAL_MASK & ~HONOR_MASK, (NUM_TARGETS, NUM_TILES))
    # 字牌：巡目分段（初盘单骑常见 → 中后盘急降；标定实测的强巡目依赖）
    if turn is not None:
        t = np.asarray(turn, dtype=np.float64)
        honor_w = np.select([t <= 3, t <= 7],
                            [W_HONOR_WAIT_EARLY, W_HONOR_WAIT_MID],
                            default=W_HONOR_WAIT_LATE)[:, None, None]   # [n,1,1]
    else:
        honor_w = W_HONOR_WAIT_MID
    w = w * np.where(honor[None], honor_w, 1.0)
    w = w * np.where(term[None], W_TERMINAL_WAIT, 1.0)
    if dora is not None:
        dora = np.asarray(dora, dtype=np.float64)
        dora3 = np.broadcast_to(dora[:, None, :] > 0, (n, NUM_TARGETS, NUM_TILES))
        w = w * np.where(dora3, W_DORA_WAIT, 1.0)
        dora_adj = _dilate_suit(dora[:, None, :] > 0)         # [n,1,34] 宝牌 ±1/±2
        w = w * np.where(dora_adj & ~dora3, W_DORA_ADJ_WAIT, 1.0)
    # 壁：未见张数梯度（条件于听牌后支持较弱，取温和值）
    w = w * np.select([u3 == 1, u3 == 2, u3 == 3],
                      [W_KABE_U1, W_KABE_U2, W_KABE_U3], default=1.0)
    # 筋：只对前 3 次舍牌（早切）成立；立直家的早切筋显著更强
    if early_river is not None:
        suji_e = _dilate_suit(np.asarray(early_river, dtype=bool), offsets=(-3, 3))
        suji_e = suji_e & ~river
        if riichi is not None:
            ri3 = np.broadcast_to(np.asarray(riichi, dtype=bool)[:, :, None],
                                  (n, NUM_TARGETS, NUM_TILES))
            w = w * np.where(suji_e & ri3, W_SUJI_RICHI, 1.0)
            w = w * np.where(suji_e & ~ri3, W_SUJI, 1.0)
        else:
            w = w * np.where(suji_e, W_SUJI, 1.0)
    # 邻接先验：舍牌同色 ±1/±2（条件于听牌后弱负相关）
    adj = _dilate_suit(river)
    w = w * np.where(adj & ~river, W_ADJ_WAIT, 1.0)
    return w, support


# --------------------------------------------------------------------------- #
# 立直枚举（硬支撑集）：牌池 ∪ {t} 是否存在含 t 的完整和牌形
# --------------------------------------------------------------------------- #
def _can_form(counts: list[int], n_sets: int, need_pair: bool,
              budget: list[int]) -> bool:
    """多重集 counts 是否包含 n_sets 个面子（+1 雀首 if need_pair）作为子集。

    经典分支限界：按牌型序处理首个非零牌，枚举 刻子/顺子/雀首/整叠跳过 四种分支。
    预算耗尽保守返回 True（支撑集保持真实听牌的超集）。
    """
    budget[0] -= 1
    if budget[0] <= 0:
        return True
    need = 3 * n_sets + (2 if need_pair else 0)
    if sum(counts) < need:
        return False
    if n_sets == 0:
        if need_pair:
            return any(c >= 2 for c in counts)
        return True
    i = 0
    while i < NUM_TILES and counts[i] == 0:
        i += 1
    if i == NUM_TILES:
        return False
    # 刻子
    if counts[i] >= 3:
        counts[i] -= 3
        ok = _can_form(counts, n_sets - 1, need_pair, budget)
        counts[i] += 3
        if ok:
            return True
    # 顺子（仅数牌，i%9 <= 6）
    if i < 27 and (i % 9) <= 6 and counts[i + 1] > 0 and counts[i + 2] > 0:
        counts[i] -= 1
        counts[i + 1] -= 1
        counts[i + 2] -= 1
        ok = _can_form(counts, n_sets - 1, need_pair, budget)
        counts[i] += 1
        counts[i + 1] += 1
        counts[i + 2] += 1
        if ok:
            return True
    # 雀首
    if need_pair and counts[i] >= 2:
        counts[i] -= 2
        ok = _can_form(counts, n_sets, False, budget)
        counts[i] += 2
        if ok:
            return True
    # 跳过该牌型全部副本（不作为面子/雀首素材）
    saved = counts[i]
    counts[i] = 0
    ok = _can_form(counts, n_sets, need_pair, budget)
    counts[i] = saved
    return ok


def _sets_containing(t: int) -> list[tuple[int, int, int]]:
    """全部含 t 的面子（刻子 + 同色三处顺起点）。"""
    out = [(t, t, t)]
    s, j = divmod(t, 9)
    if s < 3:
        for start in (t - 2, t - 1, t):
            if 0 <= start - s * 9 <= 6:
                out.append((start, start + 1, start + 2))
    return out


def _subset_of(counts: list[int], tiles: tuple[int, ...]) -> bool:
    need = [0] * NUM_TILES
    for x in tiles:
        need[x] += 1
    return all(counts[x] >= need[x] for x in range(NUM_TILES) if need[x])


def _remove(counts: list[int], tiles: tuple[int, ...]) -> None:
    for x in tiles:
        counts[x] -= 1


def _riichi_can_wait(pool: list[int], t: int, n_sets: int) -> bool:
    """牌池 pool 中是否存在 (13-3*n_melds) 张听牌手牌，其等待包含 t。

    等价于 R = pool + {t} 中存在含 t 的完整和牌形 W（|W| = 3*n_sets + 2）：
    normal 路径为 n_sets 面子 + 雀首；七对/国士仅无副露时可能（n_sets == 4）。
    n_sets = 4 - 副露数（暗杠不破门清：立直家可能有暗杠，此时 n_sets=3）。

    关于 R 的构造：点出的 t 是**已存在的一张实体牌**（牌山或他家手牌——都在 pool 中；
    或在观察者手中——pool 外的一张额外副本），故 R[t] = min(pool[t]+1, 4)
    （全局仅 4 张；pool[t]=4 时点出的就是其中一张，不能再造出第 5 张）。
    """
    R = list(pool)
    R[t] = min(R[t] + 1, 4)
    budget = [ENUM_NODE_BUDGET]

    # --- normal：雀首 = (t,t)，其余 n_sets 面子 ---
    if R[t] >= 2:
        R2 = list(R)
        R2[t] -= 2
        if _can_form(R2, n_sets, False, budget):
            return True
    # --- normal：t 在某个面子中，其余 (n_sets-1) 面子 + 雀首 ---
    for S in _sets_containing(t):
        if _subset_of(R, S):
            R2 = list(R)
            _remove(R2, S)
            if _can_form(R2, n_sets - 1, True, budget):
                return True
    if n_sets < 4:
        return False                              # 有副露则无七对/国士路径
    # --- 七对：雀首 (t,t) + 其余 6 对 ---
    if R[t] >= 2:
        R2 = list(R)
        R2[t] -= 2
        if sum(1 for c in R2 if c >= 2) >= 6:
            return True
    # --- 国士：13 幺九 + 1 重复；t 为单张或重复张皆可 ---
    if t in YAOCHUU_SET:
        others = [x for x in YAOCHUU.tolist() if x != t]
        if all(pool[x] >= 1 for x in others):
            if pool[t] >= 1:
                return True                      # 重复张 = t（H 持 1 张 t）
            if any(pool[x] >= 2 for x in others):
                return True                      # 重复张 = 别家幺九（t 仅为单张）
    return False


def riichi_possible_waits(
    unseen: np.ndarray,           # [34]
    river: np.ndarray,            # [34] bool 目标家本局河
    own_hand: np.ndarray,         # [34] bool 观察者手牌
    n_melds: int = 0,             # 目标家副露数（吃/碰/大明杠/暗杠；立直家只可能有暗杠）
) -> np.ndarray:
    """立直枚举：返回 bool[34] 的可能听牌支撑集（含见逃排除）。"""
    unseen = np.asarray(unseen, dtype=np.float64)
    pool = [int(x) for x in np.clip(np.round(unseen), 0, 4)]
    river = np.asarray(river, dtype=bool)
    own_hand = np.asarray(own_hand, dtype=bool)
    expressible = (np.round(unseen) > 0) | own_hand
    n_sets = 4 - int(n_melds)
    out = np.zeros(NUM_TILES, dtype=bool)
    for t in range(NUM_TILES):
        if not expressible[t] or river[t]:
            continue                                   # 见逃排除 + 不可表达
        if _riichi_can_wait(pool, t, n_sets):
            out[t] = True
    return out


# --------------------------------------------------------------------------- #
# 顶层：与神经网络同构的输出
# --------------------------------------------------------------------------- #
def rule_predict(batch: dict, *, dora: np.ndarray | None = None,
                 order_blind: bool = False) -> dict:
    """规则法读牌前向。batch 为 dataset.collate 口径的 numpy 字典（或子集）。

    必需键：unseen / river / riichi / own_hand / turn / tiles_left / called /
            meld_counts / tl_tokens / tl_mask。
    返回 {'alloc_probs': [n,3,34,5], 'wait_probs': [n,3,35], 'exp_counts': [n,3,34]}。

    ``order_blind=True`` 时先把时序 token 规范化排序（`canonicalize_tl_order`），
    使 A3 成为**真顺序盲**预测器——用于给 §12.4/RM13~RM15 提供纯组成地板。
    默认 False：与历史存档数字（README §12、diagnostics/results/）逐位可比。
    """
    unseen = np.asarray(batch['unseen'], dtype=np.float64)
    river = np.asarray(batch['river'], dtype=bool)
    riichi = np.asarray(batch['riichi'], dtype=np.float64)
    own_hand = np.asarray(batch['own_hand'], dtype=bool)
    turn = np.asarray(batch['turn'], dtype=np.float64)
    tiles_left = np.asarray(batch['tiles_left'], dtype=np.float64)
    called = np.asarray(batch['called'], dtype=np.float64)
    meld_counts = np.asarray(batch['meld_counts'], dtype=np.float64)
    n = unseen.shape[0]
    # 宝牌：指示牌计数 → 实际宝牌计数（缺失字段时退化为无宝牌先验）
    if batch.get('dora_indicators') is not None:
        dora = indicators_to_dora(batch['dora_indicators']).astype(np.float64)
    else:
        dora = None

    # 手牌数 = 13 - 3*副露数（公开信息；加杠净贡献 0 不计）
    hand_sizes = 13.0 - 3.0 * meld_counts               # [n,3]

    # 时序 token（order_blind 时先规范化排序 ⇒ 破坏顺序、保留多重集）
    tl_tokens, tl_mask = batch['tl_tokens'], batch['tl_mask']
    if order_blind:
        tl_tokens, tl_mask = canonicalize_tl_order(tl_tokens, tl_mask)

    # ---- alloc ----
    early_river = early_discard_mask(tl_tokens, tl_mask)
    alloc_probs = rule_alloc_probs(unseen, river, hand_sizes, tiles_left, dora,
                                   early_river=early_river)

    # ---- wait ----
    feat = timeline_features(tl_tokens, tl_mask)
    p_ten = rule_tenpai_prob(feat, riichi, turn, called, meld_counts)   # [n,3]
    w, support = rule_wait_weights(unseen, river, own_hand, dora, turn=turn,
                                   early_river=early_river, riichi=riichi)

    # 立直家：枚举支撑集收紧（硬约束）；枚举为空时保留规则支撑（防御性回退）
    ri = riichi > 0.5
    for s in range(n):
        for j in range(NUM_TARGETS):
            if not ri[s, j]:
                continue
            enum = riichi_possible_waits(unseen[s], river[s, j], own_hand[s],
                                         n_melds=int(meld_counts[s, j]))
            if enum.any():
                support[s, j] = enum
    w = np.where(support, w, 0.0)

    wsum = w.sum(-1, keepdims=True)                     # [n,3,1]
    tile_probs = np.where(wsum > 0, w / np.maximum(wsum, 1e-12), 0.0)
    # 有支撑才放置听牌质量（支撑空 = 即使听牌也无可荣牌 → 全部质量归「未听」）
    tile_mass = np.where(wsum[..., 0] > 0, p_ten, 0.0)  # [n,3]

    wait_probs = np.zeros((n, NUM_TARGETS, NUM_WAIT_CLASSES), dtype=np.float64)
    wait_probs[..., :NUM_TILES] = tile_probs * tile_mass[..., None]
    wait_probs[..., NOT_TENPAI] = 1.0 - tile_mass
    wait_probs = wait_probs / wait_probs.sum(-1, keepdims=True).clip(min=1e-12)

    return {
        'alloc_probs': alloc_probs.astype(np.float32),
        'wait_probs': wait_probs.astype(np.float32),
        'exp_counts': expected_counts(alloc_probs).astype(np.float32),
    }
