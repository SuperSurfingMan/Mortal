"""手牌结构分解枚举器：从 tehai 计数向量产出「向听数 + 搭子构造」显式标签。

用途：读牌模型 v2 的结构估计流（显式监督 向听数/面子/搭子 构造，再作为信念瓶颈
反哺 H1/H2 头）。与 libriichi 的 ``shanten::calc_all`` 严格对齐：

- normal 路径：枚举全部 面子/搭子/雀头/孤立 分解，按标准公式
  ``向聴 = 8 - 2*M - min(T+K, 4-M)`` 加雀首修正计算；
- 七对路径：``calc_chitoi``；国士路径：``calc_kokushi``；
- 三条路径取 min（与 Rust 的提前退出条件一致：normal<=0 或 len_div3<4 时不再试
  七对/国士）。

交叉验证见 ``test_pipeline.py``：在数千个真实手牌上与 ``PlayerState.shanten``
getter 比对（注意：立直家的 shanten 被 Rust 冻结在宣言时刻的值，见 update.rs
的 ``if !self.riichi_accepted[0]`` 守卫——这类样本不参与比对，标签用本枚举器
的真值）。

标签语义（每个样本 x 目标家）：
    form    3 类：normal / chiitoi / kokushi（哪条路径达到最小向听）
    counts  7 个软分布（0..4）：shuntsu/kotsu/toitsu/ryanmen/kanchan/penchan/isolated
            在全部最小向听的 normal 分解上取边际直方图（保留真实歧义，例如
            "3 个顺子 + 两面搭" vs "3 个顺子 + 嵌张搭" 的模糊性）
"""
from __future__ import annotations

import numpy as np
import torch

NUM_TILES = 34
STRUCT_KEYS = ('shuntsu', 'kotsu', 'toitsu', 'ryanmen', 'kanchan', 'penchan', 'isolated')
NUM_STRUCT_COUNTS = len(STRUCT_KEYS)          # 7
MAX_COUNT_BIN = 4                             # 每个结构元素计数的分布域 0..4
COUNT_BINS = MAX_COUNT_BIN + 1                # 5
NUM_FORMS = 3                                 # normal / chiitoi / kokushi
FORM_NORMAL, FORM_CHIITOI, FORM_KOKUSHI = 0, 1, 2
MAX_SHANTEN_BUCKET = 7                        # 标签桶 0..7（7 = 向听 >= 6）
NUM_SHANTEN_BUCKETS = MAX_SHANTEN_BUCKET + 1  # 8
KOKUSHI_TILES = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def _decompose_normal(counts: list[int], len_div3: int) -> list[tuple]:
    """枚举标准分解。返回 [(结构计数 7 元组, shanten)]（含全部 shanten 层级）。

    len_div3 = 手牌张数 // 3（= 4 - 副露数）；向听公式按 L 参数化，见 shanten_of。
    """
    out: list[tuple] = []

    def shanten_of(acc: list[int]) -> int:
        # 与 libriichi shanten::calc_normal 同口径（L=len_div3=4-副露数，
        # M=手牌内面子数，T=搭子数，K=对子数）：
        #   向聴 = 2L - 2M - min(T+K, L-M) - 雀首修正
        # 雀首修正：搭子/对子总数超过还差的面子槽位（T+K > L-M）且手里有对子时，
        # 多出来的对子可直接充当雀首（0 抽），再省 1 抽。
        # 推导（逐路径数 draw）：面子可由搭子/对子晋升（各 1 抽）或孤立牌拼成
        # （2 抽）；雀首可由现成对子（0 抽）或孤立牌+1 抽得到。
        m = acc[0] + acc[1]
        t = acc[3] + acc[4] + acc[5]
        k = acc[2]
        slots = len_div3 - m                     # 还差的面子数
        s = 2 * len_div3 - 2 * m - min(t + k, slots)
        if k >= 1 and (t + k) > slots:
            s -= 1
        return s

    def rec(i: int, c: list[int], acc: list[int]) -> None:
        while i < NUM_TILES and c[i] == 0:
            i += 1
        if i == NUM_TILES:
            out.append((tuple(acc), shanten_of(acc)))
            return
        suit, num = divmod(i, 9)
        # 顺子
        if suit < 3 and num <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1; acc[0] += 1
            rec(i, c, acc)
            acc[0] -= 1; c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        # 刻子
        if c[i] >= 3:
            c[i] -= 3; acc[1] += 1
            rec(i, c, acc)
            acc[1] -= 1; c[i] += 3
        # 对子
        if c[i] >= 2:
            c[i] -= 2; acc[2] += 1
            rec(i, c, acc)
            acc[2] -= 1; c[i] += 2
        # 两面/边张搭子 (i, i+1)
        if suit < 3 and num <= 7 and c[i + 1] > 0:
            c[i] -= 1; c[i + 1] -= 1
            slot = 5 if num in (0, 7) else 3      # 12/89 = 边张，其余两面
            acc[slot] += 1
            rec(i, c, acc)
            acc[slot] -= 1
            c[i] += 1; c[i + 1] += 1
        # 嵌张搭子 (i, i+2)
        if suit < 3 and num <= 6 and c[i + 2] > 0:
            c[i] -= 1; c[i + 2] -= 1; acc[4] += 1
            rec(i, c, acc)
            acc[4] -= 1; c[i] += 1; c[i + 2] += 1
        # 孤立牌
        c[i] -= 1; acc[6] += 1
        rec(i, c, acc)
        acc[6] -= 1; c[i] += 1

    rec(0, list(counts), [0] * NUM_STRUCT_COUNTS)
    return out


def _cached_decompose(counts: tuple, len_div3: int):
    """按 (counts, len_div3) 记忆化——同一事件流里 4 个观察者重复 3 个目标家。"""
    key = (counts, len_div3)
    hit = _DECOMP_CACHE.get(key)
    if hit is None:
        hit = _decompose_normal(list(counts), len_div3)
        if len(_DECOMP_CACHE) < 200_000:
            _DECOMP_CACHE[key] = hit
    return hit


#: 进程级分解 memo（键=手牌计数，值=**全部**向听层级的叶子分解）。
#: ⚠️ 生命周期必须限定在单文件（2026-10-21 实测）：每条 value 均值 236 个
#: 分解、深尺寸均值 38KB（中位 20.5KB、最大 360KB），200k 上限会让每个
#: DataLoader worker 常驻约 7.6GB 活对象（叠加分配器高水位后 RSS 11.4GB/个，
#: NW=16 时逼近 251GB 上限）。跨文件手牌几乎不重复，全局 memo 没有收益；
#: 文件内复用由 GameReplayer.struct_cache 承担，故 dataset.replay_file 每次
#: 回放结束调用 clear_decomp_cache() 释放。
_DECOMP_CACHE: dict = {}


def clear_decomp_cache() -> None:
    """释放进程级分解 memo（按文件作用域使用，见 dataset.replay_file）。"""
    _DECOMP_CACHE.clear()


def shanten_chitoi(counts) -> int:
    kinds = pairs = 0
    for c in counts:
        if c > 0:
            kinds += 1
            if c >= 2:
                pairs += 1
    return 7 - pairs + max(0, 7 - kinds) - 1


def shanten_kokushi(counts) -> int:
    kinds = pairs = 0
    for i in KOKUSHI_TILES:
        if counts[i] > 0:
            kinds += 1
            if counts[i] >= 2:
                pairs += 1
    return 14 - kinds - (1 if pairs > 0 else 0) - 1


def structure_labels(counts, len_div3: int) -> dict:
    """由 tehai 计数向量计算结构标签。

    len_div3 = (手牌张数) // 3（13 张 -> 4；10 -> 3；...）。
    返回：
        form    int            normal/chiitoi/kokushi
        shanten int            0..6（>=0，与 Rust 的 max(0) 一致）
        counts  np.ndarray     [7, 5] 软分布（在最小向听的 normal 分解上取边际）
        n_decomp int           最小向听的 normal 分解个数（歧义度，诊断用）
    """
    decs = _cached_decompose(tuple(counts), len_div3)
    s_normal = min(s for _, s in decs)
    # Rust calc_all 的提前退出：normal<=0 或 len_div3<4 时不再试七对/国士
    if s_normal <= 0 or len_div3 < 4:
        s_all, form = s_normal, FORM_NORMAL
    else:
        s_chiitoi = shanten_chitoi(counts)
        s_all, form = s_normal, FORM_NORMAL
        if s_chiitoi < s_all:
            s_all, form = s_chiitoi, FORM_CHIITOI
        if s_all > 0:
            s_kokushi = shanten_kokushi(counts)
            if s_kokushi < s_all:
                s_all, form = s_kokushi, FORM_KOKUSHI

    # 最小向听的 normal 分解上的边际直方图（保留结构歧义）
    best = [acc for acc, s in decs if s == s_normal]
    hist = np.zeros((NUM_STRUCT_COUNTS, COUNT_BINS), dtype=np.float32)
    for acc in best:
        for k, v in enumerate(acc):
            hist[k, min(v, MAX_COUNT_BIN)] += 1.0
    hist /= hist.sum(-1, keepdims=True)

    return {
        'form': form,
        'shanten': int(max(s_all, 0)),
        'counts': hist,
        'n_decomp': len(best),
    }


def labels_from_bytes(tehai_bytes: bytes, len_div3: int) -> dict:
    """便捷入口：tehai 为 [u8;34] 的 bytes（PlayerState.tehai getter 的返回）。"""
    counts = np.frombuffer(tehai_bytes, dtype=np.uint8).astype(int).tolist()
    return structure_labels(counts, len_div3)


# --------------------------------------------------------------------------- #
# 可微软向听（分解 -> 向听的一致性约束 / 诊断共用）
# --------------------------------------------------------------------------- #
def soft_shanten_from_counts(count_probs: torch.Tensor,
                             len_div3: torch.Tensor | float) -> torch.Tensor:
    """由预测的搭子构造分布计算**可微软向听数**（与硬枚举器同一公式）。

    用于「分解-向听一致性约束」：预测的分解直方图必须对应一个自洽向听数。
    对预测分布取期望得到 M/T/K，代入与 ``_decompose_normal.shanten_of``
    完全同口径的公式：

        向听 = 2L - 2M - min(T+K, L-M) - 雀首修正
        L = len_div3 = 手牌张数 // 3 = 4 - 副露数
        M = 顺子 + 刻子；T = 两面 + 嵌张 + 边张；K = 对子
        雀首修正：有对子 且 搭子+对子总数超过还差的面子槽位时再省 1 抽

    与硬版的差异仅在「期望」与「逐分解」：硬公式对每个具体分解精确成立，
    软公式对边际分布的期望做同构计算（min 取次梯度、雀首修正用 sigmoid 软 AND），
    全程可微——梯度经 count_probs 回流到分解头，经返回值进入一致性损失。

    **只覆盖 normal 路径**（与硬枚举器的 normal 项同口径）：七对/国士形制的
    counts 标签取的仍是 normal 分解边际，不承载其向听信息。实测（202601 树
    8 文件 18855 个目标）：normal 形制上 round(soft)==硬标签 **100%**
    （mean|d|=0.039），非 normal 形制恒偏（mean|d|≈1.4）——故一致性约束
    按形制门控（见 train.py），诊断脚本同样只解读 normal 子集。

    Args:
        count_probs: [..., NUM_STRUCT_COUNTS, COUNT_BINS]，最后一维是 0..4 的分布
        len_div3:   [...] 或标量，L = 4 - 副露数（由公开信息精确推导）
    Returns:
        [...] 软向听（实数；>=6 的远处牌不截断，与标签比对时由调用方 clamp）
    """
    k = torch.arange(COUNT_BINS, dtype=count_probs.dtype, device=count_probs.device)
    exp = (count_probs * k).sum(-1)                          # [...,7] 期望计数
    m = exp[..., 0] + exp[..., 1]                            # 面子数（顺+刻）
    t = exp[..., 3] + exp[..., 4] + exp[..., 5]              # 搭子数（两面+嵌张+边张）
    kotsu = exp[..., 2]                                      # 对子数
    len_div3 = torch.as_tensor(len_div3, dtype=count_probs.dtype,
                               device=count_probs.device)
    slots = len_div3 - m                                    # 还差的面子槽位
    s = 2.0 * len_div3 - 2.0 * m - torch.minimum(t + kotsu, slots)
    # 雀首修正的软 AND：sigmoid(k-0.5) * sigmoid((t+k)-slots-0.5)（sharpness 8）
    sharp = 8.0
    corr = (torch.sigmoid(sharp * (kotsu - 0.5))
            * torch.sigmoid(sharp * (t + kotsu - slots - 0.5)))
    return s - corr
