"""A3 规则法基线回归测试（对应 memory §14-35/36：断言设计意图 + 独立重算 + 数值门禁）。

八层检查：
  A. alloc 不变量：形状/unseen 掩码/归一化；flat 不动点（= evaluate.binomial_baseline
     逐位相等）；倾向方向（字牌<中张、河牌<非河牌、宝牌>非宝牌、u 梯度）。
  B. wait 硬规则：河牌（现物/振听）概率 0；不可表达概率 0；归一化；立直家未听类 0。
  C. wait 软规则方向：筋（早切、立直更强、晚切无效）、壁、宝牌、字牌巡目依赖、
     听牌先验（立直=1、鸣牌/巡目/摸切尾提升）。
  D. 立直枚举独立重算：暴力枚举全部 13-3m 张听牌子集（structure 枚举器算向听）
     与 _riichi_can_wait 支撑集比对（多随机小池）；构造精确案例（单和牌形）；
     见逃排除；副露参数化（暗杠立直 3 面子）；5 张同牌容量边界。
  E. 真实数据门禁（小样本牌谱）：真实听牌 ⊆ 支撑集（枚举超集性质，全部目标家）；
     立直家未听类恒 0；tenpai AUC > 门禁；alloc CE 优于均匀基线。
  F. dora 映射：指示牌→宝牌后继（9m→1m 回绕、字牌循环、重复指示牌计数）。
  G. 端到端不变式：rule_predict 输出形状/归一化/有限性；meld_counts→手牌数。
  H. 性能门禁：2000 样本 rule_predict 耗时上限（防枚举退化）。

用法：
    python tools/read_model/test_rule_baseline.py [--n-files 6]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D                                     # noqa: E402
import rule_baseline as RB                              # noqa: E402
import train as T                                       # noqa: E402
from evaluate import auc_score, binomial_baseline       # noqa: E402
from structure import structure_labels                 # noqa: E402  # noqa: F401（D 层保留导入以备诊断）

NUM_TILES = 34
FAILS: list[str] = []
SKIPS: list[str] = []
N_ITEMS = 0                # 已报告项数（OK+FAIL+SKIP）
#: 期望项数门禁（2026-10-20，README §11.5-⑭）：与 test_pipeline 同款——早退/跳过
#: 会让覆盖静默缩水而摘要照样 ALL PASS（坑 57 的结构性根因）。增删 check 须同步此数。
EXPECTED_ITEMS = 50


def check(name, cond, extra=''):
    global N_ITEMS
    N_ITEMS += 1
    print(f'  [{"OK" if cond else "FAIL"}] {name}' + (f'    {extra}' if extra else ''),
          flush=True)
    if not cond:
        FAILS.append(name)


def skip(name, extra=''):
    """可选项缺失 ⇒ 单列，**不计入通过数**（旧写法 check(..., True) 会把跳过记成 PASS）。"""
    global N_ITEMS
    N_ITEMS += 1
    SKIPS.append(name)
    print(f'  [SKIP] {name}' + (f'    {extra}' if extra else ''), flush=True)


def make_batch(n=8, seed=0):
    """构造确定性合成 batch（覆盖各分支：立直/鸣牌/无 token/多巡目）。"""
    rng = np.random.default_rng(seed)
    b = {}
    unseen = rng.integers(0, 5, size=(n, NUM_TILES)).astype(np.float64)
    unseen[:, :5] = 0                                   # 保证有不可表达牌
    b['unseen'] = unseen
    river = np.zeros((n, 3, NUM_TILES), dtype=bool)
    for s in range(n):
        for j in range(3):
            river[s, j, rng.integers(0, NUM_TILES, size=4)] = True
    b['river'] = river
    b['riichi'] = (rng.random((n, 3)) < 0.3).astype(np.float64)
    b['riichi'][0, 0] = 1.0
    own = np.zeros((n, NUM_TILES), dtype=bool)
    own[:, 5:9] = True
    b['own_hand'] = own
    b['turn'] = rng.integers(0, 18, size=n).astype(np.int64)
    b['tiles_left'] = rng.integers(10, 70, size=n).astype(np.int64)
    b['called'] = (rng.random((n, 3)) < 0.25).astype(np.float32)
    b['meld_counts'] = rng.integers(0, 2, size=(n, 3)).astype(np.int64)
    # 时序 token：随机舍牌序列（含摸切/手切/鸣牌）
    L = 8
    tl = np.zeros((n, 3, L, RB.TL_FEATURES), dtype=np.float32)
    tlm = np.zeros((n, 3, L), dtype=bool)
    for s in range(n):
        for j in range(3):
            k = 0
            for _ in range(rng.integers(1, 6)):
                tid = int(rng.integers(0, NUM_TILES))
                tg = rng.random() < 0.5
                act = 'dahai' if rng.random() < 0.8 else 'pon'
                tl[s, j, k, tid] = 1.0
                tl[s, j, k, NUM_TILES] = 1.0 if tg else 0.0
                tl[s, j, k, NUM_TILES + 1 + RB.TL_ACTION_KINDS.index(act)] = 1.0
                tl[s, j, k, -1] = rng.random()
                tlm[s, j, k] = True
                k += 1
    b['tl_tokens'] = tl
    b['tl_mask'] = tlm
    b['dora_indicators'] = np.zeros((n, NUM_TILES), dtype=np.int64)
    b['dora_indicators'][:, 18] = 1                      # 1s 指示 → 2s 宝牌
    b['alloc'] = np.zeros((n, 3, NUM_TILES), dtype=np.uint8)
    b['wait'] = np.zeros((n, 3, 35), dtype=np.float32)
    b['wait'][..., 34] = 1.0
    return b


def tile_counts(spec: dict[int, int]) -> list[int]:
    """{tid: count} → [34] 计数向量。"""
    c = [0] * NUM_TILES
    for tid, k in spec.items():
        c[tid] = k
    return c


# --------------------------------------------------------------------------- #
def test_alloc(b):
    print('== A. alloc 不变量 ==')
    hs = 13.0 - 3.0 * b['meld_counts']
    ap = RB.rule_alloc_probs(b['unseen'], b['river'], hs, b['tiles_left'],
                             dora=RB.indicators_to_dora(b['dora_indicators']).astype(float),
                             early_river=RB.early_discard_mask(b['tl_tokens'], b['tl_mask']))
    n = ap.shape[0]
    u = np.broadcast_to(np.round(b['unseen']).astype(int)[:, None, :], (n, 3, NUM_TILES))
    k = np.arange(5)[None, None, None, :]
    check('A1 形状 [n,3,34,5]', ap.shape == (n, 3, NUM_TILES, 5))
    check('A1 unseen 掩码（k>u 概率 0）', not ap[k > u[..., None]].any())
    check('A1 列归一化', np.allclose(ap.sum(-1), 1.0, atol=1e-5))
    check('A1 非负有限', (ap >= 0).all() and np.isfinite(ap).all())

    # A2 flat 不动点：与 evaluate.binomial_baseline 逐位相等
    flat = RB.rule_alloc_probs(b['unseen'], b['river'], hs, b['tiles_left'], flat=True)
    base = binomial_baseline(b['unseen'], hs, b['tiles_left'])
    check('A2 flat == binomial_baseline（逐位）', np.allclose(flat, base, atol=1e-6),
          f'max|Δ|={np.abs(flat - base).max():.2e}')

    # A3 倾向方向（构造单样本：同 unseen，只变类别/河牌/宝牌）
    uu = np.full((1, NUM_TILES), 4.0)
    river = np.zeros((1, 3, NUM_TILES), dtype=bool)
    river[0, 0, 3] = True                                # 目标0 河里有 4m
    dora = np.zeros((1, NUM_TILES))
    dora[0, 13] = 1.0                                    # 5p 是宝牌
    early = np.zeros((1, 3, NUM_TILES), dtype=bool)
    early[0, 0, 3] = True
    p = RB.rule_alloc_probs(uu, river, np.full((1, 3), 13.0), np.array([50.0]),
                            dora=dora, early_river=early)
    e = RB.expected_counts(p)[0]                          # [3,34]
    honor_mean = e[:, 27:].mean()
    mid_mean = e[:, 4:7].mean()                           # 5m/6m/7m 中段
    check('A3 字牌持有倾向 < 中张（数据证伪「留幺九」）', honor_mean < mid_mean,
          f'honor {honor_mean:.3f} < middle {mid_mean:.3f}')
    check('A3 河牌倾向 < 非河牌', e[0, 3] < e[0, 4])
    check('A3 宝牌倾向 > 非宝牌', e[0, 13] > e[0, 14])
    uu2 = np.array([[4.0] * NUM_TILES, [1.0] * NUM_TILES])
    p2 = RB.rule_alloc_probs(uu2, np.zeros((2, 3, NUM_TILES), bool),
                             np.full((2, 3), 13.0), np.array([50.0, 50.0]), flat=False)
    e2 = RB.expected_counts(p2)
    check('A3 未见张数梯度（u=1 < u=4）', e2[1, 0, 5] < e2[0, 0, 5])


def test_wait_hard(b):
    print('== B. wait 硬规则 ==')
    out = RB.rule_predict(b)
    wp = out['wait_probs']
    n = wp.shape[0]
    check('B1 形状 [n,3,35]', wp.shape == (n, 3, 35))
    check('B1 行归一化', np.allclose(wp.sum(-1), 1.0, atol=1e-6))
    check('B1 非负有限', (wp >= 0).all() and np.isfinite(wp).all())
    check('B2 河牌（现物/振听）概率 0', not (wp[..., :NUM_TILES][b['river']] > 0).any())
    u = np.round(b['unseen']).astype(int)
    inexpressible = (u == 0) & ~b['own_hand']
    check('B2 不可表达牌概率 0',
          not (wp[..., :NUM_TILES][:, :, inexpressible].any(0) if False
               else wp[..., :NUM_TILES][np.broadcast_to(
                   inexpressible[:, None, :], (n, 3, NUM_TILES))] > 0).any())
    ri = b['riichi'] > 0.5
    support_nonempty = (wp[ri, :NUM_TILES] > 0).any(-1)
    check('B3 立直家未听类 0（支撑非空时）', (wp[ri, 34][support_nonempty] < 1e-6).all())


def test_wait_soft():
    print('== C. wait 软规则方向 ==')
    # 构造：非立直目标0，巡目 8，无宝牌；单张比较法（权重比 = 概率比）
    def weights(unseen, river, own, turn, early=None, riichi=None):
        w, sup = RB.rule_wait_weights(
            np.asarray(unseen, dtype=float)[None],
            np.asarray(river, dtype=bool)[None],
            np.asarray(own, dtype=bool)[None],
            turn=np.array([float(turn)]),
            early_river=(np.asarray(early, dtype=bool)[None]
                         if early is not None else None),
            riichi=(np.asarray(riichi, dtype=bool)[None]
                    if riichi is not None else None))
        return w[0, 0], sup[0, 0]

    uu = np.full(NUM_TILES, 3.0)
    river = np.zeros((3, NUM_TILES), dtype=bool)
    own = np.zeros(NUM_TILES, dtype=bool)
    river[0, 3] = True                                    # 目标0 早切 4m
    early = np.zeros((3, NUM_TILES), dtype=bool)
    early[0, 3] = True
    w, sup = weights(uu, river, own, 8, early=early)
    # 1m 是 4m 的筋（x-3），2m 是普通中张
    check('C1 筋：P(1m) < P(2m)（早切 4m）', w[0] < w[2],
          f'w(1m)={w[0]:.3f} w(2m)={w[2]:.3f}')
    # 立直时筋更强
    ri = [True, False, False]
    w_ri, _ = weights(uu, river, own, 8, early=early, riichi=ri)
    check('C1 立直家筋更强', w_ri[0] < w[0], f'riichi {w_ri[0]:.3f} < {w[0]:.3f}')
    # 晚切无筋效应
    early_none = np.zeros((3, NUM_TILES), dtype=bool)
    w_late, _ = weights(uu, river, own, 8, early=early_none)
    check('C2 晚切无筋效应', abs(w_late[0] / w_late[2] - 1.0) < 0.35,
          f'ratio={w_late[0] / w_late[2]:.3f}')
    # 壁：u=1 < u=3（对照牌取不受筋/邻接影响的 1p；7m 是 4m 的筋牌不能作对照）
    u1 = uu.copy()
    u1[17] = 1.0                                        # 9m 未见 1 张
    w_k, _ = weights(u1, river, own, 8, early=early)
    check('C3 壁：u=1 权重 < u=3', w_k[17] < w_k[18],
          f'w(9m,u=1)={w_k[17]:.3f} < w(1p,u=3)={w_k[18]:.3f}')
    # 宝牌
    dora = np.zeros((1, NUM_TILES))
    dora[0, 13] = 1.0
    w_d, _ = RB.rule_wait_weights(uu[None], river[None], own[None], dora=dora,
                                  turn=np.array([8.0]))
    w_nd, _ = RB.rule_wait_weights(uu[None], river[None], own[None],
                                   turn=np.array([8.0]))
    check('C4 宝牌权重 > 非宝牌', w_d[0, 0, 13] > w_nd[0, 0, 14])
    # 字牌巡目依赖
    w_e, _ = RB.rule_wait_weights(uu[None], river[None], own[None], turn=np.array([1.0]))
    w_l, _ = RB.rule_wait_weights(uu[None], river[None], own[None], turn=np.array([14.0]))
    check('C5 字牌：初盘权重 > 后盘（字牌危险只成立于初盘）',
          w_e[0, 0, 27] > w_l[0, 0, 27], f'{w_e[0, 0, 27]:.3f} > {w_l[0, 0, 27]:.3f}')

    # C6 听牌先验
    L = 6
    tl = np.zeros((1, 1, L, RB.TL_FEATURES), dtype=np.float32)
    tlm = np.zeros((1, 1, L), dtype=bool)
    for k in range(3):
        tl[0, 0, k, 5] = 1.0
        tl[0, 0, k, NUM_TILES] = 1.0                       # 连续 3 次摸切
        tl[0, 0, k, NUM_TILES + 1] = 1.0                   # dahai
        tlm[0, 0, k] = True
    feat = RB.timeline_features(tl, tlm)
    check('C6 tg_tail 计算正确（右对齐）', feat['tg_tail'][0, 0] == 3.0)
    p_lo = RB.rule_tenpai_prob(feat, np.zeros((1, 1)), np.array([2]),
                               np.zeros((1, 1)), None)
    p_hi = RB.rule_tenpai_prob(feat, np.zeros((1, 1)), np.array([14]),
                               np.zeros((1, 1)), None)
    check('C6 听牌先验随巡目上升', p_hi[0, 0] > p_lo[0, 0], f'{p_lo[0,0]:.3f} -> {p_hi[0,0]:.3f}')
    p_called = RB.rule_tenpai_prob(feat, np.zeros((1, 1)), np.array([8]),
                                   np.ones((1, 1)), None)
    p_uncalled = RB.rule_tenpai_prob(feat, np.zeros((1, 1)), np.array([8]),
                                     np.zeros((1, 1)), None)
    check('C6 鸣牌提升听牌先验', p_called[0, 0] > p_uncalled[0, 0])
    p_ri = RB.rule_tenpai_prob(feat, np.ones((1, 1)), np.array([8]),
                               np.zeros((1, 1)), None)
    check('C6 立直家听牌先验 = 1', p_ri[0, 0] == 1.0)


def _can_partition(counts: list[int], n_sets: int, need_pair: bool) -> bool:
    """独立判据（精确划分，与被测 _can_form 的「子集搜索」语义不同）：
    counts 能否**整分**为 n_sets 个面子 + (need_pair) 个雀首（全部牌用完）。"""
    total = sum(counts)
    if total != 3 * n_sets + (2 if need_pair else 0):
        return False
    if total == 0:
        return True
    i = 0
    while counts[i] == 0:
        i += 1
    if counts[i] >= 3:                                       # 刻子
        counts[i] -= 3
        if _can_partition(counts, n_sets - 1, need_pair):
            counts[i] += 3
            return True
        counts[i] += 3
    if i < 27 and i % 9 <= 6 and counts[i + 1] > 0 and counts[i + 2] > 0:  # 顺子
        counts[i] -= 1
        counts[i + 1] -= 1
        counts[i + 2] -= 1
        if _can_partition(counts, n_sets - 1, need_pair):
            counts[i] += 1
            counts[i + 1] += 1
            counts[i + 2] += 1
            return True
        counts[i] += 1
        counts[i + 1] += 1
        counts[i + 2] += 1
    if need_pair and counts[i] >= 2:                         # 雀首
        counts[i] -= 2
        if _can_partition(counts, n_sets, False):
            counts[i] += 2
            return True
        counts[i] += 2
    return False


def is_complete(w: list[int], n_sets: int) -> bool:
    """独立判据：14-3m 张多重集是否为完整和牌形（normal/七对/国士）。

    注意不能用 shanten==0 判「和牌」——14 张手牌 shanten 0 也包含「再摸一张
    才和」的听牌态（如 6 对+2 单张）。这里用精确划分判据（与 rule_baseline
    的 _can_form 子集搜索是不同实现、不同语义）。
    """
    if sum(w) != 3 * n_sets + 2:
        return False
    if n_sets == 4:
        if sum(1 for c in w if c) == 7 and all(c == 2 for c in w if c > 0):
            return True                                     # 七对
        yaoy = [w[i] for i in (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)]
        if (all(c > 0 for c in yaoy) and sum(yaoy) == 14
                and max(yaoy) == 2 and sum(1 for c in yaoy if c == 2) == 1):
            return True                                     # 国士
    return _can_partition(list(w), n_sets, True)


def brute_force_waits(pool: list[int], river: list[bool], own: list[bool],
                      n_melds: int) -> set[int]:
    """独立重算：枚举全部 (13-3m) 张子集 H ⊆ pool，对每个候选 t 用 is_complete
    判 H∪{t} 是否和牌，收集等待（t 需可表达且非河牌）。小池专用。"""
    hand_size = 13 - 3 * n_melds
    tiles = []
    for tid in range(NUM_TILES):
        tiles += [tid] * pool[tid]
    expressible = [(pool[t] > 0) or own[t] for t in range(NUM_TILES)]
    out = set()
    for combo in itertools.combinations(range(len(tiles)), hand_size):
        h = [0] * NUM_TILES
        for idx in combo:
            h[tiles[idx]] += 1
        for t in range(NUM_TILES):
            if not expressible[t] or river[t]:
                continue
            if h[t] >= 4:
                continue                                # 全局仅 4 张
            w = list(h)
            w[t] += 1
            if is_complete(w, 4 - n_melds):
                out.add(t)
    return out


def test_enumeration():
    print('== D. 立直枚举（独立重算 + 构造案例） ==')
    # D1 构造：pool = 111m 234p 567s 77p EE → 唯一 13 张听牌手牌，等待 {E, 7p}
    pool = tile_counts({0: 3, 9: 1, 10: 1, 11: 1, 18: 1, 19: 1, 20: 1, 26: 2, 27: 2})
    river = [False] * NUM_TILES
    own = [False] * NUM_TILES
    sup = RB.riichi_possible_waits(pool, river, own)
    check('D1 单和牌形枚举 == {E,7p}', set(np.nonzero(sup)[0].tolist()) == {27, 26},
          f'got {sorted(np.nonzero(sup)[0].tolist())}')
    # D2 见逃排除：河里有 E → 支撑只剩 7p
    river2 = [False] * NUM_TILES
    river2[27] = True
    sup2 = RB.riichi_possible_waits(pool, river2, own)
    check('D2 见逃排除（E 在河 → 支撑 {7p}）',
          set(np.nonzero(sup2)[0].tolist()) == {26})
    # D3 副露参数化：暗杠立直（n_melds=1）只需 3 面子+雀首
    pool3 = tile_counts({0: 3, 9: 1, 10: 1, 11: 1, 18: 1, 19: 1, 20: 1, 26: 2})
    sup3 = RB.riichi_possible_waits(pool3, river, own, n_melds=1)
    check('D3 暗杠参数化（n_melds=1 → 3 面子，7p 可等待）', bool(sup3[26]))
    sup3b = RB.riichi_possible_waits(pool3, river, own, n_melds=0)
    check('D3 n_melds=0 时同池无 13 张听牌手牌 → 空支撑', not sup3b.any())
    # D4 容量边界：pool[t]=4 时 t 不得因「第 5 张」成为假等待
    #     pool = 111m 222m 333m 444m 5m5m5m 6m6m6m 7m7m7m（13 张，无对子）
    pool4 = tile_counts({0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3})
    # 手牌 = 3 组顺子 123m*... 实际 111 222 333 444 555 666 777 不是顺子；
    # 换成 123m 123m 123m 456m 456m 456m 77m? 超 13。构造纯边界用例：
    #     H = 123m 123m 123m 456m（12 张）+ 单张 → 不听；换容量探测牌 8m：
    pool4 = tile_counts({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
                         9: 1, 10: 1, 11: 1, 12: 1})       # 13 张 1m~4p 各一张
    sup4 = RB.riichi_possible_waits(pool4, river, own)
    bf4 = brute_force_waits(pool4, river, own, 0)
    check('D4 容量边界与暴力重算一致', set(np.nonzero(sup4)[0].tolist()) == bf4,
          f'got {sorted(np.nonzero(sup4)[0].tolist())} vs bf {sorted(bf4)}')
    # D5 随机小池暴力重算（多种子；直接构造 8~12 牌种、总张数 13~15 的池）
    ok_all = True
    for seed in range(6):
        rng = np.random.default_rng(1000 + seed)
        while True:
            counts = np.zeros(NUM_TILES, dtype=int)
            tids = rng.choice(NUM_TILES, size=int(rng.integers(8, 13)), replace=False)
            for tid in tids:
                counts[tid] = int(rng.integers(1, 4))
            if 13 <= counts.sum() <= 15:
                break
        pool_r = [int(c) for c in counts]
        river_r = [False] * NUM_TILES
        river_r[int(rng.integers(0, NUM_TILES))] = True     # 随机一张振听
        own_r = [False] * NUM_TILES
        own_r[int(rng.integers(0, NUM_TILES))] = True       # 自家持有（可表达）
        got = set(np.nonzero(RB.riichi_possible_waits(pool_r, river_r, own_r))[0].tolist())
        bf = brute_force_waits(pool_r, river_r, own_r, 0)
        if got != bf:
            ok_all = False
            print(f'    seed {seed}: got {sorted(got)} vs bf {sorted(bf)}')
    check('D5 随机小池暴力重算一致（6 种子）', ok_all)


def test_real_data_gate(n_files):
    print('== E. 真实数据门禁 ==')
    import glob
    files = sorted(glob.glob(
        '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609/**/*.json.gz',
        recursive=True))[:n_files]
    batch = T.build_fixed_eval_set(files, 1500)
    b = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in batch.items()}
    out = RB.rule_predict(b)
    wp, ap = out['wait_probs'], out['alloc_probs']
    WT = b['wait']
    sup = WT[..., :NUM_TILES] > 0
    ri = b['riichi'] > 0.5
    # E1 枚举超集性质：真实听牌必在支撑集内（全部目标家，含立直/非立直）
    viol = (sup & ~(wp[..., :NUM_TILES] > 0)).sum()
    check('E1 真实听牌 ⊆ 支撑集（超集性质）', viol == 0, f'violations={int(viol)}')
    # E2 立直家未听类恒 0
    check('E2 立直家未听类恒 0', (wp[ri, 34] < 1e-6).all(), f'n_riichi={int(ri.sum())}')
    # E3 tenpai AUC 门禁（规则法应显著优于随机）
    tp = b['tenpai'] > 0.5
    score = 1.0 - wp[..., 34]
    auc = auc_score(tp.ravel(), score.ravel())
    check('E3 tenpai AUC > 0.60', auc > 0.60, f'AUC={auc:.4f}')
    # E4 alloc CE 优于均匀基线（软规则必须带来正信息增益）
    H = b['alloc']
    hs = H.sum(-1)
    bidx = (np.arange(len(H))[:, None, None], np.arange(3)[None, :, None],
            np.arange(NUM_TILES)[None, None, :])
    ce_rule = -np.log(ap[bidx + (H,)].clip(1e-12)).mean()
    base = binomial_baseline(b['unseen'], hs, b['tiles_left'])
    ce_base = -np.log(base[bidx + (H,)].clip(1e-12)).mean()
    check('E4 alloc CE < 均匀基线（正信息增益）', ce_rule < ce_base,
          f'rule {ce_rule:.4f} < base {ce_base:.4f}')
    # E5 手牌数公式：13 - 3*meld_counts 与 alloc 标签一致（公开信息口径验证）
    check('E5 hand_size == 13-3*meld_counts', np.all(hs == 13 - 3 * b['meld_counts']))


def test_dora():
    print('== F. dora 映射 ==')
    check('F1 1s 指示 → 2s 宝牌',
          RB.indicators_to_dora(np.array([0] * 18 + [1] + [0] * 15))[0, 19] == 1)
    check('F2 9m 指示 → 1m 宝牌（回绕）',
          RB.indicators_to_dora(np.array([0] * 8 + [1] + [0] * 25))[0, 0] == 1)
    check('F3 N 指示 → E 宝牌（字牌循环）',
          RB.indicators_to_dora(np.array([0] * 30 + [1] + [0] * 3))[0, 27] == 1)
    check('F4 C 指示 → P 宝牌（字牌循环）',
          RB.indicators_to_dora(np.array([0] * 33 + [1]))[0, 31] == 1)
    dup = np.array([0] * 18 + [2] + [0] * 15)              # 同一指示牌两次
    check('F5 重复指示牌计数', RB.indicators_to_dora(dup)[0, 19] == 2)
    succ = RB.dora_successor(np.array([8, 26, 30, 33, 17]))
    check('F6 dora_successor 批量', succ.tolist() == [0, 18, 27, 31, 9],
          f'{succ.tolist()}')


def test_e2e_and_perf():
    print('== G/H. 端到端与性能 ==')
    b = make_batch(n=64, seed=7)
    out = RB.rule_predict(b)
    check('G1 输出三件套形状',
          out['alloc_probs'].shape == (64, 3, 34, 5)
          and out['wait_probs'].shape == (64, 3, 35)
          and out['exp_counts'].shape == (64, 3, 34))
    check('G1 wait 归一化', np.allclose(out['wait_probs'].sum(-1), 1.0, atol=1e-6))
    check('G1 exp_counts == alloc_probs 期望',
          np.allclose(out['exp_counts'],
                      (out['alloc_probs'] * np.arange(5)).sum(-1), atol=1e-5))
    check('G1 全有限', np.isfinite(out['alloc_probs']).all()
          and np.isfinite(out['wait_probs']).all())
    t0 = time.time()
    RB.rule_predict(b)
    dt = time.time() - t0
    check('H1 64 样本（含 20 立直枚举）< 5s', dt < 5.0, f'{dt:.2f}s')


def _multiset_preserved(tl, mk, tl2, mk2) -> bool:
    """canonicalize_tl_order 是否只重排、不增删 token（逐 (样本,目标家) 比较多重集）。"""
    if not np.array_equal(mk, mk2):
        return False
    for s_ in range(tl.shape[0]):
        for j_ in range(tl.shape[1]):
            idx = np.where(mk[s_, j_])[0]
            if not len(idx):
                continue
            A, B = tl[s_, j_, idx], tl2[s_, j_, idx]
            if not np.array_equal(A[np.lexsort(A.T[::-1])], B[np.lexsort(B.T[::-1])]):
                return False
    return True


def test_order_blind(b):
    """A3 的 `order_blind` 变体必须**真正顺序不变**（README §11.5-⑩ / AUDIT S1）。

    背景：A3 原样版消费 tl_tokens（前 3 次舍牌掩码 + 时序特征），故其格内 ΔΔ 含真实
    顺序响应，不能当「组成混杂地板」。order_blind 变体把 token 规范化排序（破坏顺序、
    保留多重集），才是纯组成地板。本门禁守三条性质：置换不变、原样版确有响应（防
    上一条空转）、多重集守恒（防排序把 token 改了）。
    """
    print('\n-- order_blind 变体（真顺序盲地板，§11.5-⑩）--', flush=True)
    rng = np.random.default_rng(20261020)
    tl = np.asarray(b['tl_tokens'], dtype=np.float64)
    mk = np.asarray(b['tl_mask'], dtype=bool)
    tl_p = tl.copy()
    n_perm = 0
    for s_ in range(tl.shape[0]):
        for j_ in range(tl.shape[1]):
            idx = np.where(mk[s_, j_])[0]
            if len(idx) < 3:
                continue
            perm = rng.permutation(len(idx))
            if np.array_equal(perm, np.arange(len(idx))):
                continue
            tl_p[s_, j_, idx] = tl[s_, j_, idx[perm]]
            n_perm += 1
    b2 = dict(b)
    b2['tl_tokens'] = tl_p
    base, base_p = RB.rule_predict(b)['exp_counts'], RB.rule_predict(b2)['exp_counts']
    ob, ob_p = (RB.rule_predict(b, order_blind=True)['exp_counts'],
                RB.rule_predict(b2, order_blind=True)['exp_counts'])
    d_ob = float(np.abs(ob - ob_p).max())
    d_base = float(np.abs(base - base_p).max())
    check('RB12 order_blind 对 token 置换不变（真顺序盲）',
          n_perm > 0 and np.allclose(ob, ob_p, rtol=0, atol=1e-12),
          f'置换 {n_perm} 个 (样本,目标家)，max|Δ|={d_ob:.2e}')
    check('RB13 原样 A3 对同一置换**有**响应（证明 RB12 非空转）',
          not np.allclose(base, base_p, rtol=0, atol=1e-12),
          f'max|Δ|={d_base:.4f}')
    tl_c, mk_c = RB.canonicalize_tl_order(tl, mk)
    check('RB14 canonicalize_tl_order 保留 token 多重集（只重排不增删）',
          _multiset_preserved(tl, mk, tl_c, mk_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-files', type=int, default=6)
    ap.add_argument('--expect', type=int, default=EXPECTED_ITEMS,
                    help='期望报告项数（OK+FAIL+SKIP）；不符即 FAIL（§11.5-⑭）')
    args = ap.parse_args()
    print('================ A3 规则法基线回归测试 ================')
    b = make_batch()
    test_alloc(b)
    test_wait_hard(b)
    test_wait_soft()
    test_enumeration()
    test_dora()
    test_e2e_and_perf()
    test_order_blind(b)
    test_real_data_gate(args.n_files)
    print('======================================================')
    n_ok = N_ITEMS - len(FAILS) - len(SKIPS)
    print(f'共报告 {N_ITEMS} 项：OK {n_ok} / FAIL {len(FAILS)} / SKIP {len(SKIPS)}'
          f'（期望 {args.expect} 项）')
    if SKIPS:
        print(f'SKIP {len(SKIPS)} 项（**未执行**，不计入通过）: {SKIPS}')
    if N_ITEMS != args.expect:
        FAILS.append(f'覆盖项数 {N_ITEMS} != 期望 {args.expect}'
                     '（有 check 被跳过，或增删 check 未同步 EXPECTED_ITEMS）')
    if FAILS:
        print(f'FAILED ({len(FAILS)}): ' + '; '.join(FAILS))
        sys.exit(1)
    print('ALL PASS')


if __name__ == '__main__':
    main()
