"""读牌模型数据集：从 mjai 牌谱生成「观察者 obs + 三家暗牌/听牌/辅助标签」样本。

对应监督学习路线方案（memory/NEXT_STEPS_ANALYSIS 的读牌模型方案）§2/§3：

- **标签来源（全知）**：同一事件流同时驱动 4 个 ``libriichi.state.PlayerState``
  （绝对座位 0..3）。观察者 o 的输入 obs 取自其自身状态（与主训练管线同一
  ``encode_obs(version=4)``，零新增特征通道）；另外三家的标签直接取自对应
  PlayerState 的 ``tehai`` / ``waits`` getter —— mjai 日志自足，无需 oracle。
- **快照时点**：观察者的决策点（``cans.can_act()``，与 GameplayLoader 的样本
  语义一致）+ 局内公告（ReachAccepted / Dora）+ StartKyoku（巡目 0 先验）。
  此时所有目标家均处于 3n+1 静止态（自家摸牌后未打牌前无人可行动），
  ``waits`` 是新鲜计算的（立直家则为冻结值，恰为真实听牌集）。
- **公开信息独立推算**：``tiles_seen`` / 各家河 / 立直标志 / 牌数守恒全部由
  事件流在 Python 侧推算，不依赖任何 Rust 新增 getter（对现有代码零侵入）。
  推算正确性由 ``test_pipeline.py`` 与 obs 内嵌的 tiles_seen 通道、以及与
  Rust ``GameplayLoader`` 的 obs 逐位比对来验证。
- **增广**：万↔筒置换在事件流层面完成（与 ``GameplayLoader.augmented`` 同语义），
  obs 与标签随之一致变换；test 中验证等变性。

样本字段（dict，collate 时 stack）：

    obs           (1012, 34) f32   观察者 v4 obs（at_kan_select=False）
    unseen        (34,)      f32   4 - tiles_seen（H1 容量掩码，列守恒上界）
    alloc         (3, 34)    u8    三家暗牌计数标签（相对座位 1/2/3 的 tehai）
    wait          (3, 35)    f32   有效荣和集上的均匀分布（剔除自家河牌；第 35 类=无可荣牌）
    next_discard  (3,)       i64   下一打牌（0..33；34=本局无后续打牌）
    next_tedashi  (3,)       i64   下一张**手切**牌（跳过摸切；0..33；34=本局无后续
                                 手切。RM14 辅助任务标签：预测目标家下一张手切牌）
    shanten       (3,)       i64   向听数（0..6，>=6 归 7 桶；枚举器真值，
                                 立直家为 0 而非 Rust 的冻结值）
    tl_tokens     (3,L,F)    f32   目标家公开动作时序 token（v2 时序流；末位巡目
                                 仍写快照常数 = 旧口径兼容，见 tl_turn）
    tl_mask       (3,L)      bool  有效 token 位置（False=padding）
    tl_turn       (3,L)      f32   **逐 token 真实巡目**（min(at_turn/18,1.5)；RM6/RM12）
    tl_age        (3,L)      f32   逐 token 距快照的巡目差（min(Δturn/18,1.5)，turn 衰减用）
    tl_consumed   (3,L,2)    i8    chi/pon 被鸣的两张（tile id，-1=无；da/ankan 只存前 2）
    tl_flags      (3,L,2)    f32   [立直后打出, 该打牌被鸣]（RM6/RM12 token 内容）
    struct_form   (3,)       i64   结构路径：normal/chiitoi/kokushi
    struct_counts (3,7,5)    f32   搭子构造软分布（最小向听分解上的边际直方图）
    riichi        (3,)       f32   是否已立直（accepted）
    called        (3,)       f32   是否已鸣牌（吃/碰/杠）
    meld_counts   (3,)       i64   各家吃/碰/大明杠/暗杠数（公开副露；加杠净贡献 0
                                 不计）。手牌数 = 13 - 3*meld_counts（静止态硬事实），
                                 供规则法基线/硬约束层使用；注意暗杠不破门清，
                                 立直家可能有暗杠（此时手牌 10 张）
    turn          ()         i64   观察者 at_turn（样本权重/课程用）
    tiles_left    ()         i64   牌山剩余（70 - 本局摸牌数）
    tenpai        (3,)       f32   目标家是否**结构性**听牌（waits 非空；注意与 wait
                                 标签的区别：振听家的 wait 标签为「未听」但 tenpai=1，
                                 wait 损失仍在其上激活，监督模型学会识别振听）
    river         (3, 34)    bool  各家全部打牌集合（H2 硬掩码用）
    post_riichi_river (3, 34) bool 各家立直后的打牌（诊断用；掩码已统一按 river）
    dora_indicators (34,)     i64   当前宝牌指示牌计数（deaka；实际宝牌 = 后继牌，
                                 规则法基线的宝牌先验用；与 c_total 同口径）
    _observer/_event/_kyoku    调试元数据（collate 前剔除）
"""
from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from libriichi.consts import obs_shape
from libriichi.state import PlayerState

from structure import (
    COUNT_BINS as S_COUNT_BINS,
    NUM_STRUCT_COUNTS as S_NUM_COUNTS,
    clear_decomp_cache,
    structure_labels,
)

NUM_TILES = 34
NUM_WAIT_CLASSES = NUM_TILES + 1          # 34 种听牌 + 未听
NO_DISCARD = NUM_TILES                    # next_discard 的「本局无后续打牌」
MAX_SHANTEN_BUCKET = 7                    # 0..6 + (>=6 归 7)

# ---- obs 牌河顺序消融（P1b 输入消融：遮 obs 牌河顺序、只留集合）----
# libriichi v4 obs（``obs_repr.rs``，with_sp_tables=False）的牌河块布局：
#   自家河：35 起 4ch×6（正序）+ 4ch×18（倒序），recency 平面 131；
#   他家河：132/327/522 起 8ch×6 + 8ch×18，recency 平面 base+192..194；
#   最近手切：836/839/842 起各 3ch（tile/aka/dora）；835 是 tiles_seen，
#   其前 831..834 是 ankan_overview（不能动）。
# 消融做法：在每个窗口内对「有效 item 块」做随机置换（item 连同 aka/dora/
# tedashi/riichi/kan 标记整体搬动），并清零 recency 平面与最近手切平面
# ⇒ 集合与标记不变，时间顺序不可恢复。默认 ``keep``（旧口径逐位不变）。
OBS_RIVER_ORDER = os.environ.get('RM_OBS_RIVER_ORDER', 'keep')
_OBS_RIVER_BLOCKS = ((35, 4, 131), (132, 8, 324), (327, 8, 519), (522, 8, 714))
_OBS_LAST_TEDASHI_ROWS = (836, 839, 842)


def set_obs_river_order(mode: str) -> None:
    """切换 obs 牌河顺序消融（``keep`` 原样 / ``set`` 只留集合）。"""
    global OBS_RIVER_ORDER
    assert mode in ('keep', 'set'), mode
    OBS_RIVER_ORDER = mode


def _ablate_river_order(obs: np.ndarray, seed: int) -> None:
    """就地抹掉 obs 里的牌河顺序（P1b 输入消融）。

    每个窗口内的 item 块随机重排，recency 与最近手切平面清零。牌河集合、
    张数与 aka/tedashi/riichi/kan 标记全部保留，但「哪张先切」在 obs 里
    不可恢复——顺序只能从 tile-indexed 时序流（tl_turn/tl_age）读到。
    """
    rng = np.random.default_rng(seed)
    for base, ch, rec in _OBS_RIVER_BLOCKS:
        slots = [base + ch * i for i in range(6)]
        slots += [base + ch * (6 + j) for j in range(18)]
        occupied, blocks = [], []
        for row in slots:
            if obs[row:row + ch].any():
                occupied.append(row)
                blocks.append(obs[row:row + ch].copy())
        for row in slots:
            obs[row:row + ch] = 0.0
        obs[rec:rec + (1 if ch == 4 else 3)] = 0.0
        if not blocks:
            continue
        perm = rng.permutation(len(blocks))
        for row, k in zip(occupied, perm):
            obs[row:row + ch] = blocks[k]
    for row in _OBS_LAST_TEDASHI_ROWS:
        obs[row:row + 3] = 0.0

# ---- 时序 token（v2 时序流）----
# 每目标家一条「公开动作序列」：打牌（含摸切标志）/ 立直宣言 / 鸣牌。
TL_ACTION_KINDS = ('dahai', 'reach', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan')
TL_NUM_ACTIONS = len(TL_ACTION_KINDS)
TL_FEATURES = NUM_TILES + 1 + TL_NUM_ACTIONS + 1   # 牌 one-hot + 摸切 + 动作 + 归一化巡目
TL_MAX_LEN = 26                            # 一局内单家公开动作上限（实测远小于此）

# mjai 牌字符串 → deaka 后的 0..33 tile id
_SUIT_BASE = {'m': 0, 'p': 9, 's': 18}
_HONOR_ID = {'E': 27, 'S': 28, 'W': 29, 'N': 30, 'P': 31, 'F': 32, 'C': 33}

_PAI_BY_TID = (
    [f'{n}m' for n in range(1, 10)]
    + [f'{n}p' for n in range(1, 10)]
    + [f'{n}s' for n in range(1, 10)]
    + ['E', 'S', 'W', 'N', 'P', 'F', 'C']
)


def tile_id(pai: str) -> int:
    """'5mr' → 4（deaka）；'E' → 27。只接受合法 mjai 牌名。"""
    if pai.endswith('r'):  # aka：5mr/5pr/5sr
        base = pai[:-1]
        return _SUIT_BASE[base[1]] + int(base[0]) - 1
    head = pai[0]
    if head.isdigit():
        return _SUIT_BASE[pai[1]] + int(head) - 1
    return _HONOR_ID[head]


def _tid_to_pai(tid: int) -> str:
    return _PAI_BY_TID[tid]


def augment_tid(tid: int) -> int:
    """万↔筒置换（与 Event::augment 同语义：kind 0↔1，字牌不动）。"""
    if tid < 9:
        return tid + 9
    if tid < 18:
        return tid - 9
    return tid


def augment_perm() -> np.ndarray:
    """34 牌种的万↔筒置换排列（perm[i] = i 的置换像）。"""
    return np.array([augment_tid(i) for i in range(34)], dtype=np.int64)


def _augment_event_tids(ev: dict) -> None:
    """对单个已解析事件做万↔筒置换（风牌/宝牌指示牌中的万筒也要置换）。

    与 Rust ``Event::augment`` 严格同语义：aka（5mr/5pr/5sr）置换后保持 aka
    （5mr → 5pr），丢失 aka 标记会改变 akas_seen/doras_seen 与 obs 的 aka 通道。
    """
    def swap_tile_field(key: str) -> None:
        if key in ev and isinstance(ev[key], str):
            ev[key] = _swap_pai(ev[key])

    def swap_tile_list(key: str) -> None:
        if key in ev and isinstance(ev[key], list):
            ev[key] = [_swap_pai(t) for t in ev[key]]

    etype = ev.get('type')
    if etype == 'start_kyoku':
        if 'tehais' in ev:
            ev['tehais'] = [[_swap_pai(t) for t in hand] for hand in ev['tehais']]
        swap_tile_field('bakaze')
        swap_tile_field('dora_marker')
    elif etype in ('tsumo', 'dahai'):
        swap_tile_field('pai')
    elif etype in ('chi', 'pon', 'daiminkan', 'kakan'):
        swap_tile_field('pai')
        swap_tile_list('consumed')
    elif etype == 'ankan':
        swap_tile_list('consumed')
    elif etype == 'dora':
        swap_tile_field('dora_marker')
    elif etype == 'hora':
        swap_tile_list('ura_markers')


def _swap_pai(pai: str) -> str:
    """单张牌名的万↔筒置换，保持 aka 标记（5mr → 5pr）。"""
    is_aka = pai.endswith('r')
    base = pai[:-1] if is_aka else pai
    new_tid = augment_tid(tile_id(base))
    out = _PAI_BY_TID[new_tid]
    if is_aka and new_tid in (4, 13, 22):     # 5m/5p/5s 才能带 aka
        out += 'r'
    return out


class GameReplayer:
    """单局回放器：4 个 PlayerState + Python 侧公开信息台账。

    公开信息台账（全部 deaka tile id，观察者无关的部分只算一次）：

    - ``c_total[i]``    ：进入公共视野的 tile 计数 = 四家打牌 + 鸣牌消耗 + 暗杠消耗
                         + 加杠牌 + 宝牌指示牌。对观察者 A 有
                         ``tiles_seen_A = A 的 tehai + c_total``
                         （自家摸到的牌在 tehai/自家消耗里各计一次，不会重复计数）。
    - ``river[p]``      ：p 的全部打牌集合（H2 舍牌振听软先验用）。
    - ``post_riichi[p]``：p 立直 accepted 之后的打牌集合（立直家硬掩码用）。
    - ``riichi[p]``     ：p 是否已立直 accepted。
    - ``meld_tiles[p]`` ：p 手牌之外的 tile 数（吃/碰/大明杠 3、暗杠 4、加杠 1），
                         仅供守恒校验。
    - ``n_melds[p]``    ：p 的吃/碰/大明杠/暗杠次数（加杠不计——净贡献 0）。
                         手牌数 = 13 - 3*n_melds（静止态硬事实；暗杠不破门清，
                         立直家可能有暗杠）。逐局重置。
    - ``dora_count``    ：当前宝牌**指示牌**计数（deaka tile id，含杠后新增；
                         实际宝牌 = 指示牌的后继，见 rule_baseline.dora_successor）。
                         逐局重置；与 c_total 中的宝牌指示分量同口径。
    - ``timeline[p]``   ：p 的公开动作 token 序列（v2 时序流输入；逐局重置）。
                         每条 token = (事件序号, 牌 tid 或 -1, 是否摸切, 动作类别)。
                         全部来自公开事件（打牌/鸣牌/立直宣言都是明信息），
                         不含任何隐藏信息；摸切打牌的牌就是当回摸到的牌，
                         手切则摸到的牌仍在手里（模型自行推断）。
    """

    def __init__(self, events: list[dict], lines: list[str]):
        self.events = events
        self.lines = lines
        self.states = [PlayerState(i) for i in range(4)]

        self.c_total = np.zeros(NUM_TILES, dtype=np.int64)
        self.dora_count = np.zeros(NUM_TILES, dtype=np.int64)   # 宝牌指示牌计数（逐局重置）
        self.river = [np.zeros(NUM_TILES, dtype=bool) for _ in range(4)]
        self.post_riichi = [np.zeros(NUM_TILES, dtype=bool) for _ in range(4)]
        self.riichi = [False] * 4
        self.meld_tiles = [0] * 4
        self.n_melds = [0] * 4                # 吃/碰/大明杠/暗杠次数（加杠不计）
        self.n_tsumo = 0                 # 本局摸牌数（含岭上）
        self.kyoku = 0
        self.event_kyoku = np.zeros(len(events), dtype=np.int64)
        self.samples: list[dict] = []
        self.stats = {'n_snapshots': 0, 'n_samples': 0, 'n_kyoku': 0}
        # 时序 token 与结构标签的按 (事件, 座位) 缓存：同一事件的 4 个观察者
        # 视角共享，避免重复计算（结构枚举 ~0.1ms/目标家）。
        self.timeline = [[] for _ in range(4)]       # 当前 kyoku 的 token 列表
        self.tl_cache: dict = {}                     # (ev_idx, seat) -> 6 元组（RM6/RM12）
        self.struct_cache: dict = {}                 # (ev_idx, seat) -> 结构标签 dict

    # -- 事件流公开信息更新（只用 ev 的字段，与 PlayerState 无关） --
    def _track_public(self, ev_idx: int, ev: dict) -> None:
        etype = ev.get('type')
        if etype == 'start_kyoku':
            # 与 PlayerState.start_kyoku 一致：所有逐局台账在新一局重置
            # （tiles_seen/river/riichi/meld/dora 都是局内概念，漏重置会跨局累积）
            self.kyoku += 1
            self.n_tsumo = 0
            self.c_total = np.zeros(NUM_TILES, dtype=np.int64)
            self.dora_count = np.zeros(NUM_TILES, dtype=np.int64)
            self.river = [np.zeros(NUM_TILES, dtype=bool) for _ in range(4)]
            self.post_riichi = [np.zeros(NUM_TILES, dtype=bool) for _ in range(4)]
            self.riichi = [False] * 4
            self.meld_tiles = [0] * 4
            self.n_melds = [0] * 4
            self.timeline = [[] for _ in range(4)]
            # 换局清空两个 memo 缓存（2026-10-20，README §11.5-⑮）：键是全局 ev_idx，
            # 跨局本就不会复用，留着只是白占内存（每 token 表 26×43 f32 ≈ 4.5KB，
            # 结构标签更大）。worker RSS 随文件累积的一个次要来源。
            self.tl_cache.clear()
            self.struct_cache.clear()
            self.c_total[tile_id(ev['dora_marker'])] += 1
            self.dora_count[tile_id(ev['dora_marker'])] += 1
            return
        if etype == 'dora':
            self.c_total[tile_id(ev['dora_marker'])] += 1
            self.dora_count[tile_id(ev['dora_marker'])] += 1
            return
        if etype == 'tsumo':
            self.n_tsumo += 1
            return
        if etype == 'reach_accepted':
            self.riichi[ev['actor']] = True
            return
        actor = ev.get('actor')
        if actor is None:
            return
        if etype == 'dahai':
            tid = tile_id(ev['pai'])
            self.c_total[tid] += 1
            self.river[actor][tid] = True
            if self.riichi[actor]:
                self.post_riichi[actor][tid] = True
            self._tl_push(ev_idx, actor, tid, bool(ev.get('tsumogiri', False)), 'dahai',
                          post_riichi=self.riichi[actor])
        elif etype in ('chi', 'pon', 'daiminkan'):
            consumed = tuple(tile_id(t) for t in ev['consumed'])
            for t in ev['consumed']:
                self.c_total[tile_id(t)] += 1
            self.meld_tiles[actor] += 3
            self.n_melds[actor] += 1
            self._tl_push(ev_idx, actor, tile_id(ev['pai']), False, etype,
                          consumed=consumed)
            # RM12/RM6：标出「这张打牌被鸣了」——被吃/碰/大明杠的那张
            # 必然是 target 的最后一个 dahai token（鸣牌事件紧随其后）。
            target = ev.get('target')
            if target is not None and self.timeline[target]:
                last = self.timeline[target][-1]
                if last[3] == 'dahai':
                    last[7] = True
        elif etype == 'kakan':
            self.c_total[tile_id(ev['pai'])] += 1
            self.meld_tiles[actor] += 1
            self._tl_push(ev_idx, actor, tile_id(ev['pai']), False, 'kakan',
                          post_riichi=self.riichi[actor])
        elif etype == 'ankan':
            consumed = tuple(tile_id(t) for t in ev['consumed'])
            for t in ev['consumed']:
                self.c_total[tile_id(t)] += 1
            self.meld_tiles[actor] += 4
            self.n_melds[actor] += 1
            self._tl_push(ev_idx, actor, consumed[0], False, 'ankan',
                          consumed=consumed, post_riichi=self.riichi[actor])
        elif etype == 'reach':
            # 立直宣言（随后必有同 actor 的 dahai）——单独一个 token，
            # 让时序流显式看到「第几巡宣言了立直」
            self._tl_push(ev_idx, actor, -1, False, 'reach')

    def _tl_push(self, ev_idx: int, seat: int, tid: int, tsumogiri: bool, action: str,
                 consumed: tuple = (), post_riichi: bool = False) -> None:
        """向 seat 的时序 token 序列追加一条公开动作（v2 时序流）。

        token 结构（RM6/RM12）：
          [事件序号, 牌 tid, 摸切, 动作, 该动作发生时 seat 的 at_turn,
           被鸣牌 tuple, 立直后打出, 该打牌被鸣]
        最后一项是可变的（被鸣要等后续 chi/pon 事件才能知道）。
        """
        self.timeline[seat].append([
            ev_idx, tid, tsumogiri, action, int(self.states[seat].at_turn),
            consumed, bool(post_riichi), False,
        ])

    def _tl_of(self, ev_idx: int, seat: int) -> tuple:
        """seat 在事件 ev_idx 处理完毕时的时序 token 序列。

        返回 ``(tokens[L,F], mask[L], turn[L], age[L], consumed[L,2], flags[L,2])``：
        tokens 是旧 43 维口径（末位巡目为快照常数），其余是 RM6/RM12 侧车字段
        （逐 token 真实巡目、距快照巡目差、chi/pon consumed、[立直后打出, 被鸣]）。
        按 (事件, 座位) 缓存：同一事件的 4 个观察者视角共享。
        """
        key = (ev_idx, seat)
        hit = self.tl_cache.get(key)
        if hit is not None:
            return hit
        toks = self.timeline[seat][-TL_MAX_LEN:]      # 超长保护（实际远达不到）
        now_turn = int(self.states[seat].at_turn)
        # 旧口径保持：token[..., -1] 仍是**快照时目标家的 at_turn 常数**，
        # 让 A2b/A2c 旧 ckpt 在新数据集上逐位复现（README 坑 48）。
        snapshot_turn = min(now_turn / 18.0, 1.5)
        arr = np.zeros((TL_MAX_LEN, TL_FEATURES), dtype=np.float32)
        mask = np.zeros(TL_MAX_LEN, dtype=bool)
        turns = np.zeros(TL_MAX_LEN, dtype=np.float32)
        ages = np.zeros(TL_MAX_LEN, dtype=np.float32)
        consumed_arr = np.full((TL_MAX_LEN, 2), -1, dtype=np.int8)
        flags = np.zeros((TL_MAX_LEN, 2), dtype=np.float32)
        for k, (_ei, tid, tg, action, turn, consumed, post_riichi, called) in enumerate(toks):
            if tid >= 0:
                arr[k, tid] = 1.0
            arr[k, NUM_TILES] = 1.0 if tg else 0.0
            arr[k, NUM_TILES + 1 + TL_ACTION_KINDS.index(action)] = 1.0
            arr[k, -1] = snapshot_turn
            turns[k] = min(turn / 18.0, 1.5)
            ages[k] = min(max(now_turn - turn, 0) / 18.0, 1.5)
            for c, t in enumerate(consumed[:2]):
                if t >= 0:
                    consumed_arr[k, c] = t
            flags[k, 0] = 1.0 if post_riichi else 0.0
            flags[k, 1] = 1.0 if called else 0.0
            mask[k] = True
        out = (arr, mask, turns, ages, consumed_arr, flags)
        self.tl_cache[key] = out
        return out

    def _struct_of(self, ev_idx: int, seat: int) -> dict:
        """seat 在事件 ev_idx 时的结构标签（向听/搭子构造），按 (事件, 座位) 缓存。"""
        key = (ev_idx, seat)
        hit = self.struct_cache.get(key)
        if hit is not None:
            return hit
        tehai = np.frombuffer(self.states[seat].tehai, dtype=np.uint8)
        len_div3 = int(tehai.sum()) // 3
        labels = structure_labels(tehai.astype(int).tolist(), len_div3)
        # 立直家的 shanten 被 Rust 冻结在宣言时刻的值（update.rs 守卫），
        # 标签用枚举器真值（手牌实际向听）；getter 值仅用于交叉验证测试。
        self.struct_cache[key] = labels
        return labels

    def _next_discard_label(self, from_idx: int, target_abs: int) -> int:
        """从 from_idx 之后扫本局内 target 的第一张打牌；跨局/无则 NO_DISCARD。"""
        for k in range(from_idx + 1, len(self.events)):
            ev = self.events[k]
            if ev.get('type') == 'end_kyoku':
                break
            if ev.get('type') == 'dahai' and ev.get('actor') == target_abs:
                return tile_id(ev['pai'])
        return NO_DISCARD

    def _next_tedashi_label(self, from_idx: int, target_abs: int) -> int:
        """从 from_idx 之后扫本局内 target 的下一张**手切**（跳过摸切）。

        RM14：辅助任务「预测目标家下一张手切牌」。摸切打出的牌是当回摸到的
        牌（模型可直接用），手切才携带「他手里留了什么/拆了什么」的顺序信息；
        因此这里跳过 tsumogiri，直到本局结束；无则 NO_DISCARD。
        """
        for k in range(from_idx + 1, len(self.events)):
            ev = self.events[k]
            if ev.get('type') == 'end_kyoku':
                break
            if (ev.get('type') == 'dahai' and ev.get('actor') == target_abs
                    and not bool(ev.get('tsumogiri', False))):
                return tile_id(ev['pai'])
        return NO_DISCARD

    def _emit(self, ev_idx: int, observer: int, stage: int = 3) -> None:
        """在事件 ev_idx 处理完毕后，为观察者 observer 生成一条样本。

        标签（tehai/waits/shanten/riichi 等 getter）都是廉价操作；obs 编码
        （encode_obs，约 2.2ms，含 SP 计算器）占绝对多数耗时。因此先算标签、
        做课程阶段过滤，通过后才编码 obs——阶段 1（仅有立直家）可省掉约 85%
        的编码开销。
        """
        st_obs = self.states[observer]
        own_tehai = np.frombuffer(st_obs.tehai, dtype=np.uint8)
        turn = int(st_obs.at_turn)
        riichi_flag = np.zeros(3, dtype=np.float32)

        # --- 阶段过滤（先于昂贵的 obs 编码）---
        if stage != 3:
            if stage == 1:                       # 仅「**他家**有立直」（不含观察者自身：
                # rel 只取 1/2/3 ⇒ 读牌目标是别人，自己立直时手牌已公开、无读牌意义）
                for rel in (1, 2, 3):
                    if self.riichi[(observer + rel) % 4]:
                        break
                else:
                    return
            elif stage == 2:
                # 中盘及以后（巡目 >= 4）**全量**样本。⚠️ 不是「阶段 1 ∪ 中盘」：
                # 前 3 巡的立直样本在本阶段被排除（README §11.5-④，2026-10-20 更正注释）。
                if turn < 4:
                    return
            else:
                raise ValueError(stage)

        obs_arr, _mask = st_obs.encode_obs(4, False)  # (obs, mask)；mask 对读牌无用
        obs = np.asarray(obs_arr, dtype=np.float32)
        if OBS_RIVER_ORDER != 'keep':          # P1b 输入消融：抹掉 obs 牌河顺序
            _ablate_river_order(obs, (ev_idx << 3) ^ (observer + 1) ^ 0x5EED)

        tiles_seen = own_tehai.astype(np.int64) + self.c_total
        unseen = (4 - tiles_seen).astype(np.float32)
        # 可表达集：x 可能被某家打出 ⟺ unseen_x > 0（有副本在别家手牌或牌山），
        # 或 x 在观察者自己手里（观察者亲自打出）。注意 unseen==0 不等于「不是听牌」——
        # 4 张全在「自家手牌+公共区」时，观察者自己手里的副本依然能点炮。
        expressible = (unseen > 0) | (own_tehai > 0)

        alloc = np.zeros((3, NUM_TILES), dtype=np.uint8)
        wait = np.zeros((3, NUM_WAIT_CLASSES), dtype=np.float32)
        next_discard = np.full(3, NO_DISCARD, dtype=np.int64)
        next_tedashi = np.full(3, NO_DISCARD, dtype=np.int64)
        called_flag = np.zeros(3, dtype=np.float32)
        meld_counts = np.zeros(3, dtype=np.int64)
        tenpai_flag = np.zeros(3, dtype=np.float32)
        dora_count = self.dora_count.copy()
        river = np.zeros((3, NUM_TILES), dtype=bool)
        post_riichi_river = np.zeros((3, NUM_TILES), dtype=bool)

        for rel in (1, 2, 3):
            tgt = (observer + rel) % 4
            st = self.states[tgt]
            alloc[rel - 1] = np.frombuffer(st.tehai, dtype=np.uint8)

            # 听牌标签 = 「有效荣和集」：PlayerState.waits 是结构性听牌集（可能含
            # 自己河里的牌；见 update.rs 的 update_waits_and_furiten——waits 只把
            # at_furiten 记在别处，不缩小自身）。而 H2 的输出语义是「打这张牌给他
            # 能不能荣和」，按规则自己河里的待牌永远无法荣和，故标签必须剔除河牌；
            # 再剔除「无人能打出」的牌（4 张全在公共区且自家没有）。
            # 立直家：宣言时不允许振听 → 冻结听牌集与立直前河牌不相交，剔除立直后
            # 河牌后恰为当前可荣集（含极罕见的「立直后摸到听牌却见逃」情形）。
            waits = np.asarray(st.waits, dtype=bool)
            structural_tenpai = bool(waits.any())
            effective = waits & ~self.river[tgt] & expressible
            n_eff = int(effective.sum())
            if n_eff > 0:
                wait[rel - 1, :NUM_TILES][effective] = 1.0 / n_eff
            else:
                wait[rel - 1, NUM_TILES] = 1.0
            tenpai_flag[rel - 1] = 1.0 if structural_tenpai else 0.0

            next_discard[rel - 1] = self._next_discard_label(ev_idx, tgt)
            next_tedashi[rel - 1] = self._next_tedashi_label(ev_idx, tgt)
            riichi_flag[rel - 1] = 1.0 if self.riichi[tgt] else 0.0
            called_flag[rel - 1] = 1.0 if self.meld_tiles[tgt] > 0 else 0.0
            meld_counts[rel - 1] = self.n_melds[tgt]
            river[rel - 1] = self.river[tgt]
            post_riichi_river[rel - 1] = self.post_riichi[tgt]

        # ---- v2 新增：时序 token + 结构标签（按 (事件, 座位) 缓存，4 视角共享）----
        tl_tokens = np.zeros((3, TL_MAX_LEN, TL_FEATURES), dtype=np.float32)
        tl_mask = np.zeros((3, TL_MAX_LEN), dtype=bool)
        tl_turn = np.zeros((3, TL_MAX_LEN), dtype=np.float32)
        tl_age = np.zeros((3, TL_MAX_LEN), dtype=np.float32)
        tl_consumed = np.full((3, TL_MAX_LEN, 2), -1, dtype=np.int8)
        tl_flags = np.zeros((3, TL_MAX_LEN, 2), dtype=np.float32)
        struct_form = np.zeros(3, dtype=np.int64)
        struct_counts = np.zeros((3, S_NUM_COUNTS, S_COUNT_BINS), dtype=np.float32)
        struct_shanten = np.zeros(3, dtype=np.int64)
        for rel in (1, 2, 3):
            tgt = (observer + rel) % 4
            arr, mask, turns, ages, cons, flags = self._tl_of(ev_idx, tgt)
            tl_tokens[rel - 1] = arr
            tl_mask[rel - 1] = mask
            tl_turn[rel - 1] = turns
            tl_age[rel - 1] = ages
            tl_consumed[rel - 1] = cons
            tl_flags[rel - 1] = flags
            st_lab = self._struct_of(ev_idx, tgt)
            struct_form[rel - 1] = st_lab['form']
            struct_counts[rel - 1] = st_lab['counts']
            struct_shanten[rel - 1] = min(st_lab['shanten'], MAX_SHANTEN_BUCKET)

        self.samples.append({
            'obs': obs,
            'unseen': unseen,
            'alloc': alloc,
            'wait': wait,
            'next_discard': next_discard,
            'next_tedashi': next_tedashi,
            'shanten': struct_shanten,
            'riichi': riichi_flag,
            'called': called_flag,
            'meld_counts': meld_counts,
            'dora_indicators': dora_count,
            'turn': np.int64(st_obs.at_turn),
            'tiles_left': np.int64(70 - self.n_tsumo),
            'tenpai': tenpai_flag,
            'river': river,
            'post_riichi_river': post_riichi_river,
            'own_hand': own_tehai > 0,
            'tl_tokens': tl_tokens,
            'tl_mask': tl_mask,
            'tl_turn': tl_turn,
            'tl_age': tl_age,
            'tl_consumed': tl_consumed,
            'tl_flags': tl_flags,
            'struct_form': struct_form,
            'struct_counts': struct_counts,
            '_observer': observer,
            '_event': ev_idx,
            '_kyoku': int(self.event_kyoku[ev_idx]),
        })

    def _decision_observers(self, etype: str) -> list[int]:
        """哪些观察者在本事件处理完毕后有合法动作（决策点）。

        判据与 GameplayLoader 的 ``cans.can_act()`` 对齐，但按事件类型直接推导
        （避免对 4 个状态重复查询）：

        - tsumo                → 只有 actor（要打牌/立直/自摸和/九种九牌）
        - dahai                → actor 之外所有 can_act 的人（吃/碰/大明杠/荣和）
        - chi / pon / reach    → actor（随后必须打牌；reach 即立直宣言）
        - kakan                → actor 之外可以枪杠的人（can_ron_agari）
        - daiminkan/ankan/hora/ryukyoku/end_* → 无人（杠后摸岭上，决策在后续事件）
        """
        if etype == 'tsumo':
            actor = self.events[self._cur_idx]['actor']
            st = self.states[actor]
            return [actor] if st.last_cans.can_act else []
        if etype == 'dahai':
            actor = self.events[self._cur_idx]['actor']
            return [p for p in range(4)
                    if p != actor and self.states[p].last_cans.can_act]
        if etype in ('chi', 'pon', 'reach'):
            actor = self.events[self._cur_idx]['actor']
            st = self.states[actor]
            return [actor] if st.last_cans.can_act else []
        if etype == 'kakan':
            actor = self.events[self._cur_idx]['actor']
            return [p for p in range(4)
                    if p != actor and self.states[p].last_cans.can_act]
        return []

    def run(self, stage: int = 3) -> list[dict]:
        for idx, (ev, line) in enumerate(zip(self.events, self.lines)):
            self._cur_idx = idx
            self._track_public(idx, ev)
            self.event_kyoku[idx] = self.kyoku
            for st in self.states:
                st.update(line)

            etype = ev.get('type')
            if etype in ('start_kyoku', 'reach_accepted'):
                observers = list(range(4))
            elif etype == 'dora':
                # dora 公告全部紧跟杠（ankan→dora→岭上摸牌，或 kakan→岭上摸牌→dora，
                # 实测 50 个牌谱 37 个 dora 事件无一例外）：此时杠家处于岭上循环中间态
                # （已杠未摸 = 13-3m-1，或已摸未打 = 13-3m+1），不在 3n+1 静止态，
                # 其 waits 标签可能是摸牌前的陈旧值。跳过该快照——新指示牌信息在紧随
                # 其后的下一个快照中必然可见，无信息损失，但保证全部标签语义干净。
                recent_kan = any(
                    self.events[k].get('type') in ('ankan', 'kakan', 'daiminkan')
                    for k in range(max(0, idx - 2), idx)
                )
                observers = [] if recent_kan else list(range(4))
            else:
                observers = self._decision_observers(etype)
            if observers:
                self.stats['n_snapshots'] += 1
                for observer in observers:
                    self._emit(idx, observer, stage=stage)

        self.stats['n_samples'] = len(self.samples)
        self.stats['n_kyoku'] = self.kyoku
        return self.samples


def replay_file(
    path: str | Path,
    *,
    version: int = 4,
    augment: bool = False,
    stage: int = 3,
    max_samples: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """回放单个 mjai .json.gz 牌谱，返回 (samples, stats)。

    version 只支持 4（v4 obs 是现役口径）；保留参数是为与主管线对齐。

    ⚠️ **内存口径（README §11.5-⑮，2026-10-20 审计）**：本函数把整个文件的样本
    **全部物化**后才返回（`ReplayFile.samples`），每条 obs 为 (1012,34) float32
    = 137KB ⇒ 实测单文件 1071 样本 ≈ **147MB 常驻**。DataLoader 每 worker 同时
    持有 1 个文件 + prefetch 批次，故 worker RSS ≈ 文件峰值 × NW（NW=16 时实测
    15.7GB/个，251GB 机器可用内存一度只剩 24GB）。`MALLOC_ARENA_MAX=2` 只缓解
    glibc arena 保留，**根治需流式化（逐样本 yield）或 obs 降精度**（通道多为
    0/1 与小整数，fp16/uint8 近无损）——该改动会变更数值/迭代口径，必须安排
    在轮次间隙，不可与在训实验并行。
    ⚠️ **2026-10-21 P0 已修**：worker RSS 线性增长的主因不是本节的文件物化
    高水位，而是 `structure._DECOMP_CACHE` 进程级缓存跨文件累积（值均值 236
    个分解/手、深尺寸 38KB，200k 上限 ⇒ 7.6GB 活对象/worker）。现在每次回放
    结束（含异常路径）调用 `clear_decomp_cache()` 释放；剩余 RSS 为单文件
    物化的阶段性高水位，会随文件切换复用而不再线性增长。
    stage：课程阶段过滤器（1=仅有立直家，2=巡目>=4，3=全部），在 obs 编码前生效。
    max_samples：早停保护（调试用），达到即裁剪。
    """
    assert version == 4, '读牌模型 v1 只支持 v4 obs'
    assert stage in (1, 2, 3), stage
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        raw = f.read()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]
    if augment:
        for ev in events:
            _augment_event_tids(ev)
        lines = [json.dumps(ev, separators=(',', ':')) for ev in events]

    rep = GameReplayer(events, lines)
    try:
        samples = rep.run(stage=stage)
    finally:
        # P0（2026-10-21）：分解 memo 是进程级缓存，值=全部向听层级的叶子
        # 分解（均值 236 条/手、深尺寸均值 38KB），跨文件累积会让每个 worker
        # 常驻 ~7.6GB 活对象（实测 RSS 11.4GB/个）。跨文件手牌几乎不重复，
        # 文件内复用已由 struct_cache 承担 ⇒ 回放结束（含异常）即清。
        clear_decomp_cache()
    stats = dict(rep.stats)
    stats['file'] = str(path)
    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]
        stats['truncated'] = True
    return samples, stats


def glob_files(pattern: str) -> list[str]:
    """展开 glob（支持 **）；返回排序后的列表。"""
    import glob as _glob
    return sorted(_glob.glob(pattern, recursive=True))


def month_of(path: str | Path) -> str:
    """'.../houou4/202601/xxx.json.gz' → '202601'。"""
    return Path(path).parent.name


def worker_shard(file_list: list[str], worker_id: int, num_workers: int) -> list[str]:
    """按 worker 交错切分文件列表（与 dataloader.worker_init_fn 同思路）。"""
    if num_workers <= 1:
        return list(file_list)
    return file_list[worker_id::num_workers]


def iter_files(file_list: list[str], rng: Optional[random.Random] = None) -> Iterator[str]:
    files = list(file_list)
    if rng is not None:
        rng.shuffle(files)
    yield from files
