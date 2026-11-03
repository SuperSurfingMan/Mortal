#!/usr/bin/env python3
"""Phase 3：逐牌「在不在」二分类的 token 特征。

目标：给定**观察者自家手牌 + 全场公开信息**，对每个目标家 j（相对座位 1/2/3）
输出 34 类「该牌是否在他手里」的概率（不预测张数）。

两类 token（交叉注意力的两个流）：

- **牌 token** `[3, 34, FT]`：每张牌一个 query。内容是"这张牌自己的公开履历"：
  目标家对它的手切/摸切次数与最早巡目、立直宣言牌、鸣牌占用、
  河里是否出现、对当前立直家的现物/筋计数、危险度、剩余张数、自家是否持有等。
- **动作 token** `[3, 26, FE]`：目标家的每一条公开动作（打牌/吃/碰/杠/立直），
  带 **手切 vs 摸切**、动作类型、巡目/距快照年龄、立直后、被鸣、
  以及"这张打牌对当前立直家有多安全"（现物/筋/危险度）。
- 另有一条 **context 向量** `[3, FC]`：巡目、牌山、立直/副露/手牌数、
  该家最近打牌的安全比例、自家手牌花色分布与宝牌数。

危险度口径（纯公开信息，无泄漏）
--------------------------------
对每个立直家 k，`danger_k(x)` 在 [0,1]：
  现物（x 在 k 河里）→ 0；`unseen[x]==0` → 0（无人可能持有）；
  筋（与宣言牌同花色同 0-based rank%3 组）→ ×0.45；幺九 → ×0.8；
  `unseen[x]<=1` → ×0.7；基值 0.12。
`risk[j,x]` = 对 j 的**立直对手**取均值；`risk_max` 取最大值。
这不是精确放铳率，只是"他是否在打安全牌"的可读信号。

注：样本里没有观察者自己的立直标志，故安全/危险只按另外三家计算。
"""
from __future__ import annotations

import numpy as np

NUM_TILES = 34
TL_MAX_LEN = 26
ACTION_KINDS = ('dahai', 'reach', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan')
A_DAHAI, A_REACH, A_CHI, A_PON, A_DAIMINKAN, A_ANKAN, A_KAKAN = range(7)
MELD_ACTIONS = (A_CHI, A_PON, A_DAIMINKAN, A_ANKAN, A_KAKAN)

FT = 22          # 牌 token 特征维
FE = 22          # 动作 token 特征维
FC = 18          # context 特征维

TILE_FEATURE_NAMES = (
    'tedashi_n', 'tsumogiri_n', 'has_tedashi', 'tedashi_first_turn',
    'has_tsumogiri', 'tsumogiri_first_turn', 'post_riichi_n', 'last_age',
    'is_decl_tile', 'meld_used_n', 'in_river', 'post_riichi_river',
    'unseen', 'own_hand', 'walled', 'is_dora', 'is_dora_indicator',
    'safe_riichi_cnt', 'suji_cnt', 'risk_mean', 'opp_river_cnt', 'unseen_le1',
)
EVENT_FEATURE_NAMES = (
    'tsumogiri', 'act_dahai', 'act_reach', 'act_chi', 'act_pon',
    'act_daiminkan', 'act_ankan', 'act_kakan', 'turn', 'age',
    'post_riichi', 'called', 'has_consumed1', 'has_consumed2',
    'safe_riichi_n', 'safe_vs_all_riichi', 'risk_mean', 'risk_max',
    'unseen', 'is_last', 'position', 'is_decl_discard',
)
CONTEXT_FEATURE_NAMES = (
    'turn', 'tiles_left', 'riichi', 'called', 'meld_count', 'hand_size',
    'n_riichi_other', 'n_riichi_total', 'defense_frac', 'last3_risk',
    'own_man', 'own_pin', 'own_sou', 'own_honor', 'own_dora', 'dora_total',
    'discard_n', 'tedashi_frac',
)


def _successor(tids: np.ndarray) -> np.ndarray:
    """宝牌指示牌 tid → 实际宝牌 tid（与 rule_baseline.dora_successor 同口径）。"""
    tids = np.asarray(tids, dtype=np.int64)
    out = tids.copy()
    suit = tids // 9
    num = tids % 9
    out = np.where((suit < 3) & (num == 8), tids - 8, tids + 1)
    out = np.where(tids == 30, 27, out)
    out = np.where(tids == 33, 31, out)
    return out


def _suji_mask(decl: int) -> np.ndarray:
    """立直宣言牌 decl 的筋牌（同花色、0-based rank%3 相同、非自身）。"""
    m = np.zeros(NUM_TILES, dtype=bool)
    if decl < 0:
        return m
    suit, num = divmod(int(decl), 9)
    if suit >= 3:
        return m
    base = suit * 9
    for r in range(9):
        if r % 3 == num % 3 and r != num:
            m[base + r] = True
    return m


def _decl_tiles(tl: np.ndarray, mask: np.ndarray):
    """reach token 之后该家的第一张 dahai = 立直宣言牌（-1=未立直）。"""
    tile_oh = tl[:, :, :NUM_TILES]
    act = np.argmax(tl[:, :, NUM_TILES + 1:NUM_TILES + 1 + len(ACTION_KINDS)], -1)
    decl = np.full(3, -1, dtype=np.int64)
    decl_idx = np.full(3, -1, dtype=np.int64)
    for j in range(3):
        rk = np.where((act[j] == A_REACH) & mask[j])[0]
        if rk.size == 0:
            continue
        r = int(rk[0])
        for k in range(r + 1, tl.shape[1]):
            if not mask[j, k]:
                continue
            if act[j, k] == A_DAHAI and tile_oh[j, k].sum() > 0.5:
                decl[j] = int(np.argmax(tile_oh[j, k]))
                decl_idx[j] = k
                break
    return decl, decl_idx, act


def _risk_matrix(river: np.ndarray, decl: np.ndarray, riichi: np.ndarray,
                 unseen: np.ndarray):
    """按当前立直家算 34 张牌的危险度（对每个目标家 j）。"""
    base = np.full(NUM_TILES, 0.12, dtype=np.float64)
    terminal = np.zeros(NUM_TILES, dtype=bool)
    terminal[[0, 8, 9, 17, 18, 26]] = True
    d0 = np.where(terminal, base * 0.8, base)
    d0 = np.where(unseen <= 1, d0 * 0.7, d0)
    d0 = np.where(unseen <= 0, 0.0, d0)

    risk = np.zeros((3, NUM_TILES), dtype=np.float32)
    risk_max = np.zeros((3, NUM_TILES), dtype=np.float32)
    safe_cnt = np.zeros((3, NUM_TILES), dtype=np.float32)
    suji_cnt = np.zeros((3, NUM_TILES), dtype=np.float32)
    safe_all = np.zeros((3, NUM_TILES), dtype=np.float32)

    for j in range(3):
        threats = [k for k in range(3) if k != j and riichi[k] > 0.5]
        if not threats:
            continue
        acc = np.zeros(NUM_TILES, dtype=np.float64)
        hi = np.zeros(NUM_TILES, dtype=np.float64)
        safe = np.zeros(NUM_TILES, dtype=np.float64)
        suji = np.zeros(NUM_TILES, dtype=np.float64)
        for k in threats:
            suji_k = _suji_mask(int(decl[k]))
            dk = np.where(suji_k, d0 * 0.45, d0)
            dk = np.where(river[k], 0.0, dk)
            acc += dk
            hi = np.maximum(hi, dk)
            safe += river[k]
            suji += suji_k
        n = len(threats)
        risk[j] = acc / n
        risk_max[j] = hi
        safe_cnt[j] = safe
        suji_cnt[j] = suji
        safe_all[j] = np.all([river[k] for k in threats], axis=0)
    return risk, risk_max, safe_cnt, suji_cnt, safe_all


def build_sample(s: dict) -> dict:
    """一个 replay 样本 → 紧凑 token 特征（numpy，无 torch）。"""
    tl = np.asarray(s['tl_tokens'], dtype=np.float32)
    mask = np.asarray(s['tl_mask'], dtype=bool)
    turn_n = np.asarray(s['tl_turn'], dtype=np.float32)        # 已 /18 并 clip 1.5
    age_n = np.asarray(s['tl_age'], dtype=np.float32)           # 已 /18 并 clip 1.5
    flags = np.asarray(s['tl_flags'], dtype=np.float32)
    cons = np.asarray(s['tl_consumed'], dtype=np.int64)
    unseen = np.asarray(s['unseen'], dtype=np.float32)
    own = np.asarray(s['own_hand'], dtype=bool)
    river = np.asarray(s['river'], dtype=bool)
    post = np.asarray(s['post_riichi_river'], dtype=bool)
    riichi = np.asarray(s['riichi'], dtype=np.float32)
    called = np.asarray(s['called'], dtype=np.float32)
    melds = np.asarray(s['meld_counts'], dtype=np.int64)
    dora_ind = np.asarray(s['dora_indicators'], dtype=np.float64)
    alloc = np.asarray(s['alloc'], dtype=np.uint8)
    tenpai = np.asarray(s['tenpai'], dtype=np.float32)
    turn = float(s['turn'])
    tiles_left = float(s['tiles_left'])

    L = tl.shape[1]
    tile_oh = tl[:, :, :NUM_TILES]
    has_tile = tile_oh.sum(-1) > 0.5
    tid = np.argmax(tile_oh, -1)
    tsumogiri = tl[:, :, NUM_TILES] > 0.5
    decl, decl_idx, act = _decl_tiles(tl, mask)
    is_dahai = (act == A_DAHAI) & mask
    is_meld = np.isin(act, MELD_ACTIONS) & mask
    post_flag = flags[:, :, 0] > 0.5
    called_flag = flags[:, :, 1] > 0.5

    risk, risk_max, safe_cnt, suji_cnt, safe_all = _risk_matrix(
        river, decl, riichi, unseen)

    # ---- 逐牌聚合（目标家 j 的公开履历）----
    ted_n = np.zeros((3, NUM_TILES), dtype=np.float32)
    tsu_n = np.zeros((3, NUM_TILES), dtype=np.float32)
    disc_n = np.zeros((3, NUM_TILES), dtype=np.float32)
    post_n = np.zeros((3, NUM_TILES), dtype=np.float32)
    meld_n = np.zeros((3, NUM_TILES), dtype=np.float32)
    first_ted = np.full((3, NUM_TILES), 1.5, dtype=np.float32)
    first_tsu = np.full((3, NUM_TILES), 1.5, dtype=np.float32)
    last_age = np.full((3, NUM_TILES), 1.5, dtype=np.float32)
    for j in range(3):
        m = mask[j] & has_tile[j]
        d_m = m & is_dahai[j]
        t_m = d_m & ~tsumogiri[j]
        s_m = d_m & tsumogiri[j]
        np.add.at(ted_n[j], tid[j][t_m], 1.0)
        np.add.at(tsu_n[j], tid[j][s_m], 1.0)
        np.add.at(disc_n[j], tid[j][d_m], 1.0)
        np.minimum.at(first_ted[j], tid[j][t_m], turn_n[j][t_m])
        np.minimum.at(first_tsu[j], tid[j][s_m], turn_n[j][s_m])
        np.minimum.at(last_age[j], tid[j][d_m], age_n[j][d_m])
        np.add.at(post_n[j], tid[j][d_m & post_flag[j]], 1.0)
        mm = m & is_meld[j]
        np.add.at(meld_n[j], tid[j][mm], 1.0)

    dora_succ = _successor(np.arange(NUM_TILES))
    dora_cnt = np.zeros(NUM_TILES, dtype=np.float32)
    dora_cnt[dora_succ] = dora_ind.astype(np.float32)
    ar = np.arange(NUM_TILES)
    opp_river = river.sum(0)[None, :] - river

    T = np.zeros((3, NUM_TILES, FT), dtype=np.float32)
    T[..., 0] = ted_n / 4.0
    T[..., 1] = tsu_n / 4.0
    T[..., 2] = (ted_n > 0)
    T[..., 3] = np.where(ted_n > 0, first_ted, 0.0) / 1.5
    T[..., 4] = (tsu_n > 0)
    T[..., 5] = np.where(tsu_n > 0, first_tsu, 0.0) / 1.5
    T[..., 6] = post_n / 4.0
    T[..., 7] = np.where(disc_n > 0, last_age, 0.0) / 1.5
    T[..., 8] = (ar[None, :] == decl[:, None])
    T[..., 9] = meld_n / 2.0
    T[..., 10] = river
    T[..., 11] = post
    T[..., 12] = unseen[None, :] / 4.0
    T[..., 13] = own[None, :]
    T[..., 14] = (unseen[None, :] <= 0)
    T[..., 15] = dora_cnt[None, :] > 0
    T[..., 16] = (dora_ind[None, :] > 0)
    T[..., 17] = safe_cnt / 3.0
    T[..., 18] = suji_cnt / 3.0
    T[..., 19] = risk
    T[..., 20] = opp_river / 3.0
    T[..., 21] = (unseen[None, :] <= 1)

    # ---- 动作 token ----
    e_risk = np.take_along_axis(risk, tid, axis=-1)
    e_riskm = np.take_along_axis(risk_max, tid, axis=-1)
    e_safe = np.take_along_axis(safe_cnt, tid, axis=-1)
    e_safe_all = np.take_along_axis(safe_all, tid, axis=-1)
    last_idx = mask.sum(1) - 1
    E = np.zeros((3, L, FE), dtype=np.float32)
    E[..., 0] = tsumogiri
    E[..., 1:8] = tl[:, :, NUM_TILES + 1:NUM_TILES + 1 + len(ACTION_KINDS)]
    E[..., 8] = turn_n
    E[..., 9] = age_n
    E[..., 10] = flags[..., 0]
    E[..., 11] = flags[..., 1]
    E[..., 12] = (cons[..., 0] >= 0)
    E[..., 13] = (cons[..., 1] >= 0)
    E[..., 14] = e_safe / 3.0 * is_dahai
    E[..., 15] = e_safe_all * is_dahai
    E[..., 16] = e_risk * is_dahai
    E[..., 17] = e_riskm * is_dahai
    E[..., 18] = unseen[tid] / 4.0
    for j in range(3):
        if last_idx[j] >= 0:
            E[j, int(last_idx[j]), 19] = 1.0
        if decl_idx[j] >= 0:
            E[j, int(decl_idx[j]), 21] = 1.0
    E[..., 20] = (np.arange(L, dtype=np.float32) / L)[None, :]
    E *= mask[..., None]
    ev_tile = np.where(mask, tid, 0).astype(np.int16)

    # ---- context ----
    Cx = np.zeros((3, FC), dtype=np.float32)
    own_f = own.astype(np.float32)
    dora_own = float((own_f * dora_cnt).sum())
    for j in range(3):
        dm = mask[j] & is_dahai[j]
        n_d = float(dm.sum())
        if riichi.sum() > 0.5:
            Cx[j, 8] = float((e_safe[j] > 0.5)[dm].mean()) if n_d > 0 else 0.0
        else:
            Cx[j, 8] = 0.0
        idx_d = np.where(dm)[0]
        if idx_d.size > 0:
            Cx[j, 9] = float(e_risk[j][idx_d][-3:].mean())
        Cx[j, 16] = n_d / 18.0
        n_ted = float((dm & ~tsumogiri[j]).sum())
        Cx[j, 17] = n_ted / max(n_d, 1.0)
    Cx[:, 0] = min(turn / 18.0, 1.5)
    Cx[:, 1] = tiles_left / 70.0
    Cx[:, 2] = riichi
    Cx[:, 3] = called
    Cx[:, 4] = np.clip(melds, 0, 4) / 4.0
    Cx[:, 5] = np.clip(13 - 3 * melds, 0, 13) / 13.0
    Cx[:, 6] = (riichi.sum() - riichi) / 3.0
    Cx[:, 7] = riichi.sum() / 3.0
    Cx[:, 10] = own_f[:9].sum() / 14.0
    Cx[:, 11] = own_f[9:18].sum() / 14.0
    Cx[:, 12] = own_f[18:27].sum() / 14.0
    Cx[:, 13] = own_f[27:].sum() / 14.0
    Cx[:, 14] = dora_own / 4.0
    Cx[:, 15] = float(dora_cnt.sum()) / 4.0

    return {
        'tile_feat': T,
        'ev_tile': ev_tile,
        'ev_feat': E,
        'ev_mask': mask,
        'cons': np.where(cons >= 0, cons, -1).astype(np.int8),
        'ctx': Cx,
        'y': (alloc > 0).astype(np.uint8),
        'alloc': alloc,
        'unseen': unseen.astype(np.float16),
        'own_hand': own,
        'tenpai': tenpai,
        'turn': np.int16(min(turn, 30)),
        'riichi': riichi > 0.5,
        'called': called > 0.5,
        'meld_counts': melds.astype(np.int8),
    }


ARRAY_KEYS = ('tile_feat', 'ev_tile', 'ev_feat', 'ev_mask', 'cons', 'ctx', 'y',
              'alloc', 'unseen', 'own_hand', 'tenpai', 'turn', 'riichi', 'called',
              'meld_counts')


def stack(samples: list[dict]) -> dict:
    """把 build_sample 的列表按 key 堆叠并压缩 dtype。"""
    out = {}
    for k in ARRAY_KEYS:
        arr = np.stack([s[k] for s in samples])
        if k in ('tile_feat', 'ev_feat', 'ctx', 'unseen', 'tenpai'):
            arr = arr.astype(np.float16)
        out[k] = arr
    return out
