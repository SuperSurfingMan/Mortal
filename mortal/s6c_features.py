"""S6c：主模型「自家速度」2 通道（自家向听 + 实战有效牌数），纯 Python 侧计算。

**为什么是这两通道（2026-10-23 探测结论）**

- v4 obs 第 862..868 行已有向听 one-hot，但它在**打牌后**才更新（`state/update.rs`
  的 `update_shanten` 挂在 dahai 上），所以 3n+2（自摸后 14 张）状态里仍是摸牌前的
  向听：实测 13 张状态 260/260 全等、14 张只有 827/1092（失配全部是 obs = 真实 − 1）；
  且该 one-hot 在 shanten≥6 处塌成一格。故「当前向听」仍有新信息。
- SP 表只在 `SHANTEN_THRES=3` 以内计算，shanten≥4 占决策点 ~8.7%，此时
  `required_tiles` 等 119 个 SP 通道全零 —— **有效牌数是已知输入缺失**。

**口径（与 libriichi 逐位对齐）**

- 向听用 `libriichi/src/algo/data/shanten_{suhai,jihai}.bin.gz` 的 tomohxx 表，
  本模块是 `algo/shanten.rs` 的 numpy 向量化移植（含 `len_div3` 副露口径、
  `L<4` 不算七对/国士、和了返回 −1 的语义）。与 Rust 单测 19 例逐位一致（见
  `tools/test_s6c_features.py`）。
- 3n+2：逐个可打牌求 13 张手牌的向听，`best` 取最小；候选 = 达到 best 的打牌。
  3n+1：`best` 即当前手牌向听。
- `ch0 = clamp(best, 0, 8) / 8`；`ch1 = max_候选(有效牌剩余枚数) / 136`。
  有效牌 = 摸进后向听前进 1 的牌种，剩余枚数 = `4 − tiles_seen`（实战口径，
  扣掉弃牌/副露/宝牌指示；13 张向听=0 时即和了牌）。`best ≤ −1`（已和了）
  两通道置 0。
- 两通道沿 34 列广播（与 S6a 一致），首层 Conv1d 从 1012 扩到 1014。

**布局依赖**：`obs[0:4]` = 自家手牌计数；`obs[835]` = `tiles_seen/4`（与
`defense_guard.py` / `tools/read_model/precompute_danger.py` 同一常量，均已逐位
验证）。换 obs 版本必须同步改这里并 bump `data_version`。

**开销**：2579 样本实测 0.158 ms/样本（按文件批量，8 worker 下约 2% 单核）。
"""
from __future__ import annotations

import gzip
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

SPEC = 's6c'
NUM_EXTRA_CHANNELS = 2

NUM_TILES = 34
NUM_COPIES = 4
MAX_SHANTEN = 8
OWN_HAND_ROWS = slice(0, 4)      # v4：自家 tehai 计数 0..4（0/1 填充）
TILES_SEEN_ROW = 835             # v4：tiles_seen = count/4
_KAWA_DIV = 136.0                # 有效牌归一化分母（4*34，与 tiles_left/69 同量级）

_TABLE_SIZES = (1_940_777, 78_032)   # shanten_suhai / shanten_jihai
_POW5_9 = (5 ** np.arange(8, -1, -1)).astype(np.int64)
_POW5_7 = (5 ** np.arange(6, -1, -1)).astype(np.int64)
_KOKUSHI = np.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
_CHUNK = 4096                    # 有效牌逐候选分块（控峰值内存）

_TABLES: Optional[Tuple[np.ndarray, np.ndarray]] = None


# --------------------------------------------------------------------------- #
# 表：tomohxx shanten table（与 libriichi 同源，字节级一致）
# --------------------------------------------------------------------------- #
def _data_dir() -> Path:
    env = os.environ.get('MORTAL_S6C_DATA_DIR')
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / 'libriichi' / 'src' / 'algo' / 'data'


def _load_table(path: Path, size: int) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 5)
    out = np.empty((b.shape[0], 10), dtype=np.uint8)
    out[:, 0::2] = b & 0b1111
    out[:, 1::2] = (b >> 4) & 0b1111
    if out.shape[0] != size:                      # 表损坏/版本不符时立刻炸
        raise RuntimeError(f'{path} 条目数 {out.shape[0]} != {size}')
    return np.ascontiguousarray(out)


def tables() -> Tuple[np.ndarray, np.ndarray]:
    """懒加载 (SUHAI, JIHAI)；缺失时给出可执行的修法而不是 IndexError。"""
    global _TABLES
    if _TABLES is None:
        d = _data_dir()
        suhai, jihai = d / 'shanten_suhai.bin.gz', d / 'shanten_jihai.bin.gz'
        for p in (suhai, jihai):
            if not p.exists():
                raise FileNotFoundError(
                    f'S6c 需要向听表 {p}（libriichi 源码树自带）；'
                    '部署包请把两个 bin.gz 放到同一目录并设 MORTAL_S6C_DATA_DIR')
        _TABLES = (_load_table(suhai, _TABLE_SIZES[0]),
                   _load_table(jihai, _TABLE_SIZES[1]))
    return _TABLES


# --------------------------------------------------------------------------- #
# 向听（algo/shanten.rs 的 numpy 移植，返回未截断值，可为 −1）
# --------------------------------------------------------------------------- #
def _lookup(tab: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """同 Rust `.get(idx).unwrap_or_default()`：越界返回全 0。"""
    out = np.zeros((idx.shape[0], 10), dtype=np.int32)
    ok = idx < tab.shape[0]
    out[ok] = tab[idx[ok]]
    return out


def _add_suhai(lhs: np.ndarray, tab: np.ndarray, m: int) -> None:
    for j in range(5 + m, 4, -1):
        cand = np.minimum(lhs[:, j] + tab[:, 0], lhs[:, 0] + tab[:, j])
        for k in range(5, j):
            cand = np.minimum(cand, lhs[:, k] + tab[:, j - k])
            cand = np.minimum(cand, lhs[:, j - k] + tab[:, k])
        lhs[:, j] = cand
    for j in range(m, -1, -1):
        cand = lhs[:, j] + tab[:, 0]
        for k in range(0, j):
            cand = np.minimum(cand, lhs[:, k] + tab[:, j - k])
        lhs[:, j] = cand


def _add_jihai(lhs: np.ndarray, tab: np.ndarray, m: int) -> None:
    j = m + 5
    cand = np.minimum(lhs[:, j] + tab[:, 0], lhs[:, 0] + tab[:, j])
    for k in range(5, j):
        cand = np.minimum(cand, lhs[:, k] + tab[:, j - k])
        cand = np.minimum(cand, lhs[:, j - k] + tab[:, k])
    lhs[:, j] = cand


def shanten_normal(counts: np.ndarray, len_div3: int) -> np.ndarray:
    """标准形向听（单一路径，不截断）。counts (N,34) 整数，len_div3 标量。"""
    suhai, jihai = tables()
    counts = np.asarray(counts, dtype=np.int64)
    ret = _lookup(suhai, counts[:, :9] @ _POW5_9).copy()
    _add_suhai(ret, _lookup(suhai, counts[:, 9:18] @ _POW5_9), len_div3)
    _add_suhai(ret, _lookup(suhai, counts[:, 18:27] @ _POW5_9), len_div3)
    _add_jihai(ret, _lookup(jihai, counts[:, 27:34] @ _POW5_7), len_div3)
    return ret[:, 5 + len_div3] - 1


def shanten_all(counts: np.ndarray, len_div3: int) -> np.ndarray:
    """同 `shanten::calc_all`：normal/七对/国士取 min，**不**做 max(0)。

    `len_div3 < 4` 或 normal ≤ 0 时提前退出（与 Rust 一致，此时七对/国士不可能更优）。
    """
    out = shanten_normal(counts, len_div3)
    if len_div3 < 4:
        return out
    counts = np.asarray(counts, dtype=np.int16)
    kinds = (counts > 0).sum(1)
    pairs = (counts >= 2).sum(1)
    chitoi = 7 - pairs + np.maximum(0, 7 - kinds) - 1
    use = out > 0
    if use.any():
        out[use] = np.minimum(out[use], chitoi[use])
    k = counts[:, _KOKUSHI]
    kokushi = 14 - (k > 0).sum(1) - ((k >= 2).sum(1) > 0) - 1
    use = out > 0
    if use.any():
        out[use] = np.minimum(out[use], kokushi[use])
    return out


def _shanten_grouped(counts: np.ndarray, len_div3: np.ndarray) -> np.ndarray:
    """按 len_div3 分组算（同一文件里副露数分布很窄，分组数通常 1~5）。"""
    out = np.empty(counts.shape[0], dtype=np.int8)
    for lv in np.unique(len_div3):
        sel = len_div3 == lv
        out[sel] = shanten_all(counts[sel], int(lv))
    return out


# --------------------------------------------------------------------------- #
# 特征
# --------------------------------------------------------------------------- #
def compute(obs: np.ndarray) -> np.ndarray:
    """(N,1012,34) 或 (1012,34) float32 -> (N,2) float32：向听/8、有效牌/136。"""
    obs = np.asarray(obs, dtype=np.float32)
    squeeze = obs.ndim == 2
    if squeeze:
        obs = obs[None]
    counts = np.rint(obs[:, OWN_HAND_ROWS, :].sum(1)).astype(np.int16)
    seen = np.rint(obs[:, TILES_SEEN_ROW, :] * NUM_COPIES).astype(np.int16)
    if np.any(seen < counts):
        raise RuntimeError(
            f'obs[{TILES_SEEN_ROW}] 不再是 tiles_seen（出现 seen < 自家手牌）；'
            'obs 布局变了，需同步更新 s6c_features 并 bump data_version')

    n = counts.sum(1).astype(np.int32)
    len_div3 = (n // 3).astype(np.int8)
    out = np.zeros((counts.shape[0], 2), dtype=np.float32)
    usable = (n % 3) != 0
    two = (n % 3) == 2
    if not np.all(usable):
        warnings.warn(f'S6c: {(~usable).sum()} 个样本手牌张数 %3==0，两通道置 0')
    if not usable.any():
        return out if not squeeze else out[0]

    best = np.full(counts.shape[0], 99, dtype=np.int8)

    # --- 3n+1：当前 13 张手牌 ---
    idx1 = np.where(usable & ~two)[0]
    if len(idx1):
        best[idx1] = _shanten_grouped(counts[idx1], len_div3[idx1])

    # --- 3n+2：逐个可打牌，取最优 ---
    idx2 = np.where(usable & two)[0]
    cand_s = cand_t = vals = None
    if len(idx2):
        c2 = counts[idx2]
        pos, tile = np.nonzero(c2)               # (M,)：样本内位置 / 牌号
        cand_s = idx2[pos]
        cand_t = tile.astype(np.int64)
        hand13 = counts[cand_s].copy()
        hand13[np.arange(len(cand_s)), cand_t] -= 1
        vals = _shanten_grouped(hand13, len_div3[cand_s])
        np.minimum.at(best, cand_s, vals)
    out[:, 0] = np.clip(best, 0, MAX_SHANTEN) / MAX_SHANTEN

    # --- 有效牌：候选手牌 × 34 种摸牌，向听前进 1 即有效 ---
    s_parts = [idx1]
    h_parts = [counts[idx1]]
    b_parts = [best[idx1]]
    l_parts = [len_div3[idx1]]
    if len(idx2):
        # 候选 = 达到最优向听的打牌；best<0（已和了）不参与
        keep = (vals == best[cand_s]) & (best[cand_s] >= 0)
        if keep.any():
            ks, kt = cand_s[keep], cand_t[keep]
            hk = counts[ks].copy()
            hk[np.arange(len(ks)), kt] -= 1
            s_parts.append(ks)
            h_parts.append(hk)
            b_parts.append(best[ks])
            l_parts.append(len_div3[ks])
    s_all = np.concatenate(s_parts)
    h_all = np.concatenate(h_parts)
    b_all = np.concatenate(b_parts)
    l_all = np.concatenate(l_parts)

    eff_best = np.zeros(counts.shape[0], dtype=np.int32)
    for start in range(0, len(s_all), _CHUNK):
        stop = min(start + _CHUNK, len(s_all))
        h = h_all[start:stop].copy()
        base = b_all[start:stop]
        lv = l_all[start:stop]
        rem = np.maximum(NUM_COPIES - seen[s_all[start:stop]], 0)
        eff = np.zeros(stop - start, dtype=np.int32)
        for u in range(NUM_TILES):
            need = rem[:, u]
            if not (need > 0).any():
                continue
            h[:, u] += 1
            v = _shanten_grouped(h, lv)
            h[:, u] -= 1
            eff += np.where((v == base - 1) & (need > 0), need, 0)
        np.maximum.at(eff_best, s_all[start:stop], eff)
    out[:, 1] = eff_best / _KAWA_DIV

    # 已和了（best == −1）：不需要速度信号，两通道归零
    win = best < 0
    if win.any():
        out[win] = 0.0
    return out if not squeeze else out[0]


def attach(obs: np.ndarray, feat: Optional[np.ndarray] = None) -> np.ndarray:
    """把 2 通道广播后拼到 obs 尾部：(N,1012,34) -> (N,1014,34)（单样本同理）。"""
    obs = np.asarray(obs, dtype=np.float32)
    squeeze = obs.ndim == 2
    o = obs[None] if squeeze else obs
    if feat is None:
        feat = compute(o)
    feat = np.asarray(feat, dtype=np.float32)
    if feat.shape != (o.shape[0], NUM_EXTRA_CHANNELS):
        raise ValueError(f'feat shape {feat.shape} != {(o.shape[0], NUM_EXTRA_CHANNELS)}')
    rows = np.repeat(feat[:, :, None], NUM_TILES, axis=2)
    out = np.concatenate([o, rows], axis=1)
    return out[0] if squeeze else out


def resolve(control) -> Tuple[str, int]:
    """从 `[control]` 解析 (spec, 额外通道数)；未知组合直接报错而非静默按 0 处理。"""
    control = control or {}
    spec = str(control.get('extra_features', '') or '')
    n = int(control.get('extra_channels', 0) or 0)
    if spec == SPEC:
        if n not in (0, NUM_EXTRA_CHANNELS):
            raise ValueError(f'extra_features={SPEC} 需 extra_channels={NUM_EXTRA_CHANNELS}，实际 {n}')
        return SPEC, NUM_EXTRA_CHANNELS
    if n:
        raise ValueError(
            f'extra_channels={n} 但 extra_features={spec!r} 不可识别'
            f'（S6c 注入需显式写 extra_features = "{SPEC}"）')
    return '', 0
