"""批量向听/待牌计算：直接复用 libriichi 的 tomohxx 查表（与 Rust 逐位同口径）。

为什么需要它：Bayes 上界的 MCMC / 粒子滤波要在一次快照上评估 **上千个候选暗牌**
（每个粒子 × 34 种听牌候选），`structure.py` 的分解枚举（~0.1ms/手牌）与
`PlayerState`（无法注入任意手牌）都不适用。本模块把 Rust `algo::shanten` 的
查表算法向量化到 numpy：输入 [N,34] 计数矩阵，一次算完 N 手牌。

算法严格照抄 `libriichi/src/algo/shanten.rs`：
  - `SUHAI_TABLE`（1,940,777 × 10）按「每 5 字节 → 10 个 nibble」解包；
  - `calc_normal` = 三个数牌花色依次 `add_suhai` + 字牌 `add_jihai`，
    返回 `ret[5 + len_div3] - 1`（和牌 = -1）；
  - `calc_all` = normal，若 `<=0` 或 `len_div3<4` 提前返回，否则再与七对/国士取 min。
表文件直接读 `libriichi/src/algo/data/*.bin.gz`（构建时 include_bytes 的同一份）。

对外接口：
    calc_all(counts, len_div3)             单张手牌 → int（-1 = 和牌）
    calc_all_vec(counts[N,34], len_div3)   → [N] int8
    tenpai_vec(counts[N,34], len_div3)     → [N] bool（shanten == 0）
    waits_vec(counts[N,34], len_div3, cand)→ (waits[N,C] bool, tenpai[N] bool)
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

NUM_TILES = 34
_TABLE_DIR = Path(__file__).resolve().parents[2] / 'libriichi' / 'src' / 'algo' / 'data'
_JIHAI_SIZE = 78_032
_SUHAI_SIZE = 1_940_777


def _load_table(name: str, size: int) -> np.ndarray:
    raw = gzip.decompress((_TABLE_DIR / name).read_bytes())
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(size, 5)
    out = np.empty((size, 10), dtype=np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F
    assert out.shape == (size, 10)
    return out


JIHAI_TABLE = _load_table('shanten_jihai.bin.gz', _JIHAI_SIZE)
SUHAI_TABLE = _load_table('shanten_suhai.bin.gz', _SUHAI_SIZE)


def _sum_tiles(t: np.ndarray) -> np.ndarray:
    """[N,k] 计数 → 5 进制下标（Rust sum_tiles 同口径）。"""
    acc = np.zeros(t.shape[0], dtype=np.int64)
    for i in range(t.shape[1]):
        acc = acc * 5 + t[:, i]
    return acc


def _add_suhai(lhs: np.ndarray, index: np.ndarray, m: int) -> None:
    tab = SUHAI_TABLE[index]
    for j in range(5 + m, 4, -1):
        sht = np.minimum(lhs[:, j] + tab[:, 0], lhs[:, 0] + tab[:, j])
        for k in range(5, j):
            sht = np.minimum(sht, np.minimum(lhs[:, k] + tab[:, j - k],
                                             lhs[:, j - k] + tab[:, k]))
        lhs[:, j] = sht
    for j in range(m, -1, -1):
        sht = lhs[:, j] + tab[:, 0]
        for k in range(j):
            sht = np.minimum(sht, lhs[:, k] + tab[:, j - k])
        lhs[:, j] = sht


def _add_jihai(lhs: np.ndarray, index: np.ndarray, m: int) -> None:
    tab = JIHAI_TABLE[index]
    j = m + 5
    sht = np.minimum(lhs[:, j] + tab[:, 0], lhs[:, 0] + tab[:, j])
    for k in range(5, j):
        sht = np.minimum(sht, np.minimum(lhs[:, k] + tab[:, j - k],
                                         lhs[:, j - k] + tab[:, k]))
    lhs[:, j] = sht


def calc_normal_vec(counts: np.ndarray, len_div3) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int16)
    n = counts.shape[0]
    ret = SUHAI_TABLE[_sum_tiles(counts[:, :9])].astype(np.int16)
    ld = np.full(n, int(len_div3), dtype=np.int64) if np.isscalar(len_div3) \
        else np.asarray(len_div3, dtype=np.int64)
    # 同一批里 len_div3 通常相同；不同时按值分组，避免逐行 m 循环
    for m in np.unique(ld):
        rows = np.flatnonzero(ld == m)
        sub = ret[rows]
        for lo in (9, 18):
            _add_suhai(sub, _sum_tiles(counts[rows, lo:lo + 9]), int(m))
        _add_jihai(sub, _sum_tiles(counts[rows, 27:]), int(m))
        ret[rows] = sub
    return ret[np.arange(n), 5 + ld] - 1


def _chitoi(counts: np.ndarray) -> np.ndarray:
    kinds = (counts > 0).sum(-1)
    pairs = (counts >= 2).sum(-1)
    return 7 - pairs + np.maximum(7 - kinds, 0) - 1


def _kokushi_counts():
    return np.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], dtype=np.int64)


def _kokushi(counts: np.ndarray) -> np.ndarray:
    sub = counts[:, _kokushi_counts()]
    kinds = (sub > 0).sum(-1)
    pairs = (sub >= 2).sum(-1)
    return 14 - kinds - (pairs > 0).astype(np.int64) - 1


def calc_all_vec(counts: np.ndarray, len_div3) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int16)
    n = counts.shape[0]
    ld = np.full(n, int(len_div3), dtype=np.int64) if np.isscalar(len_div3) \
        else np.asarray(len_div3, dtype=np.int64)
    s = calc_normal_vec(counts, ld)
    # 与 Rust calc_all 的提前退出同口径：normal<=0 或 len_div3<4 不再查七对/国士
    mask = (s > 0) & (ld >= 4)
    if mask.any():
        rows = np.flatnonzero(mask)
        s[rows] = np.minimum(s[rows], _chitoi(counts[rows]))
        still = rows[s[rows] > 0]
        if len(still):
            s[still] = np.minimum(s[still], _kokushi(counts[still]))
    return s.astype(np.int8)


def calc_all(counts, len_div3: int) -> int:
    c = np.asarray(counts, dtype=np.int16).reshape(1, NUM_TILES)
    return int(calc_all_vec(c, int(len_div3))[0])


def tenpai_vec(counts: np.ndarray, len_div3) -> np.ndarray:
    return calc_all_vec(counts, len_div3) == 0


def wait_candidates(counts: np.ndarray) -> np.ndarray:
    """一批手牌的听牌候选并集（同色 ±2 邻域 + 国士 13 种 + 手内牌）。

    和牌张必然参与某个面子/雀首：数牌必与手内某张同色距离 <=2；国士另算；
    七对 = 手内已有的牌。这里是**候选集**（超集），最终仍由 calc_all 判定。
    """
    counts = np.asarray(counts, dtype=np.int16)
    out = np.zeros(NUM_TILES, dtype=bool)
    out[_kokushi_counts()] = True
    for s in range(3):
        sub = counts[:, s * 9:(s + 1) * 9] > 0
        for off in (1, 2):
            if off < 9:
                out[s * 9:s * 9 + 9 - off] |= sub[:, off:].any(0)
                out[s * 9 + off:s * 9 + 9] |= sub[:, :-off].any(0)
    out |= (counts > 0).any(0)
    return out


def waits_vec(counts: np.ndarray, len_div3, cand: np.ndarray | None = None):
    """批量待牌：返回 (waits[N,C] bool, tenpai[N] bool)。cand 默认 34 种全查。"""
    counts = np.asarray(counts, dtype=np.int16)
    n = counts.shape[0]
    if cand is None:
        cand = np.arange(NUM_TILES)
    cand = np.asarray(cand, dtype=np.int64)
    c = len(cand)
    expanded = np.repeat(counts, c, axis=0)
    t = np.tile(cand, n)
    ok = expanded[np.arange(n * c), t] < 4          # 第 5 张不存在
    expanded[np.arange(n * c), t] = np.where(ok, expanded[np.arange(n * c), t] + 1,
                                             expanded[np.arange(n * c), t])
    ld = np.full(n * c, int(len_div3), dtype=np.int64) if np.isscalar(len_div3) \
        else np.repeat(np.asarray(len_div3, dtype=np.int64), c)
    sh = calc_all_vec(expanded, ld).reshape(n, c)
    waits = (sh == -1) & ok.reshape(n, c)
    return waits, (calc_all_vec(counts, len_div3) == 0)
