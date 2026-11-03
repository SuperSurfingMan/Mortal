#!/usr/bin/env python3
"""语义标签：把 27 类计数形状映射到 10 类局部结构角色。

## 动机

27 类计数形状 `(c_a, c_b, c_c)` 把**语义**抹平了：

- `(1,1,0)` 在 a=1（1-2）时是**边张**，只能等 3；
- `(1,1,0)` 在 a=2（2-3）时是**两面**，等 1 或 4。

两者对下游（守卫 risk、危险度、EV）的价值完全不同，但计数形状归为同一类。
"每中心先验"部分吸收了位置差异，但模型不知道**为什么**这两类不同。

## 10 类定义（局部、无歧义、不依赖全局分解）

三张连续窗口 (a, b, c)，持有计数截断到 0/1/2：

| idx | 名称 | 判定 |
|---:|---|---|
| 0 | EMPTY | 三张都不持有 |
| 1 | MENJEN | 三张全持有 = **顺子** |
| 2 | PAIR_C | 中心对子 + 至少一侧邻张 |
| 3 | PAIR_C_ONLY | 中心对子、两侧都无（**雀头候选**） |
| 4 | PAIR_SIDE | 一侧对子（a 或 c ≥2）、中心非对子 |
| 5 | RYANMEN | 相邻两张且非边张 = **两面** |
| 6 | PENCHAN | 相邻两张且为 1-2 / 8-9 = **边张** |
| 7 | KANCHAN | 隔张两张（a,c 持有、b 无）= **嵌张** |
| 8 | SINGLE_C | 只有中心一张 |
| 9 | SINGLE_SIDE | 只有一侧一张 |

关键增益：**RYANMEN vs PENCHAN 用中心位置判定**，这是 27 类计数形状
无法区分的自由度；对同花色的 `(1,1,0)`，a=1 归 PENCHAN、a≥2 归 RYANMEN。

## 完备性

27 个 `(c_a,c_b,c_c) ∈ {0,1,2}³` 到 10 类的映射是**确定且完备**的
（`mapping_matrix` 每行恰有一个 1），因此 tag 转换不丢样本、不产生歧义。
"""
from __future__ import annotations

import numpy as np

NUM_SEMANTIC = 10
SEM_NAMES = ('EMPTY', 'MENJEN', 'PAIR_C', 'PAIR_C_ONLY', 'PAIR_SIDE',
             'RYANMEN', 'PENCHAN', 'KANCHAN', 'SINGLE_C', 'SINGLE_SIDE')


def semantic_index(counts, center) -> np.ndarray:
    """counts [...,3]（a,b,c 的持有数）→ 语义类索引 [...]。

    center = (tile_a, tile_b, tile_c)；边张判定只看同花色点数：
      a-b 相邻且 a 是 1（num_a==0）⇒ 边张；
      b-c 相邻且 c 是 9（num_c==8）⇒ 边张。
    """
    c = np.clip(np.asarray(counts, dtype=np.int64), 0, 2)
    a, b, d = c[..., 0], c[..., 1], c[..., 2]
    num_a = int(center[0]) % 9
    num_d = int(center[2]) % 9

    # 默认 SINGLE_SIDE；后续 where 逐个覆盖（顺序敏感，勿随意调整）
    out = np.full(a.shape, 9, dtype=np.int8)
    out = np.where((a == 0) & (b == 0) & (d == 0), 0, out)
    out = np.where((a >= 1) & (b >= 1) & (d >= 1), 1, out)
    out = np.where((b >= 2) & ((a >= 1) | (d >= 1)), 2, out)
    out = np.where((b >= 2) & (a == 0) & (d == 0), 3, out)
    out = np.where(((a >= 2) | (d >= 2)) & (b < 2), 4, out)

    adj_ab = (a == 1) & (b == 1) & (d == 0)
    adj_bc = (b == 1) & (d == 1) & (a == 0)
    out = np.where(adj_ab & (num_a > 0), 5, out)
    out = np.where(adj_ab & (num_a == 0), 6, out)
    out = np.where(adj_bc & (num_d < 8), 5, out)
    out = np.where(adj_bc & (num_d == 8), 6, out)

    out = np.where((a >= 1) & (d >= 1) & (b == 0), 7, out)
    out = np.where((a == 0) & (b == 1) & (d == 0), 8, out)
    return out


def mapping_matrix(center) -> np.ndarray:
    """[27, 10] 确定性映射矩阵：27 类计数形状 → 10 类语义。

    每行恰有一个 1，因此 `joint27 @ M` 是合法的边缘化（不重不漏）。
    """
    m = np.zeros((27, NUM_SEMANTIC), dtype=np.float64)
    for idx in range(27):
        ca, cb, cc = idx // 9, (idx // 3) % 3, idx % 3
        m[idx, int(semantic_index(np.asarray([ca, cb, cc]), center))] = 1.0
    return m


def to_semantic(joint27, center) -> np.ndarray:
    """[N,27] → [N,10]。"""
    return np.asarray(joint27, dtype=np.float64) @ mapping_matrix(center)


def coverage_check(centers) -> dict:
    """自检：每个中心上 27 类是否都被映射（无遗漏）且分布合理。"""
    rows = []
    for center in centers:
        m = mapping_matrix(center)
        rows.append({
            'center': [int(x) for x in center],
            'classes_used': int((m.sum(0) > 0).sum()),
            'row_ok': bool(np.all(m.sum(1) == 1.0)),
            'has_both_edges': bool(m[12, 5] > 0 and m[12, 6] > 0
                                   or m[4, 5] > 0 and m[4, 6] > 0),
        })
    return {'n_centers': len(rows),
            'all_rows_ok': all(r['row_ok'] for r in rows),
            'min_classes_used': min(r['classes_used'] for r in rows),
            'max_classes_used': max(r['classes_used'] for r in rows),
            'rows': rows}
