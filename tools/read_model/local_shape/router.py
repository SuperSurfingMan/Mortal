#!/usr/bin/env python3
"""A2c / P2c 的模型选择：按场景整批选，而非逐窗口路由。

## 为什么不做逐窗口路由

`eval_router.py` 用缓存的概率测算（cond 口径，n=172942 非空样本）：

| 策略 | Δtop1（all/active_any） | ΔCE |
|---|---:|---:|
| `both_a2c`（active_both→A2c） | **+0.00059** | +0.00317 |
| `nonri_both_a2c`（非立直&both→A2c） | +0.00068 | +0.00195 |

`active_both` 只占 `active_any` 的 **5.63%**，路由收益被稀释到 0.07pp；
而逐窗口路由需要**同时跑两个模型**（P2c 解冻了骨干，不能复用 A2c 的前向），
推理成本翻倍。**收益 << 成本，不实现。**

## 正确形态：按场景整批选

同一批数据整体属于哪个场景，就用对应模型，只跑一次前向。

cond 口径下的分层结果（`eval_cond_recheck.py`）：

| 场景 | A2c | P2c | 选择 |
|---|---:|---:|---|
| all/active_any | 0.3549 | **0.3670** | p2c |
| riichi/all_centers | 0.2322 | **0.2832** | p2c |
| riichi/active_any | 0.2973 | **0.3525** | p2c |
| riichi/active_both | 0.6314 | **0.6483** | p2c |
| clean_t8/all_centers | 0.2229 | **0.2363** | p2c |
| clean_t8/active_any | 0.3425 | 0.3428 | p2c（持平，CE 更好） |
| all/active_both | **0.5067** | 0.4893 | a2c |
| clean_t8/active_both | **0.4758** | 0.4390 | a2c |

规律：**P2c 在立直层领先 5pp 级，A2c 只在「非立直 + 两端都手切」占优**。

## 用法

    from router import select
    model = select('riichi')          # -> 'p2c'
    model = select('clean_t8_active_both')  # -> 'a2c'

注意：`active_both` 场景若混有立直样本，应优先用 p2c（立直子集 P2c 更好）；
只有当**确认整批都是非立直 active_both** 时，才切 a2c。
"""
from __future__ import annotations

#: 场景 → (模型, 依据)
DECISION: dict[str, tuple[str, str]] = {
    'default': ('p2c',
                'all/active_any cond: P2c +1.20pp top1 / −0.177 nats CE'),
    'riichi': ('p2c',
               'riichi/all_centers +5.10pp、riichi/active_any +5.52pp、'
               'riichi/active_both +1.69pp'),
    'clean_t8': ('p2c',
                 'clean_t8/active_any 持平（0.3428 vs 0.3425），CE 更好；'
                 'all_centers +1.34pp'),
    'active_both': ('p2c',
                    'active_both 混有立直样本时 P2c 更好（riichi/both +1.69pp）；'
                    '仅当确认全为非立直时切 a2c'),
    'nonriichi_active_both': ('a2c',
                              'all/active_both A2c 0.5067 vs P2c 0.4893（+1.74pp）'),
    'clean_t8_active_both': ('a2c',
                             'clean_t8/active_both A2c 0.4758 vs P2c 0.4390（+3.67pp）'),
}


def select(scene: str) -> str:
    """按场景返回模型名（'a2c' 或 'p2c'）；未知场景回退 default。"""
    return DECISION.get(scene, DECISION['default'])[0]


def rationale(scene: str) -> str:
    """返回该场景的选择依据（供日志/报告引用）。"""
    m, why = DECISION.get(scene, DECISION['default'])
    return f'{scene} → {m}（{why}）'


if __name__ == '__main__':
    print(f'{"场景":<24s} {"模型":<6s} 依据')
    for k, (m, why) in DECISION.items():
        print(f'{k:<24s} {m:<6s} {why}')
