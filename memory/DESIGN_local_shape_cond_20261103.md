# 局部形状：cond 口径重算与路线修正（2026-11-03）

## 0. 一句话

`active_any` 下最大类是 **EMPTY（空窗口）**，占 56.04%，导致 P0 以来全部 micro top1
被「永远猜空」主导。改用 cond 口径（筛非空 + EMPTY 置 0 重归一）重算后，
A2c 的真实增益从 3.23% 升到 11.08%，P2c 的 FAIL 判定翻正但**分层互有胜负**。

## 1. 口径定义

| 口径 | 样本 | 概率 | 对应问题 |
|---|---|---|---|
| micro | 全部 | 原样 | 原报告口径（被 EMPTY 主导）|
| nonempty | y≠0 | 原样 | 已知非空但未条件化（EMPTY 抢 argmax）|
| **cond** | y≠0 | EMPTY 置 0 重归一 | **已知有牌，问是什么结构** |

## 2. 标签结构

| 标签 | 类数 | 最大类占比 | 熵 | 有效类数 |
|---|---:|---:|---:|---:|
| 27 类计数形状（active_any，micro） | 27 | 0.5604 (EMPTY) | 1.678 | 5.4 |
| 27 类（active_any，cond） | 27 | 0.2745 | 2.273 | 9.7 |
| 10 类语义（active_any，cond） | 10 | 0.3716 | 1.774 | 5.9 |

## 3. cond 重算结果（`eval_cond_recheck.py`，549s）

### 3.1 主口径（all/active_any，n=172942）

| 模型 | top1 | top3 | CE | 归一化增益 | CE 降幅 |
|---|---:|---:|---:|---:|---:|
| prior | 0.2745 | — | 2.2725 | 0 | 0 |
| A2c | 0.3549 | 0.6297 | 2.0359 | 0.1108 | 0.1041 |
| P1 | 0.3492 | 0.6750 | 1.9442 | 0.1029 | 0.1444 |
| P2a | 0.3248 | 0.6094 | 2.0874 | 0.0693 | 0.0814 |
| P2b | 0.3575 | 0.6784 | 1.9225 | 0.1143 | 0.1540 |
| **P2c** | **0.3670** | **0.6939** | **1.8585** | **0.1274** | **0.1822** |

### 3.2 分层胜负（rel A2c，正=局部头好）

| 模型 | all/any | all/both | riichi/both | clean_t8/both | all/any CE |
|---|---:|---:|---:|---:|---:|
| P1 | −0.0058 | −0.0319 | −0.0572 | −0.0479 | +0.0917 |
| P2a | −0.0301 | −0.0498 | −0.0551 | −0.0587 | −0.0514 |
| P2b | +0.0025 | −0.0331 | −0.0328 | −0.0467 | +0.1134 |
| **P2c** | **+0.0120** | −0.0174 | **+0.0169** | −0.0367 | **+0.1774** |

### 3.3 立直盲区（cond）

| | top1 | CE |
|---|---:|---:|
| 先验（riichi/active_both） | 0.6875 | 0.6454 |
| A2c | 0.6314 | 1.4052 |
| P2c | 0.6483 | 1.1822 |

三者 top1 全部低于先验、CE 全是先验的 2 倍 ⇒ **现有读牌器在「已立直 + 两端都手切」
上比瞎猜还差**，且不是加头能解决的。

## 4. 结论修正清单

| 原结论 | 处置 | 修正后 |
|---|---|---|
| P2c「strict top1 未过 A2c ⇒ FAIL」 | **改** | cond 下 +1.20pp，改为「分层互有胜负」 |
| 「局部形状线关闭」 | **改** | 「有条件成立」：校准/立直组件可用，通用头不重启 200k |
| 「A2c 只比先验高 1.4pp，信号稀薄」 | **改** | cond 下 +8.0pp / 归一化 11.1% |
| P1/P2b PASS via CE | 保留 | 注明是校准而非判别 |
| P2a FAIL | 保留 | cond 下更差，结论更稳 |
| `bayes_ceiling` SIGNAL_SHALLOW | 待定 | 同一口径问题，需 cond 重算 tier |

## 5. 四条技术启示

1. **「判有无」与「判形状」是两个任务**。A2c 在 nonempty（不重归一）下仅 0.0873，
   说明概率质量大量压在 EMPTY 上，本质是非空检测器；拆成两级更合理。
2. **P2c 与 A2c 应按子集路由**（点预测场景）：
   `active_both → A2c`，其余 → `P2c`。零训练成本。
3. **立直家是盲区**，只能靠换任务定义（待牌命中而非 27 类形状）突破。
4. **「CE 赢、top1 输」是稳定模式**：P1/P2b 的容量被校准吃掉；它们只对概率型下游
   （守卫 risk、EV 重排）有价值。

## 6. 推荐路线（按性价比）

| 优先级 | 动作 | 成本 | 收益 |
|---:|---|---|---|
| 1 | 标签诊断（最大类占比 + 有效类数），扫全部在读任务 | 30 min | 找出还有多少门禁建在错误口径上 |
| 2 | A2c/P2c 按子集路由 | 1 h | 零训练拿到两者各自最好的一面 |
| 3 | P1/P2b 定位为「概率校准组件」，在守卫线验证 | 半天 | CE −0.18 nats 可落地 |
| 4 | `bayes_ceiling` tier 在 cond 下重算 | 90 min | 决定「上限」叙事是否成立 |
| 5 | 立直子集换任务定义（待牌命中） | 1-2 天 | 唯一能突破盲区的方向 |

## 7. 局限

- cond 口径只在「已知窗口非空」时成立；下游若未知非空，仍应走两阶段或 micro。
- 只覆盖 27 类局部形状；其他读牌任务是否有同类偏差，需第 1 项诊断回答。
- P2c 的 +1.20pp 在 n=172942 下显著，但绝对量级仍小（21 中心 × 27 类，
  每中心约多猜对 0.03 类）。
- `bayes_ceiling` 的三层 tier（counts/hard/behave）同样受 EMPTY 主导，
  其「信息量≈0」的结论未经 cond 验证。

## 8. 产物

| 文件 | 说明 |
|---|---|
| `local_shape/eval_cond_recheck.py` | 重算脚本（`--reuse-probs`；概率缓存 `/tmp/ls_cond_probs.npz` 589MB）|
| `local_shape/results/COND_RECHECK_REPORT.md` | 完整报告（4 表 + 结论）|
| `local_shape/results/cond_recheck.json` | 全量指标（6 层 × 3 子集 × 3 口径 × 7 模型）|
| `local_shape/semantic_label.py` / `eval_semantic.py` | 语义标签定义与验证（NO-GO）|
| `local_shape/results/SEMANTIC_LABEL_REPORT.md` | 语义标签报告 |
| `local_shape/README.md` §5b/§5c/§5d | 三节结论 |

## 9. 标签诊断结果（`tools/read_model/label_diag.py`，4s）

扫了 8 个单标签任务 + 1 个多标签任务（待牌），覆盖 `all` / `riichi` / `clean_t8`
三层与 `all_centers` / `active_any` / `active_both` 三子集。

| 任务 | K | p_max | eff | 风险 | 当前指标 | 被 top1 坑 |
|---|---:|---:|---:|---|---|---|
| shape27 | 27 | 0.5604 | 5.8 | HIGH | top1 + CE | **是** |
| shape10 | 10 | 0.5604 | 4.4 | HIGH | top1 + CE | **是** |
| count5 | 5 | 0.7062 | 2.2 | HIGH | CE + MAE | 否 |
| presence2 | 2 | 0.7062 | 1.8 | HIGH | AUC + BCE | 否 |
| tenpai2 | 2 | 0.8526 | 1.5 | HIGH | AUC | 否 |
| riichi2 | 2 | 0.9291 | 1.3 | HIGH | BCE / AUC | 否 |
| called2 | 2 | 0.8347 | 1.6 | HIGH | BCE / AUC | 否 |
| shanten8 | 8 | 0.2864 | 5.0 | MED | accuracy | 需双报 |

**四条结论**

1. **只有 3 个任务真正被 top1 坑：`shape27` / `shape10` / `shanten8`**。
   其余 5 个虽集中（p_max 0.71~0.93），但用 AUC/CE/MAE/hit@k，**未受影响**。
   这把“口径债务”的范围从“全部任务”缩到 3 个。
2. **5 个退化项**（p_max = 1.0，标签恒定）：`tenpai2@riichi`、`riichi2@riichi`、
   `riichi2@clean_t8`、`called2@clean_t8`、`shanten8@riichi`。成因：立直必听牌、
   clean_t8 必不立直/不副露。这些子集上的指标恒为 100%，**应直接从评估中移除**。
3. **mask 越严标签越集中**：shape27 的 p_max 在 all_centers 0.3304 → active_any 0.5604
   → active_both 0.7350（eff 11.4 → 5.8 → 3.1）。
   ⇒ `active_both` 本身就是一个人为制造的高集中子集，在此上做 top1 门禁尤其危险。
4. **立直层的待牌标签反而是最健康的**：多标签最大正例率 **0.117**、平均待牌 1.81、
   无待牌率 0。对比同层 shape27 的 p_max 0.6547 ⇒
   支持“立直子集换成待牌命中任务”的建议（路线第 5 项）。

产物：`diagnostics/results/LABEL_DIAG_REPORT.md` / `label_diag.json`。

## 10. 路由决策（`eval_router.py` + `router.py`）

### 10.1 逐窗口路由已证伪

| 策略 | Δtop1（all/active_any） | ΔCE |
|---|---:|---:|
| `both→A2c` | **+0.00059** | +0.00317 |
| `非立直&both→A2c` | +0.00068 | +0.00195 |

原因：`active_both` 只占 `active_any` 的 **5.63%**（cond 非空），收益被稀释到 0.07pp；
而逐窗口路由需**双模型推理**（P2c 解冻了骨干，无法复用 A2c 前向），成本翻倍。
**0.07pp << 2× 成本 ⇒ 不实现。**

### 10.2 正确形态：按场景整批选

| 场景 | A2c | P2c | 选择 |
|---|---:|---:|---|
| all/active_any | 0.3549 | **0.3670** | p2c |
| riichi/all_centers | 0.2322 | **0.2832** | p2c |
| riichi/active_any | 0.2973 | **0.3525** | p2c |
| riichi/active_both | 0.6314 | **0.6483** | p2c |
| clean_t8/all_centers | 0.2229 | **0.2363** | p2c |
| clean_t8/active_any | 0.3425 | 0.3428 | p2c（CE 更好） |
| all/active_both | **0.5067** | 0.4893 | a2c |
| clean_t8/active_both | **0.4758** | 0.4390 | a2c |

**新发现**：P2c 在 `riichi/all_centers` 领先 **+5.10pp**、`riichi/active_any` **+5.52pp**，
远大于此前只看 `riichi/active_both` 的 +1.69pp。**立直层是 P2c 最大的优势区**。

规则：**默认 P2c；只有「确认整批为非立直 active_both」才切 A2c。**

实现：`router.select(scene)` 一行调用，决策集中在 `router.DECISION` 一处。

## 11. P0-a：BAYES_CEILING 三层 tier 按 cond 重算（2026-11-03，5451s）

目标：回答「P1/P2a/P2c 的 FAIL 是否属于后验本身接近均匀的预期结果」。

实现：`local_shape/p0a_bayes_cond.py`。MC 采样顺序与 `bayes_ceiling.py` **逐位
相同**（同 seed、同循环结构），只在累加时跳过 `y == 0` ⇒ cond 与 micro 是同一批
后验样本的两种打分口径，可配对比较。counts 用精确多元超几何（全量，无 MC 误差）。
自检 6/6（counts micro top1 与旧 JSON 逐位一致；cond prior 的 n 与
`eval_cond_recheck.json` 完全相同）。

### 11.1 主表（cond CE，越小越好）

| cell | n | prior | counts | hard | behave | A2c | P2c | strict 信息 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all/all_centers | 168.7万 | 2.662 | 2.624 | **2.609** | 2.808 | 2.473 | **2.409** | +0.038 | INFO_NEGLIGIBLE |
| all/active_any | 17.3万 | 2.272 | 2.249 | **2.209** | 2.496 | 2.036 | **1.858** | +0.023 | INFO_NEGLIGIBLE |
| all/active_both | 5865 | **1.674** | 1.834 | 1.819 | 2.231 | 1.593 | **1.499** | −0.160 | TIER_MISFIT |
| riichi/all_centers | 11.7万 | **2.429** | 2.805 | 2.502 | 3.977 | 2.494 | **2.192** | −0.376 | TIER_MISFIT |
| riichi/active_any | 22672 | **2.022** | 2.513 | 2.167 | 3.370 | 2.257 | **1.847** | −0.490 | TIER_MISFIT |
| riichi/active_both | 944 | **0.642** | 1.925 | 1.786 | 2.338 | 1.405 | 1.182 | −1.284 | TIER_MISFIT |
| clean_t8/all_centers | 40.2万 | 2.750 | **2.665** | 2.675 | 2.961 | 2.467 | **2.408** | +0.084 | INFO_SMALL |
| clean_t8/active_any | 80145 | 2.339 | **2.211** | 2.213 | 2.474 | 2.040 | **1.942** | +0.129 | INFO_SMALL |
| clean_t8/active_both | 3321 | **1.630** | 1.799 | 1.766 | 2.171 | 1.660 | 1.633 | −0.168 | TIER_MISFIT |

（粗体 = 该行最低。注意 `active_both` 系列与立直/`called` 层的 **prior 才是最低**。）

### 11.2 四条结论

1. **tier 信息确实很小，但不是零**：strict（精确 counts）在 all/active_any 只有
   **+0.023 nats**，clean_t8/active_any **+0.129**，turn>=8/active_any **+0.076**。
   ⇒ 「27 类形状在给定可观测信息下接近不可预测」的方向成立。
2. **但 tier 不是上界**：在 **8 个“tier 优于 prior”的 cell** 上，A2c 的 CE 比最强
   tier 低 **0.135~0.209 nats**（all/active_any 0.173、nonriichi/active_any 0.206、
   turn>=8/all_centers 0.209）。其余 10 个 cell 的 prior 已优于所有 tier
   （TIER_MISFIT），谈“tier 上界”无意义。
3. **失配集中在 active_both / 立直 / called**（strict 信息 −0.08~−1.28）。原因：
   这几层的 **mask 本身含信息**（「这两张曾被手切」⇒ 窗口非空概率极高），
   而 tier 的均匀子集假设完全不用 mask ⇒ 条件化后分布错位。这反而是「tier 是
   信息子集」的最直接证据：**mask 就是 tier 不知道的信息**。
4. **P2c 在 15/18 个 cell 上 CE 最低**（3 个例外全是 active_both 系，那里
   评估集 oracle prior 更低）；P1/P2b 也全面优于 A2c。
   ⇒ **P1/P2a/P2c 的 FAIL 是 top1 门禁的判定，不是能力不足**；在 CE/校准口径
   上它们已经超过 A2c。

### 11.3 副产品：Dirichlet α 语义错误（`bayes_ceiling.py`，已修）

旧实现 `p = (cnt + α/27)/(wsum + α)` 中 `wsum = 1`，`α=0.5` 相当于 **33% 均匀混合**：

| cell | 精确 counts | MC（旧 α） | MC（新 α） |
|---|---:|---:|---:|
| all/active_any | 2.2486 | 2.4708 | 2.2514 |
| all/active_both | 1.8338 | 2.1857 | 1.8395 |
| clean_t8/active_any | 2.2105 | 2.4390 | 2.2151 |

- 与精确 counts 的 **L1 距离 0.43 → 0.04**；MC tier 的 CE 偏高 **0.22~0.35 nats**。
- **top1 不受影响**（加常数不改 argmax）⇒ §3 的 top1 结论仍有效，但其 CE 结论
  （「CE 全部高于边缘熵 ⇒ 均匀子集被手切打破」）**归因不完整**：主因是平滑口径。
- 修正：按**等效样本数**计，`bayes_ceiling.py` 加 `--alpha-mode {legacy,count}`
  （默认 legacy 保旧结果可复现，新分析用 count）。

### 11.4 限制与处置

- **behave 未收敛**：M 收敛 Δ 达 0.157（riichi/active_both 0.494）⇒ M=4096 对 27 类
  长尾不足，其 CE 有正偏（Jensen），**「behave 比 hard 差」有相当部分是采样偏差**。
  behave 只作上界参考，不作判据。
- **prior 分组口径**：本脚本按 (中心, 目标家) 分 **63 组**（与 micro prior 同口径）；
  `eval_cond_recheck.py` 按中心分 **21 组**。细分组更强 ⇒ 小样本 cell 的 prior top1
  差可达 0.3pp，两处的 `norm_gain` 不可混用。
- **对「关闭」的处置**：原「按预注册门禁关闭」保留操作性（不重启 200k），但理由应改为
  「**实现未达 A2c（top1 门禁），但上限未被证明**」，而不是「任务信号稀薄、已达天花板」。

产物：`local_shape/results/P0A_BAYES_COND_REPORT.md` / `p0a_bayes_cond.json`；
脚本 `p0a_bayes_cond.py` + `run_p0a.sh`。
