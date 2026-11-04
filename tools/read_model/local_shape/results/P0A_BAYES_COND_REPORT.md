# P0-a：BAYES_CEILING 三层 tier 按 cond 口径重算

- 缓存：`/tmp/read_model_diag_cache_a2c/cache.npz`（n=40000；alloc 校验 PASS）
- MC 子集：6000 快照（seed=20261103；tiers=counts,hard,behave；M=[256, 1024, 4096]；chains=256；iters=128；β=1.0；K≤10）
- counts 精确自检：max|P(a=0) − CPMF0| = 0.00e+00（0 处）
- 口径：P0 的 eval 划分与 mask；**cond = 筛 `y != 0` + EMPTY 置 0 重归一**（「已知窗口非空，问是什么结构」）。
- 采样顺序与 `bayes_ceiling.py` **逐位相同**（同 seed、同循环结构），只在累加时跳过 `y == 0` ⇒ cond 与 micro 是同一批后验样本的两种打分。
- `counts` = 精确多元超几何（全量、无 MC 误差）；`hard` = counts + 立直⇒听牌（MC）；`behave` = hard + 舍牌行为似然（MC，构造性近似）。
- **口径限制**：三 tier 只含「张数 + 立直约束 + 静态舍牌似然」，不含牌河顺序/宣言牌/逐巡事件 ⇒ 它是信息**子集**，实测头可以超过它。

## 1. 自检与交叉校验

| 项 | 本脚本 | 参考 | 一致 |
|---|---:|---:|---|
| counts micro top1 all/active_any | 0.5540667983416075 | 0.5540667983416075 | ✓ |
| cond prior all/active_any | 0.274537/n=172942 | 0.27452555298805237/n=172942 | ✓ |
| counts micro top1 all/active_both | 0.7243934396602358 | 0.7243934396602358 | ✓ |
| cond prior all/active_both | 0.400682/n=5865 | 0.4006820023059845/n=5865 | ✓ |
| counts micro top1 riichi/active_both | 0.6225310899780542 | 0.6225310899780542 | ✓ |
| cond prior riichi/active_both | 0.690678/n=944 | 0.6875/n=944 | ✓ |

`counts` 的 micro 自检值同时与旧 `bayes_ceiling.json` 对齐，证明两次运行的 eval 划分与 mask 完全相同。

**prior 分组口径（必读）**：本脚本的 prior 按 **(中心, 目标家)** 分 **63 组**（与 `bayes_ceiling` 的 micro prior 同口径，保证 micro/cond 配对可比）；`eval_cond_recheck.json` 的 prior 按 **(中心)** 分 **21 组**（三家合并，先验更弱）。两者样本数完全相同，先验值差 0.0032（最大，见下表）；
`cond_recheck` 报的 A2c/P2c `norm_gain` 基于 21 组，在小样本 cell 上偏乐观。本报告主表用 63 组先验重算 `norm_gain63`。

| cell | 63 组 prior top1（本） | 21 组 prior top1（recheck） | 差 |
|---|---:|---:|---:|
| all/active_any | 0.2745 | 0.2745 | 0.0000 |
| all/active_both | 0.4007 | 0.4007 | 0.0000 |
| riichi/active_both | 0.6907 | 0.6875 | 0.0032 |
| clean_t8/active_both | 0.3917 | 0.3887 | 0.0030 |

## 2. 采样量 M 收敛检查（cond CE）

**all / active_any**

| tier | M=256 | M=1024 | M=4096 | Δ(max−min) |
|---|---|---|---|---|
| counts | 2.3228 | 2.2656 | 2.2514 | 0.07141 |
| hard | 2.2737 | 2.2192 | 2.2087 | 0.06502 |
| behave | 2.6536 | 2.5545 | 2.4962 | 0.15734 |

**riichi / active_both**

| tier | M=256 | M=1024 | M=4096 | Δ(max−min) |
|---|---|---|---|---|
| counts | 1.9745 | 1.9592 | 1.9479 | 0.02664 |
| hard | 1.7940 | 1.7901 | 1.7862 | 0.00783 |
| behave | 2.8316 | 2.5568 | 2.3379 | 0.49369 |

**解读**：counts/hard 的 Δ≤0.07 已基本收敛；`behave` 的 Δ 达 0.157（riichi/active_both 甚至 0.494）⇒ M=4096 对 behave （27 类长尾）**仍不足**，其 CE 被正偏（`-log p̂` 的 Jensen 效应）抬高，“behave 比 hard 差”有相当部分是采样偏差，不是纯失配。

## 2b. MC 后验的 Dirichlet 平滑口径（α 语义修正）

`bayes_ceiling.py` 的原始实现把 `α` 作用在**归一化到 1 的权重**上：`p = (cnt + α/27) / (wsum + α)`，`wsum = 1`。`α=0.5` ⇒ **33% 的均匀混合**，把分布压平。

- 与精确 counts 的 **L1 距离 0.43**；大类概率被压低约 25% ⇒ CE 系统偏高。
- **top1 不受影响**（加常数不改变 argmax），所以旧结果的 top1 仍有效。
- 修正后 α 按**等效样本数**解释：`p = (cnt + α/27)/(M + α)`，L1 降到 **0.04**。

| cell | 精确 counts CE | MC counts（旧 α） | MC counts（新 α） |
|---|---:|---:|---:|
| all/active_any | 2.2486 | 2.4708 | 2.2514 |
| all/active_both | 1.8338 | 2.1857 | 1.8395 |
| clean_t8/active_any | 2.2105 | 2.4390 | 2.2151 |

**影响范围**：`BAYES_CEILING_REPORT.md`（micro）里所有 MC tier（`counts_mc`/`hard`/`behave`）的 CE 都被旧 α 抬高约 0.2~0.3 nats，其 top1 仍有效。因此 micro 时代「CE 全部高于标签边缘熵 ⇒ 均匀子集假设被手切打破」的**归因不完整**：主因是平滑口径，行为失配是次要项。

## 3. 熵分解（cond 口径；CE 为主判据，top1 仅对照）

`micro` 列取自旧 `bayes_ceiling.json`（同一后验的旧打分口径），`cond` 列为本脚本。


**all / all_centers**（cond n=1687499；micro n=2520000）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 2.6618 | 0.1665 | 0.4115 | 2.4098 | 0.3304 |
| A2c 独立边际乘积 | 2.4733 | 0.2172 | 0.4868 | 2.1732 | 0.3363 |
| P1 冻结局部头 | 2.4506 | 0.2178 | 0.5018 | 2.1639 | 0.3319 |
| P2a 立直显式特征 | 2.5156 | 0.2023 | 0.4787 | 2.2067 | 0.3323 |
| P2b behave 蒸馏 | 2.4442 | 0.2205 | 0.5043 | 2.1592 | 0.3372 |
| P2c 受控解冻 | 2.4091 | 0.2280 | 0.5135 | 2.1263 | 0.3402 |
| Bayes counts 边际乘积 | 2.6130 | 0.1647 | 0.4163 | 2.3399 | 0.3289 |
| Bayes counts 联合（精确全量） | 2.6242 | 0.1655 | 0.4157 | 2.3492 | 0.3276 |
| Bayes counts 联合（MC 子集） | 2.6304 | 0.1661 | 0.4160 | 2.4528 | 0.3270 |
| Bayes hard（+立直听牌） | 2.6086 | 0.1659 | 0.4194 | 2.4421 | 0.3290 |
| Bayes behave（+行为似然） | 2.8084 | 0.1618 | 0.3872 | 2.4793 | 0.3289 |

判定（**CE 主判据**）：**INFO_NEGLIGIBLE（tier 信息 ≤0.05 nats） + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）**；strict（counts）信息 0.0375 nats；full（behave）-0.1466 nats（34.1σ；最强 tier=hard，CE=2.609）；A2c 比最强 tier 低 0.135 nats；P2c 低 0.200 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.005，6.5σ）。

**riichi / all_centers**（cond n=116685；micro n=178584）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 2.4288 | 0.1943 | 0.5115 | 2.2204 | 0.3466 |
| A2c 独立边际乘积 | 2.4944 | 0.2322 | 0.4809 | 2.1872 | 0.3471 |
| P1 冻结局部头 | 2.2917 | 0.2629 | 0.5933 | 2.0529 | 0.3412 |
| P2a 立直显式特征 | 2.3974 | 0.2525 | 0.5580 | 2.1166 | 0.3555 |
| P2b behave 蒸馏 | 2.2975 | 0.2621 | 0.5846 | 2.0470 | 0.3633 |
| P2c 受控解冻 | 2.1922 | 0.2832 | 0.6137 | 1.9602 | 0.3745 |
| Bayes counts 边际乘积 | 2.7772 | 0.1039 | 0.2926 | 2.4249 | 0.3340 |
| Bayes counts 联合（精确全量） | 2.8052 | 0.1069 | 0.2912 | 2.4484 | 0.3268 |
| Bayes counts 联合（MC 子集） | 2.8155 | 0.1088 | 0.2983 | 2.5340 | 0.3217 |
| Bayes hard（+立直听牌） | 2.5016 | 0.1072 | 0.3496 | 2.3815 | 0.3449 |
| Bayes behave（+行为似然） | 3.9773 | 0.0928 | 0.2379 | 2.7776 | 0.3270 |

判定（**CE 主判据**）：**TIER_MISFIT（counts CE 高于 prior ⇒ 假设失效）**；strict（counts）信息 -0.3765 nats；full（behave）-1.5485 nats（87.2σ；最强 tier=hard，CE=2.502）；A2c 比最强 tier 低 0.007 nats；P2c 低 0.309 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.102，46.1σ）。

**clean_t8 / all_centers**（cond n=402053；micro n=578424）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 2.7496 | 0.1564 | 0.3827 | 2.5136 | 0.3049 |
| A2c 独立边际乘积 | 2.4674 | 0.2229 | 0.4812 | 2.1948 | 0.3256 |
| P1 冻结局部头 | 2.4574 | 0.2243 | 0.4943 | 2.1972 | 0.3165 |
| P2a 立直显式特征 | 2.5661 | 0.2055 | 0.4556 | 2.2709 | 0.3148 |
| P2b behave 蒸馏 | 2.4480 | 0.2300 | 0.5000 | 2.1906 | 0.3226 |
| P2c 受控解冻 | 2.4083 | 0.2363 | 0.5068 | 2.1515 | 0.3273 |
| Bayes counts 边际乘积 | 2.6519 | 0.1618 | 0.3863 | 2.4090 | 0.3028 |
| Bayes counts 联合（精确全量） | 2.6652 | 0.1629 | 0.3860 | 2.4207 | 0.3000 |
| Bayes counts 联合（MC 子集） | 2.6753 | 0.1635 | 0.3836 | 2.5168 | 0.2980 |
| Bayes hard（+立直听牌） | 2.6745 | 0.1624 | 0.3831 | 2.5166 | 0.2996 |
| Bayes behave（+行为似然） | 2.9610 | 0.1329 | 0.3185 | 2.5949 | 0.3038 |

判定（**CE 主判据**）：**INFO_SMALL（0.05~0.15 nats） + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）**；strict（counts）信息 0.0844 nats；full（behave）-0.2114 nats（37.7σ；最强 tier=counts，CE=2.665）；A2c 比最强 tier 低 0.198 nats；P2c 低 0.257 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.023，16.9σ）。

**all / active_any**（cond n=172942；micro n=393393）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 2.2718 | 0.2745 | 0.5393 | 1.6781 | 0.5604 |
| A2c 独立边际乘积 | 2.0359 | 0.3549 | 0.6297 | 1.4323 | 0.5746 |
| P1 冻结局部头 | 1.9442 | 0.3492 | 0.6750 | 1.4089 | 0.5603 |
| P2a 立直显式特征 | 2.0874 | 0.3248 | 0.6094 | 1.4683 | 0.5659 |
| P2b behave 蒸馏 | 1.9225 | 0.3575 | 0.6784 | 1.4036 | 0.5685 |
| P2c 受控解冻 | 1.8585 | 0.3670 | 0.6939 | 1.3536 | 0.5714 |
| Bayes counts 边际乘积 | 2.2494 | 0.2475 | 0.5260 | 1.7255 | 0.5568 |
| Bayes counts 联合（精确全量） | 2.2486 | 0.2512 | 0.5269 | 1.7364 | 0.5541 |
| Bayes counts 联合（MC 子集） | 2.2514 | 0.2541 | 0.5296 | 1.9790 | 0.5513 |
| Bayes hard（+立直听牌） | 2.2087 | 0.2534 | 0.5431 | 1.9577 | 0.5553 |
| Bayes behave（+行为似然） | 2.4962 | 0.2675 | 0.5335 | 1.8454 | 0.5582 |

判定（**CE 主判据**）：**INFO_NEGLIGIBLE（tier 信息 ≤0.05 nats） + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）**；strict（counts）信息 0.0232 nats；full（behave）-0.2244 nats（7.0σ；最强 tier=hard，CE=2.209）；A2c 比最强 tier 低 0.173 nats；P2c 低 0.350 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.007，2.6σ）。

**riichi / active_any**（cond n=22672；micro n=42401）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 2.0224 | 0.2526 | 0.6027 | 1.7582 | 0.4653 |
| A2c 独立边际乘积 | 2.2570 | 0.2973 | 0.5338 | 1.7701 | 0.4779 |
| P1 冻结局部头 | 1.9924 | 0.3215 | 0.6735 | 1.6575 | 0.4444 |
| P2a 立直显式特征 | 2.1181 | 0.2981 | 0.6125 | 1.7033 | 0.4726 |
| P2b behave 蒸馏 | 1.9662 | 0.3293 | 0.6737 | 1.6383 | 0.4723 |
| P2c 受控解冻 | 1.8472 | 0.3525 | 0.6956 | 1.5477 | 0.4815 |
| Bayes counts 边际乘积 | 2.5034 | 0.1537 | 0.3902 | 2.0393 | 0.4487 |
| Bayes counts 联合（精确全量） | 2.5129 | 0.1583 | 0.3883 | 2.0530 | 0.4413 |
| Bayes counts 联合（MC 子集） | 2.4905 | 0.1583 | 0.4067 | 2.2401 | 0.4291 |
| Bayes hard（+立直听牌） | 2.1669 | 0.1690 | 0.5101 | 2.0380 | 0.4576 |
| Bayes behave（+行为似然） | 3.3701 | 0.1503 | 0.3746 | 2.3523 | 0.4460 |

判定（**CE 主判据**）：**TIER_MISFIT（counts CE 高于 prior ⇒ 假设失效）**；strict（counts）信息 -0.4905 nats；full（behave）-1.3477 nats（60.0σ；最强 tier=hard，CE=2.167）；A2c 比最强 tier 低 -0.090 nats；P2c 低 0.320 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.102，16.6σ）。

**clean_t8 / active_any**（cond n=80145；micro n=147841）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 2.3392 | 0.2558 | 0.5161 | 1.9491 | 0.4579 |
| A2c 独立边际乘积 | 2.0401 | 0.3425 | 0.6289 | 1.6641 | 0.4839 |
| P1 冻结局部头 | 2.0124 | 0.3284 | 0.6484 | 1.6678 | 0.4647 |
| P2a 立直显式特征 | 2.1648 | 0.3047 | 0.5859 | 1.7465 | 0.4672 |
| P2b behave 蒸馏 | 1.9877 | 0.3384 | 0.6558 | 1.6587 | 0.4727 |
| P2c 受控解冻 | 1.9424 | 0.3428 | 0.6668 | 1.6121 | 0.4752 |
| Bayes counts 边际乘积 | 2.2121 | 0.2556 | 0.5358 | 1.8994 | 0.4539 |
| Bayes counts 联合（精确全量） | 2.2105 | 0.2595 | 0.5366 | 1.9071 | 0.4495 |
| Bayes counts 联合（MC 子集） | 2.2151 | 0.2640 | 0.5389 | 2.1279 | 0.4487 |
| Bayes hard（+立直听牌） | 2.2126 | 0.2590 | 0.5385 | 2.1275 | 0.4513 |
| Bayes behave（+行为似然） | 2.4735 | 0.2478 | 0.5221 | 2.0934 | 0.4607 |

判定（**CE 主判据**）：**INFO_SMALL（0.05~0.15 nats） + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）**；strict（counts）信息 0.1287 nats；full（behave）-0.1343 nats（27.0σ；最强 tier=counts，CE=2.211）；A2c 比最强 tier 低 0.170 nats；P2c 低 0.268 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.008，2.0σ）。

**all / active_both**（cond n=5865；micro n=22133）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 1.6738 | 0.4007 | 0.7327 | 1.0037 | 0.7350 |
| A2c 独立边际乘积 | 1.5927 | 0.5067 | 0.7613 | 0.9114 | 0.7389 |
| P1 冻结局部头 | 1.6044 | 0.4749 | 0.7741 | 0.9300 | 0.7265 |
| P2a 立直显式特征 | 1.7762 | 0.4569 | 0.7294 | 0.9676 | 0.7351 |
| P2b behave 蒸馏 | 1.5851 | 0.4737 | 0.7826 | 0.9143 | 0.7349 |
| P2c 受控解冻 | 1.4991 | 0.4893 | 0.7951 | 0.8741 | 0.7343 |
| Bayes counts 边际乘积 | 1.8338 | 0.3096 | 0.7178 | 1.2633 | 0.7279 |
| Bayes counts 联合（精确全量） | 1.8338 | 0.3142 | 0.7188 | 1.2773 | 0.7244 |
| Bayes counts 联合（MC 子集） | 1.8395 | 0.3130 | 0.7004 | 1.5712 | 0.7340 |
| Bayes hard（+立直听牌） | 1.8188 | 0.3410 | 0.7089 | 1.5386 | 0.7432 |
| Bayes behave（+行为似然） | 2.2309 | 0.2984 | 0.6297 | 1.3066 | 0.7489 |

判定（**CE 主判据**）：**TIER_MISFIT（counts CE 高于 prior ⇒ 假设失效） + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）**；strict（counts）信息 -0.1600 nats；full（behave）-0.5571 nats（9.7σ；最强 tier=hard，CE=1.819）；A2c 比最强 tier 低 0.226 nats；P2c 低 0.320 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.102，6.4σ）。

**riichi / active_both**（cond n=944；micro n=2734）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 0.6418 | 0.6907 | 0.9894 | 0.7918 | 0.6770 |
| A2c 独立边际乘积 | 1.4052 | 0.6314 | 0.8284 | 1.0173 | 0.6686 |
| P1 冻结局部头 | 1.2929 | 0.5742 | 0.8591 | 1.0261 | 0.6412 |
| P2a 立直显式特征 | 1.5176 | 0.5763 | 0.7977 | 1.0792 | 0.6587 |
| P2b behave 蒸馏 | 1.2955 | 0.5985 | 0.8379 | 1.0064 | 0.6668 |
| P2c 受控解冻 | 1.1822 | 0.6483 | 0.8824 | 0.9386 | 0.6602 |
| Bayes counts 边际乘积 | 1.9314 | 0.2331 | 0.7246 | 1.4611 | 0.6269 |
| Bayes counts 联合（精确全量） | 1.9253 | 0.2479 | 0.7341 | 1.4714 | 0.6225 |
| Bayes counts 联合（MC 子集） | 1.9479 | 0.2286 | 0.6929 | 1.7358 | 0.6244 |
| Bayes hard（+立直听牌） | 1.7862 | 0.3357 | 0.7143 | 1.4838 | 0.6714 |
| Bayes behave（+行为似然） | 2.3379 | 0.2857 | 0.6643 | 1.6096 | 0.6667 |

判定（**CE 主判据**）：**TIER_MISFIT（counts CE 高于 prior ⇒ 假设失效） + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）**；strict（counts）信息 -1.2835 nats；full（behave）-1.6961 nats（39.8σ；最强 tier=hard，CE=1.786）；A2c 比最强 tier 低 0.381 nats；P2c 低 0.604 nats。
同 cell 的 top1 判据（仅供对照）：**SIGNAL_SHALLOW（tier 信息≈0）**（tier 信息 -0.405，10.6σ）。

**clean_t8 / active_both**（cond n=3321；micro n=10543）

| 行 | cond CE | cond top1 | cond top3 | micro CE | micro top1 |
|---|---:|---:|---:|---:|---:|
| prior（每中心边缘） | 1.6303 | 0.3917 | 0.7495 | 1.1168 | 0.6853 |
| A2c 独立边际乘积 | 1.6598 | 0.4758 | 0.7332 | 1.0855 | 0.6897 |
| P1 冻结局部头 | 1.7680 | 0.4279 | 0.7284 | 1.1328 | 0.6735 |
| P2a 立直显式特征 | 1.9124 | 0.4170 | 0.6889 | 1.1692 | 0.6821 |
| P2b behave 蒸馏 | 1.7346 | 0.4291 | 0.7453 | 1.1049 | 0.6818 |
| P2c 受控解冻 | 1.6326 | 0.4390 | 0.7552 | 1.0569 | 0.6838 |
| Bayes counts 边际乘积 | 1.7993 | 0.3427 | 0.7239 | 1.3663 | 0.6794 |
| Bayes counts 联合（精确全量） | 1.7987 | 0.3445 | 0.7263 | 1.3794 | 0.6739 |
| Bayes counts 联合（MC 子集） | 1.7646 | 0.3714 | 0.7209 | 1.6574 | 0.6849 |
| Bayes hard（+立直听牌） | 1.7660 | 0.3802 | 0.7253 | 1.6581 | 0.6914 |
| Bayes behave（+行为似然） | 2.1708 | 0.3121 | 0.6396 | 1.4486 | 0.7018 |

判定（**CE 主判据**）：**TIER_MISFIT（counts CE 高于 prior ⇒ 假设失效） + HEAD_BEYOND_TIER（A2c 超出最强 tier ⇒ 非紧上界）**；strict（counts）信息 -0.1684 nats；full（behave）-0.5404 nats（7.7σ；最强 tier=hard，CE=1.766）；A2c 比最强 tier 低 0.106 nats；P2c 低 0.133 nats。
同 cell 的 top1 判据（仅供对照）：**TIER_NOT_TIGHT（tier 信息≈0 但实测头超额>2pp）**（tier 信息 -0.080，3.7σ）。

## 4. micro → cond：判定与数值变化

`CE 判定` 是 cond 下的主判据（跨模型可比）；`top1 判定 micro→cond` 用同一 top1 判据，只反映口径变化的影响。

| 层 | 子集 | prior top1 micro→cond | tier 信息(CE) | A2c norm_gain63 | P2c norm_gain63 | CE 判定 | top1 判定 micro→cond |
|---|---|---:|---:|---:|---:|---|---|
| all | all_centers | 0.330 → 0.167 | 0.038 | 0.061 | 0.074 | **INFO_NEGLIGIBLE** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| all | active_any | 0.560 → 0.275 | 0.023 | 0.111 | 0.127 | **INFO_NEGLIGIBLE** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| all | active_both | 0.735 → 0.401 | -0.160 | 0.177 | 0.148 | **TIER_MISFIT** | HEADROOM → TIER_NOT_TIGHT |
| riichi | all_centers | 0.347 → 0.194 | -0.376 | 0.047 | 0.110 | **TIER_MISFIT** | TIER_NOT_TIGHT → TIER_NOT_TIGHT |
| riichi | active_any | 0.465 → 0.253 | -0.490 | 0.060 | 0.134 | **TIER_MISFIT** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| riichi | active_both | 0.677 → 0.691 | -1.284 | -0.192 | -0.137 | **TIER_MISFIT** | SIGNAL_SHALLOW → SIGNAL_SHALLOW |
| nonriichi | all_centers | 0.329 → 0.168 | 0.045 | 0.058 | 0.068 | **INFO_NEGLIGIBLE** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| nonriichi | active_any | 0.572 → 0.281 | 0.049 | 0.114 | 0.122 | **INFO_NEGLIGIBLE** | SIGNAL_SHALLOW → HEAD≈CEILING |
| nonriichi | active_both | 0.746 → 0.375 | -0.113 | 0.173 | 0.135 | **TIER_MISFIT** | HEADROOM → TIER_NOT_TIGHT |
| called | all_centers | 0.511 → 0.178 | -0.083 | 0.085 | 0.103 | **TIER_MISFIT** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| called | active_any | 0.685 → 0.276 | -0.077 | 0.137 | 0.156 | **TIER_MISFIT** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| called | active_both | 0.813 → 0.482 | -0.604 | 0.029 | 0.017 | **TIER_MISFIT** | PARTIAL → SIGNAL_SHALLOW |
| turn>=8 | all_centers | 0.368 → 0.160 | 0.027 | 0.088 | 0.112 | **INFO_NEGLIGIBLE** | TIER_NOT_TIGHT → TIER_NOT_TIGHT |
| turn>=8 | active_any | 0.517 → 0.253 | 0.076 | 0.120 | 0.135 | **INFO_SMALL** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| turn>=8 | active_both | 0.712 → 0.399 | -0.145 | 0.183 | 0.145 | **TIER_MISFIT** | HEADROOM → TIER_NOT_TIGHT |
| clean_t8 | all_centers | 0.305 → 0.156 | 0.084 | 0.079 | 0.095 | **INFO_SMALL** | TIER_NOT_TIGHT → TIER_NOT_TIGHT |
| clean_t8 | active_any | 0.458 → 0.256 | 0.129 | 0.116 | 0.117 | **INFO_SMALL** | SIGNAL_SHALLOW → TIER_NOT_TIGHT |
| clean_t8 | active_both | 0.685 → 0.392 | -0.168 | 0.138 | 0.078 | **TIER_MISFIT** | HEADROOM → TIER_NOT_TIGHT |

## 5. 完整分层 CE（cond；JSON 同步）

| 层 | 子集 | n | prior | A2c | P1 | P2a | P2b | P2c | c_prod | c_joint | hard | behave |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | all_centers | 1687499 | 2.6618 | 2.4733 | 2.4506 | 2.5156 | 2.4442 | 2.4091 | 2.6130 | 2.6242 | 2.6086 | 2.8084 |
| all | active_any | 172942 | 2.2718 | 2.0359 | 1.9442 | 2.0874 | 1.9225 | 1.8585 | 2.2494 | 2.2486 | 2.2087 | 2.4962 |
| all | active_both | 5865 | 1.6738 | 1.5927 | 1.6044 | 1.7762 | 1.5851 | 1.4991 | 1.8338 | 1.8338 | 1.8188 | 2.2309 |
| riichi | all_centers | 116685 | 2.4288 | 2.4944 | 2.2917 | 2.3974 | 2.2975 | 2.1922 | 2.7772 | 2.8052 | 2.5016 | 3.9773 |
| riichi | active_any | 22672 | 2.0224 | 2.2570 | 1.9924 | 2.1181 | 1.9662 | 1.8472 | 2.5034 | 2.5129 | 2.1669 | 3.3701 |
| riichi | active_both | 944 | 0.6418 | 1.4052 | 1.2929 | 1.5176 | 1.2955 | 1.1822 | 1.9314 | 1.9253 | 1.7862 | 2.3379 |
| nonriichi | all_centers | 1570814 | 2.6556 | 2.4717 | 2.4624 | 2.5244 | 2.4551 | 2.4252 | 2.6008 | 2.6108 | 2.6165 | 2.7224 |
| nonriichi | active_any | 150270 | 2.2582 | 2.0026 | 1.9370 | 2.0827 | 1.9160 | 1.8602 | 2.2111 | 2.2087 | 2.2149 | 2.3663 |
| nonriichi | active_both | 4921 | 1.7036 | 1.6286 | 1.6642 | 1.8258 | 1.6407 | 1.5599 | 1.8150 | 1.8163 | 1.8255 | 2.2089 |
| called | all_centers | 203495 | 2.5799 | 2.3622 | 2.3144 | 2.4146 | 2.3043 | 2.2549 | 2.6370 | 2.6629 | 2.6695 | 2.7515 |
| called | active_any | 32812 | 2.1553 | 1.9197 | 1.8452 | 1.9994 | 1.8252 | 1.7552 | 2.2310 | 2.2327 | 2.2424 | 2.3533 |
| called | active_both | 1273 | 1.2178 | 1.5427 | 1.4179 | 1.6404 | 1.4256 | 1.4194 | 1.8125 | 1.8214 | 1.9306 | 2.4262 |
| turn>=8 | all_centers | 616461 | 2.7064 | 2.4320 | 2.3851 | 2.4998 | 2.3776 | 2.3247 | 2.6600 | 2.6790 | 2.6412 | 3.1130 |
| turn>=8 | active_any | 125464 | 2.3316 | 2.0419 | 1.9707 | 2.1240 | 1.9473 | 1.8854 | 2.2543 | 2.2556 | 2.2069 | 2.6090 |
| turn>=8 | active_both | 5360 | 1.6742 | 1.5868 | 1.6061 | 1.7850 | 1.5861 | 1.5075 | 1.8186 | 1.8195 | 1.8015 | 2.2455 |
| clean_t8 | all_centers | 402053 | 2.7496 | 2.4674 | 2.4574 | 2.5661 | 2.4480 | 2.4083 | 2.6519 | 2.6652 | 2.6745 | 2.9610 |
| clean_t8 | active_any | 80145 | 2.3392 | 2.0401 | 2.0124 | 2.1648 | 1.9877 | 1.9424 | 2.2121 | 2.2105 | 2.2126 | 2.4735 |
| clean_t8 | active_both | 3321 | 1.6303 | 1.6598 | 1.7680 | 1.9124 | 1.7346 | 1.6326 | 1.7993 | 1.7987 | 1.7660 | 2.1708 |

## 6. 结论

- cond 下 all/active_any 的 n 从 micro 的 393393 降到 **172942**（筛掉空窗口）；prior CE 从 1.678 变为 **2.272**（尺度变了，不能跨口径直接比）。
- all/active_any 的 cond CE：prior 2.272、counts 联合 **2.249**（MC 子集 2.251）、counts 边际乘积 2.249、hard 2.209、behave 2.496、A2c 2.036、P1 1.944、P2b 1.923、P2c 1.858。
- tier 信息（CE）：**strict（counts）= 0.0232 nats**、full（behave）= -0.2244 nats（7.0σ）⇒ INFO_NEGLIGIBLE（tier 信息 ≤0.05 nats）。
- A2c 的 cond CE 比**最强 tier**（hard，CE 2.209）低 **0.173 nats** ⇒ tier 是信息子集、**不是紧上界**。
- **top1 判据在 cond 下依然失效**：counts 的 cond top1 0.2512 低于 prior 0.2745，但其 CE 2.2486 低于 prior 2.2718 ⇒ tier 后验更平坦、argmax 天然吃亏，**top1 不能当信息量判据**。
- 立直层「均匀子集」假设失效：riichi/active_any 的 counts CE 2.513 vs prior 2.022（0.490）⇒ TIER_MISFIT，该层 tier 既不能当上界也不能当信息下界。
- CE 判定分布（18 cell）：TIER_MISFIT 10、INFO_NEGLIGIBLE 5、INFO_SMALL 3、INFO_MEANINGFUL 0；其中 **16 个 cell** 的 A2c 显著超出 tier（CE 低于 tier >0.05 nats）。
- top1 判据（仅供对照）：SIGNAL_SHALLOW 2、TIER_NOT_TIGHT 15、HEADROOM 0。
- micro → cond 的 **top1 判定**翻转 14 处：all/all_centers SIGNAL_SHALLOW→TIER_NOT_TIGHT；all/active_any SIGNAL_SHALLOW→TIER_NOT_TIGHT；all/active_both HEADROOM→TIER_NOT_TIGHT；riichi/active_any SIGNAL_SHALLOW→TIER_NOT_TIGHT；nonriichi/all_centers SIGNAL_SHALLOW→TIER_NOT_TIGHT；nonriichi/active_any SIGNAL_SHALLOW→HEAD≈CEILING；nonriichi/active_both HEADROOM→TIER_NOT_TIGHT；called/all_centers SIGNAL_SHALLOW→TIER_NOT_TIGHT。
- M 收敛：all/active_any 上三 tier 的 |CE(M_max) − CE(M_min)| 最大 0.1573 nats（Dirichlet α=0.5 平滑后）。

**判定：cond 下 tier 信息 0.0232 nats（7.0σ），且 16/18 个 cell 的 A2c 显著超出 tier ⇒ **A2c/P2c 用的是 tier 覆盖不到的信息**（牌河顺序/宣言牌/逐巡事件）。因此 P1/P2a/P2c 的 FAIL **不是「后验接近均匀」的预期结果**，而是「没打过 A2c」；micro 下的 SIGNAL_SHALLOW 是 top1 被 EMPTY 主导造成的误判。局部形状线的「关闭」应改为「实现未达 A2c，但上限未被证明」。**

