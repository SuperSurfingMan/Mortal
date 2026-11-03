# 在读任务标签诊断：最大类占比与有效类数

- 缓存：`/tmp/read_model_diag_cache_a2c/cache.npz`（n=40000）
- 判定：`HIGH` = p_max≥0.40 且 eff<8；`MED` = p_max≥0.25 或 eff<K/3；`LOW` = 其余。

口径说明：`p_max` = 最大类占比 = 常数预测器的 top1；`eff` = exp(熵) = 有效类别数；`H_norm` = 熵 / ln K。

## 1. 任务级标签结构（all 层，单标签任务）

| 任务 | K | n | p_max | 最大类 | 熵 | eff | H_norm | 风险 | 当前指标 | 被 top1 坑 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| shape27 | 27 | 393393 | 0.5604 | 0 | 1.7500 | 5.8 | 0.531 | **HIGH** | top1 + CE | **是** |
| shape10 | 10 | 393393 | 0.5604 | 0 | 1.4809 | 4.4 | 0.643 | **HIGH** | top1 + CE | **是** |
| count5 | 5 | 4080000 | 0.7062 | 0 | 0.7784 | 2.2 | 0.484 | **HIGH** | CE + MAE | 否 |
| presence2 | 2 | 4080000 | 0.7062 | 0 | 0.6055 | 1.8 | 0.874 | **HIGH** | AUC + BCE | 否 |
| tenpai2 | 2 | 120000 | 0.8526 | 0 | 0.4182 | 1.5 | 0.603 | **HIGH** | AUC | 否 |
| riichi2 | 2 | 120000 | 0.9291 | 0 | 0.2559 | 1.3 | 0.369 | **HIGH** | BCE / AUC | 否 |
| called2 | 2 | 120000 | 0.8347 | 0 | 0.4484 | 1.6 | 0.647 | **HIGH** | BCE / AUC | 否 |
| shanten8 | 8 | 120000 | 0.2864 | 2 | 1.6113 | 5.0 | 0.775 | **MED** | accuracy（辅助） | 需双报 |

## 2. 关键任务的分层


**shape27**（HIGH）

| 层 / 子集 | n | p_max | 熵 | eff | H_norm |
|---|---:|---:|---:|---:|---:|
| all/all_centers | 2520000 | 0.3304 | 2.4346 | 11.4 | 0.739 |
| all/active_any | 393393 | 0.5604 | 1.7500 | 5.8 | 0.531 |
| all/active_both | 22133 | 0.7350 | 1.1181 | 3.1 | 0.339 |
| riichi/all_centers | 178584 | 0.3466 | 2.3374 | 10.4 | 0.709 |
| riichi/active_any | 42401 | 0.4653 | 1.9704 | 7.2 | 0.598 |
| riichi/active_both | 2734 | 0.6547 | 1.2260 | 3.4 | 0.372 |
| clean_t8/all_centers | 578424 | 0.3049 | 2.5643 | 13.0 | 0.778 |
| clean_t8/active_any | 147841 | 0.4579 | 2.0348 | 7.7 | 0.617 |
| clean_t8/active_both | 10543 | 0.6850 | 1.2836 | 3.6 | 0.389 |

**shape10**（HIGH）

| 层 / 子集 | n | p_max | 熵 | eff | H_norm |
|---|---:|---:|---:|---:|---:|
| all/all_centers | 2520000 | 0.3304 | 1.9553 | 7.1 | 0.849 |
| all/active_any | 393393 | 0.5604 | 1.4809 | 4.4 | 0.643 |
| all/active_both | 22133 | 0.7350 | 0.9791 | 2.7 | 0.425 |
| riichi/all_centers | 178584 | 0.3466 | 1.8480 | 6.3 | 0.803 |
| riichi/active_any | 42401 | 0.4653 | 1.6208 | 5.1 | 0.704 |
| riichi/active_both | 2734 | 0.6547 | 1.0454 | 2.8 | 0.454 |
| clean_t8/all_centers | 578424 | 0.3049 | 1.9970 | 7.4 | 0.867 |
| clean_t8/active_any | 147841 | 0.4579 | 1.6918 | 5.4 | 0.735 |
| clean_t8/active_both | 10543 | 0.6850 | 1.1222 | 3.1 | 0.487 |

**count5**（HIGH）

| 层 / 子集 | n | p_max | 熵 | eff | H_norm |
|---|---:|---:|---:|---:|---:|
| all | 4080000 | 0.7062 | 0.7784 | 2.2 | 0.484 |
| riichi | 289136 | 0.6970 | 0.8048 | 2.2 | 0.500 |
| clean_t8 | 936496 | 0.6996 | 0.8056 | 2.2 | 0.501 |

**presence2**（HIGH）

| 层 / 子集 | n | p_max | 熵 | eff | H_norm |
|---|---:|---:|---:|---:|---:|
| all | 4080000 | 0.7062 | 0.6055 | 1.8 | 0.874 |
| riichi | 289136 | 0.6970 | 0.6134 | 1.8 | 0.885 |
| clean_t8 | 936496 | 0.6996 | 0.6112 | 1.8 | 0.882 |

**tenpai2**（HIGH）

| 层 / 子集 | n | p_max | 熵 | eff | H_norm |
|---|---:|---:|---:|---:|---:|
| all | 120000 | 0.8526 | 0.4182 | 1.5 | 0.603 |
| riichi | 8504 | 1.0000 | -0.0000 | 1.0 | -0.000 |
| clean_t8 | 27544 | 0.9316 | 0.2496 | 1.3 | 0.360 |

**shanten8**（MED）

| 层 / 子集 | n | p_max | 熵 | eff | H_norm |
|---|---:|---:|---:|---:|---:|
| all | 120000 | 0.2864 | 1.6113 | 5.0 | 0.775 |
| riichi | 8504 | 1.0000 | -0.0000 | 1.0 | -0.000 |
| clean_t8 | 27544 | 0.4545 | 1.2529 | 3.5 | 0.603 |

## 3. 多标签任务（待牌）

- **all**：n=119984，平均待牌数 1.11，单类最高正例率 0.853（类 34），无待牌率 0.853
- **riichi**：n=8504，平均待牌数 1.81，单类最高正例率 0.117（类 4），无待牌率 0.000
- **clean_t8**：n=27544，平均待牌数 1.03，单类最高正例率 0.932（类 34），无待牌率 0.932

> 待牌是**多标签**任务，门禁用 hit@1/hit@5 而非 top1；本节只报基础统计，不作口径判定。


## 4. 结论

- 标签集中的任务（HIGH）：shape27, shape10, count5, presence2, tenpai2, riichi2, called2；MED：shanten8；LOW：无；退化（p_max=1）：tenpai2@riichi, riichi2@riichi, riichi2@clean_t8, called2@clean_t8, shanten8@riichi。
- **真正被 top1 坑的（集中且现用 top1/accuracy）：shape27, shape10, shanten8**。其余 count5, presence2, tenpai2, riichi2, called2 虽集中但用 AUC/CE/MAE/hit@k，**未受影响**。
- 退化项 tenpai2@riichi, riichi2@riichi, riichi2@clean_t8, called2@clean_t8, shanten8@riichi：标签恒定（立直必听牌、向听恒 0）⇒ 这些子集上的 tenpai/shanten 指标恒为 100%，**不应参与任何比较**。
- `shape27`：K=27、p_max=0.5604（类 0）、eff=5.8；当前指标 P0/P1/P2a/P2b/P2c 门禁用 top1 ⇒ 已确认被 EMPTY 稀释 ⇒ **⚠ 已受影响**。
- `shape10`：K=10、p_max=0.5604（类 0）、eff=4.4；当前指标 语义标签验证用 top1 ⇒ 同样受影响 ⇒ **⚠ 已受影响**。
- `count5`：K=5、p_max=0.7062（类 0）、eff=2.2；当前指标 A2c alloc 指标不用 top1 ⇒ 未受影响 ⇒ **指标安全**。
- `presence2`：K=2、p_max=0.7062（类 0）、eff=1.8；当前指标 Phase 3 用 AUC/BCE ⇒ 未受影响 ⇒ **指标安全**。
- `tenpai2`：K=2、p_max=0.8526（类 0）、eff=1.5；当前指标 bayes_bound.tenpai_auc ⇒ 未受影响 ⇒ **指标安全**。
- `riichi2`：K=2、p_max=0.9291（类 0）、eff=1.3；当前指标 辅助头 ⇒ 未受影响 ⇒ **指标安全**。
- `called2`：K=2、p_max=0.8347（类 0）、eff=1.6；当前指标 辅助头 ⇒ 未受影响 ⇒ **指标安全**。
- `shanten8`：p_max=0.2864、eff=5.0 ⇒ **⚠ 需双报**。

**判定：标签集中且**现用** top1/accuracy 的任务：shape27, shape10, shanten8 ⇒ 这些门禁必须改 cond/macro 或补「非平凡类条件」；其余 5 个高集中任务已用 AUC/CE/MAE/hit@k，无需改口径。 另：5 个退化项（tenpai2@riichi, riichi2@riichi, riichi2@clean_t8, called2@clean_t8, shanten8@riichi）因标签恒定，已无信息，应直接从评估中移除。**

