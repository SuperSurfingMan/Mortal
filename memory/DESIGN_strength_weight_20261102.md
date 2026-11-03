# 数据质量加权探针（P0-b，2026-11-02 结案）

> 结论先行：100k 探针**门禁未过**（dqn_loss +5.6% > +1%），按预注册**不烧 76k 臂**，
> 数据质量加权（当前方案）关闭；数据侧机制至此全部走完。详细数字见 §4~§5。

## 0. 目标与假设

- **假设**：R2 的 top400 是文件/座位级白名单，只保证一局至少 1 名强手；另外 3 席可能偏弱。
  BC 可能连弱手的坏习惯一起模仿，因此「牌局质量」还有未被利用的方差。
- **变量**：只改「每条牌谱被抽到的概率」，不改模型、损失、数据集合、超参。
- **预注册门禁**（holdout 202609，digest `9c5362bf`，与 S6c/956k 同口径）：
  1. top1 ≥ **0.8117**（父代 0.8167 − 0.5pp）
  2. dqn_loss ≤ **0.365**（父代 0.3613 + 1%）
  3. 行为画像（agari/houjuu/fuuro/riichi）不劣化
  两项数值门禁同时满足 → 上 76k sets 双向臂；否则关闭该轴。

## 1. 权重设计

对每条牌谱只读首行 `start_game.names`，取 4 名玩家的长期 `avg_pt`
（`dataset/index/player_ratings_2022_2026H1_v3.csv`，要求每人 ≥100 局），

```
quality = min(avg_pt of 4 seats)
```

按 quality 分位切档，映射为整数重复次数（所有文件至少 1 次，不做过滤）：

| 档位 | 重复 | 文件数 | 样本占比 |
|---|---|---|---|
| `min_pt < −0.765`（底部 75%） | ×1 | 369,428 | 55.5% |
| `−0.765 ~ +0.374`（75–90 分位） | ×2 | 74,089 | 22.3% |
| `> +0.374`（前 10%） | ×3 | 49,293 | 22.2% |

- 切点 `[0.5, 0.75, 0.9]`、重复 `[1,1,2,3]`；前两档重复相同 ⇒ **q50 切点惰性**，
  实际生效的是「底部 75% 原样、前 25% 加权」。
- 均值 **1.350**，文件表 492,810 → **665,485** 条；492,798/492,810 局可评分
  （4/4 评分 461,254；3/4 30,709；2/4 835；1/4 12）。
- 为什么用 `min` 而不是均值：4 人 avg_pt 接近零和，均值几乎恒为 0（sd 1.9），
  最弱一席的分布 sd 5.0，信息量最大。

## 2. 实现（不改损失/索引）

- 权重表：`tools/training/build_strength_weights.py` → `mortal/indexes/strength_weights_v1.pth`
  （多进程只读首行 names，492k 文件约 5 s @16 workers）。
- 训练接入：`train.py` 在 `extra_logs` 之后、DataLoader 之前按 `[dataset.strength_weight]`
  把 file_list 条目重复 `repeats` 次；不改 `file_index` manifest、不改 Rust 数据管道。
- 配置：`mortal/config.strength_probe.toml`（100k）与 `config.strength_probe_smoke.toml`（2k）。
- 编排：`mortal/run_strength_probe.sh`（预检 → 训练 → holdout；`STAGE=train|holdout|all`）。
- 每条牌谱约 235 个决策样本，文件级重复等价于这些样本整体被加权；增强视图随机，
  重复条目不会逐位相同。

## 3. 预算与执行

| 阶段 | 结果 |
|---|---|
| 2k 冒烟 | 加权日志正确（命中 492,810/492,810、直方图 `{1:369428,2:74089,3:49293}`、file list 665,485），预算 958k 到达，RC=0，无 traceback/NaN |
| 100k 正式 | 2026-11-02 03:45→10:36 UTC，**24,673 s（6h51m）**，4.41 batch/s，1,056,000 步，RC=0 |

## 4. 结果

**holdout（202609，12,288 样本，digest `9c5362bf`）**

| 指标 | 加权 1,056k | S6c/956k 父代 | 变化 | 门禁 |
|---|---|---|---|---|
| top1 | **0.8141** | 0.8167 | −0.26pp | ≥0.8117 ✅ |
| dqn_loss | **0.3816** | 0.3613 | **+5.6%** | ≤0.365 ❌ |
| R2 | −1.4199 | −1.2911 | 变差 0.129 | — |
| var_q/var_t | **0.5105**/0.3154 | 0.4722/0.3154 | Q 尺度 +8.1% | — |
| agari_top1 | 149/149 | 149/149 | 持平 | — |
| logsumexp_q | +0.471 | +0.525 | −0.054 | — |

**test_play 中途画像**（2000 局 vs 基线/200k）

| 步数 | avg_rank | avg_pt | agari | houjuu | fuuro | riichi |
|---|---|---|---|---|---|---|
| 1,000k | 2.487 | 0.968 | 21.43% | 13.03% | 26.89% | 19.32% |
| 1,050k | 2.4635 | 3.0825 | 22.60% | **13.61%** | 30.08% | 19.25% |
| S6c 末次 | 2.448 | 3.375 | 21.67% | 12.27% | 31.09% | 18.55% |

## 5. 判读与关闭

- **top1 在噪声内**（−0.26pp；12,288 样本 SE≈0.35pp），行为一致率没有被破坏；
- **价值拟合明显恶化**：dqn_loss +5.6%、R2 变差、Q 尺度膨胀 8.1%
  ⇒ 分布被推向“四强桌”后 Q 目标幅度变大、拟合变差，这比 top1 更敏感；
- 按预注册（两项同时满足），**探针未通过** ⇒ 不烧 76k sets 臂（省约 10 h），
  **关闭数据质量加权（当前方案）**；数据侧机制（reshuffle/补量/混年代/自博弈/质量加权）
  至此全部走完，无正信号。

**口径边界**

1. 本轮没有“同样 100k 步、不加权”的对照；dqn_loss 上升理论上可能混入续训漂移。
   但同类续训（tierA 200k）dqn_loss 是下降的（−4.7%），因此更可能归因于加权。
2. 评分表是 2022-2026H1 全窗口静态均值，对早期牌局有轻微未来信息泄漏；
   若要严谨复现，需按时间切片重建评分。
3. 重复条目降低了 100k 步内的**唯一牌谱数**（约 16 万 vs 父代 200k 步的 22 万），
   这本身也会影响 dqn_loss；因此本探针只能回答“该加权方案是否值得继续”，
   不能单独证明“质量加权这一机制必然无效”。
4. `best.pth` 因 `reset_best_perf` 重置（哨兵 −135），与父代 best_perf 不可比；
   判定对象是最终 `mortal.pth`（1,056k）。

## 6. 复现

```bash
# 1) 权重表
python tools/training/build_strength_weights.py \
    --index mortal/indexes/reward_ablation_r4_v2.pth \
    --ratings dataset/index/player_ratings_2022_2026H1_v3.csv \
    --out mortal/indexes/strength_weights_v1.pth \
    --stat min --min-games 100 --cuts 0.5,0.75,0.9 --repeats 1,1,2,3 --workers 16

# 2) 训练 + holdout（tmux 承载）
cd mortal
tmux new -d -s sp100k "cd /mnt/data8/mortal/Mortal/mortal && STAGE=all bash run_strength_probe.sh > logs/strength_probe_run.log 2>&1"
```

## 7. 产物

- 权重表：`mortal/indexes/strength_weights_v1.pth`（50 MB）
- ckpt：`mortal/checkpoints/strength_probe/`（mortal.pth、1,000k/1,020k/1,040k 快照、
  best.pth；共 874 MB）
- 日志：`mortal/logs/strength_probe_{train,holdout,run,smoke}.log`
- 代码/配置：`tools/training/build_strength_weights.py`、`mortal/train.py`（strength_weight 段）、
  `mortal/config.strength_probe{,_smoke}.toml`、`mortal/run_strength_probe.sh`
