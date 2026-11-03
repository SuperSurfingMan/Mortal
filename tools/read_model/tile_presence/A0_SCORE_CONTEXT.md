# A0：当前分数/局况特征（2026-11-03，结论：无效）

> 结论：把 A2c obs 里有、而本模型漏掉的**当前分数/名次/局数/本场/供托/亲家/场风**
> 补进 context，**在 120k 主配置上是精确的 zero effect（+0.02pp）**；
> 在 40k 上反而 −0.6pp；推理时把这 11 维清零甚至略好（+0.11pp）。
> 数据侧也印证：**presence 率在名次/分差上几乎是平的**。
> ⇒ 「1.5pp 差距来自缺分数特征」这个假设被证伪。

## 1. 补了什么

`dataset.py` 在 `start_kyoku` 记录本局 `scores/kyoku/honba/kyotaku/oya/bakaze`
（一局内分数不变），样本新增两个字段；`tokens.py` 把它展开成 **11 维 context**：

| 维度 | 含义 |
|---|---|
| `score` | 目标家当前点数 / 1e5 |
| `rank` | 名次（0-based，并列同分） |
| `gap_leader` | 与第一名分差 / 5e4（≤0） |
| `gap_next` | 与上一名分差 / 3e4（≤0） |
| `is_oya` | 是否亲家 |
| `kyoku` / `honba` / `kyotaku` | 第几局 / 本场 / 供托 |
| `bakaze_south` | 南场标志 |
| `is_last` | 是否第 4 名 |
| `n_negative` | 负分人数 / 3 |

`FC`：18 → **29**。其余（牌 token / 动作 token）不变。

## 2. 结果

| 配置 | 基线 | +A0 | Δ |
|---|---:|---:|---:|
| 40k（d160/2 层，run2 同配置） | 0.7153 | 0.7093 | **−0.60pp** |
| 120k（d224/3 层，run3 同配置） | 0.7231 | **0.7233** | **+0.02pp** |
| A2c 对照 | 0.7383 | — | — |

分层（120k A0）：turn≥8 0.7673（run3 0.7674）、立直 0.7387（0.7397）——全域持平。

**推理期消融**（40k 模型，把 ctx[18:] 清零）：
0.7093 → **0.7104**（反而略好）⇒ 模型基本没用这些维度。

## 3. 为什么没用：数据侧的直接证据

在 train（40k）的 informative 单元上统计 presence 率：

| 名次 | 单元数 | P(在手里) |
|---|---:|---:|
| 第 1 | 1,248,637 | 0.3086 |
| 第 2 | 1,004,390 | 0.3043 |
| 第 3 | 798,174 | 0.3058 |
| 第 4 | 856,326 | 0.3031 |

| 与第一名分差 | 单元数 | P(在手里) |
|---|---:|---:|
| < −30k | 1,319,987 | 0.3041 |
| −30k~−10k | 871,975 | 0.3043 |
| −10k~0 | 466,928 | 0.3050 |
| = 0（并列第一） | 1,248,637 | 0.3086 |

全体均值 0.3057。**极差只有 0.55pp**。

解释：分数压力改变的是**留哪张牌**（攻/守的选择），而不是**手里有几张牌**。
手牌张数由 `13−3·melds` 锁死，逐牌存在率的分母不变；至于"留哪张"，
已经被逐牌手切/摸切统计和牌河吸收——所以分数在这里是**冗余的**。

## 4. 与既有结论的关系

- 这条与 [`ANALYSIS_WHY_BELOW.md`](ANALYSIS_WHY_BELOW.md) 一致：
  1.5pp 差距是**表示/优化**层面的，不是"少给了 A2c 有的特征"。
- 也解释了为什么 40k 会 −0.6pp：多 11 维输入在小数据上只是增加优化噪声；
  120k 后模型学会忽略它们（回到 zero effect）。
- 与 [`FEASIBILITY_BEAT_A2C.md`](FEASIBILITY_BEAT_A2C.md) 的建议一致：
  **同信息集内的特征补全已经做完**，再想提升只能换信息源（玩家风格/局内行为履历）。

## 5. 复现

```bash
# 导出（保留旧缓存，另存 .cache_a0 / .cache120_a0）
CACHE=.../.cache_a0 TRAIN_N=40000 VAL_N=8000 EVAL_N=40000 \
  bash tools/read_model/tile_presence/run_export.sh
CACHE=.../.cache120_a0 TRAIN_N=120000 VAL_N=8000 EVAL_N=40000 \
  bash tools/read_model/tile_presence/run_export.sh

# 训练
CACHE=.../.cache_a0 OUT=.../results/run4_a0_40k D=160 BLOCKS=2 ... run_train.sh
CACHE=.../.cache120_a0 OUT=.../results/run5_a0_120k D=224 BLOCKS=3 ... run_train.sh
```

产物：`results/run4_a0_40k/`、`results/run5_a0_120k/`；日志
`mortal/logs/read_model/tile_presence/train_20261103_08*.log`。
