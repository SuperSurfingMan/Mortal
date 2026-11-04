# Oracle Guiding + 退火：最小实现与冒烟（2026-11-03）

> 目标：把 Suphx 的 Oracle Guiding 做成本仓库可跑的最小版本——**训练期**把三家暗牌
> （`invisible_obs`，217 行）按 α 缩放后拼进首层，α 从 1 线性退到 0；**推理期 α=0**，
> engine 自动喂全零，不读取任何暗牌。
>
> 状态：**实现完成 + 全链路冒烟通过**；**100k 探针已完成（2026-11-04 01:03 UTC）**：
> α=0 终点 top1 **−0.49pp**（容差 −0.3pp）⇒ **GATE FAIL**，按预注册停止，不上 16k 臂。
> 但行为显著转向防守（放铳 −2.4pp、副露 −3.5pp），见 §5b。

---

## 1. 机制

```
输入 = cat( obs(1012) + s6c(2),  α · invisible_obs(217) )
α(t) = alpha_start,                       t < warmup
     = 线性 → alpha_end,                  warmup ≤ t < warmup + anneal
     = alpha_end,                         t ≥ warmup + anneal
推理：invisible_obs = zeros → 与父代逐位一致（新通道零初始化）
```

- `Brain(is_oracle=True)` 已存在（`model.py` 的 `torch.cat((obs, invisible_obs), 1)`），
  本实现只补 **α 调度、数据管线、推理零填充** 三块。
- α 以**本次暖启后的步数**为 0 点（父代 956k → `og_start_step`），否则 α 会永远钉在 0。
- 续训安全：`oracle_start_step` 写进 checkpoint，重启从断点继续调度，不会把 α 拉回 1.0。

## 2. 代码改动

| 文件 | 改动 |
|---|---|
| `mortal/train.py` | 新增 `_oracle_alpha(cfg, step)`；`[oracle_guiding]` 解析与断言（只支持 `cql_mc`）；`FileDatasetsIter(oracle=...)`；`_reorder` 把 `invisible_obs` 从 idx 1 挪到 `rest` 首位；`og_start_step` 相对调度；α 写 tensorboard + 日志 |
| `mortal/engine.py` | v2/3/4 且 `is_oracle` 时，`invisible_obs=None` → 自动 `zeros(B, 217, 34)` 并传给 brain |
| `mortal/mortal.py` | `is_oracle` 从 ckpt 的 `[resnet]` 读，传给 engine |
| `mortal/holdout_metrics.py` | 评估时喂全零（与推理同口径） |
| `mortal/player.py` / `ladder.py` / `one_vs_three.py` | 所有 engine 的 `is_oracle` 跟随 `model.is_oracle`（否则 test_play/ladder 会在 brain 的 assert 处崩） |
| `tools/expand_ckpt_oracle.py` | 首层手术：`Conv1d(192,1014,3) → (192,1231,3)`，新 217 通道零初始化；内置前向等价自检 |
| `mortal/config.og_probe.toml` | 100k 探针：`max_train_steps=1056000`、`oracle_guiding`、新 seed |
| `mortal/config.og_smoke.toml` | 1k 冒烟（不触发 test_play） |
| `mortal/run_og.sh` | 编排：`surgery`（含透明性校验）/ `smoke` / `probe` / `all` |

## 3. 冒烟结果（2026-11-03）

**手术**（`checkpoints/s6c/mortal.pth` 956k → `checkpoints/og/mortal.pth`）：

```
首层 (192, 1014, 3) -> (192, 1231, 3)；params 10,944,785
[verify] phi max|diff| = 0.000e+00
```

**透明性**（α=0，同一 holdout 12,288 样本）：

| ckpt | top1 | dqn_loss | R2 | var_q/var_t | agari |
|---|---:|---:|---:|---:|---:|
| S6c/956k 父代 | 0.8167 | 0.3613 | −1.2911 | 0.4722/0.3154 | 149/149 |
| OG 手术产物 | 0.8167 | 0.3613 | −1.2911 | 0.4722/0.3154 | 149/149 |

抽样摘要 `9c5362bf…` 两侧相同 ⇒ 管线完全透明。

**1k 步冒烟**（α=1.0，batch 512，GPU2）：

- `oracle alpha @ 957000: 1.0000`（相对调度生效）；
- `training budget reached`；
- 首层 oracle 切片 `mean|w|=7.1e-4 / max=4.4e-3`（梯度已流入，不是死通道）；
- α=0 的 holdout：top1 **0.8166**（父代 0.8167，−0.01pp）、dqn_loss 0.3549（父代 0.3613）；
- `test_play`（1v3，4 局）跑通 ⇒ engine 零填充与 `is_oracle` 传递无误。

## 4. 运行

```bash
# 手术 + 透明性 + 1k 冒烟（约 12 分钟）
tmux new -d -s og "cd /mnt/data8/mortal/Mortal && STAGE=smoke bash mortal/run_og.sh"

# 100k 探针（约 14 h @ 2 batch/s；α: 10k 保持 1 → 50k 线性退到 0 → 40k 为 0）
tmux new -d -s og "cd /mnt/data8/mortal/Mortal && STAGE=probe bash mortal/run_og.sh"
```

产物：`mortal/checkpoints/og/mortal.pth`、`mortal/logs/og/`、`runs/main_og/`。

**运行记录**：`STAGE=probe` 于 2026-11-03 11:35 UTC 启动，2026-11-04 01:03 UTC
跑到 1,056,000 步（约 13.5 h，含 2 次 test_play）。结果与判定见 §5b。

## 5. 预注册门禁

1. **透明性**：α=0 的暖启 ckpt 与父代 holdout 逐位一致（已过）。
2. **100k 探针**：α=0 时 holdout top1 不退（−0.3pp 内）、dqn_loss ≤ +2%。
3. 只有 2 过，才上 **16k 标准臂**（δ_min≈0.66，新种子块 1040000/1060000）；
   若只有 +0.3 级，按分辨率账写“未决”，不写“转正”。

## 5b. 100k 探针结果（2026-11-04 01:03 UTC 完成，**FAIL**）

跑到 1,056,000 步（100k）。逐 checkpoint 的 holdout（α=0，同一 12,288 样本，
父代 S6c/956k：top1 0.8167、dqn_loss 0.3613）：

| step | α | top1 | Δ vs 父代 | dqn_loss | Δ |
|---:|---:|---:|---:|---:|---:|
| 956k（父代） | — | 0.8167 | — | 0.3613 | — |
| 980k | 0.80 | 0.8149 | −0.18pp | 0.3715 | +2.8% |
| 1,000k | 0.40 | 0.8141 | −0.26pp | 0.3693 | +2.2% |
| 1,020k | 0.00 | 0.8130 | −0.37pp | 0.3738 | +3.5% |
| 1,040k | 0.00 | 0.8158 | −0.09pp | 0.3581 | −0.9% |
| **1,056k（终点）** | 0.00 | **0.8118** | **−0.49pp** | **0.3529** | **−2.3%** |

**判定：FAIL**——终点 top1 −0.49pp，超出 −0.3pp 容差（dqn_loss 那一半通过）。

两条解读：

1. **不是崩溃式捷径**：退火期间 top1 只掉 0.2~0.4pp、loss 只升 ~3%，α=0 后
   1,040k 还回升到 −0.09pp；终点 −0.49pp 里含噪声。5 个点的均值 ≈ **−0.28pp**，
   恰好压在容差线上。
2. **但行为确实变了**（1,050k 的 test_play，800 局，vs 200k 基线）：
   和了 **21.63%**、放铳 **9.83%**、副露 **27.57%**、立直 **18.40%**；
   对照 S6c/956k：21.67% / **12.27%** / **31.09%** / 18.55%。
   ⇒ 和了持平、**放铳 −2.4pp、副露 −3.5pp**：策略明显转向防守。
   这与“Q 拟合更好（loss −2.3%）但人类动作一致性下降（top1 −0.49pp）”完全一致：
   oracle 没有提供新的可学信号，但把 Q 的景观推向了更保守的一侧。

**按预注册：停在这里，不上 16k 臂。** 若 PI 愿意破例检验“更保守是否更强”，
可用 1,040k 快照（α=0 后 top1 最好，−0.09pp）跑 16k 臂；但这属于协议例外，
需要在 MEMORY §3 记录并接受多重比较的代价。

## 6. 风险与边界

- **捷径风险**：模型可能把 oracle 当捷径，α=0 后崩塌；探针要看 α=0 的 holdout 曲线，
  而不只是训练 loss。Suphx 用长退火缓解，本实现给了 50k 步退火。
- **信息上限**：本仓库"全知铳损"探针显示给了全知信息后弃牌已与人类持平
  （96.9% vs 96.8%）⇒ Oracle Guiding 更可能改善**学习效率**，而不是抬高天花板。
- **只支持 `cql_mc`**：IQL 需要 `next_invisible_obs`，dataloader 未提供；已加断言。
- **部署兼容**：旧 ckpt（`is_oracle` 缺失）行为逐位不变；oracle ckpt 走
  `engine` 零填充，`ladder`/`one_vs_three`/`test_play` 均已适配。
