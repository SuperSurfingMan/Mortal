# Oracle Guiding + 退火：最小实现与冒烟（2026-11-03）

> 目标：把 Suphx 的 Oracle Guiding 做成本仓库可跑的最小版本——**训练期**把三家暗牌
> （`invisible_obs`，217 行）按 α 缩放后拼进首层，α 从 1 线性退到 0；**推理期 α=0**，
> engine 自动喂全零，不读取任何暗牌。
>
> 状态：**实现完成 + 全链路冒烟通过**；**100k 探针已启动**
> （2026-11-03 11:35 UTC，tmux `og`，GPU2，预计 ~14 h 到 1,056,000 步）。

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

**当前运行**：`STAGE=probe` 已于 2026-11-03 11:35 UTC 在 tmux `og` 启动；
日志 `mortal/logs/og/probe_train.log`。关键检查点：

| 绝对步 | 相对步 | α | 说明 |
|---:|---:|---:|---|
| 957k | 1k | 1.00 | 首次保存，确认相对调度生效 |
| 966k | 10k | 1.00 | 退火起点 |
| 1,016k | 60k | 0.00 | 退火结束 |
| 1,000k / 1,050k | — | — | 两次 test_play（800 局） |
| 1,056k | 100k | 0.00 | 结束 + holdout 门禁 |

## 5. 预注册门禁

1. **透明性**：α=0 的暖启 ckpt 与父代 holdout 逐位一致（已过）。
2. **100k 探针**：α=0 时 holdout top1 不退（−0.3pp 内）、dqn_loss ≤ +2%。
3. 只有 2 过，才上 **16k 标准臂**（δ_min≈0.66，新种子块 1040000/1060000）；
   若只有 +0.3 级，按分辨率账写"未决"，不写"转正"。

## 6. 风险与边界

- **捷径风险**：模型可能把 oracle 当捷径，α=0 后崩塌；探针要看 α=0 的 holdout 曲线，
  而不只是训练 loss。Suphx 用长退火缓解，本实现给了 50k 步退火。
- **信息上限**：本仓库"全知铳损"探针显示给了全知信息后弃牌已与人类持平
  （96.9% vs 96.8%）⇒ Oracle Guiding 更可能改善**学习效率**，而不是抬高天花板。
- **只支持 `cql_mc`**：IQL 需要 `next_invisible_obs`，dataloader 未提供；已加断言。
- **部署兼容**：旧 ckpt（`is_oracle` 缺失）行为逐位不变；oracle ckpt 走
  `engine` 零填充，`ladder`/`one_vs_three`/`test_play` 均已适配。
