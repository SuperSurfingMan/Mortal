# 风格模仿微调系统实施计划

## 目标与总原则

为单个天凤凤凰卓玩家微调出 `style_<slug>.pth`，使决策分布接近该玩家，同时用 KL 锚定把强度损失控制在可接受范围。三条硬约束：

1. **主模型零写入**：只读取父代 checkpoint，所有产物写入 `mortal/checkpoints/style/<slug>/`、`mortal/indexes/style/`、`mortal/runs/style/<slug>/`（均已被 `.gitignore` 覆盖）。
2. **训练选项全部可选**：`train.py` 的改动只在 `control.objective == 'bc_style'` 时生效；不新增任何现有配置必填键；现有 `config.round5.toml` / `config.s6c.toml` / `config.iql_b_v3.toml` 可原样运行且行为不变。
3. **不改 Rust 侧**：复用 `libriichi` 已有的 `player_names` 白名单座位过滤（`libriichi/src/dataset/gameplay.rs:169`）与 `Stat.from_dir`（`libriichi/src/stat.rs:448`，原生支持 `.json.gz` 与玩家名过滤）。

## 改动清单

### 1. `mortal/train.py`（唯一被修改的主线文件，改动全部门控）

- `objective` 断言扩为 `('cql_mc', 'iql', 'bc_style')`；默认仍为 `cql_mc`。
- 新增可选项读取：`control.enable_test_play`（默认 `True`，false 时跳过 `TestPlayer` 构造与 `test_every` 评测块）——这是唯一会进入老配置路径的键，用 `.get(..., True)` 保证逐位兼容。
- 新增 `load_frozen_policy(ckpt, device)` 辅助函数：仿 `TestPlayer.__init__` 从 `state['config']` 解析 `version/resnet/extra_features` 并 `eval()+requires_grad_(False)`，供 KL 锚使用。
- `bc_style` 分支（`train_batch` 内新增，约 40 行）：
  - `q_out = dqn(mortal(obs), masks)`；非法动作已是 `-inf`，直接 `log_softmax` 得合法动作分布。
  - 主损失 `ce = -logp[range, actions].mean()`（可选 `style.logit_temp` 对 logits 缩放，默认 1.0）。
  - 正则 `kl = (p * (logp - logp_base)).sum(-1)`，仅在合法 mask 上求和，乘 `style.kl_weight`；`kl_weight=0` 时不加载 base 模型、不占显存。
  - `style.trainable == 'head'` 时冻结 `mortal.encoder` 全部参数（小数据玩家的兜底方案）。
  - `aux.next_rank_weight` 沿用现有配置（风格配置里默认置 0）。CQL、Q 回归、IQL 均不进入此分支。
- 构造 `FileDatasetsIter` 时传 `objective='cql_mc' if objective=='bc_style' else objective`（复用现有条目格式，`dataloader.py` 零改动；reward 字段在训练端忽略）。
- `stats` / TensorBoard 增加 `ce_loss`、`kl_loss`、`agreement_batch`（批内 top-1 命中率，作为训练侧即时监控）；NaN 守卫与快照逻辑原样复用。

### 2. 配置模板与生成器

- 新增 `tools/style_ft/make_style_cfg.py`：输入父代 ckpt、玩家白名单文件、holdout 起止月份、`--slug`，**以父代 ckpt 内保存的 `config` 为底**逐项打补丁，输出 `mortal/config.style_<slug>.toml`：
  - `[control]`：`objective='bc_style'`、独立 `state_file/best_state_file/tensorboard_dir`、`reset_optimizer='style-<slug>-v1'`、`enable_test_play=false`、`max_train_steps=20000`、`save_every=2000`、`snapshot_every=20000`、`test_every=20000`、`seed` 固定。
  - `[dataset]`：`globs` 为训练窗（不含 holdout 月份）、`player_names_files=[单人白名单]`、`file_index='indexes/style/<slug>_train.pth'`、`data_version='style-<slug>-v1'`（manifest 机制自动失效重建）、`num_workers=2`、`file_batch_size=16`、`num_epochs=1`。
  - `[style]`：`kl_weight=0.5`、`kl_state_file=<父代 ckpt>`、`trainable='full'`、`logit_temp=1.0`、`parent_sha256=<前16位>`。
  - `[freeze_bn].mortal=true`、`[cql].min_q_weight=0`、`[aux].next_rank_weight=0`、`[optim.scheduler] peak=2e-5, final=1e-7, warm_up_steps=500, max_steps=20000`。
  - `[resnet]`、`[env]`、`[baseline]`、`[test_play]`、`[grp]` 原样继承父代配置（保证 `Brain` 形状与暖启可加载、`test_play.games` 等必读键存在）。

### 3. 数据工具（`tools/style_ft/`）

- `make_player_whitelist.py`：输入玩家名（可多次传，兼容别名）与扫描 globs，并行读 `start_game.names` 统计每人生成每月份局数；输出 `dataset/index/style/<slug>.txt`（一行一名）与 `<slug>_stats.json`（总/分月局数、建议 holdout 月份、数据量档位：`full >=2000 局 / head 500~2000 / 放弃 <500`）。
- `human_profile.py`：用白名单筛出的对局在 `runs/style/<slug>/human_logs/` 建 `.json.gz` 软链农场，调用 `libriichi.stat.Stat.from_dir(dir, player_name)`，导出全部 getter（`riichi_rate/houjuu_rate/fuuro_rate/avg_riichi_jun/avg_point_per_agari/...`）为 JSON。

### 4. 评测工具（`tools/style_ft/`）

- `eval_agreement.py`：接受多个 ckpt（风格模型、父代模型 b、可选 baseline），在 holdout 月份对局上逐决策点比对 `argmax(masked Q)` 与该玩家实际动作，输出总体与分动作类（打牌 0..36 / 立直 37 / 吃碰杠 38..42 / 和了 43 / 九种九牌 44 / pass 45）、分巡目段的 top-1 一致率 JSON。
- `compare_style.py`：合并 `eval_agreement` 与 `human_profile` 的 JSON，输出 Markdown 对比表（人类 / 父代 / 风格模型 + 差值）；行为指标用绝对差与相对差，一致率用百分点 + 局级 bootstrap 95% CI（复用 `tools/read_model/compare_bootstrap.py` 的写法）。
- 强度保持：直接复用 `mortal/ladder.py`（`python ladder.py style.pth parent.pth ...`），不新增代码。

### 5. 驱动脚本

- `tools/style_ft/run_style_ft.sh <player> <parent_ckpt> <holdout_start_month>`：白名单 → 配置生成 → （可选）训练 → 对 holdout 跑 `eval_agreement`（风格/父代）→ `human_profile` → `compare_style` → `ladder` 风格 vs 父代，全部日志落 `runs/style/<slug>/`。

## 训练配方（单玩家）

- 暖启父代权重，`reset_optimizer` 丢弃父代优化器动量；`freeze_bn=true`；lr 2e-5 cosine、20k steps、batch 512。
- 数据：该玩家 2019 至今全窗对局，留出最近 2~3 个月做 holdout（不在训练 globs 内）。
- 第一轮先跑 `kl_weight` 三点扫（0 / 0.5 / 2.0），各 20k steps，产出风格-强度曲线；默认交付 `0.5`。

## 判定标准（试点玩家）

1. **风格**：holdout 动作一致率相对父代提升 ≥ 3pp，且分动作类中打牌与立直两类均不下降。
2. **行为画像**：8 项核心指标（立直率、鸣牌率、默听和率、放铳率、立直巡目、立直和了点、和了率、平均和了点）中至少 5 项向人类方向靠近，且无一项偏离超过 0.5 个标准差。
3. **强度**：`ladder` 风格 vs 父代 δ 目标 ≥ −8 点/半庄（记录 CI；若更差则回退 kl_weight 更高的档位）。
4. **隔离**：对比改动前后主配置 smoke（同 seed、前 2000 步 loss 日志）逐行一致；主 `mortal.pth/best.pth` mtime 不变。

## 分阶段实施（含验收）

- **P0 兼容性基线**：先留一份现有 smoke 配置的 2000 步 loss 日志；完成 `train.py` 改动后重跑同配置，diff 无差异。验收：无差异 + 主 ckpt 未被写。
- **P1 数据工具**：`make_player_whitelist.py` + `human_profile.py`。验收：试点玩家局数统计与 `player_ratings_*.csv` 吻合（±0），画像 JSON 字段齐全。
- **P2 训练目标**：`bc_style` 分支 + 配置生成器 + 2k 步冒烟（单玩家小窗）。验收：loss 有限、`agreement_batch` 上升、holdout 一致率可测。
- **P3 评测链**：`eval_agreement.py` + `compare_style.py` + ladder 接入。验收：父代在 holdout 上的基线一致率可复现（两次运行相同）。
- **P4 端到端试点**：选数据量最大玩家（如 2022–2026H1 的 13,951 局玩家）跑三点 kl 扫 + 判定三项标准。
- **P5 批量扩展（可选）**：按 `top_players_2022_2026H1.txt` 批量生成配置与排队训练；不改架构。

## 风险与回退

- Q 值当 logits 的 CE 非概率校准：留 `logit_temp` 旋钮；若一致率提升饱和，再考虑温度缩放或 top-k 蒸馏。
- 小数据过拟合：`trainable='head'` + 更高 kl（1.0~2.0）+ 更少 steps；以 holdout 一致率为早停依据。
- 玩家改名：白名单支持多别名，统计脚本按总局数校验覆盖率。
- 强度损失超阈值：提高 kl_weight 重新微调，主模型不受影响，随时丢弃风格分支。
- 回退：删除 `tools/style_ft/` 与 `train.py` 中门控分支即恢复原状（无数据/索引副作用）。

## 假设

- 父代 ckpt 为当前主模型（如 `mortal/checkpoints/mortal_step_000200000.pth` 或 round5/best），其 s6c `extra_features` 口径随原 config 一并继承。
- 训练机为 P100 16GB，`batch_size=512`、`enable_amp=true` 下加一份冻结 base 前向可容纳；否则将 `kl_weight` 置 0 或降 batch。
- 单人至少 500 局数据才启动微调；少于 500 局只建议调采样温度，不做权重微调。