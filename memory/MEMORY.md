# Mortal 项目上下文记忆文档

> **用途**：沉淀本项目全部历史对话上下文，供后续会话快速恢复背景，避免重复调研。
> **创建时间**：2026-10-01
> **信息来源**：本地代码阅读（`libriichi/`、`mortal/`）+ 官方文档站 + 天凤端点实测 + 作者公开声明
> **可靠性标注**：✅ 已实测/已读源码　🔶 来自官方文档或作者声明　⬜ 尚未验证

---

## 目录

1. [项目概览](#1-项目概览)
2. [整体架构](#2-整体架构)
3. [libriichi（Rust 核心）详解](#3-libriichi-rust-核心详解)
4. [Python 训练层详解](#4-python-训练层详解)
5. [配置文件全字段说明](#5-配置文件全字段说明)
6. [从零训练的完整流程](#6-从零训练的完整流程)
7. [权重现状（重要）](#7-权重现状重要)
8. [天凤数据获取调研（实测）](#8-天凤数据获取调研实测)
9. [数据质量过滤规则](#9-数据质量过滤规则)
10. [数据集目录规范](#10-数据集目录规范)
11. [环境状态与本机限制](#11-环境状态与本机限制)
12. [历史操作记录（时间线）](#12-历史操作记录时间线)
13. [待办与下一步](#13-待办与下一步)
14. [关键坑位备忘](#14-关键坑位备忘)
14bis. [日志文件归置规范（强制执行）](#14bis-日志文件归置规范强制执行)
15. [关键文件索引](#15-关键文件索引)
16. [服务器实跑现状（2026-10-07~10-09 远程同步）](#16-服务器实跑现状2026-10-0710-09-远程同步)
17. [进展总纲（10-01→10-11 系统性汇总，新会话先读这一节）](#17-进展总纲10-0110-11系统性汇总)

---

## 1. 项目概览

| 项 | 内容 |
|---|---|
| 项目名 | Mortal（凡夫） |
| 定位 | 开源日本麻将（四人立直麻将）AI，深度强化学习驱动 |
| 作者 | Equim-chan |
| 仓库 | https://github.com/Equim-chan/Mortal |
| 文档站 | https://mortal.ekyu.moe |
| 在线服务 | https://mjai.ekyu.moe （牌谱检讨 + 个室陪打） |
| 许可证 | 代码 AGPL-3.0-or-later；Logo/素材 CC BY-SA 4.0 |
| 本地路径 | `c:\Maj\Mortal` |
| 强度自称 | 强于 akochan 及作者可获得的全部 baseline，约等于早期 Suphx 🔶 |

**特性** 🔶
- 兼容天凤四人麻将标准段位规则
- Rust 高性能模拟器 + Python 神经网络推理，最高可达 **40K 半庄/小时**（RTX 4090 + Ryzen 9 7950X，batch 2000）
- 提供易用的 mjai 接口，可作为 `mjai-reviewer` 后端

**仓库顶层结构**

```
c:\Maj\Mortal\
├── libriichi\      # Rust 核心库（麻将逻辑/特征/模拟器），编译为 Python 扩展
├── mortal\         # Python 训练与推理框架
├── exe-wrapper\    # 可执行包装器（Rust）
├── log-viewer\     # 牌谱查看器（仅 README/index.example.html/.gitignore）
├── docs\           # mdBook 文档源（本地版本较旧，多数章节为空占位）
├── dataset\        # 【本项目新增】数据集目录（空骨架）
├── memory\         # 【本项目新增】记忆文档目录
├── tools\          # 【本项目新增】数据管线工具目录
├── Cargo.toml      # workspace：libriichi + exe-wrapper
├── Cargo.lock
├── Dockerfile      # 仅用于 mjai 推理，官方明确说明「不用于训练」
├── environment.yml # conda 环境定义
├── typos.toml
└── README.md
```

---

## 2. 整体架构

```
┌────────────────────────────────────────────────────────────┐
│ Python 层 (mortal\)                                        │
│  训练循环 train.py · 网络 model.py · 奖励构造 dataloader.py │
│  在线自博弈 server.py / client.py · 评估 one_vs_three.py    │
├────────────────────────────────────────────────────────────┤
│ Rust 核心 (libriichi\)  ── PyO3 编译为 libriichi.pyd/.so ── │
│  状态机 state\ · 特征编码 obs_repr · 牌谱解析 dataset\      │
│  自博弈模拟器 arena\ · mjai 协议 mjai\ · 麻将算法 algo\     │
└────────────────────────────────────────────────────────────┘
                            ↑
              训练数据：天凤凤凰卓 mjai 牌谱 (*.json.gz)
```

**核心设计要点**
- Rust 侧负责所有性能敏感工作：牌谱解析、状态更新、特征编码、对局模拟
- 神经网络与训练逻辑全在 Python
- 两侧通过 PyO3 桥接：Rust 侧 `MortalBatchAgent` 攒批次后回调 Python 的 `react_batch()`
- **奖励不在 Rust 侧计算**，由 Python 依据最终顺位生成（见 §4.3）

---

## 3. libriichi（Rust 核心）详解

### 3.1 编译配置 ✅

- `Cargo.toml`（workspace）：成员 `libriichi`、`exe-wrapper`；release 启用 `lto`、`codegen-units = 1`
- `libriichi/Cargo.toml`：
  - crate 名 **`riichi`**，产物类型 `cdylib` + `rlib`（既作 Python 扩展又作 Rust 库）
  - features：`pymod`（默认，PyO3 extension-module）、`mimalloc`（默认，全局分配器）、`abi3`、`sp_reproduce_cpp_ver`
  - 关键依赖：`pyo3 0.25`（auto-initialize、multiple-pymethods）、`numpy`/`ndarray`、`rayon`（并行）、`flate2`（gzip）、`serde_json`、`rand_chacha` + `sha3`（牌山种子）、`crossbeam`/`parking_lot`

### 3.2 模块清单 ✅

`lib.rs` 公开：`state`、`mjai`、`dataset`、`arena`、`stat`、`algo`、`agent`、`hand`、`tile`
`#[pymodule] libriichi` 注册子模块：`consts`、`state`、`dataset`、`arena`、`stat`、`mjai`，并暴露 `__version__`

| 模块 | 职责 |
|---|---|
| `state/` | 状态机与特征编码（核心） |
| `mjai/` | mjai 事件协议与 Bot 封装 |
| `dataset/` | 牌谱加载与训练样本产出 |
| `arena/` | 自博弈模拟器（可复现牌山、批量对局、1v3/2v2） |
| `algo/` | 向听数、和牌判定、符点、单人期望打点计算器 |
| `agent/` | Agent trait 与各类实现（含 Python 桥接） |
| `hand/`、`tile/` | 手牌与牌的基础类型 |
| `stat/` | 牌谱统计 |

### 3.3 state/ —— 状态与特征编码 ✅

**`state/player_state.rs`：`PlayerState`** 维护某一座位的全部可观测信息：

- 手牌 `tehai[34]`、`waits`、`dora_factor`、`tiles_seen`
- 四家牌河 `kawa`（含 `KawaItem`：切牌/鸣牌/立直/tedashi 标记）、`kawa_overview`、`fuuro_overview`、`ankan_overview`
- 分数、顺位、本场、供托
- `shanten`（向听数）、`last_cans`（当前合法动作集）
- 由 mjai 事件驱动更新（见 `state/update.rs`）

**`state/obs_repr.rs`：`ObsEncoderContext`** 输出 `(obs, mask)`，入口 `encode_obs(version, at_kan_select)`

编码内容：
- 手牌、赤牌
- 四家分数（clamp 后按 10 万归一化）、顺位 one-hot
- 场风/自风、宝牌指示牌
- 自家牌河：前 6 巡 + 最近 18 巡（每项 4 通道，常量 `SELF_KAWA_ITEM_CHANNELS`）
- 他家牌河：每项 8 通道（`KAWA_ITEM_CHANNELS`）+ 时间衰减 `exp(-0.2Δt)`
- 副露：5 通道
- 剩余牌数/69、向听数
- **v4 额外**：单机期望打点计算器 `SPCalculator` 的向听/和率/期望打点表（`MAX_NUM_TURNS = 17`，3×17 通道）；`encode_ev` 最大期望值；进张表 2×34+2 通道

**维度常量（`consts.rs`）**

| 版本 | obs 形状 [通道, 34] | 天眼 oracle 形状 |
|---|---|---|
| v1 | (938, 34) | (211, 34) |
| v2 | (942, 34) | (217, 34) |
| v3 | (934, 34) | (217, 34) |
| v4 | **(1012, 34)** | (217, 34) |

- `MAX_VERSION = 4`
- **`ACTION_SPACE = 46`**：0–36 打牌/选杠、37 立直、38–40 吃、41 碰、42 杠、43 自摸和、44 九种九牌、45 pass
- `GRP_SIZE = 7`（GRP 网络输入维度）

### 3.4 agent/ —— Python 桥接 ✅

`agent/defs.rs`：`Agent`（单局）与 `BatchAgent`（批量）trait；输入 mjai 日志 `&[EventExt]`、`PlayerState`、可选 `InvisibleState`（oracle 特征 `Array2<f32>`）

`agent/mortal.rs`：**`MortalBatchAgent`** —— 与 Python 模型交互的关键组件
- 要求 Python 对象实现：
  - 方法 `react_batch(states, masks, invisible_states) -> (actions, q_values, masks, is_greedy)`
  - 属性 `name` / `is_oracle` / `version` / `enable_quick_eval` / `enable_rule_based_agari_guard`
- 特征编码用 `rayon::spawn` 并行；跨局攒 batch 后进入 GIL 调 Python
- 支持：quick_eval（唯一可打牌直接出）、杠选择二次推理、基于规则的和牌守卫（agari guard）
- 动作索引 → 具体 mjai 事件的转换在此完成

其他：`py_agent.rs`（按 `engine_type` 分发 mortal/mjai-log）、`tsumogiri.rs`（摸切基准）、`mjai_log.rs`（日志重放）、`akochan.rs`（stdin/stdout 管道对接 akochan，环境变量 `AKOCHAN_DIR`）、`batchify.rs`（Agent→BatchAgent 适配）

### 3.5 arena/ —— 自博弈模拟器 ✅

- `arena/board.rs`：牌山生成 `init_from_seed((nonce, key))` → SHA3-256 派生每局种子 → ChaCha12 洗牌，**保证可复现**；含 `encode_oracle_obs`（天眼特征 217×34）
- `arena/game.rs`：`BatchGame`（8 局半庄 / 4 局东风，支持西入、突然死亡 30000 点规则）驱动四家 `BatchAgent` 的 poll 循环
- 对战模式（PyO3 类）：`arena/one_vs_three.rs` 的 `OneVsThree`（1 挑战者 vs 3 卫冕者轮换座位）、`arena/two_vs_two.rs` 的 `TwoVsTwo`（2v2 队战）
- `arena/result.rs`：`GameResult` / 顺位输出

**强度评测设定** 🔶：1v3 duplicate mahjong，每组 4 局同种子，四家轮换座位；顺位点分布 `[90, 45, 0, -135]`；常用种子 `0xd5dfaa4cef265cd7`（= 字符串 `Mortal the mahjong AI` 的 SHA-256 前 8 字节）

### 3.6 dataset/ —— 训练数据格式 ✅

- 数据来源：mjai 日志（`*.json.gz`，每行一个 JSON 事件，gzip + `serde_json` 解析，`rayon` 并行加载）
- `dataset/gameplay.rs`：**`GameplayLoader`**（参数 `version` / `oracle` / `player_names` / `excludes` / `trust_seed` / `augmented` 数据增强）→ 产出 `Gameplay` 条目：
  - 每步：`obs`、`invisible_obs`（oracle）、`actions`、`masks`、`at_kyoku`、`dones`、`apply_gamma`、`at_turns`、`shantens`
  - 每局：`grp`、`player_id`、`player_name`
  - ⚠️ **奖励不在此计算**，由 Python 侧依据最终顺位生成
- `dataset/grp.rs`：`Grp` —— 每局排名预测特征，形状 `[7, 4]`（大局数、本场、供托、四家分数/1 万），附 `rank_by_player`、`final_scores`
- `dataset/invisible.rs`：`Invisible`（牌山/岭上/里宝牌），`trust_seed` 时直接由种子重放牌山

### 3.7 mjai/ —— 事件协议 ✅

`mjai/event.rs`：`Event` 枚举 —— `StartGame`（含 seed）、`StartKyoku`、`Tsumo`、`Dahai`、`Chi`、`Pon`、`Daiminkan`、`Kakan`、`Ankan`、`Dora`、`Reach`、`ReachAccepted`、`Hora`、`Ryukyoku`、`EndKyoku`、`EndGame`
`EventExt` 附加元数据：`q_values`、`mask_bits`、`eval_time_ns`、`batch_size`、`kan_select`

`mjai/bot.rs`：`Bot` pyclass —— 单玩家在线推理封装，`react(line_json, can_act)` 收发 JSON 事件

### 3.8 algo/ —— 麻将算法 ✅

| 文件 | 内容 |
|---|---|
| `algo/shanten.rs` | tomohxx 向听数计算器的 Rust 移植，内嵌压缩查找表（`JIHAI_TABLE_SIZE = 78,032`、`SUHAI_TABLE_SIZE = 1,940,777`，数据在 `data/*.bin.gz`） |
| `algo/agari.rs` | 和牌判定（boomphf 完美哈希表，启动时 `ensure_init`） |
| `algo/point.rs` | 符/番 → 点数查表（`Point { ron, tsumo_ko, tsumo_oya }`，含切上满贯） |
| `algo/sp/` | nekobean 单人麻将计算器移植（`SPCalculator`）：各打牌的向听进张率、和率、期望打点、里宝率，供 v4 特征使用（`MAX_TSUMOS_LEFT` 实际为 17） |

### 3.9 bin/ —— CLI 工具 ✅

| 命令 | 用法 |
|---|---|
| `stat` | `stat <DIR> <PLAYER_NAME>` —— 对日志目录做牌谱统计（`Stat::from_dir`） |
| `validate_logs` | `validate_logs <DIR>` —— 并行校验 `**/*.json(.gz)` 牌谱合法性（重放 PlayerState + ActionCandidate 检查） |

---

## 4. Python 训练层详解

`mortal/` 目录文件：`client.py`、`common.py`、`config.example.toml`、`config.py`、`dataloader.py`、`engine.py`、`lr_scheduler.py`、`model.py`、`mortal.py`、`one_vs_three.py`、`player.py`、`prelude.py`、`reward_calculator.py`、`server.py`、`train.py`、`train_grp.py`

### 4.1 model.py —— 网络结构 ✅

| 组件 | 结构 |
|---|---|
| `ChannelAttention` | SE 通道注意力（avg + max 双路共享 MLP → sigmoid 权重） |
| `ResBlock` | 两个 Conv1d(k=3) + norm，接 ChannelAttention，残差相加；支持 pre-activation |
| `ResNet` | 输入 Conv1d → N 个 ResBlock → Conv1d(→32) → Flatten → Linear(32×34, 1024) |
| **`Brain`** | 主干：1D ResNet（**卷积沿 34 种牌维度**），默认 `conv_channels=192`、`num_blocks=40`，Mish 激活 + BatchNorm1d，输出 1024 维潜变量 φ；`is_oracle` 时通道拼接 oracle obs；版本 1 走 VAE 风格（mu/logsig 头），v2/3/4 直接返回激活后 φ |
| `AuxNet` | `Linear(1024, sum(dims))`，用于辅助预测下一局顺位（dims=(4,)） |
| **`DQN`** | Dueling 结构：`q = v + a - a_mean`，用 mask 填充 `-inf`；v1/v2/v3 为 MLP 头，**v4 为单个 `Linear(1024, 1 + ACTION_SPACE)`** |
| **`GRP`** | GRU（input `GRP_SIZE=7`，hidden 64，2 层）+ FC → **24 维**（4 人排名全排列数）；全程 float64；`calc_matrix()` 把 24 logits 归约成 `(N, 4, 4)` 的「玩家×顺位」概率矩阵；`get_label()` 把顺位映射到排列索引 |

### 4.2 train.py —— 主训练循环 ✅

**三项损失**：

```python
q_target_mc = gamma ** steps_to_done * kyoku_rewards          # 蒙特卡洛回报
phi = mortal(obs)
q_out = dqn(phi, masks)
q = q_out[range(batch_size), actions]
dqn_loss = 0.5 * mse(q, q_target_mc)                          # ① DQN 时序/回报损失
cql_loss = q_out.logsumexp(-1).mean() - q.mean()              # ② CQL 正则（仅离线）
next_rank_loss = ce(aux_net(phi), player_ranks)               # ③ 顺位辅助损失
loss = dqn_loss + cql_loss * min_q_weight + next_rank_loss * next_rank_weight
```

- 优化器：`AdamW`（decay / no-decay 参数分组，lr 由 scheduler 控制）
- 调度器：`LinearWarmUpCosineAnnealingLR`
- 混合精度：`GradScaler`（`enable_amp`）
- 保存：每 `save_every` 步存 `state_file`（含 mortal / current_dqn / aux_net / optimizer / scheduler / scaler / steps / timestamp / best_perf / config）
- 评测：每 `test_every` 步跑 `TestPlayer.test_play(test_games // 4, ...)`，写 TensorBoard；若 `avg_rank` 与 `avg_pt` 同时不劣于历史最优则另存 `best_state_file`
  - ⚠️ 显示用 `avg_pt([90, 45, 0, -135])` **不参与训练**
- 恢复训练：若 `state_file` 存在则自动续训（离线模式下会保留 optimizer/scheduler 状态）

**在线模式特殊处理**：
- `submit_param()` 提交参数；`drain()` 取轨迹
- `main()` 在 online 模式下 **spawn 子进程**（环境变量 `MORTAL_IS_SUB_PROC`）循环拉起，因为源码注释标明「online 模式训练进程会卡死在此处，原因未知」，故 `test_every` 后直接 `sys.exit(0)`
- 离线模式只跑一个 epoch（`while True` 后 `break`）

### 4.3 dataloader.py + reward_calculator.py —— 奖励构造（关键）✅

**这是 Mortal 奖励设计的核心：奖励 = GRP 预测的「每局期望顺位点变化」**

`RewardCalculator`（`reward_calculator.py`）：
- 默认 `pts = [3, 1, -1, -3]`
- `calc_grp(grp_feature)`：把每局前缀序列喂入 GRP → `calc_matrix` → 玩家×顺位概率
- `calc_rank_prob`：拼接「GRP 预测的前缀概率」与「最终真实排名的 one-hot」
- `calc_delta_pt`：`exp_pts[1:] - exp_pts[:-1]` → **逐局期望点增量即奖励**

`FileDatasetsIter`（`dataloader.py`，`IterableDataset`）：
- 在 `build_iter()` 中构造 GRP 并载入 `config['grp']['state_file']`（注释说明不能放在 `__init__`，Windows 下会失效）
- `populate_buffer()` 逐局处理：
  - `kyoku_rewards = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)`
  - `player_ranks`：由 `(-scores_seq).argsort().argsort()` 稳定排序得到
  - `steps_to_done`：从后向前累加 `apply_gamma`
- 每条样本：`[obs, (invisible_obs), action, mask, steps_to_done, kyoku_rewards[at_kyoku], player_ranks[at_kyoku+1]]`
- `worker_init_fn`：把 `file_list` 按 worker 数切分
- `reserve_ratio` 支持留出验证集

### 4.4 其他脚本

| 文件 | 作用 | 状态 |
|---|---|---|
| `train_grp.py` | 预训练 GRP 排名预测器 | 已知存在，内容未细读 |
| `one_vs_three.py` | 挑战者 1v3 对战基线评估 | 已知存在 |
| `server.py` | 在线训练服务端：`State`(L19)、`Handler(BaseRequestHandler)`(L36)、`Server(ThreadingTCPServer)`(L126)、`main()`(L133) | 结构已知 |
| `client.py` | 在线训练客户端（挂机打牌上传轨迹） | 未细读 |
| `mortal.py` | 单文件推理入口：导入 `prelude`、`MortalEngine`(engine)、`libriichi.mjai.Bot`、`libriichi.dataset.Grp`，`main()` 在 L20 | 结构已知 |
| `engine.py` | `MortalEngine` 定义 | 未细读 |
| `player.py` | `TestPlayer.test_play()` 评测 | 未细读 |
| `common.py` | `submit_param`、`parameter_count`、`drain`、`filtered_trimmed_lines`、`tqdm` | 未细读 |
| `prelude.py` | 环境预置（日志、路径等） | 未细读 |
| `config.py` | 读取 `config.toml` | 未细读 |
| `lr_scheduler.py` | `LinearWarmUpCosineAnnealingLR` | 未细读 |

### 4.5 environment.yml ✅

```yaml
name: mortal
channels: [conda-forge]
dependencies:
  - python=3.12
  - pip
  - pip: [tqdm, toml, tensorboard]
```

⚠️ **PyTorch 故意不列入**，需自行按硬件安装（作者建议用 pip，且只需 `torch`，不需 torchvision/torchaudio）

---

## 5. 配置文件全字段说明

`mortal/config.example.toml`（示例值多为占位符，需自行调优）✅

| 节 | 字段 | 含义 |
|---|---|---|
| `[control]` | `version` | 模型版本（1–4，建议 4） |
| | `online` | 是否在线自博弈训练 |
| | `state_file` / `best_state_file` | 检查点路径 / 最优检查点路径 |
| | `tensorboard_dir` | TensorBoard 输出目录 |
| | `device` | 如 `cuda:0` |
| | `enable_cudnn_benchmark` / `enable_amp` / `enable_compile` | cudnn 基准 / 混合精度 / torch.compile |
| | `batch_size` / `opt_step_every` | 批大小 / 每 N 步更新一次 |
| | `save_every` / `test_every` / `submit_every` | 存盘点 / 评测间隔 / 提交参数间隔（断言保证整除关系） |
| `[train_play.default]` | `games` / `log_dir` / `boltzmann_epsilon` / `boltzmann_temp` / `top_p` / `repeats` | 自博弈采样参数（环境变量 `TRAIN_PLAY_PROFILE` 选择 profile） |
| `[test_play]` | `games` / `log_dir` | 评测对局数 / 日志目录 |
| `[dataset]` | `globs` | **训练数据 glob，如 `/path/to/dataset/**/*.json.gz`** |
| | `file_index` | 文件索引缓存路径 |
| | `file_batch_size`（默认 15，1 文件约 660 样本） / `reserve_ratio` / `num_workers` | 文件批/留出比例/worker 数 |
| | `player_names_files` | 玩家白名单文件（**用于只保高段位玩家**） |
| | `num_epochs` / `enable_augmentation` / `augmented_first` | 轮数 / 数据增强（他家视角翻转） |
| `[env]` | `gamma` / `pts` | 折扣因子 / 顺位点分布（示例 `[6,4,2,0]`） |
| `[resnet]` | `conv_channels` / `num_blocks` | 默认 **192 / 40**（显存不足需下调） |
| `[cql]` | `min_q_weight`（示例 5） | CQL 损失权重 |
| `[aux]` | `next_rank_weight`（示例 0.2） | 顺位辅助损失权重 |
| `[freeze_bn]` | `mortal` | 在线训练时应为 `true` |
| `[optim]` | `eps`/`betas`/`weight_decay`/`max_grad_norm` | AdamW 参数（示例 eps 1e-8、betas [0.9,0.999]、wd 0.1、max_grad_norm 0） |
| `[optim.scheduler]` | `peak`/`final`/`warm_up_steps`/`max_steps` | 示例 peak=final=1e-4 |
| `[baseline.train]` / `[baseline.test]` | `device`/`enable_compile`/`state_file` | 基线模型（1v3 对战卫冕者） |
| `[online]` | `history_window`/`enable_compile` | 在线训练历史窗口 |
| `[online.remote]` | `host`/`port` | 远端服务地址 |
| `[online.server]` | `buffer_dir`/`drain_dir`/`sample_reuse_rate`/`sample_reuse_threshold`/`capacity`(1600)/`force_sequential` | 服务端缓冲区 |
| `[1v3]` | `seed_key`/`games_per_iter`(2000)/`iters`(500)/`log_dir` | 1v3 迭代评测 |
| `[1v3.challenger]` / `[1v3.champion]` / `[1v3.akochan]` | `device`/`name`/`state_file`/`stochastic_latent`/`enable_compile`/`enable_amp`/`enable_rule_based_agari_guard`；akochan 另有 `enabled`/`dir`/`tactics` | 挑战者/卫冕者/akochan 对手 |
| `[grp]` | `state_file` | **GRP 权重路径（训主模型前必须存在）** |
| `[grp.network]` | `hidden_size`(64)/`num_layers`(2) | GRP 网络 |
| `[grp.control]` | `device`/`enable_cudnn_benchmark`/`tensorboard_dir`/`batch_size`/`save_every`/`val_steps` | GRP 训练控制 |
| `[grp.dataset]` | `train_globs`/`val_globs`/`file_index`/`file_batch_size`(50) | GRP 训练/验证数据（按年月划分） |
| `[grp.optim]` | `lr`(1e-5) | GRP 学习率 |

---

## 6. 从零训练的完整流程

### 6.1 构建（Windows）✅ 来自 `docs/src/user/build.md`

```bash
# 1. 创建并激活 conda 环境（工作目录 $MORTAL_ROOT）
conda env create -f environment.yml
conda activate mortal

# 2. 安装 PyTorch（仅需 torch，CUDA 版）
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. 编译 libriichi 并放置到 mortal/ 下
cargo build -p libriichi --lib --release
# Linux:
cp target/release/libriichi.so mortal/libriichi.so
# Windows (MSYS2):
cp target/release/riichi.dll mortal/libriichi.pyd

# 4. 验证
cd mortal
python
>>> import libriichi
>>> help(libriichi)
```

### 6.2 可选目标 ✅

```bash
# 测试
cargo test --workspace --no-default-features --features flate2/zlib -- --nocapture
# 基准
cargo test -p libriichi --no-default-features --bench bench
# 可执行工具
cargo build -p libriichi --bins --no-default-features --release
cargo build -p exe-wrapper --release
# 文档
cd docs && cargo install mdbook mdbook-admonish mdbook-pagetoc && mdbook build
```

### 6.3 训练步骤

```
① 编译 libriichi（§6.1）
        ↓
② 准备训练数据：mjai 格式牌谱 *.json.gz（见 §8）
        ↓
③ 复制 config.example.toml → config.toml，填写 [grp] 与 [dataset] 等
        ↓
④ 先训 GRP：python train_grp.py     ← 奖励计算器依赖它，必须先完成
        ↓
⑤ 离线阶段：python train.py（online = false，只跑 1 个 epoch）
        ↓
⑥ 在线阶段：online = true，起 server.py 收集自博弈轨迹 + client.py 挂机打牌
        ↓
⑦ 评估：python one_vs_three.py
```

监控：`tensorboard --logdir <tensorboard_dir>`，观察 `loss/*` 与 `test_play/avg_rank`、`test_play/avg_pt`、`test_play/ranking` 等

### 6.4 官方 Docker（仅推理）🔶

```bash
git clone https://github.com/Equim-chan/Mortal.git && cd Mortal
sudo env DOCKER_BUILDKIT=1 docker build -t mortal:latest .
# 需要自备模型权重，挂载到 /path/to/model/dir
sudo docker run -i --rm -v /path/to/model/dir:/mnt mortal 2 < log.json
```

⚠️ 官方明确：**该镜像只用于 mjai 接口推理，不适用于训练**

---

## 7. 权重现状（重要）

**结论：作者不公开预训练权重。** 🔶

- 作者 gist（`https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56`）：
  - 2022-04-13 更新：源码已开源
  - **2022-08-19 更新（最终结论）**：考虑到在线版本 https://mjai.ekyu.moe 已基本实现其期望功能，**目前没有公开模型参数的计划**
- 不公开的核心原因：防止被用于段位战外挂（akochan 已有被做成外挂售卖的先例）
- 仓库 README「Okay cool now give me the weights!」一节直接指向该 gist

**更关键的连带影响**：作者在 gist 评论区（2022-04-14）亲口说明 ——
> 「仅凭现在公开的代码要去训练也不 trivial，因为我还没有公开数据集或者它们的获取方式，也没有说明训练的配置方式」

**已核实的官方文档状态**：
- 文档站目录 `SUMMARY.md` 中，`[Training]` 下的 **Pre-Trained Model / Configurations / Offline Phase / Online Phase 全部为空占位**（Markdown 链接为空）
- 本地 `docs/src/` 同样只有：`index.md`、`online/index.md`、`perf/strength.md`、`user/build.md`、`user/docker.md`、`ref/meta.md`、`ref/references.md`、`donate.md`、`contributors_sponsors.md`

→ **本项目「从零训练」没有官方教程，§6.3 的流程是基于源码逆向 + 社区实践整理。**

---

## 8. 天凤数据获取调研（实测）

> 全部结论于 **2026-10-01 实测验证**，非推测。

### 8.1 文件清单接口 ✅

```
GET https://tenhou.net/sc/raw/list.cgi
```

返回 JS 数组（实测 15729 bytes）：

```javascript
list([
{file:'sca20260924.log.gz',size:47641},
{file:'scb2026092400.log.gz',size:53798},
{file:'scc2026092400.html.gz',size:3438},
...
]);
```

### 8.2 文件类型（官方定义原文）✅

| 前缀 | 内容 | 含牌谱链接 | 起始时间 |
|---|---|---|---|
| sca | 個室 | ✗ | — |
| scb | 段位战（全卓级，**仅结果无牌谱**） | ✗ | — |
| **scc** | **鳳凰卓** | **✓** | 2009-02-01 |
| scd | 雀荘戦 | ✗ | 2010-10-14 |
| sce | 技能戦（已关闭）+ **琥珀卓** | 琥珀卓 ✓ | 2016-06-06 |
| scf | html 汇总 | — | 2015-11-04 |

**在线窗口仅保留最近 7 天**（scb/scc 按小时滚动）

官方注意事项原文要点：
- 更新周期约 5 分钟
- 按对局开始时间排序
- **新行可能插在文件中间**（长时对局）→ 必须按 log_id 去重

### 8.3 scc 文件格式 ✅

- gzip 压缩的 **HTML**，内容 **html-escaped（`&gt; &lt; &amp;`）**，解析前需 `html.unescape`
- 每行样例：
  ```
  21:01 | 27 | 四鳳南喰赤－ | |牌譜| | ogatta(+56.0) 悠楽うらめ(+1.5) かんきち1号(-19.0) りょちょぴこ(-38.5)
  ```
- 实测一个小时的 scc 文件含 **62 个牌谱 ID**（三凤/四凤混合）

**可用正则** ✅
```python
LOGID_RE = re.compile(r'log=([0-9]{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8})')
RULE_RE  = re.compile(r'(三|四)(鳳|特|上|般)(東|南)喰赤(速)?')
```

实测样例 ID：
- `2026093021gm-00a9-0000-09a004a9`（四鳳）
- `2026093021gm-00b9-0000-3d77264f`（三鳳）

⚠️ `00a9`=四鳳、`00b9`=三鳳 是观测规律，**过滤逻辑应以行内规则文本为准，不要依赖 ID 推断**

### 8.4 牌谱下载接口 ✅

```
GET https://tenhou.net/0/log/?<log_id>
```

- 返回 **gzip 压缩的 XML**（`<mjloggm ver="2.3">...`）
- 实测单局约 **15675 bytes**
- ⚠️ 响应形态取决于客户端（2026-10-01 修正）：`urllib` 拿到原始 gzip 字节（直接存，勿二次压缩）；`requests` 会自动解压 `Content-Encoding: gzip` 得到 XML 明文（需补压一次存 `.xml.gz`）。落盘前按 `1f 8b` 魔数判断

### 8.5 历史归档（2026-10-01 实测修正：zip 当前不可用）

- ❌ `https://tenhou.net/sc/raw/scrawYYYY.zip`：页面仍显示各年体量（2025=453M、2024=461M、2023=451M、2022=488M、2021=489M、2020=600M、2019=498M、2018=457M、2017=415M…），
  但抽测 2010/2020/2024/2025 **全部 404**（curl + 会话 + Referer 交叉验证）→ **2009~2025 历史牌谱当前无可下载入口**
- ✅ **滚动日归档可用**（当前唯一日粒度来源）：
  - 清单 `GET https://tenhou.net/sc/raw/list.cgi?old`；下载 `GET https://tenhou.net/sc/raw/dat/<年>/scc<YYYYMMDD>.html.gz`（支持 Range，返回 206）
  - 覆盖**当年 1/1 至约 7 天前**，与近 7 天小时窗口无缝衔接；内容为**原始 HTML**（非 html-escaped）
  - 2026-10-01 实测：266 个日文件共 8.4MB；老牌谱 ID（如 2026-01-01）仍可经 `/0/log/?<id>` 正常下载
- ⚠️ **当年数据必须当年抓**：年底日文件预计并入 scraw2026.zip（而 2025 zip 已 404）

### 8.6 公开数据集（2026-10-02 搜索发现）

| 项 | 内容 |
|---|---|
| 项目 | `NikkeTryHard/tenhou-to-mjai`（GitHub 61★，Apache-2.0，为 Mortal 训练而做） |
| 覆盖 | **2009–2026 天凤凤凰卓四人半庄全部**，~250 万局，~19GB tar（按年分包） |
| 格式 | `.mjai.json.zst` = zstd 压缩 mjai JSONL（**实测已验证**：hora 键名即 `ura_markers`，与 Mortal 直接兼容，无需改名；仅需解压 `.zst` → `.json.gz`） |
| 渠道 | ① BitTorrent magnet（README 主推，选择性下载）② HuggingFace：HF 直连被墙，**国内用 hf-mirror.com 可下**（已实测 2009/2026 两包下载+合并成功） |
| 许可 | 数据 as-is（不声称所有权、研究归档、下游自查上游条款）——⚠️ 与天凤 ToS 禁止二次分发存在灰色地带，用户需自行判断 |
| 缺口 | 2026 年仅含 1 月（12,188 局，实测）；2026 年 2 月至今需自建管线补充（互补关系） |

**价值**：官方年度 zip 404 导致 2009–2025 无入口，此数据集可补齐 17 年历史（~250 万局），且是 Mortal 原生格式。
**合并流程（已验证）**：`tenhou_pipeline.py merge-hf <tar...>` → 解压/自查/gzip 落盘/按 log_id 去重，新增明细记 `dataset/index/hf_merge_manifest.csv`。
**⚠️ 落地约束（2026-10-02 发现，改变了原方案）**：`D:\tenhou_hf` 所在卷 **D 盘是 exFAT、簇大小 256 KiB 的外置 USB 固态盘**，散装 `.json.gz`（每个 ~4 KiB）会被放大到 256 KiB/个 → **膨胀 64×**；全量历史 2009–2025（~243 万局）散文件落 D 需 **~620 GiB**，不可行。
**最终决策（用户拍板）**：**D 盘只保留 18 个年度 `.tar.zst` 原包（16.86 GiB），不转换、暂不改 Mortal 训练加载器**；已删除 D 盘上已生成的散装 `.json.gz`（`D:\tenhou_mjai`）及计划任务 `Mortal-Tenhou-MergeHF`。

### 8.7 官方红线（必须遵守）✅ 原文摘录

来自 `https://tenhou.net/sc/raw/` 页面：

> ※天鳳と競合する製品への開発・応用を目的として牌譜を使用していただくことはできません。
> ※天鳳の牌譜は、天鳳での対戦を公正に楽しんでいただく目的で公開されています。天鳳での対戦を必要としないサービスへの応用は無償有償ともに行えません。
> ※不特定多数が天鳳の牌譜をダウンロードするサービスは作成できません。

天凤官方 X（@tsuno_s，2024-06-22）：
> 牌譜のダウンロードは同時に１セッションでお願いします。１つの接続元から同時に数百以上のセッションでダウンロードする行為はご遠慮ください。

**实践约束**：串行下载 + 每请求 ≥2 秒间隔 + 控制总量；数据仅限本地研究训练，**不得二次分发原始牌谱、不得做成公开下载服务**

### 8.7 mjlog → mjai 转换工具

| 工具 | 语言 | 说明 | 验证状态 |
|---|---|---|---|
| **jansou** (`pip install jansou`) | Python | 支持 mjlog XML ⇄ mjai JSONL、tenhou JSON；子模块实测为 `analysis`/`core`/`game`/`io`/`scoring`/`validation` | 子模块名 ✅；**具体函数名待 `dir(jansou.io)` 实测** ⬜ |
| PaifuLogger (Jim137/Tenhou-Paifu-Logger) | Python | `plog --mjai` 按 URL 逐局输出 mjai | 🔶 文档确认 |
| mjlog2json (tsubakisakura) | Rust | mjlog XML → tenhou6 JSON（需二次转换到 mjai） | 🔶 crates.io 确认 |
| Tenhou Log Converter (gh.ekyu.moe/tenhou-convlog) | Web | 天凤牌谱 → 電脳麻将格式（**非 mjai**，仅供参考） | 🔶 |

**批量下载工具参考**（社区，2025-10 汇总仍有更新）🔶
- `MahjongRepository/phoenix-logs`（Python，全年 ID 获取 + 年度日志解析 + SQLite；默认 3 并发需改成 1）
- `Apricot-S/houou-logs`（Python，含礼貌下载机制：20 分钟冷却、增量跳过、DB 导出）
- `mthrok/tenhou-log-utils`（Python，库 + CLI）
- `trineutron/mahjong`（Shell，从年度日志批量下载）

⚠️ 本项目**未直接使用**这些工具，而是自建管线（避免默认多并发违反 ToS）

---

## 9. 数据质量过滤规则

**依据**：Mortal 仅支持四人麻将，且原作即用天凤标准段位规则训练

| 过滤项 | 规则 | 理由 |
|---|---|---|
| 数据池 | 只取 **scc（鳳凰卓）** | 凤凰卓入门即 R2000+，天然高质量；琥珀卓（sce）可选加入 |
| 人数 | `四` 开头（排除三麻） | Mortal 不支持三麻 |
| 长度 | `南`（半庄） | 排除东风局，与原作训练分布一致 |
| 规则 | `喰赤` | 标准有喰タン有赤牌 |
| 玩法 | 排除 `速`（速棋） | 保留训练分布纯净 |
| 去重 | 按 log_id | 文件重叠 / 历史行插入导致重复 |
| 完整性 | 转换后校验 | 见 §13 待办 |

→ **目标行匹配：`四鳳南喰赤－`**

---

## 10. 数据集目录规范

### 10.1 已创建的目录骨架 ✅

```
c:\Maj\Mortal\
├── dataset\
│   ├── raw\
│   │   ├── lists\        # scc 索引文件（近 7 天窗口 + 滚动日归档；年度 zip 已 404）
│   │   ├── archives\     # HF 年度包 C 盘临时中转区（下载后即复制到 D，随即清空）
│   │   └── mjlog\        # <log_id>.xml.gz 原始牌谱（天凤原样字节）——2026 自建下载
│   ├── mjai\
│   │   └── houou4\YYYYMM\<log_id>.json.gz   # 转换后训练数据（Mortal 直接可读）——仅 2026
│   └── index\
│       ├── houou_games.csv                  # 对局清单：log_id, 规则, 玩家, 下载/转换状态
│       └── hf_merge_manifest.csv            # HF 合并明细（当前无有效记录）
├── memory\
│   └── MEMORY.md        # 本文档
└── tools\
    └── tenhou_data\
        └── TASKLIST.md  # 数据管线执行清单（含可运行代码）

D:\tenhou_hf\            # ⚠️ exFAT / 簇 256 KiB 外置 USB 盘：只放大文件
└── tenhou-houou-YYYY.tar   # HF 年度原包（zstd mjai JSONL），18 包共 16.86 GiB
```

**现状（2026-10-02）**：C 盘 `dataset\mjai` 仅有 2026 自建数据（35,690 局）；历史 2009–2025 以年度 `.tar.zst` 原包形式存于 `D:\tenhou_hf`（未转换，理由见 §8.6）。

### 10.2 训练时 config 对应写法

```toml
[dataset]
globs = ['C:/Maj/Mortal/dataset/mjai/**/*.json.gz']
```

**格式要求**：`**/*.json.gz`，每行一个 mjai 事件 JSON（`start_game` / `start_kyoku` / `tsumo` / `dahai` / ...），见 `libriichi/src/dataset/gameplay.rs`
**自查点**：首行应为 `{"type":"start_game"}`，且存在带 `scores` 的 `start_kyoku`

### 10.3 ⚠️ Git 风险提示

仓库 `.gitignore` 当前内容为：

```
.vscode
__pycache__
/target
*.pyd
*.dll
*.exe
*.so
*.pth
.ipynb_checkpoints
*.ipynb
/_draft
```

**`dataset/`、`memory/`、`tools/` 均不在忽略列表中**。数据集可达数百 GB，且天凤 ToS 禁止二次分发，**务必先将其加入 `.gitignore` 再执行任何 git 操作**（此操作尚未执行）。

---

## 11. 环境状态与本机限制

| 项 | 状态 |
|---|---|
| Python | ✅ 3.13.5（Anaconda，`D:\Anaconda3`） |
| git | ✅ 2.45.1.windows.1 |
| **cargo** | ❌ **本机未安装**（仅数据管线用；训练侧已在服务器装好，见 §16.2） |
| 操作系统 | Windows（LAPTOP-6J5U6AJ8），时区 Asia/Shanghai |
| GPU | 本机 = RTX 3050 Ti Laptop 4GB（仅够数据管线，不宜训练）；**训练环境 = 远程服务器 Tesla P100-PCIE-16GB + torch-2.6.0+cu118**（2026-10-02 用户确认，标准配方 192×40/batch512 可用） |

> ⚠️ **本节仅描述本机 Windows**；训练服务器实况见 [§16.2](#162-环境与硬件实况2026-10-09-复核)（Ubuntu 18.04 容器 / 56 线程 / 251 GB 内存 / 4×P100 + 1×RTX 8000 / rust 1.99 / py 3.12.11 / torch 2.6.0+cu118 已验证可用）。

**已知环境限制** ⬜→✅：
- **沙箱限制 pip 写入系统 site-packages**：曾尝试 `pip install jansou` 到 `C:\Users\黄超平\AppData\Roaming\Python\Python313\site-packages`，报 `WinError 5 拒绝访问` 并被沙箱拦截（"Not allow operate files"）
- 沙箱可写目录包含 `C:\Maj\Mortal`，但**不含**用户级 Python site-packages
- 结论：**依赖安装与数据管线执行应由用户在自有环境完成**（这也是 `TASKLIST.md` 产生的直接原因）
- ✅ 2026-10-01 补充：以非沙箱方式 `pip install jansou` 成功（0.14.0 装入默认 Anaconda 环境），数据管线已可正常运行

---

## 12. 历史操作记录（时间线）

### 会话 1：项目架构分析
- **用户诉求**：分析 Mortal 全部文件，说明技术架构与如何开始训练
- **执行**：读取 `README.md`、`config.example.toml`、`mortal/model.py`、`mortal/train.py`、`mortal/dataloader.py`、`mortal/reward_calculator.py`；派子 agent 系统梳理 `libriichi/` 全模块
- **产出**：Rust+Python 双层架构说明、Rust 模块表、Python 训练层表、算法本质（离线 CQL + 在线自博弈、GRP 奖励整形）、5 步训练起步指南、4 条重要提醒

### 会话 2：权重与数据获取可行性
- **用户诉求**：作者已声明不开放权重；阅读 GitHub 仓库；如何从零训练；mjai 数据如何获取；天凤是否有公开站点可批量下载并转换
- **执行**：
  - 抓取作者 gist → 确认 2022-08-19 最终结论「不公开模型参数」
  - 抓取官方文档站 `SUMMARY.md` → 确认 Training 全章节为空占位
  - 搜索天凤牌谱批量下载工具汇总（Qiita，2025-10 更新）
  - 搜索 mjlog→mjai 转换工具
- **产出**：从零训练路线图、天凤牌谱机制说明（sc/raw + `0/log/?id`）、下载工具对比表（4 个）、转换工具（jansou / PaifuLogger / mjlog2json）、ToS 红线、数据规模建议

### 会话 3：实际数据采集尝试（被中断）
- **用户诉求**：拉取天凤目前可获取的全部数据，质量过滤并转 mjai，放入专用数据集目录
- **执行**（全部实测）：
  1. 环境检查 → Python 3.13.5 / git 2.45.1 / **cargo 缺失**
  2. 探测 `list.cgi` → HTTP 200，15729 bytes；前缀分布 sca=8、scb=177、scc=175、scf=7
  3. 抽样 scb/sca → 仅结果无牌谱链接（`log=` 出现 0 次）
  4. 抽样 scc → 初次用错扩展名（`.log.gz`）报 404；修正为 `.html.gz` 后成功
  5. 探测 `list.cgi?wr=1/2` → 返回空列表
  6. 抓取 `sc/raw` 页面 → **发现 scc=鳳凰卓(牌谱あり)、scrawYYYY.zip 年度归档、ToS 全文**
  7. 解析 scc 样本 → 提取 62 个 log_id，确认正则可用
  8. 验证单局下载 `tenhou.net/0/log/?2026093021gm-00a9-0000-09a004a9` → 15675 bytes gzip XML
  9. 创建 `dataset\{raw,mjai,index}` 与 `tools\tenhou_data`
  10. 创建 `tools\.venv` 并安装 jansou → 子模块列表确认
  11. **卡点**：pip 写用户 site-packages 被沙箱拦截 → 用户中断
- **产出**：端点与格式的完整实测结论

### 会话 4：转为文档交付
- **用户诉求**：把执行任务清单写成 .md 文档，用户自己在已有环境操作
- **产出**：`tools/tenhou_data/TASKLIST.md` —— 含任务总览表、目录布局、环境准备、数据源结论、过滤规则、6 步可运行代码、时间/容量预算、7 条坑位备忘

### 会话 5：清理
- **用户诉求**：清理损坏的 jansou 目录 + 删除此前所有残留文件
- **执行**：
  - 删除 `C:\Users\黄超平\AppData\Roaming\Python\Python313\site-packages\jansou\`（需沙箱外执行，已验证不存在）
  - 删除 `C:\Maj\Mortal\tools\.venv\`
  - 删除 `probe_scraw.py`、`probe_scraw2.py`
  - 删除 `C:\Maj\Mortal\tools_probe_list.html`
  - 排查 `accesstest_*`、ensurepip 临时 pyc → 已被系统自行清理，无需处理
  - 同步更新 `TASKLIST.md`（移除 venv/探测脚本引用，状态表第 4 项改为待执行）
- **保留**：`TASKLIST.md`（用户要求的交付物）、`dataset\{raw,mjai,index}` 空骨架

### 会话 6：整理记忆文档
- **用户诉求**：把全部历史上下文整理成一份记忆文档，保存到 `c:\Maj\Mortal\memory`
- **产出**：本文档

### 会话 7：数据管线落地与容量精算
- **用户诉求**：按 TASKLIST 继续执行数据下载（默认 Python 环境、保持目录整洁）；随后询问全量磁盘占用并要精确值
- **执行**：
  1. 默认环境装入 jansou 0.14.0（非沙箱 pip 直接成功），确认 API 为 `parse_mjlog`/`dump_mjai`/`check_paifu`
  2. 新建 `tools/tenhou_data/tenhou_pipeline.py`：fetch-lists / fetch-archives / build-index / stats / download / convert / check / all，全步骤断点续传
  3. 冒烟全链路通过（28/28 和牌回放复核）；修正两坑：requests 自动解压、jansou `uradora_markers`→`ura_markers`
  4. 窗口全量任务：4,017 局清单，下载 115 局后**按用户要求终止**（可续传）
  5. 实测发现：**年度 zip 全部 404**；滚动日归档可用 → 新增 fetch-archives，抓全 2026 日文件（266 个/8.4MB）
  6. 精算：**2026 YTD 去重 129,475 局**；n=118 单局 raw 7.59KB + mjai 4.84KB → 全量 **≈1.53 GiB**
  7. 全量落盘启动（129,485 局）：先按 2s/局 启动（ETA 78h），用户询问速度后**经确认调至 0.5s/局**（实测 1.46 局/秒，ETA ~24.5h，随后自动转换+校验）
  8. 运行中意外中断：13:18 进程被外部强制结束（无报错/无睡眠/无 Defender 拦截/无重启，原因未确认）；`verify-raw` 校验 3,615 个文件全部完好后，改由 **Windows 计划任务 `Mortal-Tenhou-Download`** 承载（意外退出自动重启、日志追加、电池模式不中断）
- **产出**：可续传管线脚本、2026 全年索引、容量精算（旧 250~350GB 估算作废）、全量下载+转换任务运行中（计划任务托管）

### 会话 8：限流中断恢复
- **用户诉求**：下载任务又中断，检查情况并继续
- **执行**：
  1. 排查：10-02 02:30 被天凤**限流掐断**（连续 ConnectionError/SSLError 触发保护性停止）；上午实测网络已恢复
  2. 断点成果：已下载 34,250 局全部转换成功（0 失败），抽样校验结构 30/30、回放计分 280/280 全过
  3. 脚本增强：`--chunk/--chunk-sleep` 分块例行冷却 + 连续失败自动冷却重试（`--cooldown/--max-cooldowns`）
  4. 以 `--delay 1.0 --chunk 5000 --chunk-sleep 900` 重启计划任务，续传剩余 ~9.6 万局（ETA ~36h）
- **产出**：限流自愈机制；下载 2026-10-02 09:01 起恢复运行（34,311/129,897）

### 会话 9：HF 数据集合并验证
- **用户诉求**：停止下载 → 检查已下载部分 → 用 HF 的 2009/2026 包验证合并流程
- **执行**：
  1. 停止计划任务 `Mortal-Tenhou-Download`（State: Ready，不会自动续跑）并终止派生 python 进程
  2. 检查：raw 35,690 个 0 损坏（verify-raw）；补齐转换 1,440 局（34,250→35,690）
     下载进度实况：**1 月 16,179 局=完整、2 月 14,352 局=完整、3 月 5,041/14,294（35%）**，另有早期 9/10 月抽查残留 118 局
  3. HF 移动验证：从 hf-mirror 下载 2009.tar（47.6MB）+ 2026.tar（83.3MB）→ 新增 `merge-hf` 子命令
     （解压 zst→结构自查→gzip 落盘→按 log_id 去重；新增明细记 `dataset/index/hf_merge_manifest.csv`）
  4. 结果：2009 包 6,897 局全新增（200902~200912），定向抽验 **100/100 结构+回环、874/874 和牌计分全通过**；
     **2026 包 12,188 局 100% 命中自有数据被去重跳过**（反证自有 1 月数据完整）
  5. `check` 子命令增加 `--glob` 参数（定向校验子集）；mjai 树现共 **42,587 局**（190.6MB）
- **产出**：merge-hf 命令 + 合并清单 + 流程验证通过；下载任务保持停止状态（用户未要求续跑）

### 会话 10：HF 年度包全量下载（C→D 中转策略）
- **用户诉求**：C 盘空间有限、D 盘多但长写入易卡死 → 每包先下载到 C 盘、复制到 D 盘、删 C 原文件
- **执行**：新增 `fetch-hf` 子命令（逐包串行：C 临时 → 复制 D → 校验字节 → 删 C；D 已有跳过、C 残留复用）；
  2009/2026 验证流程后，2010–2025（16 包）由计划任务 `Mortal-Tenhou-FetchHF` 完成（0 失败）
- **核查（10-02 10:46）**：18/18 包字节与 HF 清单**逐一比对全部一致**；日志 16 下载/16 完成/0 失败；
  抽验 2015 包随机 3 局结构 OK；C 临时区已清空；C free 12.5 GiB / D free 36 GiB（D:\tenhou_hf 17.3 GiB）
- **关键实测**：合并体积比 **0.576×**（zstd→gzip，2009: 47.6→27.4 MiB）→ 全量合并 ≈ **9.7 GiB**
- **决策（用户选）**：**全量归 D** —— 2009–2025 合并到 `D:\tenhou_mjai\houou4\`；C 盘仅保留 2026 及后续下载
- **执行**：新增 `merge-hf-batch`（每年一批 + 完成标记 `.merged/<年>` + 批间停顿 120s，可续传）；
  2009 冒烟验证通过（6,897 局，~194 局/s）；全量经计划任务 `Mortal-Tenhou-MergeHF` 运行中（2010 起，ETA ~4h）
- **完成后收尾**：删除 C 盘重复的 2009 目录；抽样校验 D 树；训练 config globs → `D:/tenhou_mjai/**` + C 盘 2026

### 会话 11：D 盘空间异常排查 → exFAT 簇浪费根因 → 方案回退
- **用户诉求**：检查文件整理情况、删干净多余文件
- **排查过程**：合并任务运行中 D 盘空间异常下跌（暂停后止跌），一度怀疑卷影副本/隐藏进程；
  逐项排除（回收站 0、无隐藏大文件、`vssadmin` 需管理员、Event 153 为 USB I/O 重试）后，
  **发现 D 盘是 exFAT 分区**（`Get-Volume`：exFAT；`Get-CimInstance Win32_Volume -Filter "DriveLetter='D:'"`：**BlockSize = 262144 = 簇 256 KiB**）
- **决定性实测**：往 D 盘写 1 字节文件 → 占用 **262,144 字节**；`D:\tenhou_mjai` 13 万文件（逻辑 521 MiB）实占 ~33 GiB
- **根因**：每个 ~4 KiB 的 `.json.gz` 至少占 1 个 256 KiB 簇 → 膨胀 64×；
  全量历史 2009–2025（~243 万局）散文件落 D 需 **~620 GiB** → 原「全量归 D」方案不可行（此前"D 盘掉空间"谜团同时解开）
- **用户决策**：只保留 HF 年度原包（`.tar.zst` 大文件不受影响），**不转换、暂不改 Mortal 训练加载器**
- **执行**：删除 `D:\tenhou_mjai`（~13 万文件，释放 ~33 GiB，D free 4.3→36 GiB）；
  删除 6 个 chkdsk 残留 `D:\FOUND.00x`；注销计划任务 `Mortal-Tenhou-MergeHF`；
  补回此前误删的 2009/2010 原包（`fetch-hf --years 2009,2010`）→ `D:\tenhou_hf` 恢复为完整 18 包 16.86 GiB

### 会话 12：训练方案制定（网络调研 + P100 环境适配）
- **用户诉求**：基于 MEMORY.md 分析训练如何开展（数据位置、网络顺序、配方），允许网络调研；后确认实际训练环境为 Tesla P100-16GB（torch-2.6.0+cu118），要求形成本地文档并估算各阶段耗时
- **执行**：
  1. 网络调研：Homura 2026-04 从零训练笔记（三阶段流程/bootstrap 两法/多轮 baseline 迭代）、DeepWiki 训练管线页、Mahjax 论文（2026-05，确认"凤凰卓离线预训练+RL 微调"是标准路径）
  2. 本机实测：GPU=RTX 3050 Ti 4GB / 5900HX / 15.4GB；确认 dataloader 默认 `oracle=False`（开源 train.py 不用天眼特征）
  3. 产出 [TRAINING_PLAN.md](../tools/training/TRAINING_PLAN.md)：P100 完整方案（环境搭建/数据上传/config.toml 全文/bootstrap.py/M0-M5 执行序/监控判停/坑位/时间估算/里程碑）
  4. 关键配方结论：P100 用标准 192×40 + batch512；cu118 正确勿换 cu126；enable_amp 先开后兜底；num_workers Linux 可 2-4；GRP 先行不可调换；bootstrap.py 解决 test_play 冷启动
  5. 时间估算：首模型（M0→M3）≈1–2 天墙钟；迭代至平台 +1–4 天；在线增强（可选）+1–5 天；依据 5.5 万步/epoch、P100 估 4–8 it/s（启动 1 小时后须实测校准）

### 会话 13：训练服务器环境手册
- **用户诉求**：编写从 conda 开始的完整环境配置手册
- **服务器实测信息（用户提供，2026-10-06）**：Ubuntu 18.04.2 容器（glibc 2.27，EOL）、内核 6.8、2×E5-2690 v4（56 线程）、**4× Tesla P100-16GB + 1× Quadro RTX 8000（48GB, sm_75）**、驱动 580.173.02、无 nvcc（不需要）；miniconda3 在 /mnt/data8
- **产出 [ENV_SETUP_P100.md](../tools/training/ENV_SETUP_P100.md)**：基础检查→conda 环境（**必须 defaults 源，禁 conda-forge**，glibc 2.27 兼容）→torch 2.6.0+cu118（自带运行时，arch 需含 sm_60/sm_75）→rustup（glibc 报错回退 1.81.0）→编译 libriichi（.so 与 bins 两条命令）→全链路验证→GPU 分工
- **GPU 分工结论**：GRP（float64）只能用 P100（FP64 4.7T vs RTX8000 0.5T）；主训练首选 RTX 8000（Turing Tensor Core，enable_amp 收益大，48GB）；多卡用 CUDA_VISIBLE_DEVICES 切换，脚本保持 device='cuda:0'

### 会话 14：服务器全量数据落地 + GRP/主模型第一轮训练完成（2026-10-06 ~ 10-09，远程侧）
- **执行**：
  1. 环境按 `ENV_SETUP_P100.md` 落地：conda `mortal`(py3.12.11) + torch 2.6.0+cu118 + rust 1.99.0 → `libriichi.so`(10-06 15:57) 与 `validate_logs`/`stat` 编译通过，基础常量校验（obs 1012×34 / ACTION_SPACE 46）通过
  2. 代码改用本机上传的 `Mortal.rar`（覆盖 `git clone` 的官方版）；随后在服务器对 `train.py`/`train_grp.py`/`player.py` 打训练控制补丁，并新建 `config.offline.toml`（`MORTAL_CFG` 指定）
  3. 数据：`dataset/tenhou_hf/*.tar`（18 个 HF 年度包，17 GB）经 `merge-hf` 在服务器展开为 `.json.gz`，加上本机 2026 采集（130,407 局），再用新增 `prune --apply` 剔除非「四鳳南喰赤」文件 → `dataset/mjai/houou4/` = **200902~202610 共 213 月 / 2,630,640 局 / 16 GB**；抽样月份文件名规则码 100% `00a9`
  4. `train_grp.py`：P100，30,000 步（≈1.89 epoch）21 分钟，val loss 2.5429→2.4550 **仍在下降**，取 `grp_best.pth`(=30k 步) 为 `grp.pth`
  5. `train.py`：RTX 8000 + AMP，200,000 步（预算到点自动退出），墙钟 22.9 h + 4 次评测各 21 min；dqn_loss 0.467→0.392
  6. 10-09 复盘：随机 baseline 导致 `test_play` 指标饱和（avg_rank≈1.00、1 位率 99.9%）；补做 snapshot 配对对抗与未见月离线指标 → **50k/150k/200k 强度持平**；数据管道实测为瓶颈（3.12 it/s vs GPU 8.88 it/s）
- **产出**：详见 [§16](#16-服务器实跑现状2026-10-0710-09-远程同步) 与 [TRAINING_MANUAL.md](../tools/training/TRAINING_MANUAL.md) §14

### 会话 15：数据侧 P0 落地——修复 2026 牌谱 names + file_index manifest（2026-10-12，本机）
- **背景**：`memory/NEXT_STEPS_ANALYSIS_20261012.md` §2.1/§2.2 定的两条 P0。起因是白名单训练对自建 2026 的 13 万局**完全不产出样本**：jansou 的 `dump_mjai` 不写 `start_game.names`，而 Rust `StartGame.names` 是 `#[serde(default)]` → 缺字段静默变 4 个空串 → `GameplayLoader` 按座位过滤时永远匹配不上（**不报错**）。
- **坑 1（信息其实没丢）**：原始 `dataset/raw/mjlog/<log_id>.xml.gz` 的 `<UN>` 保留 `n0~n3`（另有 dan/rate/sx）→ 可恢复。
- **坑 2（差点白干，见 §14-30）**：`<UN>` 的 `n0~n3` 是 **URL 百分号编码**的玩家名（值长 45~63、纯 ASCII）。首轮实现直接把原值写进 `start_game.names`：`validate_logs` 全绿、索引照建、训练不报错，但白名单命中率 **0%**（结构合法、语义全废）。靠"抽样比对白名单命中率"才发现；`unquote` 后立刻正常。
- **产出工具**：
  - `tools/tenhou_data/repair_names.py`——从原始 mjlog 恢复 names（自动 unquote + `%XX` 残留告警），写入**独立补丁树** `dataset/mjai/houou4_names/`（不动旧树，R1~R4 可复现）；除首行外逐字节保留、gzip `mtime=0` 确定性、可断点续跑；报告写 `dataset/index/repair_names_report.json`。
  - `tools/tenhou_data/verify_names.py`——验收工具：names 覆盖率 / 席位 / 白名单与榜单池**命中率** / `%XX` 残留。**凡牌谱字段修复都必须跑它**（`validate_logs` 查不出语义错误）。
  - `tools/training/test_file_index_manifest.{py,sh}`——manifest 单测（16 项）+ 端到端回归（E1~E5 五条路径）。
- **实测结果**：
  - 修复 130,407 文件 / 1.92 min / 607.6 MiB / `suspicious_names=0`；`target/release/validate_logs` 全量 **16 s 无告警**。
  - `GameplayLoader` 同一份牌谱：旧树 **0 席位** → 补丁树 4 席位 584 样本；v2 白名单模式命中 3/4 家、725 样本。
  - `verify_names`：2 万抽样（v2 白名单）座位命中 **20.66%** / 文件命中 **60.60%**；**全量 130,407 文件复核（v3 白名单）**：带 names **100%**、521,628 席位中白名单命中 **130,415 席（25.00%）** / **89,317 文件（68.49%）**、榜单池命中 **99.36%**（即 2026 席位玩家几乎全部可在 HF 2022-2025 玩家池中找到）；旧树对照全 **0%**。
  - 强度榜 v3 重建（HF 2022-2025 + 补丁树 202601-202608，813,145 文件，3.1 min，**失败 0**）：玩家池 4,883→**5,356**、合格（≥400 局且 avg_pt≥2）472→**501**；top400 门槛 +2.45→**+2.57** 且名单 **69 人进出**；top200 门槛 **+3.98**。
  - 新池覆盖度：**有 names 的文件 813,145（100%）**；top400 席位 735,249→**789,454（+7.4%）** / 样本≈134M / **262k 步每 epoch**；top200 席位 334,907→**344,068** / **114k 步**；top100 席位 119,452 / 40k 步。
- **file_index manifest（train.py）**：索引由 `{file_list}` 升级为 `{file_list, manifest}`（`FILE_INDEX_VERSION=2`）。manifest 分两类：**参与比对**（`index_version / globs / player_names_files(sha256) / excludes_files / data_version`）与**仅记录**（`player_names_count / enable_augmentation / augmented_first / libriichi_version / built_at / n_files / names_coverage`）。加载时不一致 → warning + 自动重建；旧格式与损坏索引同样重建；新增可选配置 `dataset.data_version`（改它即可让全部旧索引失效）。
- **新增产物**：`dataset/mjai/houou4_names/`（补丁树 607.6 MiB）、`dataset/index/player_ratings_2022_2026H1_v3.csv`、`top_players_{400,200}_v3.txt`、`dataset/index/repair_names_report.json`。
- **未做（留给下一轮拍板）**：把 `data_version` 写进新配置；R5 用哪套名单（建议 v3 系列——榜单已含 2026；top200/top400 对照设计见分析文档 §2.3）。

### 会话 16：模型侧 P0 落地——奖励口径可切换 + gamma 语义验证 + 判定执行参数固化（2026-10-12，本机）
- **背景**：`memory/NEXT_STEPS_ANALYSIS_20261012.md` §3.1 / §3.2 / §3.5 三条模型侧 P0。
- **① 奖励口径（§3.1）**：训练目标一直是 `env.pts=[6,4,2,0]`（等间距顺位点），而最终判定用天凤点 `[90,45,0,-135]`（第 4 名重罚）→ 模型被训成"2、3 名差不多"，与 ladder 口径不一致。
  - `dataloader.py` 新增 `reward_mode`（`'rank_pt'` 默认＝历史行为；`'delta_points'`＝每局**真实分数差**，复用此前从未被调用的 `RewardCalculator.calc_delta_points`）与 `reward_scale`（归一化，避免换口径后 CQL 权重失衡），并新增 `always_include_kan_select` 直通 GameplayLoader（默认 True＝历史行为）。
  - `train.py` 从 `env.reward_mode`/`env.reward_scale` 读取、启动即打印口径、并新增**续训目标守卫**：`state['config']['env']` 与当前 `env` 的 `gamma/pts/reward_mode/reward_scale` 不一致时显式告警（否则会把两种目标混进同一条损失曲线；旧 ckpt 缺键按默认处理，不误报）。
  - `holdout_metrics.py` 同步同一口径并在表头打印，避免"用新配置比旧 ckpt"的静默不可比。
- **② gamma 语义（§3.2）**：新增 `tools/training/inspect_gamma_semantics.py`，在真实牌谱上验三条不变量（200 文件 / 54,526 动作样本）：`apply_gamma == (label <= 37)`、`dones` 只在局边界、**折扣视界 = 当前局内**。实测结论：只有打牌(0-36) 与立直(37) 参与折扣，**杠(42，样本中 171 次) 不折扣**（与 `add_entry` 旧注释 "only discard and kan" 不符）；每局最后一条动作 `steps_to_done=0` 不被折扣，下一局从 0 重启 → **`gamma<1` 只打折当前局内较早的动作，不含任何后续局信息，"跨局信用分配"的说法不成立**（`gameplay.rs` 注释已改写为实测口径，见 §14-31）。
- **③ 判定执行参数固化（§3.5）**：`ladder.py` 此前硬编码 `enable_amp=False / guard=True`，配置里的 `[1v3.*]` 对它无效，判定结果也不带执行口径 → 事后无法回答"这局是用什么跑的"。
  - 现优先级 **CLI > 配置 `[1v3.ladder]` > 历史默认**；默认值刻意等于 R2~R4 判定口径（保证旧结论可比）；若 `[1v3.challenger/champion]` 的值与生效值不同会**打印告警**（不再静默）；双侧强制同参数。
  - 输出新增 `exec_cfg` / `challenger_ckpt` / `champion_ckpt` 行（含 **sha256 前 16 位、steps、resnet、mtime**）并写 `<log_dir>/run_meta.json`（含完整结果）；`raw rankings` / `seed_start` 行格式未变，`agg_ladder.py` 仍兼容。
- **验证**：`tools/training/test_reward_modes.py` 全过——`Grp::final_scores` 独立重算 **12/12 座位逐位一致**、`calc_delta_points` 与独立重算最大误差 **7.3e-12**、两模式样本数一致（2304）、`delta_points` 取值全为 100 的整数倍、`reward_scale` 精确等比（多重集比较，注意 `load_files` 每轮 `random.shuffle`）、非法 mode 被拦、续训守卫 6 例；`test_file_index_manifest.sh` 复跑 U+E1~**E6**（新增 E6：续训改 pts → 恰好 1 条告警）全过；`ladder.py` 2-set 冒烟通过（两处 `[1v3.*].enable_amp=True` 不一致告警如预期）。
- **未做**：§3.3（Q 校准 / IQL，P1）、§3.4（AMP 重开评估，P1）；**真正的跨局 credit assignment** 需数据侧显式提供"下一局状态 + 未来回报"并引入 target network，属独立框架改动，不在本次范围。

---

### 会话 17：奖励对齐消融已准备并启动（2026-10-12，本机）
- **先修正的工程问题**：`test_file_index_manifest.sh` 改为 `set -euo pipefail` 并不再吞掉单测失败；`holdout_metrics.py` 增加固定 `torch.Generator`、worker seed 同步与 `holdout_sample_digest`/`holdout_target_digest`（真实 `202609`、2 批冒烟输出正常）；训练新增可选 `control.seed`；`train.py` 增加 `control.reset_optimizer` 一次性令牌，奖励目标切换时只继承模型权重并跳过旧 optimizer/scheduler/scaler。
- **回归结果**：`test_reward_modes.py`（奖励公式/接线/续训守卫/reset）全过；`inspect_gamma_semantics.py --n-files 200` 全过（800 gameplays / 133,065 动作）；`test_file_index_manifest.sh` U+E1~E6 全过；`py_compile`、`bash -n`、`git diff --check`、lint diagnostics 均通过。
- **新增工具**：`build_file_index.py`（预构建带 manifest 的共享索引）、`prepare_reward_ablation.py`（生成三组隔离 TOML/父 checkpoint/`experiment.json`）、`run_reward_ablation.sh`（三 GPU 并行训练入口）、`finish_reward_ablation.sh`（训练完成后自动 holdout + 双向 ladder + block bootstrap 收尾）。
- **实验定义**（严格固定 R4 v2 数据池，不混入 v3/修复树）：父代 `checkpoints/round4/mortal.pth`，645,000 步，sha256=`47417e67ac2eeb61dd6d10cd708491bbd3d2b24c929686af55bdef5694fd4590`；原 R4 globs（HF 2022-2025 + 原始 `202601-202608`）、`top_players_400_v2`、`enable_augmentation=true`、`augmented_first=true`；共享索引 `indexes/reward_ablation_r4_v2.pth` 有 492,810 个文件，`with_names=699,715`、`without_names=113,430`。
- 三组均 `max_train_steps=756,000`（新增 111,000）、`seed=20261012`、fp32、相同模型/GRP/CQL/辅助损失；仅奖励不同：A=`rank_pt` + `[6,4,2,0]`；B=`rank_pt` + `[2,1,0,-3]`；C=`delta_points` + `reward_scale=1e-3`。三组首次启动均确认跳过父代 optimizer/scheduler/scaler。
- 三组训练与收尾链均已完成（终局判定见会话 18）；元数据在 `mortal/reward_ablation/experiment.json`，分组日志为 `train.log`/`holdout_202609.log`/`ladder_summary.log`，共享日志为 `launcher.log`/`finish.log`。
- **中途诊断（13:15 UTC，A=652k / B=651k / C=653k，各仅训 6~8k/111k 步）**：
  - ✅ 评测基础设施验证通过：三组 `holdout_sample_digest` **完全一致**（`0e3d3cce0b7bad83`），`holdout_target_digest` **各不相同**（A=`7e60085d…` / B=`de072bcf…` / C=`c953c84c…`——奖励口径确实在管线中生效）。
  - 早期指标（12 批 × 512，202609）：A top1=0.8182 / R2=−0.54；B top1=0.8201 / R2=−0.84（⚠ B 的目标尺度不同，dqn_loss/R2 **不可跨组比**）；**C top1=0.8019 / R2=+0.06 —— R2 首次转正**（此前所有模型均为负），但 top1 略降。方向符合"从模仿人类转向预测分数变化"的预期；agari_top1 三组均 90/90（guard 完好）。
  - ⚠ 运行中的进程用的是**修改前**的 `train.py`（Python 模块启动时已加载），因此 B/C 日志里 env 告警仍显示旧文案；实际上 optimizer reset **已生效**（"跳过 checkpoint optimizer/scheduler"告警在位）。修改后的代码只影响后续新进程（如 finish 脚本调用的 ladder/holdout）。
- **后续**：三组各到 756k 后，训练完成后由 `finish_reward_ablation.sh` 自动执行：同一 holdout month/seed，核对相同 `holdout_sample_digest`（记录各组 `holdout_target_digest`）；再用全新种子块 100000+/110000+ 双向各 5×500 sets；只按 block bootstrap CI 判读，未完成前不把 holdout 视为强度结论。

### 会话 18：奖励消融终局判定（10-13 05:46 UTC）——C 显著变弱，B 方向性正向但未达显著

**训练完成情况**：A/C 于 00:02/00:04 到 756k，B 于 04:17 到 756k，三组退出码均 0、0 非有限张量；`finish_reward_ablation.sh` 自动完成三组 holdout + 双向 ladder（新种子 100000+/110000+，各 5×500 sets）+ block bootstrap，05:46:45 收尾。

**强度判定（vs 父代 round4/645k，双向同牌山 ≥5000 sets）**：

| 组 | 变量 | δ (pt/半庄) | 解析 95%CI | bootstrap 95%CI | 判读 |
|---|---|---|---|---|---|
| A | `rank_pt` `[6,4,2,0]`（现状，控制组） | **−0.04** | [−1.20, +1.13] | [−1.03, +1.02] | 无结论（与父代等价） |
| B | `rank_pt` `[2,1,0,-3]`（对齐天凤） | **+0.18** | [−0.98, +1.34] | [−0.85, +1.19] | 无结论，**方向正向** |
| C | `delta_points` scale=1e-3（真实分数差） | **−5.61** | [−6.79, −4.44] | [−6.67, −4.55] | **显著变弱**（t=−9.38） |

C 组分块明细：A 方向（C 当挑战者）5 块 avg_pt 全负（−3.62~−8.26），B 方向（父代当挑战者）5 块全正（+2.95~+5.83）——反对称形态是"确实变弱"的典型特征（对比 R2 有效时的 A +2.9/B −2.9）。

**终态 holdout（202609，24 批×512，三组 `sample_digest` 一致）**：A top1=0.8179/R2=−0.65；B top1=0.8158/R2=−1.01；C top1=0.7960/R2=**+0.01**。⚠ C 的 R2 虽首次转正，但**没有转化为对局强度**——"Q 能拟合分数差"与"会打牌"是两回事，这是本轮最重要的认知。

**结论与影响**：
1. **C（真实分数差奖励）被决定性否掉**（−5.6 pt）。推测主因：① 分数差方差约为顺位奖励的 40 倍（var_t 22.2 vs 0.54），`cql.min_q_weight=5` 的相对权重被稀释；② 每局分数差被对手行为主导（噪声大），而顺位聚合整局信息更稳；③ scale=1e-3 归一化后目标范围 ±16 仍为 rank_pt（±3.2）的 5 倍。
2. **A 确认平台期第三次**（同奖励 +111k 步 + 重置优化器 = 无变化），与 R3/R4 一起封死"同分布增量训练"路线。
3. **B 是唯一正向信号但未达显著**（+0.18，需 44 倍 sets 才能到 2σ）。理论依据充分（训练目标与判定口径一致）且 CI 下界 −0.85 证明无实害——**建议后续轮次采用 B 口径作为新基线**。
4. 分析文档 §6 推测 2（奖励对齐是否改善 ladder）标记为"方向正向未证实"；推测 4（GRP 换成真实分数差监督）**已被本轮间接否掉**。

**下一轮候选**（按优先级）：① **R5 = B 奖励 + top200_v3 名单**（B 组本身即" B 奖励+top400"的现成控制组，天然单变量对比；top200 席位 344,068 ≈114k 步/遍）；② 若 R5 仍无增益 → IQL/Bellman 带 next state 目标；③ 扩窗 2019+（需 GRP 重训 + 数据审计）。

### 会话 19：第 5 轮启动（10-13 06:01 UTC）——B 口径 + top200_v3 收紧名单

- **设计**：唯一实质变量 = 数据选择从 `top_players_400_v2` 收紧到 `top_players_200_v3`（v3 榜单已含 names 修复后的 2026 玩家）。其余全部继承 B 组：`pts=[2,1,0,-3]`、`rank_pt`、augmentation=true、fp32、同 GRP/CQL/aux、optimizer 不重置（同奖励口径，与 R2→R3→R4 一致）。
- **数据管线**（names 修复后的第一次实战）：globs = HF 2022-2025 原树 + 修复补丁树 `houou4_names/20260[1-8]`；file_index `indexes/main_r5_top200_v3.pth`：命中 **293,416 文件**、`with_names=813,145 / without_names=0`（旧管线下 2026 的 113,430 文件全部无 names、0 席位——修复前后对比的直接证据）。
- **父代与预算**：`reward_ablation/B_rank_pt_aligned/mortal.pth`（756k）；`max_train_steps=870,000`（+114k ≈ top200 增强样本一遍）。启动实锤：`loaded: 04:17:53` → `total steps: 756,000` → `reward: rank_pt(pts=[2.0,1.0,0.0,-3.0])` → 无 optimizer reset 告警、无 env 告警 → GPU0 100%（2.67 batch/s）。
- **判定**：R5(870k) vs B 组(756k)，双向各 5×500 sets，**新种子块 120000+/130000+**（不与奖励消融的 100000+/110000+ 重复）；`run_round5.sh` 链式执行（预检→暖启→训练→判定→block bootstrap 汇总）。
- **选卡修正（06:11 UTC，重要）**：R5 首次启动时误选 GPU 0（P100），实测 fp32 **2.48 batch/s**；换回 GPU 2（RTX 8000）实测 **4.36 batch/s**，**快 1.76×**，训练时长从 12.8 h 降到 7.3 h。已停掉 P100 那次（仅跑 11 分钟、无存盘、无污染）并清理残留后重启。**教训：主训练轮次一律用 `CUDA_VISIBLE_DEVICES=2`（RTX 8000）**——P100 是 Pascal 架构（FP32 ~9.3 TFLOPS）、RTX 8000 是 Turing（~13 TFLOPS），本模型 192×40 fp32 直接吃 GPU 算力（§16.21 已记 R4 的 4.24 it/s 实测）。选卡不能随手挑，估算时间时也要用对卡的基线，否则“每轮约 7 h”的既有预期会被无声拉长一倍以上。
- **预计**：训练 ≈7.3 h（约 13:40 UTC 完成）+ 判定 ≈45 min。判进标准不变：bootstrap 95%CI 下界 > 0。
- **若 R5 仍无增益**：按分析文档顺序转 IQL/Bellman（需 next state + target network）或扩窗 2019+（需 GRP 重训 + 数据审计）。

### 会话 20：第 5 轮终局判定（10-13 14:28 UTC）——top200 收紧名单【无结论、方向偏负】

**训练完成**：13:59:31 `training budget reached` → **退出码 0**；终态 `round5/mortal.pth` = **870,000 步、0 个非有限张量**（fp32 全程无 NaN，三重防护未触发）。链路 `ALL DONE 14:28:50`。

**判定结果（R5/870k top200_v3 vs B 组/756k top400_v2，新种子块 120000+/130000+，双向各 5×500 sets）**：

| 方法 | δ (pt/半庄) | SE | 95%CI | 判读 |
|---|---|---|---|---|
| 解析（按局独立） | **−0.605** | 0.594 | [−1.77, +0.56] | \|t\|=1.02 < 2 → 无结论 |
| **block bootstrap（按 set）** | **−0.605** | **0.508** | **[−1.61, +0.36]** | CI 跨 0 → 无结论；ρ≈+0.185 |

A 侧（R5 当挑战者）avg_pt **+0.121**（SE 0.851）/ B 侧（B 组当挑战者）avg_pt **+1.332**（SE 0.828）→ δ=(A−B)/2=−0.605。分块：A = +1.82/+2.18/−1.10/−1.78/−0.52（2 正 3 负），B = +1.30/+1.60/+3.10/−0.43/+1.08（4 正 1 负）。**没有出现奖励消融 C 组那种「A 全负 / B 全正」的干净反对称形态**，因此只能说"top200 收紧没换来增益"，**不能判定为确证变弱**（这是所有"无结论"轮里点估计最偏负的一次，但仍在噪声带内）。

test_play 固定种子（⚠ 不入强度结论，仅趋势）：800k avg_rank 2.464/pt 3.06、850k 2.464/pt 3.96，`best.pth` 更新到 850k。

**结论与影响**：
1. **「换信息源」里的收紧名单候选（top400→top200）失败关闭**：top200 席位 344,068 仅 top400 的 ~45%、增强一遍只 114k 步，"少而精"没兑现成更强监督；R2 的 +2.86 无法拆开归因到名单轴，单独在名单上再收紧自然没有可复现的增益。
2. **连同 R3/R4/消融 A，"同分布层面挪数据"（重复 epoch / 视角增强 / 换名单子集 / 换优化器）已连续 5 次否掉** → 数据 reshuffle 这条路基本走到头。
3. R5 相对 B 组 −0.605、B 相对 round4/645k 仅 +0.18 → **R5 ≈ round4 持平甚至略降，没有推进强度前沿**；现役最强仍是 `checkpoints/round4/mortal.pth`（645k），`round5/mortal.pth` 不替代任何基线。
4. **下一轮候选（按分析文档 §4/§6）**：② **IQL / advantage-weighted regression / 带 next-state + target network 的离线 Bellman**——跳出 MC 单样本回归天花板的唯一未试方向，应优先；③ 扩窗 2019+ + 重训 GRP + 按年份审计（成本最高、放最后）。**"收紧名单"已从候选删除。**

### 会话 21：hp_ablation 便宜前置实验（超参消融）终局判定（10-14 07:54 UTC）——cql/aux 均无增益

**背景**：R5 失败后按用户选 (B)，先做 §3.3 的两项廉价前置检查（cql 权重小扫 + aux 消融），再决定是否上 IQL/Bellman 大改。

**设计**（严格复刻 B 做控制、只动一个旋钮）：三条分支均从 `round4/645k` 暖启、`reset_optimizer`、top400_v2/[2,1,0,-3]/augmentation、+111k→756k；`cql_min1`（cql=1）、`cql_min10`（cql=10）、`aux_next0`（aux=0）；**控制组 = B 本身**（cql5/aux0.2，已 756k）。判定 vs round4/645k，双向各 5×500 sets、新种子块 **140000/150000**。

**过程插曲**：GPU1（cql_min10）持续负载下**热降频至 607 MHz**（同 79°C 的 GPU0 为 1240 MHz），速率从 4.3→1.2 batch/s，单方面拖住整条链。处置：kill 原链（aux 已完成、cql_min1 741k/cql_min10 694k 的 mortal.pth 均完好）→ **原地续训**到凉卡（cql_min1→GPU4 P100、cql_min10→GPU2 RTX，**不重跑 prepare**、reset 令牌已消费不会二次触发）→ 判定尾巴重跑。

**判定结果（vs round4/645k，均双向 ≥5000 sets）**：

| 分支 | 变量 | A侧 | B侧 | δ | 解析 95%CI | **bootstrap 95%CI** | 判读 |
|---|---|---|---|---|---|---|---|
| aux_next0 | aux 0.2→0 | +0.716 | −0.464 | **+0.59** | [−0.58,+1.76] | **[−0.42,+1.61]** | 无结论（t=0.99） |
| cql_min1 | cql 5→1 | −0.167 | −0.270 | **+0.05** | [−1.12,+1.22] | **[−0.97,+1.06]** | 无结论 |
| cql_min10 | cql 5→10 | −0.059 | +0.072 | **−0.07** | [−1.23,+1.10] | **[−1.09,+0.99]** | 无结论 |
| B（控制） | cql5/aux0.2 | — | — | +0.18 | 种子 100000/110000 | — | 无结论 |

holdout（仅诊断，sample_digest 一致）：三条 top1 0.812–0.817（持平）、R² 全负；var_q/dqn_loss 随 cql 权重升高（0.29→0.44→0.51）但不转化为强度。

**结论与影响**：
1. **超参不是瓶颈**：cql 1/5/10 对强度单调无效（全跨 0）；aux 0 vs 0.2 也无差异（aux=0 的 +0.59 与 B 统计等价）。→ **排除了"只是超参没扫对"这一替代解释**。
2. **直接支持转 ② IQL/Bellman**：瓶颈确在目标形式（MC 单样本回归、无 next-state/target net），应按计划大改，不再在超参上纠缠。
3. **现役最强仍为 `round4/mortal.pth`（645k）**；三条 hp_ablation 分支 + B 与父代均统计等价，无任何分支值得采纳为新基线。
4. 产物：`mortal/hp_ablation/<tag>/{train.log,holdout_202609.log,ladder_summary.log}`；链日志 `mortal/logs/hp_resume_chain.log`（含 ALL DONE）；脚本 `tools/training/prepare_hp_ablation.py`、`mortal/run_hp_ablation.sh`、`mortal/resume_and_judge_hp.sh`。

### 会话 22：IQL/Bellman Stage A（跨局信用分配）终局判定（10-14 17:16 UTC）——无增益、方向偏负

**背景与设计**：hp_ablation（会话 21）排除超参解释后，按「IQL/Bellman 两阶段改造」计划先做便宜的 Stage A。唯一实质变量 `env.credit_mode`：`'kyoku'`（现状：样本只回归**本局**的 GRP 期望点差，未来所有局的收益被完全丢弃）→ `'game'`（跨局折扣回报 `returns[k]=Σ_{j≥k}γ^(j−k)·r_j`）。γ=1 时有闭式 **`pts[final_rank] − E_grp[pts|第 k 局开始]`**：GRP 中途期望自动成为**控制变量**（方差远低于纯 MC），逐局项 telescope 求和使中间项精确抵消（无双重计入），终局处 GRP 收敛为 one-hot 故**无需终端特判**。数据/名单(top400_v2)/增强/B 口径 pts/CQL=5/aux=0.2/GRP_h128 全部与 B 组锁死。

**代码改动**（三处 + 验证）：`dataloader.py`（`credit_mode`/`gamma` 参数 + populate_buffer 沿局轴折扣后缀和）、`train.py`（`credit_mode` 进 `_ENV_REWARD_KEYS` 守卫；game 模式不再乘 γ^steps_to_done）、`holdout_metrics.py`（口径透传）；新增 `tools/training/test_credit_modes.py` 五层验证 **ALL PASS**（闭式恒等 max_err 8.9e-16、GRP 终局 one-hot、telescope 终点、kyoku 回归等价多重集逐位一致、γ=0.5 折扣确实生效、守卫 3 例）。冒烟链 `run_iql_a_smoke.sh` **7/7 全过**（守卫告警恰 1 条、索引 manifest 复用同批文件、无 non-finite、快照、test_play、吞吐 4.6 it/s）。

**执行**：`run_iql_a.sh`（tmux `iql_a`）10-14 09:32 UTC 从 **B/756k** 暖启（`reset_optimizer='iql-a-v1'`）→ **867,000** 步，16:47 exit=0、无 non-finite、预算达成；用时 **7.25 h**，稳态 **4.52 batch/s**（与 B 的 4.28 持平 → **后缀和零 GPU 成本**）。判定 vs B/756k，新种子块 **160000/170000**，双向各 5×500 sets（10,000 半庄/方向）。

**判定结果**：

| 方向 | 分块 avg_pt | 合计 |
|---|---|---|
| 波A（新当挑战者） | +0.02 / −2.54 / −1.91 / −1.37 / −1.37 | **−1.435**（SE 0.847） |
| 波B（旧当挑战者） | +2.79 / +0.99 / −0.32 / −0.79 / +0.43 | **+0.621**（SE 0.837） |

- **δ = −1.028 pt/半庄**；解析 SE 0.595、t=−1.73、95%CI[−2.20, +0.14]（跨 0）
- **block bootstrap：SE 0.517、95%CI = [−2.06, −0.02]、ρ_A +0.189 / ρ_B +0.191**（bseed 20261014，5000 次）→ **CI 不跨 0，工具判读「变弱」**
- 两向同号，**未出现** C 组那种「A 全负 / B 全正」反对称伪影 → 是真实方向偏负，不是测量噪声
- ⚠ 诚实标注：上界仅 −0.02，属**边际显著**；最稳妥表述是「**无增益、方向偏负**」，不宜称“确证变弱”

**holdout 202609（仅诊断）**：top1 **0.8118**（与 B 同量级）、dqn_loss 1.6623 vs const 1.2103、**R²=−0.3734**、**var_q/var_t = 0.9158/2.4208**、agari_top1 171/171、logsumexp_q +0.433；digest `sample=d83666d339dac9d9` / `target=654aa926354a76a9`（target 随口径变化属预期，不可与 B 口径横比）。诊断含义：新目标幅度更大而 Q 只拟合到约 1/2.6 的方差 → **欠拟合目标幅度**，与「更像人 ≠ 更能赢」的既有结论同向。

**结论与影响**：
1. **Stage A 不采纳**：`checkpoints/iql_a/`（867k）归档保留，不替代任何基线；现役最强仍 `round4/mortal.pth`(645k)，训练基线口径仍 B/756k。
2. **「把整场回报回传给早期动作」这条 MC / 控制变量路线被否** —— 跨局信用分配本身不是瓶颈的解；瓶颈不在“看不到长期结果”，而在**没有价值传播与政策改进机制**（Q 无法给未访问动作定价，argmax 仍停在行为克隆上）。
3. **按计划直接转 Stage B（真 IQL）**：数据侧 `next_obs` + terminal 标志，模型侧 `VHead` + EMA target 网络，损失 `L_Q(r+γV(s')) + L_V(τ-expectile) + L_AWR(优势加权 CE)`、CQL 关闭；预算 +111k，吞吐预计降至 ~1.7-2 it/s（每步 3 次 encoder 前向）≈ 18-22 h；判定种子块预 **180000/190000**，父代 B/756k。Stage B 仍无增益 → ③ 扩窗 2019+ + 重训 GRP + 按年份审计。
4. 产物：`mortal/iql_a/{train.log,holdout_202609.log,ladder_summary.log,ladder/A_16*.log,ladder/B_17*.log}`、链日志 `mortal/logs/iql_a_chain.log`（含 ALL DONE）、配置 `mortal/config.iql_a{,_smoke}.toml`、脚本 `mortal/run_iql_a{,_smoke}.sh`、索引 `mortal/indexes/iql_a_r4_v2.pth`；ckpt `checkpoints/iql_a/`=mortal.pth(867k)+best.pth(800k)+step_800k/850k。

### 会话 23：IQL Stage B 首轮判定（10-15 15:06 UTC）——结果无效，根因是 r_t 双重计入（已修复）

**执行**：03:09 UTC 启动（B/756k 暖启，`objective='iql'`，γ=0.997 / τ=0.7 / β=3.0 / ema_decay=0.995）→ 14:38 训完 **867k**（用 11.5 h，实测 **2.69 batch/s**，exit=0、无 non-finite）→ 15:06 ALL DONE。

**判定（vs B/756k，新种子 180000/190000，双向各 5×500 sets）**：
- **δ = −31.04 pt/半庄**（解析 SE 0.569、t=−54.5、95%CI[−32.15,−29.92]；bootstrap SE 0.546、CI[−32.13,−29.98]）—— 巨大且极显著的负值
- 分块：A 侧（新当挑战者）−33.66/−28.96/−35.89/−31.93/−30.73，四位率 777~836/2000；B 侧（父代当挑战者）五块全 +28.5~+31.0 → 新模型压倒性处于下风
- holdout 202609：top1 **0.6917**（父代/Stage A 约 0.811）、**logsumexp_q = −176.1**（Stage A 为 +0.43）、agari_top1 170/171

**关键判断：这不是“目标形式无效”，而是实现缺陷。**证据链：
1. `tools/training/diag_iql_reward_double_count.py`：3 文件 = 12 game / 144 局 / 2304 样本，**iql 条目的非零 r_t = 2304 条**，而按设计（每局只计一次）应为 **141 条**；放大倍数恰等于**每局平均决策点数 16.00×**。
2. 计划明写“r_t 仅局边界非零”，但首版 `dataloader.py` 把 `kyoku_rewards[at_kyoku[i]]` 发给了该局**每一个**样本 → Bellman 沿链把每局奖励重复累加 ≈16 次 → Q 目标尺度爆到 −176 量级 → AWR 优势与 argmax 全被污染 → 强度断崖。
3. 劣化在 **2k 步冒烟就已可见**（avg_rank 2.789 / avg_pt −19.34 vs 父代 2.466 / +3.06），但当时冒烟门禁只查结构性项目（无 non-finite、到预算、test_play 有输出），**未把“模型没被打坏”设为门槛** → 按计划开了正式轮，白烧 11.5 h GPU + 30 min 判定。

**修复与加固（同日）**：
1. `mortal/dataloader.py`：`r_t_i = kyoku_rewards[at_kyoku[i]] if dones[i] else 0.0`（仅局边界非零），并把该缺陷写进注释警示。
2. `tools/training/test_iql_data.py` C 段改为**断言设计意图**：非零 r_t 个数 == 局边界样本数（141）、零值个数 == 非边界数（2163）、非零多重集 == 局边界处 calc_delta_pt、r_t 总和 == 逐局奖励总和、max|r_t| ≤ max|pts|。修复后 **八层 ALL PASS**。
3. `mortal/run_iql_b_smoke.sh` 新增两条**行为门禁**：⑥ test_play avg_rank ≤ 2.60；⑦ holdout top1 ≥ 0.78 且 |logsumexp_q| ≤ 10；FAIL 即禁止启动正式轮。
4. 命名空间隔离：v1 产物（`checkpoints/iql_b/`(867k)、`checkpoints/iql_b_smoke/`）含受损权重与 v_net/EMA 键，**封存不续用**；验证走 `*_smoke_v2`，重跑正式轮用全新命名空间 + 新种子块 **200000/210000**。

**结论与影响**：
1. Stage B 首轮的 −31.04 **不作为 IQL 路线的判决**（无效实验）；重跑 v2 才是该路线的首次有效试验。
2. 基线不变：现役最强 `checkpoints/round4/mortal.pth`（645k）、训练基线口径 B/756k。
3. 方法论两条沉淀进 §14-35/§14-36（测试必须验设计意图而非实现现状；冒烟必含数值型行为门禁）。

**后续（同日 15:40）：新增的行为门禁当场抓出第二处实现偏差**——`r_t` 修正后的 v2 冒烟仍 **SMOKE FAIL**：末态 test_play avg_rank **2.840** / avg_pt **−26.54**（门禁 ≤2.60）、holdout top1 **0.6975**（门禁 ≥0.78）、logsumexp_q **−33.59**（门禁 |·|≤10）——比 v1 的 −176.1 小了 5 倍（印证 r_t 修正有效）但仍异常。定位到第二处偏离 IQL 原文：**L_V 的 expectile 应对「数据里实际执行的动作」的 Q̄(s,a) 回归（Kostrikov et al. 2021 式 9），而我误对全部合法动作（~30 个从未被监督的 Q 值）取上分位** → V 被无约束的 Q 外推拖走，再经 V(s') 反馈进 Q 目标形成自激漂移。已改（`q_bar = q_bar_all[range(bs), actions]`）；命名空间递增至 `iql_b_smoke_v3`（v1/v2 均含受损权重不可续用，就地封存取证）。**如果没有这两条行为门禁，第二个缺陷又会烧掉一整轮 11.5 h。**

**v3 验证结果（10-15 16:03 UTC，SMOKE ALL PASS 10/10）**：两处修正后模型不再被打坏——test_play avg_rank **2.527** / avg_pt **−1.58**（门禁 ≤2.60）、holdout top1 **0.8134**（门禁 ≥0.78，已高于父代 B 的 0.8112 与 Stage A 的 0.8118）、logsumexp_q **+1.343**（门禁 |·|≤10）、无 non-finite、吞吐 2.93 batch/s。三代冒烟对比（avg_rank / logsumexp_q）：v1 2.789 / −176.1 → v2 2.840 / −33.6 → **v3 2.527 / +1.34**——两个缺陷各贡献一个数量级的尺度异常，修正链与诊断推断完全自洽。**真 IQL 现在才处于可被有效检验的状态，待正式轮。**

### 会话 24：IQL Stage B 正式轮 v2 启动（10-15 16:24 UTC）——两处缺陷修复后首次有效检验

**启动前代码审查结论（dataloader.py / train.py / model.py / holdout_metrics.py 全链路复核）**：
1. `dataloader.py`：`r_t_i = kyoku_rewards[at_kyoku[i]] if dones[i] else 0.0`（仅局边界非零）✅；`next_obs = obs[i+1]`（末条自身占位 + `terminal=True`）✅；7 元组字段序 `[obs, actions, masks, next_obs, terminal, r_t, ranks]` ✅；iql + `credit_mode='game'` 组合有 AssertionError 守卫（防后缀和与 bootstrap 双重计入）✅。
2. `train.py`：L_Q = `0.5·MSE(Q(s,a), r_t + γ·V(s'))`，`torch.where(terminal, r_t, r_t+γV(s'))` 正确处理终端 ✅；L_V 仅对数据动作 `q_bar_all[range(bs), actions]` 做 τ-expectile ✅；L_AWR = 优势加权 CE（w=exp((Q−V)/β)，均值归一）✅；EMA target = 参数 polyak + BN buffer 每步跟随 online ✅；state 保存 `v_net/ema_mortal/ema_dqn` 三键，NaN 守卫覆盖全部六模块 ✅。
3. `model.py` VHead（1024→256 Mish→1）与 DQN v_head 同风格 ✅；`holdout_metrics.py` iql 口径透传、q_target=None 只出 top1/agari/lse ✅。

**审查中发现的第三处问题（测试缺陷，非实现缺陷）**：`tools/training/test_iql_data.py` C 段幅度断言 `max|r_t| ≤ max|pts|` **数学上错误**——`r_t = E[pts|k+1] − E[pts|k]` 是两个凸组合之差，正确上界是 **pts 极差 max−min**（B 口径 = 2−(−3) = 5.0）而非 max|pts|（=3.0）。10-15 当天用默认 `config.round4.toml`（pts=[6,4,2,0]，max|pts| 恰等于极差 6.0）跑所以未暴露；本轮按 `config.iql_b_smoke.toml` 复跑时 FAIL（max|r_t|=3.4573 ≤ 5.0，144 局单局奖励独立重算确认是合法单局值，多重集/计数/总和三项双重计入判据全过）。已修断言为极差上界并加注释说明"多重集/计数/总和才是双重计入的确定性判据"→ 八层 ALL PASS。（沉淀 §14-38）

**v2 正式轮配置**（全新命名空间，v1 全部产物封存不续用）：
- `config.iql_b_v2.toml` + `run_iql_b_v2.sh`：objective='iql'，γ=0.997 / τ=0.7 / β=3.0 / ema=0.995；B/756k 暖启（reset 令牌 `iql-b-v2`）→ 867k；数据/名单/增强/pts/GRP 与 B 锁死；索引复用 `indexes/iql_b_r4_v2.pth`（492,810 文件，manifest 校验通过）。
- 判定：vs B/756k，**新种子块 A=200000-/B=210000-**（180000/190000 已被 v1 无效轮消费），双向各 5×500 sets + block bootstrap；CI 下界 > 0 判进。
- 启动即核：暖启"缺 v_net/EMA"告警恰 1 条、env 日志 gamma=0.997/tau=0.7/beta=3.0/ema=0.995、索引复用同批文件、吞吐 2.93~2.95 batch/s（≈冒烟 v3 的 2.93）、三损失初值 q=0.128/v=0.093/awr=0.411、next_rank=0.742（历史区间 0.74~0.79）、lr=1e-4 恒定、无 non-finite；GPU2 97% 利用率。预计 ≈10.5 h 训练 + ≈1 h 判定。

**产物**：`mortal/iql_b_v2/`（train/holdout/ladder）、链日志 `mortal/logs/iql_b_v2_chain.log`、ckpt `checkpoints/iql_b_v2/`。

**中途观察（10-16 01:47 UTC，训练 850k 时）——价值尺度慢漂移**：q_target/q_predicted 直方图中位数 757k +0.6 → 788k +12.3 → 819k +24.3 → 850k **+46.9**（约每 3 万步翻倍的纯指数增长，无饱和迹象，预计 867k 终点 ~+70），超理论回报界 [−5,+5]（r_k=E[pts\|k+1]−E[pts\|k] 沿局 telescope 至 pts[final]−E[pts\|start]；757k 暖启时 p99 仅 +3.4 证实界正确）约 10 倍；q_loss 0.128→0.877、v_loss 0.093→0.229 单调上升与此同源（Q 与 V 同步膨胀、相对拟合仍紧）。性质判断：① 非前两处缺陷 signature（那俩 2k 步冒烟内即 logsumexp −176/−33.6、avg_rank 2.8）；v3 冒烟 2k 步内尺度健康（+0.6→+1.0）→ **2k 冒烟窗口抓不到 3 万步时间尺度的漂移**（冒烟方法论局限，v3 设计须补"多检查点尺度门禁"）；② 代码逐条对照 IQL 原文 Algorithm 1（L_V=EMA Q̄@数据动作、L_Q=r+γV_online(s')、AWR=exp((Q−V)/β)）无第三处定义错误 → 判断为 γ=0.997 近 1 + 奖励仅局边界非零（~6% 样本有锚）+ 无 replay 在线迭代下 bootstrap 反馈环增益略大于 1 的固有慢漂移；③ AWR 权重与 argmax 对 Q/V 共同尺度不变故事始门上无损（800k avg_rank 2.4645≈父代 2.466、avg_pt +3.58），850k 透出轻度下滑（2.5045/+0.47，~1.5σ 内未破 ≤2.60）。决定让自动链跑完拿完整数据点（预计 05:00 UTC 前后出 ladder 判定）；若判无增益，v3 的反漂移候选：bootstrap 改用 target V、降 γ、奖励稠密化——均属新设计变量，须单独冒烟+判定。

### 会话 25：IQL v2 终局 + 价值漂移根因分析与 4 臂探针（10-16）

**v2 终局数据（10-16 03:22 UTC 训完 867k，ladder 判定 ~05:00-05:30 出）**：
- holdout 202609：top1 **0.8069**（父代 B 0.8112、Stage A 0.8118、v3 冒烟 0.8134——轻降 ~0.5%）、agari_top1 168/171、**logsumexp_q=+45.06**（父代 ~+0.4，证实末端价值尺度膨胀）。
- 训练期 test_play：800k avg_rank 2.4645/avg_pt +3.58 → 850k 2.5045/+0.47（500 局噪声 ~1.5σ 内，未破 ≤2.60 门禁）。
- 完整尺度轨迹（q_target p50）：757k +0.58 → 760k +1.63 → 770k +5.55 → 780k +9.44 → … → 850k +46.8 → 861k +60.8；log2 在 780k 后为直线（每 10k 步 +0.31 → **倍增 ~32k 步，纯指数**）。

**根因分析（三层证据链）**：
1. **反馈环增益略大于 1**。线性化：δV ← Q̄(s,a_data)（expectile 回归），Q̄ ← r + γV(s')。纯均值通道每跳增益 γ=0.997<1 本应收缩，但 expectile 回归**不是无偏的**：L_2^τ 在 V 低于目标时以权重 τ=0.7 上拉、高于时仅以 1−τ=0.3 下拉。增长 regime 中价值从有锚的局末向前传播，中游状态的 V 持续落后于其下游目标（追而不及）→ 残差持续为负 → 每步被 τ=0.7 不对称放大 → 有效增益 γ·(1+bias)>1，bias≈2.3e-5/步（=倍增 32k 步）——小到躲过 2k 冒烟，10 万步累成 60 倍。
2. **EMA 滞后通道维持指数形态**。Q̄ 滞后在线 Q ~200 步；增长 regime 中 dQ̄/dt=λ·Q̄ → 目标噪声 ∝ 当前尺度 → expectile 取其 0.7 分位 → 偏差 ∝ 尺度 → 对数线性（纯指数）而非线性增长。实测 log2(p50) 直线正是此 signature。
3. **奖励稀疏抽掉每步锚**。94% 样本 r=0、目标=0.997·V(s') 纯 bootstrap 链；收缩全靠 γ<1 的 0.3% margin，expectile 不对称恰好从这点 margin 偷出增益。另注：AdamW weight_decay=0.1 本已提供 ~1e-5/步的收缩（约为增速的 43%），漂移是在对抗 wd 的情况下发生的。

**影响范围（参数级诊断，`tools/training/diag_iql_value_scale.py`，862k ckpt vs 父代）——决定性发现**：
- v4 DQN 的 q = v(φ)+a(φ,·)−mean(a)，**argmax_a q = argmax_a a**（v 与 a_mean 均为每状态常数）→ 膨胀完全 confined 在价值通道：dqn v_head p50 −4.86→**+27.7**、VHead p50 **+31.8**；而优势通道 **a_head 统计不变**（跨动作 std 1.74→1.60、p99 3.77→3.95）。
- 因此「行为门禁全过」的真实原因是**部署 argmax 结构上只依赖 a_head**，而非「价值学习没问题」；价值函数确实是废的（holdout logsumexp_q +45 证实）。
- 二阶污染：AWR 权重 w=exp((q−V)/β) 中 q−V 含 (v−v_ema) 滞后项 ≈ λ·T_ema·scale ≈ 0.28（vs 优势尺度 1~3）→ ~10% 权重失真；L_Q 对 a_head 的梯度 ∝ TD 残差绝对值（0.51→1.60，3 倍）→ 策略通道的 L_Q/AWR 平衡被间接改变——这是 top1 轻降与 avg_pt 波动的最可能解释。

**v3 实验设计（已启动 4 臂探针验证修复，`mortal/run_iql_probe.sh`）**：全部从 B/756k 暖启、同数据（indexes/iql_b_r4_v2）/30k 步/同 batch，只改一个旋钮：
| 臂 | 配置 | 预期 |
|---|---|---|
| A 对照 | v2 原配置（γ=0.997/τ=0.7/在线 V） | 复现漂移（30k 步 p50 ~+15） |
| B v-ema | bootstrap 改用 V 的 Polyak EMA target（新增 `iql_v_target_decay=0.99`，train.py 已实现） | 稳定（待验） |
| C γ=0.99 | 每决策步折扣收紧，直接压反馈环增益（margin 300× 于实测 bias） | 稳定（待验） |
| D τ=0.5 | expectile 对称化（机制检查：若稳定则证实不对称放大是根源） | 稳定（待验） |
判据：q_target p50 全程 ≤ 8（train.py 现每 1000 步打 `iql scale:` 行）。**v3 正式轮 = v2 + 探针选出的最小修复**（优先 B：保留 τ=0.7 与 γ=0.997，只加 target-V，标准手法且不改变 IQL 语义），其余全锁死，新种子块 220000/230000；若 v2 ladder 判进则 v2 先采纳为基线、v3 作为改进轮。

**配套改动**：train.py 新增 `iql_v_target_decay`（V 的 EMA target 用于 L_Q bootstrap，含 ckpt 存取/守卫/续训兼容）+ 每存盘点 `iql scale:` 分位数日志；`tools/training/diag_iql_value_scale.py`（模块级尺度诊断）；`tools/training/prepare_iql_probe.py` + `mortal/config.iql_probe_{a,b,c,d}.toml` + `mortal/run_iql_probe.sh`；.gitignore 补 `mortal/iql_probe*/`。

**收尾预案**：v3 判进 → 稳定版 IQL 采纳；v3 仍无增益 → IQL 路线关闭（这次是"价值稳定条件下的干净否证"），转扩窗 2019+ + GRP 重训 + 按年份审计。

### 会话 26：v2 终局判定（显著变弱）+ 探针机制判决（棘轮效应）与 v3 定稿（10-16）

**v2 终局（10-16 04:08 UTC ALL DONE，双向各 5×500 sets，种子块 200000/210000）**：
- A 侧（新当挑战者）avg_pt −3.146（SE 0.849）；B 侧（父代当挑战者）+3.686（SE 0.826）
- **δ(新−旧) = −3.416 pt/半庄，SE 0.592，t = −5.77，解析 CI[−4.58,−2.25]；bootstrap CI[−4.46,−2.36] → 显著变弱**
- 即：带价值漂移的 IQL v2 比 B/756k 弱 3.4 pt——漂移的二阶污染（a_head 跨动作 std 1.74→1.60 变钝、top1 0.811→0.807）实打实伤了强度。该结果**不作为 IQL 路线判决**（目标实现不稳定），与 v1 同一纪律。

**探针机制判决（5 臂各 30k 步，同暖启/同数据，10-16 03:25 启动；GPU2 于 05:05 补臂 E）**——8k 步时 q_target p50：

| 臂 | 旋钮 | 8k 步 p50 | 判决 |
|---|---|---|---|
| A 对照 | v2 原配置 | +2.99 | 复现漂移（与 v2 同速） |
| B | bootstrap 用 V 的 EMA target（0.99） | +2.19 | 仅慢 ~25%，**不止漂** |
| C | γ=0.99 | +2.59 | **几乎无效** |
| D | τ=0.5 | **+0.20** | **基本平坦（唯一稳定）** |
| E | τ=0.7 + L_Q 目标钳位 ±8 | 待出 | 验证"撞墙止漂"是否伤价值学习 |

**根因定论——expectile 不对称"棘轮效应"**（推翻会话 25 初步假设"反馈环增益>1"）：若根因是增益，γ=0.99（较实测增速有 300× margin）本该稳定、V-EMA lag 该显著阻尼——两者都几乎无效，唯独去掉不对称（τ=0.5）后稳定。正确机制：增长 regime 中价值从有锚的局末向前传播，V 持续落后于上升的目标 → L_V 残差持续为正 → τ=0.7 对正残差强上拉（权重 0.7）、对负残差仅弱下拉（0.3）→ 每轮净向上蠕动，**蠕动速率 ∝ 残差幅度 ∝ 目标变化率 ∝ 当前尺度** → 纯指数（倍增 ~32k 步）。γ 与 target lag 只影响传播速度、不影响棘轮注入，故无效。推论：τ=0.6 也只把棘轮减半（ doubling ~64k 步，111k 步仍发散），**稳定必须 τ≈0.5**；IQL 的"乐观"本质上是对 Q 分布的尺度相对分位，与本环境的稀疏奖励+在线迭代不相容。

**v3 定稿**：= v2 + **τ=0.5**（expectile 对称化 = fitted-Q + AWR；探针唯一验证的稳定修复），其余全锁死；种子块 220000/230000。科学定位：稳定条件下的价值学习首次干净检验——判进则"离线价值学习"路线成立；仍无增益则价值学习路线关闭，转扩窗 2019+。备选 clip8（臂 E）仅当 E 证明钳位不伤价值学习时启用。配套：train.py 已加 `iql_value_clip`（臂 E）+ `iql_v_target_decay` + 每存盘点 `iql scale:` 日志；`prepare_iql_b_v3.py --fix tau05` 一键生成配置；`run_iql_b_v3.sh` 就绪。

**方法论沉淀**（§14-39 已补）：漂移时间尺度 ~3 万步 → 冒烟必须覆盖现象时间尺度（2k 步结构性门禁对该漂移完全无感）；"超参扫描式"单旋钮探针（30k 步 × 5 臂 ≈ 4.5h）是定位此类慢漂的最便宜手段，远胜反复烧 11.5h 正式轮。

**v3 冒烟与启动（10-16 06:06-06:26 UTC）**：冒烟 **10/10 全过**且质量信号历轮最佳——test_play avg_rank **2.445**/avg_pt **+3.87**（优于父代 B 2.466/+3.06）、holdout top1 **0.8158**（历轮最高；父代 0.8112、Stage A 0.8118、v2 0.8069）、logsumexp_q **+0.531**（v2 为 +45）、价值尺度 max p50 **0.025**（零漂移）。06:26:30 正式轮启动（GPU2，B/756k 暖启 → 867k，种子块 220000/230000，预计 ~17:00 UTC 训完 + ~1h 判定）。臂 d（τ=0.5 探针，保留跑满 30k）17k 步 p50 +0.381，线性蠕变 ~+0.023/1k 步 → 预计 111k 终点 ~+2.7 在界内，长稳确认中。**注意**：2k 冒烟与 17k 探针的正信号只证明"没被打坏"，强度结论仍只认 v3 的 ladder（CI 下界 > 0 判进）。

### 读牌模型（会话 27/27b，10-16~10-17）：手牌暗牌预测，监督学习路线

- **定位与边界**：用户给出「路线 1 监督学习」方案（从可见历史预测三家暗牌后验与听牌分布，输出校准概率 + 硬约束层，最终注入 Mortal 提升防守）。实现于**全新目录 `tools/read_model/`**，对 `mortal/` 与 `libriichi/` **零改动**（Rust 都没重编）。**尚未注入 Mortal**——强度结论只认未来的路径 C 注入 + ladder 在线验收，离线指标仅作诊断。
- **数据管线（dataset.py）**：4 个 `PlayerState` 同时回放同一事件流——观察者 obs 用主管线同一 `encode_obs(4)`，三家标签取自其余状态的 `tehai`/`waits` getter（mjai 自足，全知标签无需 oracle）。公开信息（tiles_seen/河/立直/摸牌数）Python 侧独立台账推算（`tiles_seen_A = 自家tehai + c_total`，c_total=四家打牌+鸣牌消耗+暗杠+加杠+宝牌指示牌，**逐局重置**）。快照时点 = 观察者决策点（`cans.can_act()`）+ 局内公告；课程阶段过滤器前置到 obs 编码前（阶段 1 省 ~85% 编码）。对方案的 5 处修正（均有实测依据，README §2）：输入直接用 v4 obs（1012 通道）而非新建 160 通道；**听牌标签 = 有效荣和集** `waits & ~river & expressible`（`waits` 是结构性集含自家河牌，河里的待牌按规则永远无法荣和；`expressible = unseen>0 | own_hand>0`——unseen==0 时自家手里的牌仍能点炮）；河牌对所有目标家硬掩码（方案原文仅立直家硬）；dora 公告 100% 紧跟杠（实测）故跳过该快照；快照取决策点子集而非每个公开事件。
- **关键实测发现（已固化进测试）**：加杠对静止手牌数净贡献 0 → `hand = 13-3*(吃+碰+大明杠+暗杠)`；`[u8;34]` getter 返回 bytes、`[bool;34]` 返回 list，getter 都是属性不是方法；obs 第 835 行 = tiles_seen/4、第 870 行 = at_kan_select；与 `GameplayLoader` obs **逐位一致**（loader 全部 obs 除 kan_select 条目被 100% 复现）；万↔筒增广必须保持 aka（5mr→5pr）否则 aka 通道与 akas_seen 全错。
- **v1（model.py，1.13M 参数）**：单帧 1D ResNet（1012→96，12 块）→ H1 逐列计数分配头（unseen 掩码，5 类）+ H2 听牌头（cat([g, E[h_j]])→35 类）+ H3 辅助头。硬约束层（constraints.py）：H2 加性对数偏置 + IPF 投影（列守恒 Σe≤unseen、总手牌数 Σe==hand_size）。终局（15000 步 / 768 万样本）：alloc CE 0.6705 vs 二项基线 0.7271（信息增益 0.057 nats/列）、ECE 0.0038、top-1 命中 0.566；wait hit@1 0.139、AUC 0.628、立直家 hit@1 0.124。过拟合探针证明任务可学（固定 512 样本 wait CE→0.691≈标签熵）。mean/attention 池化 A/B 无差异。
- **v2（model_v2.py + structure.py，1.36M 参数）**：三流融合——① 空间流同 v1；② **时序流**（v1 最大盲区：牌河在 obs 的通道维上，沿列卷积无法按时间推理，而手切是读牌最强信号）：每目标家公开动作序列（43 维 token × ≤26：牌+摸切标志+动作类别+巡目）→ 2 层 pre-norm Transformer（d=64,4头，**手写注意力**——nn.TransformerEncoderLayer 会融合成 `aten::_transformer_encoder_layer_fwd`，ONNX opset 17 无法导出）+ attention pooling → e_T；③ **结构估计流**：枚举全部标准分解（顺子/刻子/对子/两面/嵌张/边张/孤立）按与 Rust `calc_all` 同口径公式（`2L-2M-min(T+K,L-M)-雀首修正`，L=4-副露数）算向听，标签=全部最小向听分解的边际直方图 + 形制 + 向听 8 桶，作为 soft 信念瓶颈注入 H1/H2/H3。**枚举器交叉验证：20.3 万个真实静止态手牌与 Rust 向听数零不一致**（坑：公式必须按 L 参数化；雀首修正条件 `K>=1 且 T+K>L-M`；立直家 shanten 被 Rust 冻结在宣言值）。
- **v1 vs v2 对决（10-17，同固定评估集 20000 样本 + bootstrap 200 次 CI）**：**v2 三项显著增益**——alloc top-1 0.5658→0.5830（+3% 相对，CI [+.013,+.021]）、立直家待牌 hit@1 0.1239→0.1447（+17% 相对，CI [+.009,+.034]）、**听牌检测 AUC 0.6276→0.6855（+0.058，CI [+.053,+.062]，最大单项增益**——"该不该防"比"防哪张"更基础）。wait hit@1/@5 与 argmin 型防守模拟差异均在 CI 跨 0 的噪声区。**方法论教训**：60 文件小子集曾给出"v2 全面更差"的**相反**结论——wait 指标在 ~2500 样本上方差极大，必须固定大评估集 + bootstrap CI 才能判决。**结构头本身仍弱**：向听预测 25.7% **低于** 28.4% 多数类基线（低估 shanten=2/4、过预测 shanten=0），形制 91.9%=多数类（平凡）；学到的主要是顺子数/孤立数这类从 tiles_seen 可直接推断的量——增益主要来自时序流，显式向听估计尚未学会。
- **训练配置**：813,145 训练文件（HF 2022-2025 + names 补丁树 202601-202608）/ 202609 验证（固定评估集），15000 步 × 512，三阶段课程 1:0.27/2:0.30/3:0.43，`--augment random`，AdamW+cosine，fp32。吞吐 ~2.5k 样本/s（8 workers），瓶颈是 `encode_obs` 的 SP 计算器（~2.2ms/样本，对读牌无用）——优化路径：Rust 侧 `with_sp_tables` 开关。产物：`mortal/read_model_v1/`、`mortal/read_model_v2/`（best.pth + onnx + eval_report.json + train_log.jsonl），ONNX 均经 onnxruntime 校验。
- **工程资产**：回归测试 23 项 ALL PASS（含枚举器交叉验证、时序 token 独立重算、与 GameplayLoader obs 逐位一致、136 牌守恒、tiles_seen 通道比对、增广等变、学习门禁）；`compare_v1_v2.py`（对决）+ `compare_bootstrap.py`（CI 判决）；`run_v1.sh`/`run_v2.sh`。
- **下一步**：① 结构头专项（向听损失加权 / 分解-向听一致性约束，把 25.7% 拉过基线）；② 更长训练（50k+ 步）验证时序流增益是否扩大；③ 方案 §9 路径 C 注入 Mortal + ladder 在线验收（唯一强度标准）。集成前需补：规则法基线（方案 §8.1 第 2 条）、立直振听自动检测。
- **v2s 结构头专项（会话 27c，10-17，已训练+对决判决）**：解剖 v2 发现「向听没学会」的机制根因是 **`StructureHead.shanten_head` 零监督**——`L_struct` 只有 counts+form，该头输出不进 `s_belief` 瓶颈（38 维不含向听），25.7% 是无监督头的随机水平（回归测试 G3e 把「关闭新损失后该头梯度恒 0」存档为门禁）。v2s 六改：① 向听分布注入瓶颈（38→46 维，`build_model_v2s`，+9.2k 参数）；② 结构头向听直接 CE（0.5）；③ **分解-向听一致性约束**（`structure.soft_shanten_from_counts`：预测分解边际分布取期望代入同口径向听公式，min 取次梯度、雀首修正 sigmoid 软 AND，全程可微；头侧 CE 让向听头对齐分解暗含向听 [target detach]，分解侧 MSE 让软向听拟合真标签 [梯度回流分解头]；**形制门控**——公式只覆盖 normal 路径，实测 normal 上 round(soft)==硬标签 100% [mean|d|=0.041，9495 目标，测试 G1]，七对/国士的 counts 标签仍是 normal 分解边际故排除）；④ 向听-听牌一致性（shanten==0 ⟺ tenpai，BCE 0.1）；⑤ aux 向听 CE ×3（0.05→0.15）；⑥ 向听/形制类均衡（批内频率逆平方根）。顺带修 v1/v2 潜在 bug：`l_wait` 在整批无心听牌目标时 `wait_logits.sum()*0` 遇硬掩码 -inf 得 NaN（大 batch 不触发，测试 G3a 门禁）。**对决（同固定评估集 20000 + bootstrap 200）**：向听头 0.2574→**0.5263**（CI [+.264,+.273]，原定「拉过 28.4% 基线」超额完成；逐桶召回 {0.49,0.44,0.09,0.23,0.01}→{0.72,0.51,0.61,0.41,0.40}，仅最稀有 sh5 归零；七对 0.212→0.494、国士 0→0.512）、**听牌检测 AUC 0.6855→0.7392（+0.054，CI [+.051,+.057]，tenpai-tie 的功劳）**、头↔分解一致率 0.28→0.93；**代价**：wait hit@1 −0.014（CI 显著）、立直 hit@1 −0.017、alloc top-1 −0.007。机制诊断：听牌目标上两模型行为几乎一致（P(未听) 0.0039 vs 0.0041），差别是 H2 锐度略降（top1 概率 0.082→0.076）——容量/优化预算成本（瓶颈 +8 维 + 4 项新损失分薄 15k 预算），非信号方向错误，解药是更长训练（50k+，下一步 ② 现有双重动机），各损失均有 CLI 旋钮可消融。附记：aux 头向听 v2 时代已有 0.540（×3 后 0.553），v2s 结构头 0.547——两头收敛到互洽。产物 `mortal/read_model_v2s/`（best@13000 + onnx 校验 + eval_report + train_log）；回归测试扩到八层 39 项 ALL PASS；旧 v1/v2 checkpoint 加载不受影响（已验）。
- **下一步计划（会话 27c 定稿，2026-10-17）**——三阶段，全部机时在空闲 P100（勿抢 GPU2/RTX8000）：

  **阶段 A：模型定型（2 个训练轮次，机时 ~0.5 天）**

  - **A1 瓶颈归因消融（先跑，15k 步，~2-4h；入口 `tools/read_model/run_v2x.sh`，tmux `read_v2x`，out-dir `mortal/read_model_v2x`）**：`--model v2`（38 维瓶颈不加宽）+ v2s 全部新损失（旗标显式传参防默认漂移），同数据/同步数(20261016)/同预算/同课程/同增广。三单元单因素对照（20000 固定评估集 + bootstrap 200 CI，与 v2/v2s 判决完全同口径）：v2(38维/旧损, wait 0.1464) — **v2x(38维/新损, W 待测)** — v2s(46维/新损, wait 0.1323)。**预注册判据**（正向效应分解：瓶颈效应 B = W−0.1323，损失效应 L = 0.1464−W，B+L=0.0141；加性假设，2×2 缺第四格 46维+旧损=无兴趣格；80/20 阈值——wait_hit1 的 95% CI 半宽 ≈0.007，60/40 band 窄于噪声不可分辨）：① **B ≥ 0.0113（W ≥ 0.1436，损失代价 ≤20%）→ 瓶颈加宽是主因** → 回退 38 维保留全部损失，v2x 直接晋级 A2 50k；② **L ≥ 0.0113（W ≤ 0.1351，损失代价 ≥80%）→ 新损失是主因** → 转 `--tenpai-tie-weight 0` 消融（它直接塑形 H2 消费的向听信念）；③ 混合区（0.1351<W<0.1436）→ 比增益定去留：v2x 的 AUC/向听 ≥ v2s−0.005 则按简约原则用 38 维，否则保留 46 维（瓶颈在贡献增益而非仅代价）。**次级预注册判读**：(a) v2x 结构头向听准确率 vs v2s 0.526——接近则瓶颈对「头学会」非必要（作用仅在馈通 H1/H2），差距大则瓶颈消费是头的自蒸馏训练信号；(b) wait 代价机制诊断（听牌目标 P(未听)/top1 概率/熵，6000 子集同 v2s 诊断法）——复现「行为一致仅锐度降」则确认容量/优化类成本；(c) 护栏：v2x 须保留损失收益（AUC/向听/一致率不回零），否则说明瓶颈宽是增益的载荷而非仅代价。
  - **A1 结果（10-17 跑完并判决；v2x best@12000，7678s）——三单元归因（20000 固定评估集 + bootstrap 200，全部带 CI）**：

    | 指标 | v2 (38/旧损) | v2x (38/新损) | v2s (46/新损) | 损失效应 (v2→v2x) | 瓶颈效应 (v2x→v2s) |
    |---|---|---|---|---|---|
    | wait_hit1 | 0.1464 | **0.1386** | 0.1323 | −0.0078 CI[−.016,−.001]（55%） | −0.0064 CI 跨 0（45%） |
    | **tenpai_auc** | 0.6855 | **0.6670** | **0.7392** | **−0.0185 CI[−.021,−.016]（负！）** | **+0.0722 CI[+.070,+.075]** |
    | shanten_acc | 0.2574 | 0.5257 | 0.5263 | +0.2683（≈全部） | +0.0007 ns |
    | alloc_top1 | 0.5830 | 0.5719 | 0.5762 | −0.0111 sig | +0.0043 sig（回补） |
    | riichi_hit1 | 0.1447 | 0.1295 | 0.1280 | −0.0152 sig（≈全部） | −0.0014 ns |
    | wait_hit5 | 0.4900 | 0.4993 | 0.4936 | +0.0092 sig | −0.0057 ns |

    **结论**：① 向听头学习**完全由损失驱动**（次级判读 a 证实：v2x 头 0.5257≈v2s 0.5263、一致率 0.9405——瓶颈消费对「头学会」非必要）；② **AUC 增益完全由瓶颈馈通驱动，且损失单独作用反而使 AUC 低于 v2**（−0.019）——tenpai-tie 塑造的头信念必须被 H2 消费才转化为听牌检测力；显式向听分布的**排序**能力远超 exp_counts 所能表达（两模型听牌目标上平均 P(未听) 一致 0.011，差别全在排序）；③ wait hit@1 代价 55/45 混合（W=0.1386 落预注册混合区）、riichi 代价几乎全在损失侧；④ 存在交互：瓶颈回补损失的 alloc 代价（+0.0043）但吃掉损失的 wait_hit5 收益（−0.0057）；⑤ 机制诊断：v2x 与 v2s 的 wait 分布形态几乎一致（top1 0.0756/0.0763、熵 3.0025/3.0005，均比 v2 的 0.0823/2.9772 软）——锐度损失在损失侧，与 AUC 排序增益是两条独立通路。**判决（预注册规则③）**：v2x AUC 0.667 ≪ v2s−0.005=0.734 → **保留 46 维瓶颈，v2s 架构定为 A2 基线**；wait/riichi 代价在损失侧，A2 家族内用旋钮攻（λ_w 0.5→0.75 / tenpai-tie↓），不回 38 维。
  - **A2 事故与处置（10-17，会话 27e）**：A2 首跑两次在 step ~7500 **确定性**崩溃（DataLoader worker `Bus error`/`Segmentation fault`，与 worker 数/墙钟无关，仅特定牌局触发）。根因：共享树 `libriichi` 当时被 **s6a 隔离实验**改版（`s6a_isolated/patches/v5_obs_features.patch`，obs v4→v5 +6 通道）且其编译产物 `.so` 在位；用户 14:48 还原源码并重编（`mortal/libriichi.so` 与 `target/release/libriichi.so` md5 一致，`obs_shape(4)=(1012,34)`），39 项回归 ALL PASS。处置：**作废被污染的 8000 步 checkpoint**（隔离为 `mortal/read_model_v2s_50k.contaminated_so/`，其前 8000 步在改版 .so 下训练、obs 语义不可信），从头重启 A2（50000 步规范配置）。教训：共享树 libriichi 实验必须完全隔离（s6a_isolated 模式），还原后必须**重编 .so 并跑回归门禁**再启动训练；另：`except Exception` 无法捕获原生段错误（所以坏 .so 会直接杀进程而非被 skip）。
  - **A2 v2s 50k 步正式轮（~5-6h，10-17 15:00 起）**：`STEPS=50000 bash run_v2s.sh`（同数据同种子，**同配置**——干净的单因素=预算扩展 + README 既定下一步②）。判据（bootstrap CI vs 15k v2）：wait Δ 跨 0 或转正 = 容量成本假设成立，该 checkpoint 定为**注入基线**；代价仍在 → 转 A2b 旋钮轮（λ_w 0.5→0.75 直接对冲 wait 项，或 tenpai-tie 0.1→0.05）。
  - **A2 结果（10-18 完成，best@47000 val 2.0968，50000 步 35612s）——预注册判据未满足**：wait_hit1 Δ(vs v2) = −0.0112 CI[−.021,−.002] **显著为负、未回补** → 容量成本假设**不成立**。同时出现**多目标分化**（vs v2）：**tenpai_auc 0.6855→0.6671（−0.0184 显著退化）**、alloc_top1 −0.0051 / riichi_hit1 −0.0282 显著退化；**但 wait_hit5 +0.0473、deal_pure 点炮率 0.0422→0.0286、riichi_deal_pure 0.0641→0.0428 均显著改善**（防守选牌更安全）、shanten_acc +0.2747。预算效应分离（vs 15k v2s）：AUC −0.0721 显著退化、wait_hit5 +0.0438 / deal_pure −0.0122 / riichi_deal_pure −0.0166 显著改善、wait_hit1 持平。**机制（已确认）**：`l_wait` 只在结构听牌目标（14.5%）上激活，其余 85.5% not-tenpai 目标对 H2 的 wait[34]（未听）输出**无直接监督**；长训练中"听牌目标 P(未听)→0"的梯度单向传递，导致非听牌目标 P(未听) 集体塌缩（均值 0.0122→0.0080、中位数 0.0030→0.0013）→ tenpai 判别 AUC 退化。**判决**：50k v2s **不按原判据定为注入基线**（AUC 退化最伤"该不该防"）；15k v2s 仍为均衡基线。**A2b 定稿**：在 wait 输出侧加直接 tenpai BCE（监督 H2 `wait_probs[34]` 拟合 `1−tenpai`，权重旋钮，直击塌缩机制），再跑长训练，目标同时保住 AUC 与点炮率。**附注**：`compare_bootstrap` 的"A优/B优"仅比较数值方向，对点炮率类指标（越低越好）需反转解读。
  - **A3 规则法基线（方案 §8.1 第 2 条，集成前必需；无训练，纯 numpy）**：现物/筋/壁 + 立直枚举 + 见逃排除，同一固定评估集出 wait hit@1/deal_safe/AUC。用途：① 校验模型 wait 概率相对人类百年规则的增量；② 注入时的混合/回退组件；③ 逐样本错误分析（模型在哪些局面输给规则）。

  **阶段 B：注入 Mortal（路径 C，唯一强度标准；按方案 §9 单独立项，不碰 `mortal/`/`libriichi/` 直到立项）**

  - **B0 前置**：立直振听自动检测（`constraints.renormalize_wait` 的 `riichi_furiten` 口子已留，河牌硬掩码覆盖主体，剩余=立直后见逃极稀有例）；吞吐（Rust `with_sp_tables` 开关或 sidecar 缓存，50k+ 轮与注入数据生成的乘数——当前瓶颈 `encode_obs` ~2.2ms/样本其中 SP 计算器是大头）。
  - **B1 限域注入实验**：读牌模型冻结，输出（wait_probs 35 维 + exp_counts + 向听分布）作额外特征注入 Mortal 观测，防守决策点/全决策点两档，短程微调；ladder 双向同牌山 ≥5000 sets + block bootstrap vs 现役最强（`round4/mortal.pth` 645k，基线口径 B/756k）。可选路径 B（ONNX-in-Rust）作零训练依赖通道：export 已就绪并经 onnxruntime 校验，补 Rust loader + 逐位一致性测试即可。
  - **B2 判定门禁（事前写死）**：CI 下界 >0 → 采纳；跨 0 → 无结论（按效能公式算所需 sets 再决定）；<0 → 回滚并记录。

  **阶段 C：持续项**：每轮训练后固定动作 = evaluate（20000）→ compare_bootstrap CI → diag_by_form 分形制 → ONNX 导出校验；任何 dataset/model/loss 改动全量重跑 39 项回归。已知遗留：类均衡使最稀有向听桶 sh5 召回 0.011→0.000（n=1155）——旋钮：均衡幂次 0.5→0.25 或 lo 钳制上调，留待 A2 轮次顺带观察。

  **明确不做**：离线指标不作强度结论；注入前不动共享代码。

### 会话 28：IQL Stage B v3 终局判定（10-16 18:29 UTC ALL DONE）——真 IQL 首次干净检验【无结论】，IQL 路线按预注册规则关闭

**Stage B 全轮次结构（入档）**：

| 轮次 | 状态 | 关键差异 | 结果 | 有效性 |
|---|---|---|---|---|
| v1（10-15） | ❌ 作废 | τ=0.7；**r_t 每样本重复计入 16×**（dataloader 无条件发 `kyoku_rewards[at_kyoku[i]]`，§16.26） | δ=−31.04、top1 0.692、lse −176.1 | 无效（实现缺陷，会话 23） |
| v2（10-16 凌晨） | ⚠ 不作判决 | r_t/L_V 已修；τ=0.7 → **价值指数漂移**（q_target p50 +0.6→+60.8，棘轮，§14-39） | δ=−3.416 显著变弱（bootstrap CI[−4.46,−2.36]）、top1 0.8069/lse +45 | 无效（目标实现不稳定，漂移二阶污染策略通道） |
| v3（10-16 06:26→17:41 训完） | ✅ 有效 | v2+**τ=0.5**（探针唯一验证的稳定修复），其余锁死=B；种子块 220000/230000、bseed 20261016、命名空间 iql_b_v3 | **δ=+0.646，bootstrap CI[−0.35,+1.63] → 无统计结论** | **首份干净判决** |

**v3 实测明细**：
- 训练：867k/867k exit=0、无 non-finite；111k 步 / 11.25 h（均 ~2.74 batch/s，区间 2.7–2.95；iql 模式 3× encoder 前向故慢于 B 的 4.2）；价值尺度全程监控（train.py 每存盘点 `iql scale:` 行）：q_target p50 约 2 万步内收敛 ~1.2，终点 **+1.206**、p99 +2.04，q_pred p50 +1.265 紧贴 → **max p50=1.206 ≪ 理论界 [−5,+5] 的 1/4，且远低于 v2 同点 +60.8；τ=0.5 反漂移修复在 111k 步全程成立**
- holdout（202609，仅诊断）：top1 **0.8167**（历轮最高：B 0.8118 / Stage A 0.8118 / v2 0.8069 / v1 0.6917）、logsumexp_q **+1.596**（v2 +45）、agari_top1 171/171
- ladder（双向同牌山各 10,000 半庄，两波各 5×500 sets）：A（v3 当挑战者）avg_pt +1.750（SE 0.832）/ B（B/756k 当挑战者）+0.459（SE 0.842）；**δ=+0.646 pt/半庄，t=+1.09；block bootstrap SE 0.509、95%CI[−0.35,+1.63] → CI 跨 0 = 无结论**；10 个分块中 3 块为负，信号全部落在同牌山正相关噪声内（ρ≈+0.19）；效能参考：若真实效应就是 +0.65，达 2σ 需 ~16,808 sets/侧（当前 5,000，即 ×3.4）

**判读（按 `run_iql_b_v3.sh` 事前写死的收尾守则）**：CI 下界 >0 → 判进采纳；无结论/变弱 → **IQL 路线关闭**。v3 落“无结论”格 → 与 Stage A（δ=−1.03 显著负，§16.25）合并：**目标形式改造（信用分配视界 / TD 价值传播）两条子路线全部关闭**，转 ③ 扩窗 2019+ + 重训 GRP + 按年份审计。现役最强仍 `round4/mortal.pth`（645k），基线口径仍 B/756k。

**归因分析（以下为基于实测数据的推断，非直接实测）**——为何 Stage A/B 均无正向效果：
1. **基线 B 已做完局内信用分配**：每局平均 16.00 个决策点、γ=0.997/决策步 → 局内折扣 0.997^16≈0.95，`kyoku_rewards` 本就近全权重回传给局内每一步；A/B 真正改动的只有**跨局**视界
2. **跨局视界信噪比崩溃**：终局 pts∈{2,1,0,−3} 的共同分量（std≈1.2）被全场 ~190 个决策分享，而单决策对终局 pts 的因果效应仅 ~±0.1 → 每决策 SNR≈1:10。Stage A 把该方差倒回 Q 头（且拆掉 delta_points 相对 GRP 期望的低方差残差结构）故**显著负**；Stage B 的 bootstrap 方差缩减被“V 自身要从同一跨局噪声信号学习”的误差反馈抵消，故**中性**
3. **AWR 在近最优专家数据上权重≈均匀**：模型与专家分歧仅剩 ~18%（top1 0.8167），其中相当比例是人类的合理方差/ indifferent 选择而非可纠正错误，优势加权无从发力
4. 与历史 null（R3/R4/R5/augmentation/hp_ablation 全 CI 跨 0）互相印证：**瓶颈是数据分布的信息天花板，不在目标函数/优化维度**——这为路线图预判的“扩窗换信息源”提供了又一向下的证据

**方法论沉淀（§14-40）**：慢漂移（时间尺度 ~3 万步）必须靠“每存差点分位数日志”在正式轮全程监控——探针（30k 步 ×5 臂）只验证短期稳定，v3 的 111k 步长稳靠 `iql scale:` 全程行 + 链脚本放行门禁（max p50 ≤ ~5）确认。

**产物**：`checkpoints/iql_b_v3/`（mortal.pth 867k + best.pth + step_800k/850k）、`iql_b_v3/{train.log,holdout_202609.log,ladder_summary.log}`、判定牌谱 `iql_b_v3/ladder/`（A_220000..222000 / B_230000..232000，20,000 局）、链日志 `logs/iql_b_v3_chain.log`（ALL DONE）、曲线 `runs/iql_b_v3`；配置 `config.iql_b_v3.toml`（`tools/training/prepare_iql_b_v3.py --fix tau05` 生成）+ `run_iql_b_v3.sh`；v2 产物在 `checkpoints/iql_b_v2/` + `iql_b_v2/`（漂移轮，权重不作基线）。种子块 220000/230000 已消费（记入 §17.8 清单）。

### 会话 29：S6a 紧迫度特征注入（obs version 5）实施 + 终局判定（10-17~18）——无结论；归因“零新信息”

**背景与设计**（用户观察 → NEXT_STEPS §4bis）：主模型缺少四类显式估计——牌型、逆转顺位所需分数（紧迫度）、自家做牌速度、他家做牌速度；GRP 顺位优先框架合理（B 口径 `pts=[2,1,0,-3]` 编码“顺位可比、分数不可比”）。9 次目标/数据旋钮否证后，唯一未试方向 = **加信息/归纳偏置**。S6 分四阶段：a 紧迫度 → b 读牌后验 → c 自家速度+牌型 → d GRP 扩展+攻防分支重规划；S6a 是最便宜的第一步（零学习成本、无读牌依赖）。

**特征（v5 = v4 + 6 行，`obs_shape(5)=(1018,34)`）**：与排序后 1/2/3/4 名的点差 `d1..d4`（4 行，符号模式即当前顺位）+ `overtake_gap`（紧邻上位者分数−自家）+ `lead_gap`（自家−紧邻下位者）。全部由 `state.scores`（已旋转，`[0]`=自家）确定性导出，归一化到 [0,1]（`diff.clamp(±1e5)/2e5+0.5`）。**实测修正**：牌局长度 6~11 局不等（提前结束/西入），故**放弃**“按剩余局数归一化的每局需得分”，不做任何分布假设。

**实施**：
- Rust：`consts.rs` `MAX_VERSION` 4→5 + `obs_shape(5)`；`obs_repr.rs` v5 复用 v4 全部编码 + 尾部 `encode_urgency`（6 行，且所有 `match version` 分支必须补 5——漏一处会 panic 或通道错位）
- Python：`model.py`（Brain/DQN `case 5`）、`engine.py`、`env_check.py`（新增 `obs_shape(5)` 预检，陈旧 .so 会被拦下）
- 权重手术：`expand_ckpt_v5.py`（首层 Conv1d [192,1012,3]→[192,1018,3]，新通道零初始化；**新通道输入为 0 时前向与 v4 逐位等价**，等价于“原模型 + 新特征从零学起”）
- 回归：`test_urgency_features.py` 六层 17 项 ALL PASS（v4 前 1012 行逐位一致；2304 条 obs 从 v4 分数通道独立重算，最大误差 3.7e-8；手术等价误差 0）
- **完全隔离（用户要求，§14-41）**：`s6a_isolated/`（code 副本 + 独立 v5 `libriichi.so` + 本地 index/ckpt/runs/logs + 输入经软链只读共享树 + `patches/v5_obs_features.patch`）；共享树 `git checkout` 还原 v4 并重建 `.so`；正式链含 GPU2 等待循环（避让并行 β 探针）

**冒烟（GPU1）**：ALL PASS —— version=5 / (1018,34) / 无 non-finite / 到 758k / test_play avg_rank 2.555≤2.60 / holdout top1 0.8166 / lse +0.483 / ckpt v5+1018。

**正式轮（10-17 15:28 → 23:47 UTC ALL DONE）**：867k、exit=0、7.8 h、**4.5 batch/s**；holdout top1 **0.8160**、lse +0.467、R² −0.995（既有同型）；test_play 800k 2.486/+1.58、850k 2.490/+1.89（父代 2.466/+3.06，固定种子仅趋势）。双向 ladder 各 10,000 半庄：A（S6a 挑战）avg_pt +0.499（SE 0.849）/ B（B/756k 挑战）+0.135（SE 0.837）→ **δ=+0.182 pt/半庄，SE 0.596，t=+0.31，解析 CI[−0.99,+1.35]；bootstrap SE 0.513、CI[−0.82,+1.20] → 无统计结论**。效能：若真实效应 +0.18，达 2σ 需 ~213,885 sets（42.8×）。

**判读**：按预注册守则落“无结论”格。**关键归因：S6a 注入的是“零新信息”特征**——6 个通道全部是 v4 obs 已有分数通道（行 7–14）的确定性函数，网络本就能算；因此 δ≈0 的正确解读是“**ResNet 已能自行计算点差，预计算不带来额外归纳偏置**”。这**不否定 S6 方向**，只否定“重排已知信息”的最便宜变体；真正可能移动指针的是模型无法从 obs 算出的信息（读牌后验 = S6b）。不建议再调特征形式（若要试，并入 S6d GRP 输入扩展）。

**产物/隔离**：`s6a_isolated/code/checkpoints/s6a/`（867k+best+step_800k/850k）、`s6a_isolated/code/s6a/`（train/holdout/ladder_summary）、判定牌谱 `s6a/ladder/`（A_240000–242000 / B_250000–252000，20,000 局）、TB `runs/s6a`、链日志 `logs/s6a_chain.log`；种子块 **240000/250000 已消费**，下一块 260000/270000。

### 会话 30：S6c（自家速度类）设计分析（10-18）——确定性函数，且已被 v4 obs 显式编码

**问题**：acceptance / 期望巡目 / 牌型 怎么设计？是否也是 obs 的确定性函数？

**答：是，且比 S6a 更“冗余”——v4 obs 已显式编码其中大部分**：
- 向听数：**6 槽 one-hot**（`obs_repr.rs` L395-396，所有版本）
- 自家待牌（听牌时）：34 维 one-hot（L382-388）；振听/立直标志（L390-401）
- **有效进张**：SP 表每个打牌候选的 `required_tiles`（2×34 one-hot，L576-608）+ 最佳有效牌
- **逐巡期望**：SP 表 ≤17 巡的 `tenpai_prob / win_prob / exp_value`（3×17，L644-691）
- 宝牌数：`doras_owned` 每玩家（L286-291）；可见牌：`tiles_seen` 通道

→ “acceptance/期望巡目”本质就是 SP（单人）计算器的输出，已在 obs 里（逐候选、逐巡的丰富形式，而非标量）。

**SP 盲区（唯一有增量空间的自量）**：`SHANTEN_THRES=3`（`algo/sp/calc.rs` L13）→ **shanten≥4 时 SP 放弃**，obs 对应区域写零。实测 5 文件 / 3,622 决策点：shanten 0:14.8% / 1:25.4% / 2:31.6% / 3:19.6% / **≥4:8.72%**。

**结论与建议**：
1. **S6c 纯重算版（把 acceptance/期望巡目/牌型做成标量通道）= 零新信息且比 S6a 更冗余 → 不单烧正式轮**（预期 null）
2. 真正值得做：① 自家速度 × 他家速度（与 **S6b 合并**，真新信息）② 自家速度 × 紧迫度（“够不够分”的组合语义）③ SP 盲区补齐 + 牌型潜力（覆盖仅 8.7%，边际小）
3. **推荐合并成一次 v6 注入**：读牌后验（S6b）+ SP 盲区 acceptance + 牌型潜力（紧迫度可留可删）→ 一次手术、一次训练；攻防决策需要三者同场
4. 若先验证：用现隔离工作区做 **30k 步探针**（~2h），门禁 holdout/行为画像（弃和率/流局率），不烧 11h
5. 实现：Rust `encode_obs` v6；盲区 acceptance 用 `algo/shanten.rs` + `tiles_seen` 枚举 34 种牌（降 shanten 的牌种数/总枚数）；手术沿用 `expand_ckpt_v5.py` 泛化；注意 SP 已是吞吐瓶颈（`with_sp_tables` 是既定优化）
6. 优先级 **S6b（新信息）> S6c（盲区补全）> S6d（GRP 扩展）**；种子块从 260000/270000 起

## 13. 待办与下一步

| # | 事项 | 说明 |
|---|---|---|
| 1 | ✅ ~~安装 Rust 工具链~~ | **已在服务器完成**（rustc/cargo 1.99.0，10-06）；本机仍不需要 |
| 2 | ✅ 安装转换库 | jansou 0.14.0 已装入默认 Anaconda 环境（requests/tqdm 原本已有） |
| 3 | ✅ 确认 jansou API | `parse_mjlog` / `dump_mjai` / `check_paifu` 已验证（见 TASKLIST.md §5.5） |
| 4 | ✅ 抓取近 7 天 scc 索引 | 已完成（176 个文件，TASKLIST.md §5.1） |
| 5 | ✅ 过滤生成对局清单 CSV | 已完成（窗口 4,020 局；`dataset/index/houou_games.csv`） |
| 6 | ⚠️ 2026 自建采集可收尾 | 130,407 局（202601~202610）已全部上传服务器并进树；天凤年度 zip 仍 404，**保持每月增量归档即可**（§8.5 年底并入风险） |
| 7 | ✅ mjlog → mjai 转换 | 已完成（本机 2026 部分）；历史年份直接来自 HF mjai 包，不需转换 |
| 8 | ✅ libriichi 闭环校验 | 服务器已跑 `validate_logs`（10-07）+ `prune --apply` 逐文件规则/结构自证，抽样月份规则码 100% `00a9`；训练已能正常取 obs/actions/masks |
| 9 | ✅ 历史数据来源已解决 | HF 18 个年度包已上传服务器 `dataset/tenhou_hf`（17 GB）并**已全部 merge 进 mjai 树** |
| 9b | ✅ D 盘散装残留已清 | 已删除 `D:\tenhou_mjai`；`D:\tenhou_hf` 原包亦已迁到服务器，**本机不再承担训练数据职责** |
| 9c | ✅ 历史数据训练落地（已完成） | 2,630,640 局已在服务器 NTFS 本地盘（`/mnt/data8`），**exFAT 限制不复存在** |
| 10 | ✅ `.gitignore` + 本地 git 仓库已建 | 10-09：`.gitignore` 补齐（dataset/runs/logs/backups 等）并 `git init` → 基线提交 `33e63bc`（分支 `main`、460 个文件、4.4 MB）、tag `round1-complete`。**仅缺远端推送**（服务器当前无外网）；见 §16.10 |
| 11 | ✅ 第一轮已训完 | GRP 30k 步 + 主模型 200k 步（10-08~10-09）；**结果不可用**，指标饱和且强度持平，见 §16.7 |
| 12 | ✅ 确认 GPU 与显存 | 服务器 4×P100-16GB + 1×RTX 8000-48GB；实际 GRP→P100、主训→RTX 8000，192×40/batch512 无压力 |

### 13bis 下一轮待办（2026-10-09 复盘后新列，优先级自上而下）

| 优先级 | 事项 | 要点 |
|---|---|---|
| **P0** | 先修评测，再动训练 | 停用「vs 随机 baseline」的 avg_rank 作为进度信号；改用 `ladder.py` 配对对抗 + 未见月离线指标（top1 一致率/Q 校准）；修正 `best.pth` 判定（当前 best=50k 步，不是最强模型） |
| **P0** | 提高吞吐 | `num_workers` 2→4～8、`RAYON_NUM_THREADS` 4→8（实测管道 3.12→7.48 it/s，而 GPU 上限 8.88 it/s）；内存 251 GB 充足，`file_batch_size` 可提到 15～20 |
| **P0** | 备份与可复现 | `git init` + 提交代码补丁 + scp 回本机；checkpoint 异盘备份；`.gitignore` 补 `dataset/`、`mortal/runs/`、`mortal/logs/` |
| **P1** | ~~把 GRP 训收敛~~ → **加大 GRP 容量** | 10-09 续训实验已推翻“未收敛”判断：30k→42k 步 val loss 仅在 2.453~2.457 噪声带内抹（best 2.45495→2.45207）→ **步数已到平台期**。改做容量实验（h64/l2 → h128/l3，从零 60k 步），用 `tools/training/grp_quality_probe.py` 的同口径指标（hit_last / mae_rank / nll_last）判胜负；胜出才替换奖励模型 |
| **P1** | 重设离线轮方案 | 只改“信息增量”源：高手白名单 `player_names_files`（已实现：牌谱无段位字段 → 用长期成绩自证，见 §16.12）、`enable_augmentation=true`、~~`gamma<1` 做跨局信用分配~~（⚠ **10-12 更正**：gamma<1 只影响当前局内较早动作的折扣，**不是**跨局信用分配，见 §14-31；已实测）、长程 LR 衰减、扩大时间窗（全量 2.63 M 或 2019+） |
| **P1** | 多卡并行实验 | 离线模式无数据并行，但可 5 张卡各跑一组对照配方，再用 ladder 选优 |
| **P2** | 外部强度锚点 | akochan（需装 JRE + `AKOCHAN_DIR`/`AKOCHAN_TACTICS`，`[1v3.akochan] enabled=true`）；人类牌谱不能当 arena 对手（mjai 日志无 `seed`，无法复现牌山） |
| **P2** | 在线自博弈（M5） | 仅在评测可信后开；`online=true` + `freeze_bn.mortal=true`；已知卡死 bug 靠子进程规避；P100×4 可当 client 集群 |
| **P3** | 防守行为专项 | 模型自博弈流局率 10.9% vs 人类 15.3%、放铳 0.1405 vs 0.1255（§16.8）→ 弃和不足，离线 RL 典型缺陷，可试 `cql.min_q_weight` / 奖励平滑 / oracle 特征 |
| **P3** | 数据合规 | 大文件与牌谱不入 git、不对外的红线不变（§8.7）；备份也只在自有存储内 |

**采集规模预算**（2026-10-01 精算，n=118：raw 7.59KB + mjai 4.84KB ≈ **12.43KB/局**）

| 范围 | 局数 | 下载耗时（@2s/局） | 磁盘（raw + mjai.gz） |
|---|---|---|---|
| 近 7 天窗口 | 实测 4,020 局 | ~2.4 小时 | ≈ 50 MiB |
| **当前可获取的全部（2026 YTD）** | 实测 **129,475 局** | ~94 小时 | **≈ 1.53 GiB（±47 MiB，95%）** |
| 历史 2009~2025（需 zip 恢复） | 340~680 万局 | 数月（连续） | 约 40~80 GiB（仅粗估，无法精确） |

**建议策略（已执行完毕，保留存档）**：先跑完 7 天窗口→再铺开 2026 YTD→历史年份靠 HF 包。三者均已完成；2026-10-09 起，**数据问题已转为“如何选数据”而非“如何拿数据”**（全量 2.63 M 局已在服务器）。

---

## 14. 关键坑位备忘

1. **scc 是 html-escaped**，先 `html.unescape` 再正则，否则含 `<>&` 的玩家名行会解析错位
2. 排序机制会**把新行插在文件中间**（长时对局），一切以 log_id 去重为准
3. `00a9`=四鳳、`00b9`=三鳳（观测规律），但**过滤逻辑以行内规则文本为准**
4. 牌谱 URL 的响应**已是 gzip**，不要二次 gzip，直接存字节
5. `list.cgi` 当前窗口没有 scd/sce 文件属正常（无对局时不出现在列表）
6. **2026 年度 zip 未生成**，今年数据靠 7 天窗口滚动累积 → **必须定期归档当前窗口**，否则 7 天前的今年数据永久丢失
7. `GameplayLoader` 必须**在 `build_iter()` 而非 `__init__` 中构造 GRP**（Windows 下后者会失效，源码注释已说明）
8. 训练依赖 **GRP 权重先存在**（`config['grp']['state_file']`），否则 dataloader 直接失败
9. `train.py` 中 CQL 损失**仅离线模式生效**；在线模式仅存 DQN 与 aux 损失
10. 在线模式有**已知 bug**（测试后进程卡死，原因未知），故源码用子进程 + `sys.exit(0)` 规避，修改时勿"修复"掉
11. `save_every % opt_step_every == 0` 与 `test_every % save_every == 0` 有断言，配置时需保证整除
12. `avg_pt([90,45,0,-135])` 仅用于显示与最优判定，**从不参与训练**
13. ⚠️ **D 盘是 exFAT（簇 256 KiB）外置 USB 盘**，**绝不能存海量小文件**（每个 `.json.gz` 白占 256 KiB）。历史牌谱只以年度 `.tar.zst` 原包留存；散装 mjai 只能放 NTFS 卷。排查磁盘空间异常时先看文件系统与簇大小（`Get-Volume`；`Get-CimInstance Win32_Volume -Filter "DriveLetter='D:'"` 的 `BlockSize`）。相关：D 盘 USB 链路曾报 Event 153（I/O 重试）
    → 2026-10-09 补充：数据已整体迁到服务器 NTFS 本地盘，此坑不再影响训练；但**本机 → 服务器传输必须先 tar/7z 打包**（百万级小文件直传极慢），且**服务器上无此问题也不代表可以往任意卷狂撒小文件**
14. ✅ ~~服务器代码无版本控制~~ → **10-09 已解决**：已 `git init` 并提交基线（`main` / `33e63bc` / tag `round1-complete`），`.gitignore` 确保牌谱与权重不入库。仍适用的风险：① **无远端**（服务器无外网，无法 push），单机器单副本仍未真正消除；② 提交身份用的是临时 `-c user.name='mortal-trainer'`（环境里本来没有任何 git 身份，我没改你的配置），想换成自己的：`git -c user.name='你的名字' -c user.email='you@example.com' commit --amend --reset-author`；③ 本机旧代码覆盖上传前，先 `git status` 看是否会把服务器侧改动抹掉
15. ⚠️ **随机 baseline 会让 `test_play` 指标完全失效**：`baseline.pth` 为随机初始化权重时，`avg_rank` 从第一次评测（50k 步）就顶到 1.000、`avg_pt` 顶到 90（因为随机对手几乎不会和牌，模型必然全场 1 位）。此时：① 指标无区分度；② `best.pth` 只会记录**第一个**评测点（后续 89.955 < 90 永远“不破记”，§16.6）。→ 训练内监控不要用 avg_rank，改见 §16.7 的两类可用指标
16. ⚠️ **评测种子固定 + 评测目录被重建**：`player.py::test_play` 用写死的 `seed_start=(10000, 0x2000)` 且每次 `shutil.rmtree(log_dir)` → ① 所有检查点在同一批 500 组牌山上比较（易过拟合牌山选择模型）；② 历史评测牌谱全部丢失，事后无法重算。建议：种子轮换 + 每步单独存目录（`runs/selection/<step>/`）
17. ⚠️ **吞吐瓶颈默认在 CPU 管道而非 GPU**：`num_workers=2` + `RAYON_NUM_THREADS=4` 时管道只能 3.12 it/s，而 GPU（RTX 8000+AMP, batch 512）能跑 8.88 it/s → 实得 2.56 it/s（§16.9）。改 `num_workers=4~8` + `RAYON_NUM_THREADS=8` 即可近 2.4× 免费提速；环境变量必须在启动训练**前**设好（rayon 线程池在进程内只初始化一次）
18. ⚠️ **`best_perf` 初始值与“更强 checkpoint”无关**：`train.py` 的 `best` 判定仅比较当前评测；想选最强模型必须跑配对对抗（`ladder.py`）而不是信 `best.pth`
19. ⚠️ **`CUDA_VISIBLE_DEVICES` 的序号默认不等于 `nvidia-smi` 的序号**：不设 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 时，CUDA 按算力优先枚举（本机 ordinal 0 = RTX 8000，1~4 = P100），而 `nvidia-smi` 按 PCI 序（RTX 8000 是 2）→ 同一个 `=2` 在两套序下分别是 P100 / RTX 8000。后果是“以为在用 48GB 的 8000，实际在跑 P100”且只能从速度反推。做法：**固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，并在启动后看日志 `device: cuda:0 (型号)` 那一行做校验**。逐个实测表见 `ENV_SETUP_P100.md §10`（日志 `mortal/logs/env_map.log`）
20. ⚠️ **`import libriichi` 必须在 `mortal/` 目录下**：仓库根目录有个同名 Rust crate 目录 `libriichi/`，会被 Python 当成**命名空间包**遮蔽真扩展（实测 `__path__=_NamespacePath([...])`、`dir()` 为空、`libriichi.__version__` 报 AttributeError）→ 看起来像“没编译/没装”，其实只是目录不对。同理 `config.py` 的相对路径、`checkpoints/`、`runs/` 也都依赖 cwd=`mortal/`
21. ⚠️ **`ladder.py` 并发必须传独立的日志目录**：它开头就 `shutil.rmtree(log_dir)`，原本硬编码 `runs/ladder` → 多卡并行跑时后启动的任务会把己在跑的任务牌谱删掉（表现为结果文件缺失/统计错乱）。已加第 5 个位置参数 `log_dir`；用法 `python ladder.py <chal> <cham> <seed_start> <sets> <自己的日志目录>`
22. ⚠️ **离线模式一个 epoch 就退出，与 `max_train_steps` 无关**：第 2 轮预算 450k 但只跑到 424.8k 就“静默正常退出”（无报错、无 `training budget reached`，日志末尾只有一行 `ResourceWarning: Implicitly cleaning up <TemporaryDirectory>`）。想跑更长必须提 `num_epochs`（或用 `ladder` 定量核实确实跑完了数据）。另注意：`DataLoader` 全轮只产生一个不满批（碎片在缓冲里拼接），所以进度条停在半周期是“数据吃完”的正常特征，不是被人为中断
23. ⚠️ **`LD_LIBRARY_PATH` 的作用域被旧手册写大了**：实测 `import libriichi`、GPU 前向+AMP、牌谱解码在最小环境（只有 PATH/HOME）全部正常；但 `target/release/validate_logs` 与 `stat` 链接了 `libpython3.12.so.1.0`，不给 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 会 `error while loading shared libraries`（退出码 **127**）→ **只有跑 Rust CLI 时才是必需项**
24. ⚠️ **正在运行的 bash 脚本不能被编辑**：bash 是**按字节偏移增量读取**脚本的，改动后它会从错位处继续读 → 轻则报 `$'\xxx' : command not found`（偏移落在多字节汉字中间），重则**把已执行过的命令再跑一遍**（10-11 真踩：patch `run_round4.sh` 后它重跑了 `train.py` 行，冒出幽灵训练进程）。正确做法：先停会话再改脚本，或永不编辑已启动的脚本（用新文件/新版本号）。
25. ⚠️ **`pkill -f "<模式>"` 会匹配到执行它的 shell 自己**：只要命令行里含同一字符串就会被自杀（输出截断、后续命令不执行）。改用 `pgrep -f "[t]rain.py"`（方括号断字）拿 PID 再 `kill`。
26. ⚠️ **SIGINT 停训后 DataLoader worker 会变孤儿进程**：主进程退出不回收到它们（PPID 变 1，继续占 CPU/文件句柄），必须再按 PID 补一刀，并 `pgrep` 确认归零后才算停干净（GPU 显存会随主进程释放，所以只看显存会漏掉）。
27. ⚠️ **AMP 下 BatchNorm 的 running stats 会在前向就被污染，救不回来**：一次 fp16 溢出后，GradScaler 只能跳过参数更新，但 BN 的 running_mean/var 已在那次前向里被写坏 → 前向从此 NaN，并连带把权重与优化器状态全部拉坏（10-11 实测扫描：Brain 572 个非有限张量、其中 162 个是 BN buffer，优化器状态 820 个）。发作是**数据顺序相关**的（重启后洗牌变了，于是在续跑后 1000 步内就中了），所以"上一段没事"不代表安全。处置：`control.enable_amp = false`（ENV 手册 FAQ 的既定兜底）+ train.py 存盘前的 finite 守卫（见 §16.20）。
28. ⚠️ **两处"次生灾害"要把链条关紧**：① TB 的 `add_histogram` 遇全 NaN 会抛 `ValueError: The histogram is empty` **直接打死训练进程**（看着像"训练自己挂了"）；② 链式脚本不校验训练退出码，就会**带着 NaN 模型继续跑判定**（实测浪费 5 卡 ≈12 min，结论 avg_rank 3.92 / avg_pt −125 全是垃圾）。已修：histogram 先过滤非有限值；链脚本 `train_code=$?` 非 0 即跳过判定并退出。
29. ⚠️ **同一个 TB run 目录会混入多段进程的事件文件**：每次启动（含崩溃后重启、暖启重跑）都新建一个 `events.out.tfevents.*` 写入同一 `tensorboard_dir`，于是同一 step 出现多个合法点（曲线锯齿/重复），而 NaN 段的记录**无法从已写入的事件文件里抹掉**（append-only）。`runs/main_r4` 就一度混了 4 个文件（09:25 AMP 段、14:44 幽灵段、15:50 NaN 段、16:21 fp32 段）。做法：把坏段/空段文件移到 TB logdir 之外归档（本机移到 `logs/tb_archive/…`，72 KB 取证保留），刷新即可看清；判读曲线时记住 500k–541k 是两次经过。
30. ⚠️ **「结构合法但语义全错」的数据修复最危险（10-12 实测踩到）**：`<UN>` 的 `n0~n3` 是 **URL 百分号编码**的玩家名（值长 45~63、纯 ASCII），直接写成 mjai `start_game.names` 后，`validate_logs` **全绿**、索引照常构建、训练不报错，但白名单命中率 **0%** —— 等于把整棵树变废，而且没有任何自动信号。**教训**：牌谱字段修复/重建必须做「与已知玩家池/白名单的命中率」验收（`tools/tenhou_data/verify_names.py`），结构校验（validate_logs）只看事件流合法性，查不出语义错误。相关：① 缺 names 时 Rust 侧 `#[serde(default)]` 会把 `names` 静默变 4 个空串，不报错只是不产样本；② `validate_logs <DIR>` 的 glob 含 `**/*.json`，别把报告类 json 写进数据树（会被当牌谱解析报错）。

31. ⚠️ **`gamma<1` 不是跨局信用分配（10-12 实测更正，此前 §13bis/§16.16/§16.22 的说法有误）**：`steps_to_done` 在**局边界归零**（`dataloader.py`：`if not dones[i]` 才累加），奖励取 `kyoku_rewards[at_kyoku[i]]`（仅当前局）→ `gamma**steps_to_done` 只在**当前局内**给较早动作打折，且每局最后一条动作 `steps_to_done=0` 不被折扣。实测（`tools/training/inspect_gamma_semantics.py`，200 文件 / 54,526 样本）：参与折扣的只有**打牌(0-36) 与立直(37)**（判据 `label <= 37`）；**鸣牌(38-41)、杠(42)、和了(43)、九种九牌(44)、pass(45) 都不折扣** —— 与 `add_entry` 旧注释 "only discard and kan" 不符（注释已改）。**若做真跨局 credit assignment**，需数据侧显式给出"下一局状态 + 未来回报"并引入 target network，属独立框架改动。
32. ⚠️ **主训练必须用 GPU 2（RTX 8000），别随手挑 P100（10-13 实测踩到）**：本模型 192×40 fp32 直接吃 GPU 算力，实测 P100 **2.48 batch/s** vs RTX 8000 **4.36 batch/s**（R4/R5 同配方复核一致，差 1.76×）。R5 首次启动误选 GPU 0，按 P100 速率估要 12.8 h，换卡后 7.3 h——**凭空多花 5.5 h**，而且会让"每轮约 7 h"的既有时间基线被无声拉长一倍。硬件原因：P100 是 Pascal（FP32 ~9.3 TFLOPS）、RTX 8000 是 Turing（~13 TFLOPS）。**做法**：所有 `run_roundN.sh` 固定 `CUDA_VISIBLE_DEVICES=2` + `CHECK_GPU_NAME='RTX 8000'`（env_check 会反验）；估算训练时长一律引用 RTX 8000 的实测基线；P100 留给并行 ladder 判定（那阶段吃 GPU 但每卡只跑 500 sets，不敏感）。
33. ⚠️ **暖启续训的冒烟配置不能用「绝对小步数」当预算（10-14 踩到）**：`train.py` 的 `steps` 从 checkpoint 里恢复，`max_train_steps` 是**全局绝对值**——若照直觉写 `max_train_steps=2000`，而父代 ckpt 自带 `steps=756000`，则启动即命中 `steps >= max_train_steps` → 打一行 `training budget already reached` **秒退、0 步不训**（看着像“配置生效了但没产物”）。正确写法：目标 = 父代步数 + 冒烟步数（如 **758000**）；并同步把 `test_every`/`snapshot_every` 调小（且满足 `test_every % save_every == 0`、`snapshot_every % save_every == 0`、`max_train_steps % save_every == 0` 三条断言），否则冒烟窗口内触发不到 test_play/快照，等于白跑。
34. ⚠️ **`tmux new -d -s x 'cmd > /path/日志'` 的重定向目录必须预先存在（10-14 踩到两次）**：重定向由 tmux 起的 shell 在**执行脚本之前**打开，若目标目录尚未被脚本 `mkdir -p` 创建，shell 直接 `No such file or directory` 退出 → **会话“秒退且零日志”**，`tmux ls` 看不到、`rc=0` 也照样失败（`tmux new -d` 的返回码不代表命令成功）。做法：启动前先 `mkdir -p` 日志目录，或把重定向指向已存在的 `mortal/logs/`；启动后用「脚本自己 tee 的日志文件」而非 tmux 重定向文件来确认存活。
35. ⚠️ **回归测试若照“实现现状”写断言，等于给 bug 盖章（10-15 踩到）**：Stage B 首版的 r_t 双重计入缺陷，在开发当天跑 `test_iql_data.py` 得到的是 **ALL PASS** —— 因为 C 段断言的是“iql 的 r_t == kyoku 口径逐样本值”（正是错误实现），而计划明写“仅局边界非零”。一个能让坏实现通过的测试，比没有测试更危险（它提供了虚假信心，直接放进 11.5 h 的正式轮）。**做法**：断言必须从**设计要求逐条翻译**（本例：“每局奖励在整个 game 的样本上只求和一次”→ 非零个数==局边界数、总和==Σr_k），参照系用独立口径重算（直接读牌谱算 dones/边界），而不是复用被测代码的输出。
36. ⚠️ **冒烟只查“结构活着”不查“模型没被打坏”，会白烧一整轮 GPU（10-15 踩到）**：v1 冒烟 8/8 全过（无 non-finite、到预算、快照在、test_play **有输出**），但同一次冒烟的 test_play 已经是 avg_rank 2.789 / avg_pt −19.34（父代 2.466 / +3.06）——劣化信号被“只要有行就算过”的门禁放过，代价 11.5 h GPU + 30 min 判定。**做法**：冒烟链必含**数值型行为门禁**（如 test_play avg_rank ≤ 2.60、holdout top1 ≥ 0.78、|logsumexp_q| ≤ 10），任一 FAIL 就禁止启动正式轮；新目标/新损失的首轮尤其如此（NaN 守卫只能拦异常值，拦不到“数值正常但方向错”）。**实效已验**：加上门禁后的 v2 冒烟直接 FAIL，抓出了第二个潜伏缺陷（§14-37），否则又是 11.5 h。
37. ⚠️ **IQL 的 expectile V 只能对「数据里实际执行的动作」回归，不能对全体合法动作取分位（10-15 踩到）**：`L_V = E_{(s,a)~D}[ L_2^τ(Q̄(s,a) − V(s)) ]`（Kostrikov et al. 2021 式 9）里的 a 是 **dataset action**；若改成对 masked 后的全部合法动作（本模型 ~30-40 个选项）取 τ-expectile，就把 V 绑到了**从未被任何损失监督过的 Q 外推值**的上分位上，后果：V 被无约束的 Q 外推拖走 → Q 目标（r+γV(s')）跟着漂 → 自激，表现为 logsumexp_q 到 −30 量级、top1 从 0.81 降到 0.70。**适用边界**：任何“对一个未完全被监督的 Q 分布做 max/expectile/top-k”的写法，都要先问这些动作的 Q 值到底由什么约束住。另记：算这类 masked 统计量时 `masked_fill(-inf)` 后要先 `torch.where(mask, x, 0)` 再参与乘方，否则 inf*0 会出 NaN。
38. ⚠️ **幅度类断言的上界要从数学推导，不能拿"直觉里的最大值"凑（10-15 v2 启动前审查发现）**：`test_iql_data.py` C 段原断言 `max|r_t| ≤ max|pts|` 是**错的**——`r_t = E[pts|k+1] − E[pts|k]` 是两个凸组合之差，正确上界是 **pts 极差 max(pts)−min(pts)**（B 口径 2−(−3)=5.0），max|pts|（3.0）只是其子集（旧 round4 口径 [6,4,2,0] 两者恰相等，故一直没暴露）。后果不是"漏报 bug"而是**误报**：按 B 口径复跑测试 FAIL（实测 max|r_t|=3.4573，144 局独立重算确认合法单局值），差点被当成"r_t 又出问题了"打断正式轮启动。**做法**：写"量级合理"类断言前先推导上界（差分量→极差；和/累加量→和上界）；本例双重计入的确定性判据是**计数/多重集/总和三项**（非零个数==局边界数、非零值多重集==边界处 calc_delta_pt、总和==逐局总和），幅度上界只是兜底。已修断言为极差上界。
39. ⚠️ **IQL 的 expectile V + γ≈1 + 边界稀疏奖励 = 价值尺度指数自激（10-16 v2 实测，会话 25）**：v2 实测 q_target 中位数 105k 步 +0.6→+61（倍增 ~32k 步纯指数，超理论界 [−5,+5] ~12 倍；理论界推导：r_k 沿局 telescope 至 pts[final]−E[pts|start]∈[−5,+5]）。机制：L_2^τ 在 V 低于目标时权重 τ=0.7 上拉、高于时仅 1−τ=0.3 下拉——增长 regime 中中游状态的 V 持续落后于下游目标（价值从有锚的局末向前传播、追而不及），残差持续为负 → 每步被不对称放大 → 有效反馈增益 γ·(1+bias)>1（bias≈2.3e-5/步）；EMA(0.995) 滞后使目标噪声 ∝ 尺度 → 偏差 ∝ 尺度 → 纯指数。**关键缓解事实**：v4 DQN 的 q=v(φ)+a(φ,·)−mean(a)，argmax 只依赖 a——价值膨胀不直接污染部署策略（参数级诊断：a_head 跨动作 std 1.74→1.60 统计不变），但 AWR 权重含 (v−v_ema) 滞后失真 ~10%、L_Q 对 a_head 的梯度随 TD 残差绝对值涨 3 倍 → 二阶污染策略通道（top1 0.811→0.807）。**处置**：① bootstrap 改用 V 的 EMA target（`iql_v_target_decay`，train.py 已实现）；② 或降 γ（0.997→0.99，margin 300×于实测 bias）；③ 或 τ→0.5（机制检查）。**方法论**：漂移时间尺度 ~3 万步，2k 步冒烟结构性门禁完全无感——**冒烟窗口必须覆盖待监测现象的时间尺度**（v3 冒烟改为 10-30k 步 + 每 1000 步 `iql scale:` 分位数门禁）。
40. ⚠️ **慢漂移类缺陷必须靠“每存差点分位数日志”做正式轮全程监控，探针只验证短期（10-16 v3 实测沉淀）**：expectile 棘轮的倍增周期 ~32k 步——5 臂探针每臂仅 30k 步、能判定“哪项修复止漂”，但**修复在 111k 步正式轮尺度是否长稳**只能靠训练内监控：train.py 现每存盘点打一行 `iql scale: q_target p50=… p99=… | q_pred p50=… | v_ema=on/off`（由 all_q/all_q_target 分位数算出），链脚本收尾时取全程 max p50 与放行门禁（≤~5；v2 同点为 +60.8）比对。v3 实测 103 个存点：p50 从 +0.02 收敛到 +1.206 后全程平坦 → 长稳确认。**做法**：凡“数值缓慢单向蠕变”类缺陷（漂移/泄漏/尺度膨胀），监控打点间隔须 ≪ 现象时间尺度的 1/10，且正式轮链脚本要把该指标做成放行门禁，而不是训完再事后翻 TB。
41. ⚠️ **并行实验环境下，禁止就地修改共享代码/替换共享 `.so`（10-17 实测干扰到 iql_beta）**：三个叠加机制——① Python 进程启动时 import 共享模块与扩展；② 已启动进程后续派生的子进程（DataLoader worker、holdout/test_play/ladder 子进程）会**重新 import** 当时的磁盘版本；③ `cp` 覆盖 `.so` 是“截断+写”，若正好落在别的实验的启动窗口内，对方会读到半写文件，表现为 `DataLoader worker exited unexpectedly` 或**静默 exit=0**（β 探针 attempt-1/2 即在此窗口内失败，其日志亦记录“外部 SIGINT 致静默 exit=0”）。**做法**：任何改变输入口径/obs 实现/目标函数的实验，一律建**隔离工作区**（复制 `mortal/*.py` 到独立 cwd + 独立编译产物 + 本地 index/ckpt/runs/logs，输入权重用软链只读共享树），共享树 `git checkout` 还原并重建原版 `.so`（范例：`s6a_isolated/`）；替换共享 `.so` 前先 `tmux ls` + 查进程确认无实验处于启动阶段，能不用共享副本就不用。

---

## 14bis. 日志文件归置规范（强制执行）

> **规则（2026-10-17 定，用户要求强制执行）**：所有新建/归档的 `.log` 文件**禁止散落在 `mortal/logs/` 顶层**，必须按下列类别放入对应子目录。写脚本时先 `mkdir -p` 目标目录（重定向目录不存在会导致 tmux 会话秒退且零日志，见 §14-34），启动前确认路径；整理旧日志时同样按本表归类。判定依据为**文件名前缀/语义**，不按时间。

| 目录 | 收纳内容 | 匹配示例 |
|---|---|---|
| `train/` | 主模型训练与轮次链日志 | `main_train`、`main_r{2..5}_train`、`round{4,5}_chain`、`round4_resume` |
| `grp/` | GRP 排名预测器训练与探针 | `grp_train`、`grp_v2_train`、`grp_h128_train`、`grp_capacity`、`grp_best_probe`、`grp_probe_base`、`grp_v2_launch` |
| `read_model/` | 读牌模型训练与守护进程 | `read_model_v{1,2,2s,2x}_{train,tmux}` |
| `iql/` | IQL/Bellman 路线全链路 | `iql_{a,b,b_v2,b_v3,probe}_chain`、`iql_a_check1`、`iql_{a,b}_launch_check`、`preflight_iql_a`、`status_iql_{a,b}`、`diag_iql_reward`、`forensic_iql_b_smoke`、`test_iql_data[_v2]`、`v3_launch` |
| `eval/` | ladder 判定、holdout、配对/bootstrap 统计 | `ladder_{probe,r2_run,r3_run}`、`holdout_r{3,4}`、`r2_dryrun`、`pairdiff`、`bootstrap_r{3,4}` |
| `smoke/` | 冒烟测试 | `smoke_{check1,check2,r4,round2}`、`smoke_v{2,3}_{check,final,final2}` |
| `check/` | 代码/配置/环境项检查 | `cfg_round2_check[_2]`、`gamma_check_*`、`refac_check`、`v3_bump_check`、`xcheck[_2]`、`syntax_check`、`doccheck` |
| `test/` | 功能/行为测试 | `bp_test[_2,_3,_final]`、`test_credit_modes` |
| `env/` | 环境与全链路验证 | `env_e2`、`env_map`、`env_probe`、`pre4_envcheck`、`envcheck_selftest` |
| `git/` | git 操作留痕 | `git_*`（init/fix/docs/status/verify/push_*/after_commit） |
| `ablation/` | 奖励/超参消融链 | `hp_ablation_chain`、`hp_resume_chain` |
| `data/` | 数据管线（榜单/白名单/扫描） | `whitelist[_v2]_build`、`toplist`、`refac_scan` |
| `monitor/` | 训练过程监控与估算 | `progress_{check,now,tp}`、`r2_monitor`、`r2_epoch_estimate` |
| `misc/` | 不属于以上类别 | `pack_zip`、`tensorboard` |
| `s6a/`（隔离区） | **S6a 隔离工作区日志，不在 `mortal/logs/` 下**：`s6a_isolated/code/logs/s6a_chain.log`、`s6a_isolated/code/s6a/{train,holdout_202609,ladder_summary}.log`、`s6a_isolated/code/s6a_smoke/*` | `s6a_chain`、`s6a/train` |

**例外与细则**：
1. **非 `.log` 文件不归置**（如 `agtest.out`、`commit_msg_stage_b.txt`、`reboot_handoff.txt`），留在 `mortal/logs/` 顶层或按工具约定存放。
2. **`ladder_r2/`～`ladder_r5/`、`_agtest/`、`r2_evaldone/`、`tb_archive/` 内的 `.log` 保持原位**：它们已按轮次/用途归类，且与同目录 `.gz` 判定牌谱、mortal_step 标记文件配套，移动会造成证据链断裂。
3. 命名规范：`{用途}_{轮次/版本}_{动作}.log`（如 `iql_b_v3_chain.log`、`main_r5_train.log`），保证他人可按前缀直接归类。
4. 新日志落盘前先查本表；表中没有的新类别，在 `misc/` 落盘的同时**回本表补一行**（本节随目录演进）。

---

## 15. 关键文件索引

### 训练与模型
- [model.py](file:///c:/Maj/Mortal/mortal/model.py) —— Brain / DQN / GRP / AuxNet / ResNet
- [train.py](file:///c:/Maj/Mortal/mortal/train.py) —— 主训练循环与三项损失
- [dataloader.py](file:///c:/Maj/Mortal/mortal/dataloader.py) —— 奖励构造与样本产出
- [reward_calculator.py](file:///c:/Maj/Mortal/mortal/reward_calculator.py) —— GRP 期望点增量奖励
- [train_grp.py](file:///c:/Maj/Mortal/mortal/train_grp.py) —— GRP 预训练
- [one_vs_three.py](file:///c:/Maj/Mortal/mortal/one_vs_three.py) —— 1v3 评估
- [config.example.toml](file:///c:/Maj/Mortal/mortal/config.example.toml) —— 全配置示例
- [server.py](file:///c:/Maj/Mortal/mortal/server.py) / [client.py](file:///c:/Maj/Mortal/mortal/client.py) —— 在线训练
- [mortal.py](file:///c:/Maj/Mortal/mortal/mortal.py) / [engine.py](file:///c:/Maj/Mortal/mortal/engine.py) —— 推理入口

### Rust 核心
- [lib.rs](file:///c:/Maj/Mortal/libriichi/src/lib.rs) —— 模块导出与 pymodule
- [consts.rs](file:///c:/Maj/Mortal/libriichi/src/consts.rs) —— obs_shape / ACTION_SPACE / GRP_SIZE
- `libriichi/src/state/` —— `player_state.rs`、`obs_repr.rs`、`update.rs`
- `libriichi/src/agent/` —— `mortal.rs`（Python 桥接）、`defs.rs`、`akochan.rs`
- `libriichi/src/arena/` —— `board.rs`、`game.rs`、`one_vs_three.rs`、`two_vs_two.rs`
- `libriichi/src/dataset/` —— `gameplay.rs`、`grp.rs`、`invisible.rs`
- `libriichi/src/algo/` —— `shanten.rs`、`agari.rs`、`point.rs`、`sp/`
- `libriichi/src/mjai/` —— `event.rs`、`bot.rs`
- `libriichi/src/bin/` —— `stat.rs`、`validate_logs.rs`

### 文档与配置
- [README.md](file:///c:/Maj/Mortal/README.md) —— 项目说明与权重指引
- [build.md](file:///c:/Maj/Mortal/docs/src/user/build.md) —— 构建流程（本地版）
- [docker.md](file:///c:/Maj/Mortal/docs/src/user/docker.md) —— Docker 快速开始（仅推理）
- [environment.yml](file:///c:/Maj/Mortal/environment.yml) —— conda 环境
- [.gitignore](file:///c:/Maj/Mortal/.gitignore) —— 需补充 dataset/memory/tools

### 本项目新增
- [TASKLIST.md](file:///c:/Maj/Mortal/tools/tenhou_data/TASKLIST.md) —— 数据管线执行清单（含可运行代码）
- [TRAINING_PLAN.md](file:///c:/Maj/Mortal/tools/training/TRAINING_PLAN.md) —— P100 训练完整方案（config 全文/bootstrap.py/时间估算）
- [ENV_SETUP_P100.md](file:///c:/Maj/Mortal/tools/training/ENV_SETUP_P100.md) —— 训练服务器环境配置手册（conda→torch→rust→libriichi→验证→GPU 分工）
- [MEMORY.md](file:///c:/Maj/Mortal/memory/MEMORY.md) —— 本文档

### 外部关键链接
- 作者权重说明 gist：https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- 官方文档站：https://mortal.ekyu.moe
- 在线服务：https://mjai.ekyu.moe
- 天凤日志一覧（含 ToS）：https://tenhou.net/sc/raw/
- 天凤年度归档：https://tenhou.net/sc/raw/scrawYYYY.zip

---

## 16. 服务器实跑现状（2026-10-07~10-10 远程同步）

> 本节全部为 **2026-10-09 在训练服务器上实测/复算得到**的数据（日志、TensorBoard event、checkpoint、牌谱重算），非推算。
> 目的：把远程侧已经发生的事实固定下来，供与本机较新的文档做一次双向对账（需核对项见 §16.11）。

### 16.1 一页结论

| 项 | 结论 |
|---|---|
| 数据 | ✅ 2,630,640 局凤凰卓半庄（200902~202610）已在服务器，训练窗 2022-01~2026-08 = 813,145 局 |
| GRP | ✅ 已训 30,000 步（约 1.9 epoch）；⚠️ val loss 仍在下降，**未收敛** |
| 主模型 | ✅ 第 1 轮 200k 步；✅ 第 2 轮 424k（白名单+grp_h128）**+2.86 pt 显著变强**；❌ 第 3 轮 650k（同池第 2 个 epoch）**δ=−0.03±0.59 无增益** → 见 §16.13/§16.15 |
| 评测 | ❌ **当前评测体系失效**（对手是随机权重）→ 不可用 `test_play` 判断增益，也不可用 `best.pth` 选模型 |
| 性能 | ⚠️ 吞吐被 CPU 管道卡住（实得 2.56 it/s vs GPU 上限 8.88），改 2 个参数可获 2.4× 免费加速 |
| 工程风险 | ❌ 无 git 仓库、无异地备份、评测牌谱被覆盖；需立刻处理 |

### 16.2 环境与硬件实况（2026-10-09 复核）

| 项 | 实测值 |
|---|---|
| OS / 内核 | Ubuntu 18.04.2 容器（glibc 2.27），内核 6.8 |
| CPU | 2× Xeon E5-2690 v4，`nproc` = **56 线程** |
| 内存 | MemTotal 251 GB / MemAvailable 241 GB（无 cgroup 限制），Swap 8 GB |
| GPU | 5 张，**两套序号不一致**：nvidia-smi（PCI 序）0/1/3/4 = P100、**2 = RTX 8000**；而 CUDA 默认按算力排序 → ordinal 0 = RTX 8000、1~4 = P100。逐个实测见 [ENV_SETUP_P100.md §10](../tools/training/ENV_SETUP_P100.md)；**必须加 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 才能用 nvidia-smi 的序号** |
| conda 环境 | `mortal`：Python **3.12.11**、torch **2.6.0+cu118**（arch 含 sm_60/sm_75）、tqdm/toml/tensorboard/numpy/zstandard/requests |
| Rust | rustc / cargo **1.99.0**（未降级到 1.81，glibc 兼容无问题） |
| libriichi | `mortal/libriichi.so`（10-06 15:57），`__version__=0.1.0`、`__profile__=release`；`obs_shape(4)=(1012,34)`、`ACTION_SPACE=46` 已断言通过；`validate_logs`/`stat` 已在 `target/release/`（两者链接 libpython，跑时必须给 `LD_LIBRARY_PATH`）。⚠ 只能从 `mortal/` 目录导入；在仓库根目录会被同名 Rust crate 目录遮蔽成空命名空间包 |
| 磁盘 | `/mnt/data8` 7.3 TB，已用 3.5 TB / 可用 3.5 TB |
| 运行目录 | `/mnt/data8/mortal/Mortal/mortal`；看板：`tensorboard --logdir runs --host 127.0.0.1 --port 6006`（本机 SSH 隧道访问） |
| tmux | `train_grp`（已结束）、`train_main`（训练已正常退出，会话未关闭） |

**启动训练的实际环境变量（完整清单与逐个必要性实测见 [ENV_SETUP_P100.md §10](../tools/training/ENV_SETUP_P100.md)）**：

```bash
cd /mnt/data8/mortal/Mortal/mortal                     # ★ 必须在此目录（仓库根会被 libriichi/ 目录遮蔽导入）
export MORTAL_ROOT=/mnt/data8/mortal/Mortal            # 仅脚本便利，代码不读
export MORTAL_CFG="$MORTAL_ROOT/mortal/config.offline.toml"   # ★ 必需！不设在 10-09 实测报 FileNotFoundError: config.toml
export CUDA_DEVICE_ORDER=PCI_BUS_ID                    # ★ 必需（不加则序号与 nvidia-smi 不一致，见上表）
export CUDA_VISIBLE_DEVICES=2                          # RTX 8000（PCI 序下 2 才是 8000）；GRP 用 0/1/3/4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1             # 把 CPU 核让给特征编码
export RAYON_NUM_THREADS=8                             # ← 第 1 轮用的是 4（偏慢），新轮用 8
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # 训练可省；跑 validate_logs/stat 则必需（实测退出码 127）
```

> ⚠️ **对 §16.2 早期版本的修正**：上一版本给的 `CUDA_VISIBLE_DEVICES=2`（未配 `CUDA_DEVICE_ORDER`）实际会落到 **P100**而不是 RTX 8000；已逐个实测重写（见 ENV_SETUP_P100.md §10 的映射表）。第 1 轮主训日志确实是 `Quadro RTX 8000`，说明当时用的并非 2 这个值。

### 16.3 数据落地详情

| 项 | 值 |
|---|---|
| 全树 | `dataset/mjai/houou4/` = **2,630,640** 个 `.json.gz`，213 个月目录（200902~202610），**16 GB** |
| 来源 | ① HF 18 个年度包 `dataset/tenhou_hf/*.tar`（17 GB，已在服务器）→ `merge-hf` 展开；② 本机 2026 采集 130,407 局（与 MANUAL 记录的数一致） |
| 清洗 | 新增 `tenhou_pipeline.py prune`（逐文件验规则码 `00a9` + 四人 + 半庄 + 赤牌 + 首尾事件 + 占位名过滤），已 `--apply` |
| 纯度抽检 | 201001/201506/202003/202407/202609 五个月文件名规则码 **100% `00a9`** |
| 训练窗 | `202[2-5]??` + `20260[1-8]` → **813,145 局**（训练日志 `file list size` 一致）；2022/23/24/25/26 分年 = 167,627/168,777/184,424/178,887/130,407 |
| GRP 验证 | `202609` = 15,892 局（日志实测）；`202610` = 1,085 局（本月增量） |
| 单局量级 | 实测 **10.6 局/半庄**（GRP 特征行数）；决策样本 **678 个/文件**（4 家合计，n=8 实测，区间 480~1,109；与 dataloader 注释的 660 一致） |
| ⚠ names 字段差异 | **HF 部分（2009~2025，699,715 文件）的 `start_game.names` 存在；自建 2026 部分（113,430 文件）只有 `rules`、没有 `names`** → ① `player_names_files` 过滤天然只能作用于 HF 部分；② `train.py` 构建索引时读 `start['names']` 会 KeyError（已改为 `.get('names', [])`）；③ 玩家强度榜（§16.12）覆盖不到 2026 |
| 缓存 | `indexes/main_recent_v1.pth`（82 MB）、`indexes/grp_recent_v1.pth`（84 MB）→ **数据或 globs 变动必须删除重建** |
| 其他 | `dataset/raw` 1.2 GB（2026 自建 mjlog）、`dataset/index` 197 MB（CSV） |

### 16.4 远程代码补丁清单（上游无、需与本机文档 §4.x 对账）

四个文件已偏离上游，**只存服务器一份**（无 git，见 §14-14）：

| 文件 | mtime | 改动要点 |
|---|---|---|
| `mortal/train.py` | 10-08 06:10 | 新 `max_train_steps`（预算到点 `raise SystemExit(0)` 自停，L43/L124/L404）、新 `snapshot_every`（每 N 步另存 `checkpoints/mortal_step_%09d.pth`，L320）、启动时若已达预算直接返回 |
| `mortal/train_grp.py` | 10-08 03:51 | 新 `grp.control.max_train_steps`（L87/L251）、`grp_best.pth` 按 val loss 取最优（L232/L249）、`best_val_loss` 写入 state、验证集固定 generator seed `20261008`、日志加 `est. %`（按 10 局/文件估算 epoch 百分比）、每次启动强制 `lr = cfg.optim.lr` |
| `mortal/player.py` | 10-08 02:49 | `TestPlayer` 的 `enable_amp` / `enable_rule_based_agari_guard` 改从 `[test_play]` 配置读（`.get(...)`）；评测改为固定种子块 `seed_start=(10000, 0x2000)`（可复现但永不换牌山，见 §14-16） |
| `mortal/config.offline.toml` | 10-08 03:43 | 实跑配方（§16.5/§16.6）；`.bak` 为 GBK 乱码原件，已由 `fix_encoding.py` 转 UTF-8并修正 `val_globs` 缺失的 `Mortal/` 路径段 |
| 新增小工具 | 10-09 | `mortal/ladder.py`（checkpoint 间 1v3 配对对抗）、`mortal/bench_pipe.py`（数据管道吞吞吐基准）——本次复盘产物，不需要可删 |

> ⚠️ 表中“上游无”是按行为反推，**未做精确 `git diff`**（服务器当前无外网，`curl raw.githubusercontent.com` 超时）。下次有网时第一步：`git clone` 上游到临时目录后 `diff -u` 校准，并把补丁提交进版本控制。

### 16.5 GRP 训练结果（M1）

```
配方：device=cuda:0(P100) batch=512 lr=1e-4 hidden=64 layers=2 save_every=2000 val_steps=200
数据：train 813,145 文件 / val 15,892（202609）
时长：30,000 步 = 21 分钟（1.89 epoch；单 epoch ≈ 15,882 步）
曲线：loss train 2.700→2.468｜loss val 2.543→2.455（仍在降）｜acc(全排列) val 0.218→0.229
产物：grp.pth = grp_best.pth = grp_last.pth（均为 30,000 步，best_val_loss 2.45495）
```

关键提示：**TB 里的 `acc/val` 容易误读**。它是对**每一个 prefix（每局前缀）的全排列** argmax 命中率（含大量“开局几乎不可预测”的前缀），所以 22.9% 并不差。直接用 `tools/training/grp_quality_probe.py` 按**位置**评估（grp.pth，3 个月 ×150 局，共 450 局）：

| GRP | hit@1 | hit@3 | hit@6 | hit@9 | hit@末局 | mae@3 | mae@6 | nll@末局 | rew std/absmax |
|---|---|---|---|---|---|---|---|---|---|
| grp.pth（h64/l2, 30k 步） | 0.249 | 0.368 | 0.467 | 0.575 | 0.767 | 0.833 | 0.703 | 1.2780 | 0.700 / 4.80 |
| grp_v2.pth（h64/l2, 42k 步） | 0.249 | **0.379** | **0.476** | **0.597** | 0.764 | 0.831 | **0.701** | **1.2591** | 0.704 / 4.45 |
| **grp_h128.pth（h128/l3, 60k 步）** | 0.249 | 0.379 | **0.483** | 0.589 | **0.769** | 0.834 | 0.702 | **1.2361** | 0.696 / 4.76 |

> ⚠️ 本节早期版本曾写“末局命中 0.50~0.56”，那是探针脚本把**开局前缀混进了末局统计**导致的错数（已修正为上表）。同时注意：hit@1≈0.25 就是随机水平（4!=24 的均匀猜≈ 1/24，但前 4 名分位排序后 hit 自然抬到 0.25）；hit@末局高主要是“分数已定死顺位”，**真正决定奖励质量的是 hit@3 / hit@6 这两个中段值**。

→ 结论修正后的读法：GRP 在**中段只有 ≈ 37% / 48% 的精确顺位命中率（mae 0.70~0.83 个名次）**，这直接给 `delta_pt` 奖励注入大量噪声；而 30k→42k 步只有微小真实提升（nll −0.019）→ 平台期已至，下一步该动的是**容量**而不是步数（§16.12）。

> ✔ 容量实验已跑完（10-09）：h128/l3/60k 把 nll_last 推到 **1.2361**（−0.042 vs 第 1 轮）、val loss 2.44599，但 **hit@3 几乎未动、`delta_pt` std 仍≈ 0.70**。→ 根因是 **信息量上限**（GRP 输入只有 7 维/局，中段真实顺位本身就高度不确定），不是拟合能力。想拿到更好的奖励得改**目标形式**（真实点数差监督 / 期望点数回归），而不是继续改尺寸或步数。奖励模型已定档为 `grp_h128.pth`（详 MANUAL §15.3）。

> ⚠️ **2026-10-09 自我修正**：本节早期版本写的“val loss 仍在降、未收敛”**已被续训实验推翻**。从 30k 步暖启续训到 42k 步实测：val loss 只在 **2.4530~2.4566 来回波动**（best 2.45495 → 2.45207，仅 −0.003），acc_val 恒定≈ 0.229 → **30k 步就已是平台期，加步数没有收益**（详见 §16.12 的容量实验）。

### 16.6 主模型离线第一轮（M2）

```
配方（config.offline.toml 关键项）：v4 192×40（Brain 10,787,456 + DQN 48,175 + Aux 4,096 参）
  online=false batch=512 opt_step_every=1 save_every=1000 test_every=50000
  max_train_steps=200000 snapshot_every=50000 enable_amp=true enable_compile=false
  dataset: 813,145 局 file_batch_size=8 num_workers=2 reserve_ratio=0 augmentation=false
  env: gamma=1 pts=[6,4,2,0]｜cql.min_q_weight=5｜aux.next_rank_weight=0.2
  optim: AdamW wd=0.1 max_grad_norm=5.0｜scheduler warm_up=2000 peak=final=1e-4 max_steps=2000（即 warm-up 后恒定 1e-4）
  baseline.test/challenger state_file = checkpoints/baseline.pth（**随机初始化**，由内联脚本以 manual_seed(20261007) 生成，仅含 mortal/current_dqn/config 三键）
设备：Quadro RTX 8000（日志实测 `device: cuda:0 (Quadro RTX 8000)`），2.56 it/s
墙钟：10-08 06:11 → 10-09 05:02（训练 22.9 h + 4×21 min 评测）；`training budget reached` 正常退出
```

| 阶段 | dqn_loss | cql_loss | next_rank_loss | test_play avg_rank / avg_pt |
|---|---|---|---|---|
| 1k 步 | 0.467 | 1.261 | 1.324 | — |
| 50k | — | — | — | **1.0000 / 90.0**（存为 best.pth） |
| 100k | — | — | — | 1.0010 / 89.955 |
| 150k | — | — | — | 1.0005 / 89.978 |
| 200k | 0.392 | 0.506 | 0.756 | 1.0010 / 89.955 |

**样本覆盖**：200k 步 × 512 = 1.024 亿样本；一个 epoch = 813,145 × 678 / 512 ≈ **1,076,000 步** → **本轮只看了训练池的 18.6%**（≈ 15 万局，按文件计）。
**checkpoint**：`mortal.pth`(=200k 状态) + `mortal_step_{50k,100k,150k,200k}.pth` + `best.pth`（=50k）+ `baseline.pth`（随机）+ `grp*.pth`，共 795 MB。

### 16.7 评测失效诊断（本次最重要的结论）

**（a）为什么 `test_play` 完蛋了**：直接重算最后一次的 2,000 局 `runs/selection/*.json.gz`：

| 指标 | mortal（挑战者） | baseline（3× 随机权重） |
|---|---|---|
| 和牌次数（2000 局合计） | **10,729** | **126** |
| 平均顺位 / 平均分 | 1.001 / **72,417** | 3.000 / 10,116 |
| 1 位率 | **99.9%** | ≈ 0 |

随机权重网络输出的 Q 值无意义，几乎不会宣布和牌 → 模型包揽所有第一名。因此 `avg_rank`、`avg_pt`、`behavior/*`、`agari_point/*` 全部失真（例如“放铳平均 −7,664 点”“和牌率 56%”都是跟不会和牌的对手才能出现的数值）。**这组指标只能证明“训练没崩”，不能区分 50k 与 200k。**

**（b）可用的两类信号（10-09 实测）**：

1. **未见月离线指标**（拿 `202609` 做 holdout，主模型只训到 202608；12,288 样本）：

| checkpoint | 与人类动作 top-1 一致率 | 0.5·MSE(Q) | 目标方差 var_t | 预测方差 var_q | R² | 和了动作 top-1 |
|---|---|---|---|---|---|---|
| 随机 baseline | 0.2481 | 0.2339 | 0.4679 | 0.0001 | +0.000 | 136/161 |
| 50k | 0.7812 | 0.4636 | 0.4679 | 0.5745 | −0.982 | 161/161 |
| 100k | 0.7905 | 0.4397 | 0.4679 | 0.5125 | −0.880 | 159/161 |
| 150k | 0.7932 | 0.4313 | 0.4679 | 0.5074 | −0.844 | 160/161 |
| **200k** | **0.7952** | 0.4190 | 0.4679 | 0.4903 | **−0.791** | 161/161 |

→ 两个健在的进步信号：人类一致率单调上升但未到平台期（+0.4 pp/50k）；**Q 值幅度过高估**（var_q > var_t，R² 为负，且 MSE 仍差于“预测常数”，常数基准 0.2339 vs 实得 0.4190）—— CQL + 单样本 MC 目标下的典型校准缺失，适合当监控量而非当优化目标。

2. **同牌山配对对抗**（`mortal/ladder.py`，挑战者 vs 3×卫冕者，四座位轮换，600 半庄/组）：

| 组 | 挑战者 | 卫冕者 | avg_rank | avg_pt | 1 位率 |
|---|---|---|---|---|---|
| A | 200k | 50k | 2.5383 | **−5.18** | 25.8% |
| B | 150k | 50k | 2.4400 | **+4.20** | 28.0% |

噪声尺度：单局顺位点 std ≈ 88 → 600 局 SE ≈ **±3.6 pt**（即使同牌山配对使真实方差更小，仍至少 ±2）。A 的 −5.2 与 B 的 +4.2 方向相冲且均在 1.5 SE 内 → **结论：50k→100k→150k→200k 对局强度持平（本轮离线训练已无可用增益）。**

**（c）关键推论**：dqn_loss 还在降、人类一致率还在涨，但强度不涨 → 模型在“**更接近人类行为**”而非“**更能赢**”。在 `gamma=1`（只看当局 delta_pt）+ CQL 锚定行为策略的框架下，再多堆同分布步数只会提升“模仿精度”，而不会超越数据：**下一步必须换信号源（GRP 质量、数据选择、跨局视界）或进入在线自博弈**，而不是单纯把 200k 改成 1M。

### 16.8 行为画像（模型 vs 人类凤凰卓，统一解析口径：每局每家）

| 口径 | 人类 2024-06+2025-06（4,000 局） | 模型自博弈（600 局） | 读法 |
|---|---|---|---|
| 和了率 | 0.2117（ron 0.1255 / tsumo 0.0862） | 0.2228（ron 0.1405 / tsumo 0.0823） | 略偏攻 |
| 立直率 | 0.1769 | 0.1794 | **几乎一致** |
| 副露次数 | 0.4826 | 0.4566 | 略保守 |
| 流局率 | 15.3% | 10.9% | **弃和不足的最强信号** |

→ 整体节奏已与人类凤凰卓相当接近（这本身说明特征/数据/训练链路完全跑通了）；差异集中在“不太会退”：流局率低 4.4 pp、ron 放铳高 12%。离线 RL 看不到“应该弃和”的反事实，这是预期内的缺陷，也是下一轮可以针对的点（`cql.min_q_weight`、奖励平滑、或 oracle 特征）。
⚠️ 口径提醒：两张表来自不同的“对手世界”（4 人类 vs 4 同家族模型），只能看趋势，不能绝对对齐。

### 16.9 吞吞吐实测（10-09 基准）

| 配置 | 数据管道（不算 GPU） |
|---|---|
| num_workers=2, RAYON=4（**当前训练在用**） | 3.12 it/s（1,596 样本/s） |
| num_workers=4, RAYON=8 | 6.97 it/s |
| num_workers=8, RAYON=6 | 6.51 it/s |
| num_workers=8, RAYON=8 | **7.48 it/s** |
| GPU 上限（RTX 8000、bs512、AMP、合成张量 fwd+bwd） | **8.88 it/s**（112.6 ms/步） |

→ 实得 2.56 it/s ≈ 管道上限 3.12 而非 GPU 上限 8.88 → **纯 CPU 卡脖子**。改 `num_workers=4~8` + `RAYON_NUM_THREADS=8` 后预期实得 ≈ 6~7 it/s（**2.4×**）；换算：50k 步从 5.5 h → ≈ 2 h；一个全 epoch（≈ 1.05M 步）从 ≈ 4.8 天 → ≈ 2 天。内存完全不是问题（每 worker 缓冲 file_batch_size=8 约 0.7 GB，共 251 GB 可用），可进一步 `file_batch_size` 提到 15~20。→ 复现命令：`python bench_pipe.py <num_workers> <rayon>`

### 16.10 风险与卫生（需立即处理）

| 风险 | 实况 | 建议动作 |
|---|---|---|
| 代码只存一份 | ✔ **10-09 已建本地 git**：基线提交 `33e63bc`（`main`，460 文件/4.4 MB，tag `round1-complete`）；防泄漏校验已跑：`git ls-files -- dataset` = 0，`.json.gz/.pth/.so/.dll/.exe/.tar` 被跟踪数 = 0 | 待做：有网时加私有远端并 `git push`（当前无外网，推不了）；另已有 `backups/code_docs_20261009.tar.gz`（32 MB）作双保险 |
| checkpoint 只存一份 | ✔ **10-09 已处理**：`backups/ckpt_round1/`（592 MB）= grp / grp_h128 / grp_best / baseline / 四个 mortal_step 副本，已验证可加载。仍在同一磁盘（/mnt/data8），真正异地备份待做 | 下次拉到别的机器/云盘时带一份；`backups/` 已加进 `.gitignore` |
| 评测牌谱被覆写 | `runs/selection` 每次 `rmtree`，只剩最后一次 | 改为 `runs/selection/<step>` 归档（已含本次 200k 的 2,000 局，建议先备份再开新轮） |
| `.gitignore` 未覆盖大文件 | ✔ 已补齐（`/dataset`、`mortal/runs|logs|indexes`、`backups/`、`tools/tenhou_data/logs` 等；`*.pth`/`*.so` 官方已含） | 注意 **不要把 `memory/`、`tools/` 一并忽略**（这两处是本项目唯一的文档）；旧待办 #10 的原写法会误伤文档 |
| GRP 与主模型共用同一时间窗 | 两者都用 2022-01~2026-08 | 可接受（GRP 是奖励模型），但主模型 holdout 靠 202609+，新增数据时别泄露 |
| 合规 | 牌谱与 HF 加工数据仅本地使用 | 不入 git、不对外分发（§8.7） |

### 16.11 与本机文档待对账项（本远程文档无法自确认的部分）

1. `config.offline.toml` 注释里引用的“**需 §4.1 / §4.2 / §4.3 代码修改**”——远程 `TRAINING_MANUAL.md`（10-03 版）**没有这些章节**，应在本机更新后的手册里。需把本机 §4.x 与 §16.4 的实代码逐条对校（包括是否还有其他未记录的修改）。
2. 数据基线：手册头部写的仍是 **130,407 局 / 595.5 MiB**（本机 2026 部分），而服务器已是 **2,630,640 局 / 16 GB**；需确定本机文档是否已记录 HF 包上服 + `prune` 这一整套。
3. `TRAINING_MANUAL.md` §8/§11 的健康判据（`avg_rank < 2.5 并持续下降`）已实际被证伪（会直接饱和到 1.0），需按 §16.7 重写；本机文档如果已有更新版本，以本机版为准同步到远程。
4. 里程碑状态：M0-a/M0-b/M0-c/M1/M2/M3 均已完成（但 M3 “第一轮收敛”的结论是**平填且指标不可用**）；M4/M5 尚未开始。需与本机里程碑表合并。

### 16.12 本轮（10-09 复盘后）新增的可执行资产

按用户选定路线“**先修评测 + 吞吐 → GRP → 换信号源离线**”落地，全部工具为非破坏性（不动第 1 轮 checkpoint）。

**① 玩家强度榜（把“高段位过滤”从空想变成可用文件）**

牌谱里没有段位字段，但同一批人反复打可以自证强度：按最终得分排顺位，累计每位玩家的 局数 / 平均顺位 / 平均顺位点（[90,45,0,-135]）。工具：`tools/player_stats/scan_player_ratings.py`（+ `build_whitelist.sh` 一键跑、`whitelist_coverage.py` 定 top-N）。

实测（窗口 2022-01~2026-08，813,145 文件，28 进程 **77 秒**扫完，≈ 10.8k 文件/s）：

| 量 | 值 |
|---|---|
| 有 names 的文件 | 699,715（其余 113,430 是自建 2026，无 names） |
| 出现过的玩家 | 4,883 位；其中 **≥400 局的有 1,784 位** |
| 长期 avg_pt 阈值 | top100 = +5.08、top200 = +3.97、top400 = +2.41、top600 = +1.23（榜首 +12.87/430 局） |
| 自检 | 全体加权平均 avg_pt = **0.0000**、avg_rank = **2.5000**（每个半庄的顺位点必为一排列的和=0 → 数学自洽） |

选定 **top 400**（≥400 局且 avg_pt≥+2.41）作为第 2 轮白名单：`dataset/index/top_players_400.txt`。覆盖度实测（关键：`gameplay.rs::load_events` 只保留**名单内玩家自己座位**的决策，所以决定数据量的是“席位”而不是“命中文件”）：

| 名单 | 席位 | 样本≈ | 1 epoch 步数 |
|---|---|---|---|
| top200 | 324,484 | 55 M | 108 k |
| **top400** | **704,231** | **120 M** | **234 k** ← 已选 |
| top600 | 1,098,750 | 187 M | 365 k |
| top900 | 1,527,953 | 260 M | 507 k |

**② 评测工具**

- `mortal/holdout_metrics.py`：未见月（默认 202609）一次跑多个 checkpoint，输出 top1 一致率 / 0.5·MSE vs 常数基准 / R² / var_q÷var_t / 和了动作 top1。实测（12 batch×512）：baseline top1=0.2567，200k=**0.8187**（R² −0.645）。
- `mortal/ladder.py`：checkpoint 间同牌山 1v3 配对对抗（选模型/定胜负用）。
- `mortal/bench_pipe.py`：数据管道吞吐基准（不占 GPU）。
- `tools/training/grp_quality_probe.py`：GRP 真实质量探针（**分位置** hit@1/3/6/9/末 + mae + nll_last + rew std），替掉易误读的 TB acc；基线数值见 §16.5 表。用它发现了我自己一处旧口径错误（把开局前缀混进末局命中率）。

**③ 第 2 轮配置**：`mortal/config.round2.toml`（与 `config.offline.toml` 并存，第 1 轮产物全部保留），只动三处：num_workers 2→8 / file_batch_size 8→16（配 `RAYON_NUM_THREADS=8`）；`player_names_files=top_players_400.txt` + **新的 `file_index='indexes/main_r2_whitelist_v1.pth'`**（沿用旧缓存会跳过过滤！）；`baseline.test.state_file` 换成 `mortal_step_000200000.pth`（使 test_play 重新有区分度）。奖励模型用 `checkpoints/grp_v2.pth`（第 1 轮 `grp.pth` 保持可用）；暖启：开跑前把 `mortal.pth` 拷到 `checkpoints/round2/mortal.pth`，预算 `max_train_steps=450000`（即再 +250k 步 ≈ 白名单池的 1.07 epoch）。

**④ 代码修正（上游隐患）**：`mortal/train.py` 构建文件索引时读 `start['names']` → 对无 names 的自建 2026 文件会 **KeyError**（一旦启用 `player_names_files` 必崩）。已改为 `start.get('names', [])` 并注释说明。

**⑤ 实验与任务状态（10-09 均已结束，全部用 tmux 承载，日志落 `mortal/logs/`）**

| 会话 | 内容 | 结果 |
|---|---|---|
| `whitelist` / `toplist` | 玩家强度榜 + 覆盖度分析 | ✅ `dataset/index/player_ratings_2022_2026H1.csv`（4,883 人）、`top_players_400.txt`（400 人） |
| `grp_v2` | 加步数实验（30k→42k） | ❌ 已提前终止：val loss 在噪声带内，无收益；`grp_v2.pth` 保留 |
| `grp_cap` | 加容量实验 h64/l2 → h128/l3，从零 60k 步（P100 idx1，用时 30 min） | ✅ **胜出**，已定为第 2 轮奖励模型（`grp_h128.pth`，val loss 2.44599、nll_last 1.2361） |
| `cfg_check` / `smoke_r2` | 第 2 轮 config 预检 + GRP 重建冒烟 | ✅ 白名单保留 59.1%、索引重建 ≈1.3 分钟、obs/reward 形状正常 |
| `main_r2`（**14:17 启动，在跑**） | 第 2 轮主训练（用户亲手拉，已暖启自第 1 轮 200k） | ✅ 预检预测全部命中：`loaded 400 players` → `file list size: 480,414`（预估 480,365，误差 0.01%）；实测 **6.71 it/s**（预估 6~7，第 1 轮 2.56）；device=RTX 8000、9 个进程（nw=8）→ 预计 12 h 后（≈ 10-10 02:30）到 450k 预算 |
| `r2_watch` | 自动验收监控（P100 idx1，nice+低 worker，不抢训练资源） | ✅ 每个新 `mortal_step_*.pth` 自动跑 `holdout_metrics.py` 对比第 1 轮 200k；训练一退出就自动跑**双向 ladder（各 500 sets，不同种子块）**定胜负，结果追写到 `logs/r2_monitor.log` |

> 写监控脚本时的坑：`pgrep -f "python -u train.py"` 会连 tmux 里的 **bash -c 包装进程**（cmdline 含同样字符串）一起匹上，导致循环永不可退出；必须锚定成 `pgrep -c -f '^python -u train\.py$'`（实测正确匹到 9 个真进程）。

**⑥ 备份与一处副作用（10-09 14:28~14:35）**

- 已做：`backups/code_docs_20261009.tar.gz`（32 MB，不含 dataset/target/runs/logs/checkpoints）+ `backups/ckpt_round1/`（592 MB，8 个关键 checkpoint 副本，逐个 `torch.load` 验证可加载）；`backups/` 已加入 `.gitignore`。**git 已于 14:42 建立**（见下方 ⑧）。
- 副作用（**无损**）：`train_grp.py` 把最优盘硬编码写到 `dirname(state_file)/grp_best.pth`，所以容量实验**覆盖了第 1 轮的 `checkpoints/grp_best.pth`**。现在它是 h128/l3、60k 步、best_val 2.44599，与 `grp_h128.pth` 内容一致（探针数值逐位相同）；而旧 grp_best 本就与保留着的 `grp.pth` 完全相同 → 没丢信息。
- 记住这个坑：**不同尺寸的 GRP 实验会互盖 `grp_best.pth`**。历史存档只认各自专用的 `state_file`（已按此命名）+ 定期副本到 `backups/`。

**⑦ 本地 git 仓库已建立（10-09 14:42，用户明确授权）**

脚本：`tools/git_init_repo.sh`（幂等：已有 `.git` 则直接退出；含防泄漏与必要文件齐全性双重校验，任一不过就**不提交**）。结果：

| 项 | 值 |
|---|---|
| 分支 / 基线提交 | `main` / `33e63bc`（本机 git 2.17 不支持 `init -b`，用 `git symbolic-ref HEAD refs/heads/main` 改名） |
| tag | `round1-complete`（annotated，标第 1 轮完成态，便于 baseline 换代与回退） |
| 入库规模 | 460 个文件、共 4.0 MB；`.git` 4.4 MB |
| 防泄漏校验 | `git ls-files -- dataset` = **0**；`.json.gz/.pth/.so/.dll/.exe/.tar` 被跟踪数 = **0**；`git status --ignored` 确认 `dataset/ target/ backups/ mortal/{checkpoints,indexes,runs,logs,libriichi.so}` 均被忽略 |
| 提交身份 | 环境里本来没有任何 git 身份 → **没改你的配置**，仅用临时 `-c user.name='mortal-trainer' -c user.email='mortal@localhost'`。要换成自己的：`git -c user.name='你' -c user.email='你@x' commit --amend --reset-author`（并记得用同样方式重做 tag） |
| 尚未做 | **无远端也推不了**（今天 `curl raw.githubusercontent.com` 超时）；有网后：`git remote add origin <私有仓地址> && git push -u origin main && git push origin --tags` |

> 坑：`git tag -a`（annotated）**也需要身份**。第一次只给 `commit` 加了 `-c`，结果 commit 成功、tag 报 `fatal: unable to auto-detect email address`；脚本已修正（两条命令共用 `GITID` 数组），并用 `--amend --no-edit` 合并为一个干净的基线提交。

### 16.13 第 2 轮判定：【有效】（10-10，5000 sets = 20,000 半庄）

上一轮“500 sets 看不出差异”的结论在加大样本后被推翻 —— 不是没变强，是样本不够。

| 项 | 值 |
|---|---|
| 方法 | 双向同牌山配对。A = 新当挑战者（5 块 × 500 sets）；B = 旧当挑战者（5 块 × 500 sets，独立种子块）；δ = (pt_A − pt_B) / 2 |
| A 方向 | 10,000 半庄 avg_pt = **+3.299**（SE 0.820）；5 块全正：+3.62 / +3.89 / +2.00 / +4.43 / +2.54 |
| B 方向 | 10,000 半庄 avg_pt = **−2.430**（SE 0.856）；5 块全负（等价于新方 +2.43） |
| **结论** | **δ = +2.86 pt/半庄，SE 0.59，t = +4.83，95%CI [+1.70, +4.03] → 第 2 轮（白名单 top400 + grp_h128）统计显著有效** |
| 代价 | 20,000 局共 **29 分钟**（5 卡并行，每波 5×500 sets ≈ 14 分） |

两条可复用的经验：

1. **样本量要按 SE 倒推**：要分辨效应 δ，需 `sets ≥ 当前 sets × (2·SE/δ)²`（`tools/training/agg_ladder.py` 会自动算并打印）。500 sets 时 SE≈±1.9，完全定不了 ±1~3 pt 的胜负，而且小样本还会**低估**效应（本次：+0.70 → 加大后 +3.30）。
2. 实测分块波动（5 块标准差 ≈ 1.0 pt）**小于**按独立样本算的 SE（±1.9）→ 同牌山四座位轮换确实压方差；但汇总器仍按独立样本保守估计。

工具：`mortal/run_r2_ladder.sh`（5 卡两波并行，每任务独立日志目录）、`tools/training/agg_ladder.py`（分块明细 + δ + SE + t + 判读）。**20,000 局牌谱已全部保留**（`mortal/logs/ladder_r2/{A,B}_*/`，共 806 MB），后续要算行为统计或重配对分析无需重跑。

### 16.14 第 3 轮在跑（10-10 05:34 启动）

- 配方与第 2 轮**完全一致**（白名单 top400 + `grp_h128` + nw=8/fbs=16/RAYON=8），唯一实质变量是 `num_epochs = 2`（同一池 480,414 文件再过一遍）
- 暖启：`checkpoints/round3/mortal.pth ← round2/mortal.pth`（steps=424,000）；预算 `max_train_steps = 650000`（再 +226k 步）
- 路径：`mortal/config.round3.toml`（由 `tools/training/make_round3_cfg.sh` 从 round2 派生）、看板 `runs/main_r3`、评测 `runs/selection_r3`、日志 `logs/main_r3_train.log`、tmux 会话 `train_r3`（RTX 8000）
- 启动实锤：`loaded 400 players` → `file list size: 480,414` → `total steps: 424,000 (~26,000)` → `device: cuda:0 (Quadro RTX 8000)`
- ⚠ 定成败要看 **ladder vs round2 终态（424k）**，不能 vs 第 1 轮；且 epoch 2 是重复数据，holdout top1 上涨不能当变强证据

### 16.15 第 3 轮判定：【无增益】（10-11，5000 sets = 20,000 半庄）

第 3 轮于 10-10 15:56 **正常跑到预算 650,000 步自停**（无报错；10:4 h / 实测 6.7 it/s），产物 `checkpoints/round3/{mortal.pth, mortal_step_450k…650k}`。随后用与 §16.13 完全相同的双向配对方法判定（A = 650k 挑战 424k，B = 反向，各 5×500 sets，新种子块 60000+/70000+）：

| 方向 | 结果 |
|---|---|
| A（650k 当挑战者，10,000 局） | avg_pt = **−0.059**，SE 0.845；5 块 −2.99/+0.52/+1.58/−0.20/+0.81 |
| B（424k 当挑战者，10,000 局） | avg_pt = **−0.004**，SE 0.838 |
| **强度差** | **δ = −0.03 pt/半庄，SE 0.595，t = −0.05，95%CI [−1.19, +1.14]** |
| 结论 | **第 3 轮（同一白名单池的第 2 个 epoch）对强度毫无提升**；且这次不是“测不出”而是“测到了接近 0”——增益上界只有 +1.1 pt，与第 2 轮的 +2.86±0.59 形成鲜明对比 |

旁证（三个口径互相印证）：

1. **未见月 top1 不升反降**（202609，同批 12,288 样本）：424k **0.8122** → 450k 0.8114 → 550k 0.8110 → 650k 0.8091（第 1 轮 200k 为 0.8041）→ epoch 2 已在轻微过拟合这个池，不再泛化。
2. 损失几乎不动：dqn_loss 424k 0.374 → 650k 0.363；cql 0.458→0.452；next_rank 0.741→0.725。
3. 行为指标（agari/houjuu/riichi）在 450k→650k 均无方向性变化。

**方法论警告（本次踩到）**：用固定种子块的 `test_play`（或基于它的逐局配对差）做跨轮判定会**严重偏高**。同一批 500 sets（10000-10499）已被三轮训练反复用于选模型，本轮配对法给出 Δ=+4.52±2.07（t=2.19），而全新种子块的 ladder 给出 δ=−0.03±0.59 —— **差了 4.5 pt**，全部是牌山过拟合。定成败必须用**未参与过选模型的新种子块**。

两个待修的状态残留（10-11 处理进度）：

- ✅ **已修**：`best_perf` 会随 state 一路继承：`round2/round3` 的 `mortal.pth` 里仍是第 1 轮在随机对手时代刷出的 `{avg_rank:1.0, avg_pt:90.0}` → `better` 条件永远不成立 → **`best.pth` 从第 2 轮起再没写过**。现 `train.py` 实现自动重置：存盘多写 `best_perf_baseline`（当时的 `baseline.test.state_file`），加载时命中“手动 `control.reset_best_perf` / 旧 state 无该字段（来源不明，保守重置，正好修好现有盘）/ 卫冕者已更换”任一→重置为哨兵 `{4.0, −135.0}` 并记日志。四路回归测试全过（`tools/training/test_bestperf_reset.sh`），详见 MANUAL §16.6。
- 自写解析复算名次与 Rust `Stat` 在**平分 tie-break** 上不同（r3 差 0.006 名次、r2 差 0.023 名次 ≈ 1.4 pt），且 arena 导出的 `end_game` 不带 scores，只能累加 deltas → **判定一律以 ladder（Rust Stat）为准**，自算只作参考（工具 `tools/training/check_selection_stats.py` 可做这个交叉校验）。

### 16.16 当前结论与下一步（10-11）

- 已证实有效的只有**数据选择（白名单 top400）+ 更好的奖励模型**这一次跳跃（+2.86 pt）；**单纯同一池反复过数据无效**。
- GPU 已全部空闲（10-10 15:56 至今日），磁盘 3.5 TB 可用；`checkpoints` 2.2 GB、`logs/ladder_r3` 815 MB（牌谱保留可事后重算）。
- 下一轮候选（仍保持单变量）：① 收紧名单 top200（更强但数据更少，已备好强度榜 CSV 可直接切）；② `enable_augmentation=true`（样本×2 且不引入重复）；③ ~~`gamma<1` 做跨局信用分配~~ → **10-12 更正：gamma<1 只在当前局内打折，不是跨局信用分配（实测见 §14-31）；该位置改由"奖励口径对齐"补上（`env.reward_mode`/`pts` 可切换，见 §12 会话 16）**；④ 扩大时间窗到 2019+ 并同步重训 GRP。按 §16 的教训，**“多加一个 epoch”这类重复数据的做法应排最后**。

---

### 16.17 代码评审（10-11）与修复：4 个真 bug + 第 4 轮启动

对全部本地修改（train.py/train_grp.py/player.py/ladder.py/三份 config/11 个新工具）做了一次以正确性为主的评审（含对真实牌谱回放验证），发现并修复：

| # | 问题 | 严重度 | 修法/状态 |
|---|---|---|---|
| 1 | **三个工具的供托计分错**（`scan_player_ratings.py` / `check_selection_stats.py` / `paired_selection_diff.py`）：只取首局 scores 且从不扣立直宣言的 1000 → 每人分数高估 1000×立直次数，排名偏向高频立直者（实测**11~13% 半庄名次错**） | 会污染白名单 + 错误结论 | 已按 `stat.rs::from_game` 权威口径重写（每局 start_kyoku 重置 + reach_accepted −1000 + hora/ryukyoku deltas 含其余供托）；**交叉验证：runs/selection_r2/r3 复算与 Rust Stat 完全一致**（2.5055/+0.158 与 2.4560/+3.60）；旧值（2.5280/−1.215）已作废 |
| 2 | **白名单 v1 因此有偏**（阈值附近的 53/400 人进出，约 ±1800 人次）→ 已重建 v2（阈值 +2.45；命中 492,810 文件 / 735,249 席位 / 244k 步每遍） | 影响训练数据选择 | 已重建并存为 `_v1`/`_v2` 两套；第 4 轮用 v2 |
| 3 | round2/round3 的 `best.pth` 从未生成，而两份 config 的 `[1v3.challenger]` 指向它们 → 跑 one_vs_three 会 FileNotFoundError | 悬空引用 | 已各 cp 一份终态快照为 best.pth（round2←round2/mortal.pth、round3←round3/mortal.pth），引用不再悬空 |
| 4 | `holdout_metrics.py` 未传 `worker_init_fn`（多 worker 重复遍历全表）、`--seed` 形同虚设 | 次要 | 已修（传 worker_init_fn + 真随机抽样 + 印出 seed）；**注意：修后抽样变了，旧的 holdout 绝对值不能与新值跨版比较** |
| 5 | `test_bestperf_reset.sh` 的 grep 模式停在旧步数 425,000 | 次要 | 已改为 651,000 |

评审确认无误的：训练循环不丢最后一次 save（raise SystemExit 位于 save/snapshot/test 之后且预算整除 save_every）；best_perf 重置逻辑正确；三份 run 脚本的环境变量/并发/相对路径无问题。评审建议项（**均已实现**）：① `agg_ladder.py` 新增按 set 的 block bootstrap（`--logs-root` + `--bootstrap`，不假设局间独立，并给出组内相关粗估）——对 R3 的 20,000 局实测：**组内相关 ρ≈+0.18（同牌山 4 局确实正相关），但 bootstrap SE=0.515 < 解析 SE=0.595、CI 更窄 → 解析口径偏保守**，R3 的 δ≈0 结论不变；② `reset_best_perf` 改为**一次性令牌**（state 记录 `best_perf_reset_token`，同值不重复触发，换值才再生效），四路回归测试新增 D1/D2 验证（D1 生效并写入令牌、D2 同值 0 次重置）。另外三处重复的计分实现已收敛为单一模块 `tools/mjai_score.py`（消除复制）。

### 16.18 第 4 轮已启动（10-11 09:25，tmux `r4_chain`）

- **唯一实验变量**：`enable_augmentation = true` + `augmented_first = true`（席位视角增强；冒烟已验：同一文件增强前后**样本数不变、obs 张量确实不同**——×2 来自两遍遍历而非每遍翻倍）
- 基础设施同步升级：白名单换 v2、`file_index` 换新名（`main_r4_whitelist_v2.pth`，建于 72 s）、暖启自 `round2/mortal.pth`（424k，holdout top1 峰值）；预算 `max_train_steps=645000`（再训 221k 步 ≈ 0.91 遍增强样本）
- 启动实锤：预检全绿（env_check）→ 暖启拷贝 → **`resetting best_perf {1.0,90.0} → {4.0,-135.0}, reason: 旧 state 未记录当时的卫冕者`（新补丁首次实战生效）** → `device: cuda:0 (Quadro RTX 8000)` → `loaded 400 players` → `file list size: 492,810`（与覆盖度预估一字不差）→ `total steps: 424,000`
- 判定与训练在**同一会话链式执行**（`mortal/run_round4.sh`）：训练到点自停 → 双向各 5×500 sets（新种子块 80000+/90000+）→ `agg_ladder.py` 汇总；预计 ≈9.2 h + 29 min
- 监控：`tail -f mortal/logs/round4_chain.log`（链）与 `mortal/logs/main_r4_train.log`（训练）；看板 `runs/main_r4`

---

### 16.19 重启交接（10-11 14:50，计划内主机重启）

**停点与状态**：

| 项 | 值 |
|---|---|
| 第 4 轮训练 | **541,000 / 645,000 步**（已存盘：`round4/mortal.pth`，含完整 optimizer/scheduler/steps，验证可读） |
| best 记录 | `round4/best.pth` = 500k 评测态（avg_rank 2.4545 / avg_pt +4.5675，同批固定 500 sets；best_perf 与 baseline 标识已在 state 内） |
| 快照 | `mortal_step_000450000.pth`、`mortal_step_000500000.pth` |
| git | HEAD `3f830ab` + tag `round3-complete`；本交接文档将随下一小提交入库 |
| 停机方式 | SIGINT 优雅停训（train.py 捕获退出）；孤儿 DataLoader worker 已按 PID 清理；GPU 已释放（0–21 MiB）；`tb`/`r4_chain` 会话已关闭 |
| 备用恢复点 | 若 `mortal.pth` 意外损坏：`cp round4/best.pth round4/mortal.pth` 即可从 500k 续训（含完整优化器状态） |

**重启后一条命令恢复**（会自动：预检 → 直接续训（检测到 state 存在，跳过暖启拷贝）→ 到 645k 自停 → 双向判定（新种子 80000+/90000+）→ 汇总）：

```bash
cd /mnt/data8/mortal/Mortal/mortal
tmux new -d -s r4_resume 'bash run_round4.sh >> logs/round4_resume.log 2>&1'
tmux new -d -s tb "cd $PWD && export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib && /mnt/data8/miniconda3/envs/mortal/bin/python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006 > logs/tensorboard.log 2>&1"
```

预计：续训 104k 步 ≈ 4.4 h，判定 ≈ 29 min → **重启后约 5 h 出第 4 轮结论**（`logs/ladder_r4/` 与 `round4_chain`-style 汇总日志）。训练日志用追加模式，历史 09:25–14:44 的记录保留在同文件上半段。

> ✅ **已恢复（10-11 15:49，重启后 ≈41 min）**：`r4_resume` 会话按本单启动，实锤——env_check 全过 → 提示“round4/mortal.pth 已存在，直接续训”（跳过暖启拷贝）→ `loaded: 2026-10-11 14:42:21` → `device: cuda:0 (Quadro RTX 8000)` → `total steps: 541,000`；**无 `resetting best_perf` 行（baseline 标识匹配，best 记录 {2.4545, +4.5675} 正确保留）**；训练日志追加标记行已见“14:44 幽灵进程”之后，历史完整。看板 `tb` 已重启（HTTP 200）。预计 20:15 UTC 前后到 645k，20:45 前后出判定结论。

**本次真踩的三个坑（已写进 §14 坑位索引）**：

1. **运行中的 bash 脚本不能编辑**：「按字节偏移增量读取」→ 我 patch `run_round4.sh` 后它从错位处重读，把 `$PY -u train.py` 又执行了一遍（幽灵进程 247719 跑了 50 s，被后续清理 SIGTERM 掉；未到存盘点、未污染任何产物）。以后要么先停会话再改脚本，要么用独立的新脚本文件。
2. `pkill -f "python -u train.py"` 会匹配到执行它的 shell 自己（命令行里含同串）→ 输出被截断；改用 `pgrep -f "[t]rain.py"` 拿 PID 再 kill。
3. SIGINT 停训后 DataLoader worker 会变**孤儿进程**（父进程退出不回收到它们），需再按 PID 补一刀；确认 `pgrep` 为 0 再收工。

> ⚠ 后续：按本单恢复后 ≈3 分钟即触发 **NaN 崩坏事故**，复盘与三重防护见 §16.20。

---

### 16.20 第 4 轮 NaN 崩坏事故复盘与三重防护（10-11 15:49–16:21）

**起因**（也正是"TB 上 r3/r4 的 loss 为什么看着杂乱无章"）：三层原因，第三层是事故。

| 层 | 原因 | 实测 |
|---|---|---|
| ① 显示口径 | 各轮 step 是**暖启延续**的：r2 画 201k–424k、r3 画 **425k–650k**、r4 又从 **425k** 起 → 同一段 x 轴叠着两条互不对应的曲线 | 三轮 dqn_loss 全在 0.356~0.384 窄带内 |
| ② 噪声本底 | `loss/*` 是每 1000 步一个**窗口均值**，目标是单样本 MC 回报 → 抖动与轮次间真实差异同量级 | 逐点 mean\|Δ\|≈0.0033、max\|Δ\|≈0.0115 |
| ③ **NaN 事故** | 542k 步起 dqn/cql/next_rank 全 NaN，TB 表现为断线掉落 | 见下方时间线 |

**事故时间线（UTC）**：

| 时刻 | 事件 |
|---|---|
| 14:42 | 重启前最后存盘 541,000 步，dqn_loss=0.3706（有限）✅ |
| 15:49 | 按 §16.19 交接单恢复：预检全过 → 跳过暖启 → 从 541k 续训 |
| 15:53 | **542,000 步起 loss 全 NaN**（恢复后头 1000 步内发作；重启前那 117k 步一直健康） |
| ~16:00 | `add_histogram` 遇全 NaN 抛 ValueError → 训练进程崩；链未校验退出码，自动进判定 |
| 16:10–16:12 | NaN 模型判定跑出 avg_rank **3.92** / avg_pt **−125**（局局第 4 名），已掐停 |
| 16:13 | 扫描定污染范围：Brain 572 个非有限张量（含 162 个 BN buffer）+ DQN/Aux 权重 + 优化器 820 个 |

**根因**：AMP(fp16) 偶发溢出 + **BN running stats 在前向即被污染**（GradScaler 只跳参数更新、救不了 buffer）→ 前向永久 NaN → 权重与优化器随后全坏。触发与**数据顺序**有关（重启后洗牌变了），所以"之前几十万步都没事"不等于安全。

**损失**：41k 步（541k→500k，≈1.7 h）+ 一次被浪费的判定（5 卡 ≈12 min）。**能救回来靠的是 `snapshot_every=50000`**：`best.pth`(500k) 与 `mortal_step_000500000.pth` 扫描均 0 非有限张量 ✅。

**已实施的三重防护**：

1. **finite 守卫**（`train.py` 存盘段）：三个模型 state_dict 与本周期 loss 累加值若非有限 → `logging.error` 并 `SystemExit(1)`，**绝不覆盖 checkpoint**（这次正是被 NaN 反复覆盖了 `mortal.pth`）。自检：坏 state 抓到 572 个，两个干净快照放行 0 个。
2. **histogram 守卫**（同文件）：先过滤非有限值再 `add_histogram`，为空则跳过 → NaN 不再打崩训练进程。
3. **退出码守卫**（`run_round4.sh`）：`train_code=$?` 非 0 → 打印回滚指引并退出，**不带坏模型进判定**。

**恢复与当前状态**：`cp checkpoints/round4/best.pth checkpoints/round4/mortal.pth`（500k，含 500k 评测后的 best_perf 与 baseline 标识）→ 16:21 重启链条（会话 `r4_fix`），实况 `loaded: 12:47:49` → `total steps: 500,000` → GPU2 94%、4.7 GB。按决策 **`enable_amp = false`**（fp32）；剩 145k 步，实测吞吐出来后再更新 ETA（见 MANUAL §16.10 末）。取证文件与 NaN 判定残留目录已清理。

> **TB 看图正确姿势**：① 左侧只勾一个 run（或分轮看）；② 记住 step 跨轮延续，**同 x 不一定同义**；③ 要看趋势就调大 smoothing，或直接看 50k 一次的 `test_play/avg_pt` 与 `holdout_metrics.py` 的 top1（比逐 1000 步的 loss 曲线可靠得多）。

> 后续：链条于 10-12 02:11 到预算（退出码 0，守卫放行）→ 02:39 判定完成，终局结论见 §16.21。

---

### 16.21 第 4 轮终局判定（10-12 02:39）：augmentation 无增益

**链条收尾情况**：训练 02:11:04 `training budget reached` → **退出码 0**（新加的退出码守卫据此才放行判定）→ 双向各 5×500 sets（新种子 80000+/90000+）→ 02:39 `ALL DONE`。终态 `round4/mortal.pth` = 645,000 步、**0 个非有限张量**、`best_perf={2.4545, +4.5675}` 与 `best_perf_baseline` 均在位；`best.pth` 仍停在 500k（后面三次评测没超过它，属正常）。

**判定结果（round4/645k vs 其父 round2/424k）**：

| 方法 | δ (pt/半庄) | SE | 95%CI | 判读 |
|---|---|---|---|---|
| 解析（按局独立） | **+0.097** | 0.593 | [−1.07, +1.26] | \|t\|=0.16 → 无结论 |
| **block bootstrap（按 set）** | **+0.097** | **0.508** | **[−0.90, +1.07]** | 无结论；再次证实解析法偏保守（r3 也如此：0.595 vs 0.515） |

组内相关 ρ≈+0.185/0.188，与 r3 的 0.184/0.189 一致（工具在新数据上二次验证通过）。旁证：未见月 202609 的 top1 由 round2/424k 的 0.8052 → r4/600k 的 0.8112（12 批小样本，±0.5 pt 内，只算方向为正）。

**三轮累计证据**：

| 轮 | 变量 | δ vs 父代 | 结论 |
|---|---|---|---|
| R2 | 白名单 top400 + GRP h128/l3 | **+2.86 ± 0.59**（t=4.83） | ✅ 有效 |
| R3 | 同池再跑一个 epoch | −0.03 ± 0.59 | ❌ 无效 |
| R4 | 席位视角增强 +145k 步新样本 | +0.10 ± 0.59 | ❌ 无效 |

→ **“增加同分布样本量”这条路已被两条独立尝试否掉**（重复 epoch、视角旋转增强都不产生新信息）。第 5 轮必须换**信息源**：收紧名单（top200，席位 334,907→1 epoch 111k 步）、跨局信用分配（gamma<1）、扩大时间窗到 2019+ 并同步重训 GRP，三者按此顺序试。

**运维侧要点（本轮实测）**：fp32（`enable_amp=false`）稳态 **4.24 it/s**，比 AMP 的 6.7 it/s 慢 36%；GPU 显存 2.7→5.2 GB、利用率 91–94%（瓶颈已从 CPU 管道转到 GPU）。三重防护本轮全程未触发（16:21 之后 0 条 non-finite）。TB 的 `runs/main_r4` 已把 NaN 段与空幽灵段的事件文件移到 `logs/tb_archive/main_r4_nan_and_ghost/`（保留取证），因此该 run 现在只剩 AMP 段 + fp32 段，500k–541k 区间每个 step 有两个合法点（两次经过），别当成抖动。

**判定分块明细**（挑战者侧 avg_pt，每块 500 sets = 2000 半庄；A=新当挑战者，B=旧当挑战者）：

| 块 | seed | 1st/2nd/3rd/4th | avg_pt | 块 | seed | 1st/2nd/3rd/4th | avg_pt |
|---|---|---|---|---|---|---|---|
| A_80000 | 80000 | [530,479,481,510] | +0.20 | B_90000 | 90000 | [490,508,519,483] | +0.88 |
| A_80500 | 80500 | [526,477,491,506] | +0.25 | B_90500 | 90500 | [501,519,494,486] | +1.42 |
| A_81000 | 81000 | [518,496,493,493] | +1.19 | B_91000 | 91000 | [483,522,500,495] | +0.07 |
| A_81500 | 81500 | [512,485,496,507] | −0.27 | B_91500 | 91500 | [472,541,492,495] | +0.00 |
| A_82000 | 82000 | [483,527,514,476] | +1.46 | B_92000 | 92000 | [474,521,508,497] | −0.49 |

A 侧合计 +0.567（SE 0.844）、B 侧合计 +0.373（SE 0.835）——两个方向都近乎对称地散在 0 附近，是“等强度”的典型形态（对比 R2 判定当时 A +2.9 / B −2.9 的反对称形态）。

**本轮终态资产**：`checkpoints/round4/` = `mortal.pth`(645k) + `best.pth`(500k) + `step_450k/500k/550k/600k`；判定牌谱 `logs/ladder_r4/`（10 个任务、20,000 局、807 MB）；曲线 `runs/main_r4`；TB 取证归档 `logs/tb_archive/main_r4_nan_and_ghost/`（NaN 段 + 空幽灵段事件文件，72 KB）。

---

### 16.22 第 5 轮方案（已定稿待批准，未开工）

**唯一变量 = 收紧名单到 top200**，其余全部回到 R2/R4 已验证的配置。理由：R3/R4 已把“增加同分布样本量”这条路否掉，剩下未试的增益来源只有**换信息源**，而名单收紧是与 R2 的 +2.86 同源（数据质量）、且成本最低的一条。

| 项 | 值 | 说明 |
|---|---|---|
| 名单 | `dataset/index/top_players_200_v2.txt`（阈值 avg_pt ≥ +3.9，**v2 口径**） | 由 `build_whitelist.sh` 改 `--top 200` 产出，尚未生成 |
| 数据量（v2 实测） | 命中 279,713 文件（34.4%）/ 席位 **334,907** / 样本 ≈57 M / **1 epoch ≈ 111k 步** | 席位才是决定量（`load_events` 只保留名单内座位） |
| augmentation | **改回 false** | R4 实测无增益，不带着无用变量进下一轮 |
| 暖启 | `checkpoints/round4/mortal.pth`（645k，0 非有限张量） | 现役最强；与 424k/650k 统计等价，选它因未见月 top1 最高 |
| 预算 | `max_train_steps = 756000`（645k + 111k = 恰好一整遍新池） | fp32 ≈ 7.3 h；若重开 AMP ≈ 4.6 h |
| 索引 | 新名 `indexes/main_r5_whitelist_top200_v2.pth` | **名单变了必须换名**（§14 坑位） |
| 卫冕者/判定 | `[baseline.test]` 仍用 round1-200k（趋势连续）；ladder **vs round4/645k**，新种子块 **100000+/110000+**，双向各 5×500 sets | 判进 δ > 0 且超 ±0.6 pt |
| 链脚本 | 待创建 `run_round5.sh`（沿用 R4 的 env 重置 + 三重防护 + 退出码守卫） | 配 `make_round5_cfg.sh` 派生 |

**两个待你拍板的选项**：

1. **AMP 开还是关**？关掉（当前）稳但 4.24 it/s；开 AMP 6.7 it/s 省 40% 时间，风险已被 finite 守卫框住（最坏 ≤1000 步 + 一次明确停等；这条谱系累计 ≈77 万步只出过 1 次 NaN）。
2. **要不要并行**？4 张 P100 全闲。可同时跑第二路线 `env.gamma = 0.995`（跨局信用分配）做对照，各自独立判定；代价是 CPU 争用让双方慢约 20–30%。

若不批准 top200，后备顺序：② ~~gamma<1 跨局信用~~ → **奖励口径对齐（`env.reward_mode='delta_points'` 或 `pts=[2,1,0,-3]`；10-12 已实现，见 §12 会话 16）** → ③ 扩窗 2019+ 并同步重训 GRP → ④ GRP 目标形式改造（真实点数差监督，当前 7 维输入已到信息量上限）。

---

### 16.23 第 5 轮终局判定（10-13 14:28）：top200 收紧名单 = 无增益（方向偏负）

> 详情与分块明细见 §12 会话 20。本节只归档结论与更新后的强度账。

**实际执行与 §16.22 方案的差异**（开工前定稿时未预见）：名单从 v2 升到 **v3**（含 names 修复后的 2026 玩家）；augmentation **仍为 true**（与 B 组保持一致，未改回 false）；奖励口径改为 **B 组 [2,1,0,-3]**（而非 top200 独立变量）；暖启父代为 **B 组/756k**（而非 round4/645k）。因此 R5 = **以 B 组为父代、唯一变量 = top400_v2→top200_v3 的单变量对照**（B 组即天然控制组）。

**判定（R5/870k vs B 组/756k，新种子 120000+/130000+，双向各 5×500 sets）**：δ = **−0.605 pt/半庄**，解析 SE 0.594、t=−1.02、95%CI[−1.77,+0.56]；**block bootstrap SE 0.508、95%CI[−1.61,+0.36]、ρ≈+0.185** → **无结论、方向偏负**（未出现 A 全负/B 全正的反对称形态，不能定确证变弱）。

**四轮+消融累计证据（强度账、全部≥5000 sets 双向同牌山）**：

| 轮 | 变量 | δ vs 父代 | 结论 |
|---|---|---|---|
| R2 | 白名单 top400 + GRP h128/l3 | **+2.86 ± 0.59**（t=4.83） | ✅ 有效 |
| R3 | 同池再跑一个 epoch | −0.03 ± 0.59 | ❌ 无效 |
| R4 | 席位视角增强 +145k 步 | +0.10 ± 0.59 | ❌ 无效 |
| 消融 B | 奖励口径 top400→[2,1,0,-3] | +0.18（bootstrap CI[−0.85,+1.19]） | ❌ 无结论（方向正向，已采纳为基线） |
| **R5** | **top400_v2 → top200_v3 收紧名单** | **−0.61 ± 0.59**（bootstrap CI[−1.61,+0.36]） | ❌ **无结论、方向偏负** |

→ **“换信息源”里的收紧名单候选已死**。连同 R3/R4/消融 A，“同分布层面挤数据”（重复 epoch / 视角增强 / 换名单子集 / 换优化器）已连续 **5 次**否掉 → 数据 reshuffle 路线到头。下一轮按分析文档 §4/§6 优先级推进到 **② IQL/Bellman（带 next-state + target network）**；若仍无增益再考虑 **③ 扩窗 2019+ + 重训 GRP**。“收紧名单”从候选删除。现役最强仍为 `checkpoints/round4/mortal.pth`（645k）。

**本轮终态资产**：`checkpoints/round5/` = `mortal.pth`(870k) + `best.pth`(850k) + `step_800k/850k`；判定牌谱 `logs/ladder_r5/`（10 个任务、双向共 20,000 局）；汇总链 `logs/round5_chain.log`（含分块明细与 bootstrap）；曲线 `runs/main_r5`。

---

### 16.24 hp_ablation 便宜前置实验（超参消融，10-13~14）：cql/aux 均无增益

> 详情与分块见 §12 会话 21。本节只归档结论。

**目的**：按 §3.3 顺序 2/3，在进入 IQL/Bellman 大改前先验证"是否只是超参没扫对"。三条分支从 round4/645k 同 B 配方暖启、只改一个旋钮（cql=1/cql=10/aux=0），控制组=B 本身。

**判定（vs round4/645k，新种子 140000/150000，双向 ≥5000 sets）**：

| 分支 | 变量 | δ | bootstrap 95%CI | 判读 |
|---|---|---|---|---|
| aux_next0 | aux 0.2→0 | **+0.59** | [−0.42,+1.61] | 无结论 |
| cql_min1 | cql 5→1 | **+0.05** | [−0.97,+1.06] | 无结论 |
| cql_min10 | cql 5→10 | **−0.07** | [−1.09,+0.99] | 无结论 |
| B（控制） | cql5/aux0.2 | +0.18 | 种子 100000/110000 | 无结论 |

→ **超参（cql.min_q_weight / aux.next_rank_weight）不是瓶颈**，排除"只是超参没扫对"的替代解释→ **直接支持转 ② IQL/Bellman（目标形式改造）**。现役最强仍为 `round4/mortal.pth`（645k），无新分支值得采纳。

**运维插曲**：GPU1 持续负载下热降频（SM 1240→607 MHz、4.3→1.2 batch/s）拖住整条链 → kill 后原地续训到凉卡（cql_min10→GPU2 RTX）。已沉淀为坑位记忆“GPU 持续负载下热降频”。

---

### 16.25 IQL/Bellman Stage A（跨局信用分配）终局判定（10-14 17:16）：无增益、方向偏负

> 详情与分块明细见 §12 会话 22。本节只归档结论与更新后的强度账。

**方案（相对原计划的差异）**：原拟“先跨局信用、再真 IQL”两步走；Stage A 实际采用 **GRP 作控制变量的折扣后缀和**（`env.credit_mode='game'`，γ=1 闭式 `pts[final_rank] − E_grp[pts|第 k 局开始]`），因此**零 GPU 成本**（实测 4.52 batch/s，与 B 持平）且无需 next-state/target net——它是 Bellman 思想的廉价近似，而不是真 IQL。

**判定（IQL-A/867k vs B 组/756k，新种子 160000/170000，双向各 5×500 sets）**：δ = **−1.028 pt/半庄**，解析 SE 0.595、t=−1.73、95%CI[−2.20,+0.14]（跨 0）；**block bootstrap SE 0.517、95%CI=[−2.06,−0.02]、ρ≈+0.19** → CI 不跨 0（工具判“变弱”），但上界仅 −0.02 属**边际显著**，保守表述为“无增益、方向偏负”。两向同号（新做挑战者 −1.435；旧做挑战者 +0.621），无 C 组式反对称伪影。

**累计强度账（全部 ≥5000 sets 双向同牌山）**：

| 轮 | 变量 | δ vs 父代 | 结论 |
|---|---|---|---|
| R2 | 白名单 top400 + GRP h128/l3 | **+2.86 ± 0.59**（t=4.83） | ✅ 有效 |
| R3 | 同池再跑一个 epoch | −0.03 ± 0.59 | ❌ 无效 |
| R4 | 席位视角增强 +145k 步 | +0.10 ± 0.59 | ❌ 无效 |
| 消融 B | 奖励口径 →[2,1,0,−3] | +0.18（bootstrap CI[−0.85,+1.19]） | ❌ 无结论（已采纳为基线） |
| 消融 C | 真实分数差奖励 | −5.61（bootstrap CI[−6.67,−4.55]） | ❌ 确证变弱，路线关 |
| R5 | top400_v2 → top200_v3 收紧名单 | −0.61（bootstrap CI[−1.61,+0.36]） | ❌ 无结论、方向偏负 |
| hp_ablation | cql 1/10、aux 0 | +0.05 / −0.07 / +0.59（均跨 0） | ❌ 超参非瓶颈 |
| **IQL-A** | **credit_mode: kyoku → game（跨局信用）** | **−1.03 ± 0.60（bootstrap CI[−2.06,−0.02]）** | ❌ **无增益、方向偏负 → MC 回传路线关** |

→ **目标形式改造的第一条子路线（把整场回报用 GRP 控制变量回传）已否**。剩下的未试于段是**价值传播 + 政策改进**本身：Stage B 真 IQL（`next_obs` + terminal 标志、`VHead`、EMA target 网络、`L_Q + L_V(τ-expectile) + L_AWR`、CQL 关）。吞吐预估因每步 3 次 encoder 前向降至 ~1.7-2 it/s（+111k 步 ≈ 18-22 h）；判定种子块预 180000/190000。Stage B 仍无增益 → ③ 扩窗 2019+（需 GRP 重训 + 按年份审计）。现役最强仍 `checkpoints/round4/mortal.pth`（645k），基线口径仍 B/756k。

**本轮终态资产**：`checkpoints/iql_a/` = `mortal.pth`(867k) + `best.pth`(800k) + `step_800k/850k`；判定牌谱 `mortal/iql_a/ladder/`（10 任务、双向 20,000 局）；汇总 `mortal/iql_a/ladder_summary.log`；链日志 `mortal/logs/iql_a_chain.log`（含 ALL DONE）；曲线 `runs/iql_a`。

---

### 16.26 IQL Stage B 首轮（10-15 15:06）：结果无效——两处实现偏差（r_t 双重计入 + L_V 误用全体动作），已修待重跑

> 详情见 §12 会话 23。本节只归档结论（强度账不把该行当作 IQL 路线的判决）。

**方案**：`control.objective='iql'` 替换整套目标——next_obs bootstrap（`L_Q = MSE(Q(s,a), r+γV(s'))`）+ expectile V（`τ=0.7`，Q̄ 用 EMA target 网络，decay 0.995）+ AWR 优势加权 CE（β=3.0），CQL 关闭；γ=0.997 为每决策步折扣。其余与 B 组锁死。

**首轮实测（vs B/756k，种子 180000/190000，双向各 5×500 sets）**：δ = **−31.04 pt/半庄**（t=−54.5，bootstrap CI[−32.13,−29.98]）；holdout top1 0.6917（父代 0.811）、logsumexp_q −176.1；训练 11.5 h / 2.69 batch/s / 无 non-finite。

**根因（确定性，已实测）**：计划要求 r_t 仅局边界非零，但首版 `dataloader.py` 把 `kyoku_rewards[at_kyoku[i]]` 发给了该局**每个**样本 → Bellman 沿链重复累加。`tools/training/diag_iql_reward_double_count.py` 实测：3 文件（12 game / 144 局 / 2304 样本）非零 r_t = **2304 条 vs 应有 141 条**，放大倍数恰 = **每局平均决策点数 16.00×** → Q 目标尺度爆到 −176 量级，AWR/argmax 全被污染。**所以 −31.04 度量的是“重复计入 16 倍奖励”的坏目标，不是 IQL 本身。**

**两道防线为何都没拦住**（已分别沉淀 §14-35/§14-36）：① 回归测试 C 段断言了“实现现状”（iql 的 r_t == kyoku 逐样本值）而非设计要求 → 给 bug 盖章；② 冒烟只查结构性项目（无 non-finite、到预算、test_play **有输出**），而同一份冒烟日志里 test_play 已是 avg_rank 2.789/avg_pt −19.34（父代 2.466/+3.06）→ 劣化信号没当门禁。

**第二处偏差（由新增的行为门禁当场抓出）**：`r_t` 修正后的 v2 冒烟仍 **SMOKE FAIL**（avg_rank 2.840 / avg_pt −26.54；top1 0.6975；logsumexp_q **−33.59**——比 v1 的 −176.1 小 5 倍，印证 r_t 修正有效，但仍异常）。根因：**L_V 的 expectile 应只对「数据里实际执行的动作」的 Q̄(s,a) 回归（IQL 原文式 9），却误对全部合法动作（~30 个从未被监督的 Q 值）取 τ 上分位** → V 被无约束的 Q 外推拖走，再经 V(s') 反馈进 Q 目标自激漂移。已改为 `q_bar = q_bar_all[range(bs), actions]`。**若无这两条门禁，第二个缺陷会再烧掉一整轮 11.5 h。**

**修复与重跑准备（10-15 同日）**：`r_t = kyoku_rewards[at_kyoku[i]] if dones[i] else 0.0`；`L_V` 改为只对数据动作的 Q̄(s,a) 做 expectile；`test_iql_data.py` 改为断言设计意图（非零个数==局边界数 141、零值==非边界 2163、总和==逐局总和）→ 八层 ALL PASS（幅度兜底断言 10-15 当晚又发现上界写错，已修，见 §14-38）；`run_iql_b_smoke.sh` 新增行为门禁⑥ avg_rank ≤ 2.60、⑦ top1 ≥ 0.78 且 |logsumexp_q| ≤ 10（FAIL 禁开正式轮，已证实有效）；命名空间递增隔离：v1 `checkpoints/iql_b/`(867k) + `iql_b_smoke/`、v2 `iql_b_smoke_v2/` 均含受损权重 → **封存不续用**，v3 为两缺陷都修后的验证轮；重跑正式轮用全新命名空间 + 新种子块 **200000/210000**。

**基线不变**：现役最强 `checkpoints/round4/mortal.pth`（645k），判定基线 B/756k；目标形式改造的两种子路线（Stage A 跨局信用分配 = 已否；Stage B 真 IQL = 两处缺陷已修、v3 冒烟 10/10 通过，待正式轮）；若 v2 正式轮仍无增益 → ③ 扩窗 2019+（需 GRP 重训 + 按年份审计）。

**v3 冒烟已全过（10-15 16:03）**：avg_rank 2.527 / avg_pt −1.58、top1 **0.8134**（高于父代 0.8112）、logsumexp_q **+1.343**、无 non-finite、吞吐 2.93 batch/s；三代对比 avg_rank/logsumexp_q = 2.789/−176.1 → 2.840/−33.6 → 2.527/+1.34，两缺陷各贡献约一个数量级尺度异常，修正链与诊断自洽。**真 IQL 至此才处于可被有效检验的状态。**

**v2 正式轮已启动（10-15 16:24 UTC，会话 24）**：启动前对 dataloader/train/model/holdout 全链路复审确认两处修复无第三种偏差，另修掉测试幅度断言的上界错误（§14-38）；`config.iql_b_v2.toml` + `run_iql_b_v2.sh`（全新命名空间 `iql_b_v2`，种子块 200000/210000，索引复用 `iql_b_r4_v2.pth`），启动即核全绿（暖启告警恰 1 条、吞吐 2.93~2.95 batch/s、三损失初值 q=0.128/v=0.093/awr=0.411、无 non-finite）。预计 ≈10.5 h 训练 + ≈1 h 判定；结论待训完后补记。

→ **后续（10-16~17）**：v2 已训完判定（δ=−3.42 显著变弱但属漂移污染、不作路线判决，会话 25/26），v3（τ=0.5 干净轮）已训完判定（δ=+0.646 无结论 → IQL 路线关闭）——完整结构、数据与归因见 §16.27 与会话 28。

### 16.27 IQL Stage B v3 终局判定（10-16 18:29 UTC ALL DONE）：真 IQL 首次干净检验 = 无结论，IQL 路线关闭

> 详情见 §12 会话 28。本节只归档结论。

v3 = v2 + **τ=0.5**（expectile 对称化；5 臂探针唯一验证的反漂移修复，§14-39），其余与 B 组锁死；种子块 220000/230000、bseed 20261016、命名空间 `iql_b_v3`。训练 867k exit=0、无 non-finite；**价值尺度全程稳定**（max q_target p50=1.206，v2 同点 +60.8，理论界 [−5,+5]）；holdout top1 0.8167（历轮最高）/ lse +1.596。双向 ladder 各 10,000 半庄：**δ=+0.646 pt/半庄，block bootstrap SE 0.509、95%CI[−0.35,+1.63] → 无统计结论**。按预注册守则 → **IQL 路线关闭**（与 Stage A δ=−1.03 合并：目标形式改造两子路线均无可采纳增益），转 ③ 扩窗 2019+ + 重训 GRP + 按年份审计。现役最强仍 `round4/mortal.pth`（645k），基线口径仍 B/756k。

### 16.28 S6a 紧迫度特征注入（v5）终局判定（10-17 23:47 UTC ALL DONE）：无结论，“零新信息”归因

> 详情见 §12 会话 29/30。本节只归档结论。

**设计**：obs version 4→5（`obs_shape(5)=(1018,34)`），尾部 6 行紧迫度特征（与 1/2/3/4 名的点差 d1..d4 + overtake_gap + lead_gap），全部由 `state.scores` 确定性导出、零分布假设（牌局长度实测 6~11 局，不做剩余局归一化）；首层 conv 手术（新通道零初始化，等价于原模型 + 新特征从零学起）。**隔离**：全部代码/编译产物/配置/脚本在 `s6a_isolated/`，共享树还原 v4（§14-41）。

**实测**：训练 867k、exit=0、7.8 h、4.5 batch/s；holdout top1 0.8160 / lse +0.467 / R² −0.995；双向 ladder 各 10,000 半庄 **δ=+0.182 pt/半庄，SE 0.596，t=+0.31，解析 CI[−0.99,+1.35]；bootstrap SE 0.513、CI[−0.82,+1.20] → 无统计结论**（若真实效应 +0.18 需 ~213,885 sets）。

**归因**：6 个特征全是 v4 obs 已有分数通道（行 7–14）的确定性函数 → **零新信息**；网络本就能算，预计算不带来额外归纳偏置。该 null 只否定“重排已知信息”变体，**不否定 S6 方向**；下一子步 = S6b（读牌后验 = 模型算不出的新信息）。S6c（自家速度类）经分析同样已被 obs 显式编码（shanten one-hot/待牌/SP 表 3×17），SP 盲区仅 shanten≥4 的 8.72%——故 S6c 不单独烧轮，与 S6b 合并成一次 v6 注入（会话 30）。

---

## 17. 进展总纲（10-01→10-11 系统性汇总）

> 新会话只读本节即可接手；需细节时按小节号回查 §8～§16。
> **写任何新 `.log` 前必读 [§14bis 日志文件归置规范](#14bis-日志文件归置规范强制执行)**（禁止散落顶层，必须按类别入目录）。

### 17.1 编年

| 日期 | 里程碑 | 产出/要点 |
|---|---|---|
| 10-01 | 架构与可行性调研 | Rust+Python 双层分析；确认作者不开放权重且训练文档全为空占位（§3～§7） |
| 10-01 | 天凤端点实测 | list.cgi / scc 格式 / 牌谱下载 / 年度 zip 已 404（§8） |
| 10-02 | 发现 HF 数据集 | `tenhou-to-mjai` 2009–2025 全量，可用 hf-mirror（§8.6） |
| 10-02~10-03 | 自建采集 + 文档 | 2026 YTD 130,407 局；TASKLIST/MANUAL 成文 |
| 10-06 | 服务器环境 | conda mortal + torch 2.6.0+cu118 + rust 1.99 + libriichi.so/bins |
| 10-07 | 数据全量上服 | HF 18 包 merge-hf + prune --apply → 2,630,640 局（§16.3） |
| 10-08 | GRP + 第 1 轮训练 | grp 30k 步（21 min）；主模型 200k 步/22.9 h（§16.5/§16.6） |
| 10-09 | 首次复盘 | 发现评测饱和与 best.pth 陷阱；吞吐瓶颈；代码补丁清单（§16.4/§16.7/§16.9） |
| 10-09 | 工具与数据选择落地 | 玩家强度榜 + 白名单 top400；holdout_metrics / ladder / bench_pipe；git 基线；**第 2 轮开跑** |
| 10-10 | GRP 两假设 + 第 2 轮完成 | 加步数无效、加容量有效（grp_h128）；第 2 轮 424,847 步因 `num_epochs=1` 静默退出 |
| 10-11 | 双向判定与第 3 轮收尾 | 第 2 轮 δ=+2.86 显著；第 3 轮（epoch 2）δ=−0.03 无增益；best_perf 重置已实现并验证 |
| 10-11 | 代码评审 + 第 4 轮启动 | 修 4 个真 bug（供托计分/白名单 v2/best.pth 悬空/holdout worker）+ 实现两条建议（block bootstrap、一次性令牌）；R4=augmentation 起跑 |
| 10-12 | **主机重启 + NaN 事故 + R4 终局** | 重启后 1000 步内 AMP/BN 污染致全状态 NaN → 三重防护 + fp32 回滚 500k 重跑 → 645k 完成；判定 δ=+0.10±0.59 **无增益** |
| 10-13 | R5 终局 + 奖励消融判定 | names 修复与 v3 榜单上线；消融 C 确证变弱（−5.61）、B 采纳为基线；R5 top200 收紧名单 δ=−0.61 无增益 → 数据 reshuffle 路线到头 |
| 10-14 | hp_ablation + IQL Stage A 全天 | 上午：cql/aux 三分支均跨 0 → 超参非瓶颈（会话 21）；下午：**Stage A 跨局信用分配**实现→冒烟 7/7→训完 867k→判定 **δ=−1.03（bootstrap CI[−2.06,−0.02]）无增益、方向偏负**（会话 22/§16.25）→ 下一步转 Stage B 真 IQL |
| 10-15 | IQL Stage B 首轮 + 当日发现并修复实现缺陷 | 凌晨实现真 IQL（next_obs + VHead + EMA + AWR，回归 15 项过、冒烟 8/8）→ 11.5 h 训完 867k → **δ=−31.04、top1 0.692、logsumexp_q −176** → 诊断确认为 **r_t 双重计入 16×**（非零 2304 vs 应有 141）→ 首轮作废；已修（`dones[i]` 限定）+ 测试改验设计意图（八层 ALL PASS）+ 冒烟加行为门禁（会话 23/§16.26，§14-35/36）；下午冒烟 v3 10/10 全过（行为门禁当场抓出第二处缺陷 L_V 误用全体动作，已修，§14-37） |
| 10-15 | **IQL Stage B 正式轮 v2 启动（真 IQL 首次有效检验）** | 启动前全链路复审（无第三种偏差）+ 修测试幅度断言上界（§14-38）→ `config.iql_b_v2.toml`/`run_iql_b_v2.sh` 全新命名空间 + 种子块 200000/210000，16:24 UTC 启动，预计 ≈10.5 h 训练 + ≈1 h 判定（会话 24） |
| 10-16 | IQL v2 训完 + 发现价值尺度指数漂移 + 探针机制判决 | v2 训完 867k（top1 0.8069/lse +45）→ ladder **δ=−3.42 显著变弱**（漂移二阶污染策略通道，不作路线判决）；5 臂探针判决根因为 **expectile 不对称棘轮**（γ/v-ema 无效、τ=0.5 唯一稳定，§14-39/会话 25-26）；v3 定稿 = v2+τ=0.5，种子块 220000/230000 |
| 10-16~17 | **IQL Stage B v3 终局判定 + IQL 路线关闭** | v3（τ=0.5 稳定版）06:26→17:41 UTC 训完 867k（价值尺度全程稳定：max q_target p50 **1.206** vs v2 同点 +60.8；holdout top1 **0.8167** 历轮最高 / lse +1.596）→ 18:29 ALL DONE：双向各 10,000 半庄 **δ=+0.646 pt/半庄、bootstrap CI[−0.35,+1.63] → 无统计结论** → 按预注册守则 **IQL 路线关闭**（会话 28/§16.27）；目标形式改造（Stage A/B）两子路线全部关闭，转扩窗 2019+ + 重训 GRP + 按年份审计 |
| 10-16~17 | **读牌模型 v1+v2 实现并对决**（会话 27/27b，监督学习路线） | 新目录 `tools/read_model/`（零侵入 mortal/ 与 libriichi/）：4×PlayerState 回放出全知标签；v1=单帧 ResNet+H1/H2/H3（1.13M），v2=三流融合（+手切时序 Transformer+显式结构估计，1.36M，枚举器与 Rust calc_all 20.3 万手牌零不一致）；硬约束层；23 项回归 ALL PASS；同固定评估集 bootstrap CI 判决 **v2 三项显著增益**（alloc top-1 +3%、立直待牌 hit@1 +17%、听牌检测 AUC 0.628→0.686）；结构头向听估计仍弱（25.7%<28.4% 基线） |
| 10-17 | **读牌模型 v2s 结构头专项训练+对决**（会话 27c） | 根因解剖：v2 结构头向听头**零监督**（无损失引用 + 不入瓶颈）。v2s 六改（向听分布入瓶颈 38→46 维、向听直接 CE、分解-向听一致性约束[可微软向听公式，形制门控 normal]、tenpai 一致性、aux 向听 ×3、类均衡）+ 修 v1/v2 潜在 NaN；39 项回归 ALL PASS；bootstrap CI 判决：向听 0.257→**0.526**、AUC 0.686→**0.739**、一致率 0.28→0.93、七对/国士向听 0.21/0→0.49/0.51；代价 wait hit@1 −0.014（容量成本，解药=更长训练） |
| 10-17~18 | **S6a 紧迫度特征注入（v5）实施 + 终局判定** | 设计（NEXT_STEPS §4bis）→ v5=+6 通道（点差剖面 d1..d4 + overtake/lead gap，零假设）→ **完全隔离工作区** `s6a_isolated/`（共享树还原 v4，§14-41）→ 回归 17/17 + 冒烟 ALL PASS → 训完 867k（7.8 h / 4.5 batch/s）→ holdout 0.8160/lse +0.467 → **δ=+0.182、bootstrap CI[−0.82,+1.20] 无结论** → 归因“零新信息”（特征=已有分数通道的确定性函数）→ 不否定 S6 方向（会话 29/§16.28） |
| 10-18 | **S6c 自家速度类设计分析**（会话 30） | 结论：acceptance/期望巡目/牌型均为 obs 确定性函数且 **v4 已显式编码**（shanten 6 槽 one-hot、待牌、SP 表 required_tiles 2×34 + 逐巡 3×17、doras_owned）；SP 盲区 shanten≥4 = **8.72%** 决策点（3,622 点实测）；**S6c 纯重算版不单烧轮**，与 S6b 合并成一次 v6 注入；优先级 S6b > S6c > S6d；种子块下一块 260000/270000 |

### 17.2 模型谱系与强度账（全部用 ladder 双向同牌山 ≥5000 sets 判定）

| 世代 | 数据 | 奖励模型 | 步数 | 实测吞吐 | δ vs 上一代 | holdout top1（202609） | 产物 |
|---|---|---|---|---|---|---|---|
| R1 | 2022-01~2026-08 全体 813,145 局 | grp h64/l2 (30k) | 0→200k | 2.56 it/s / 22.9 h | —（基准） | 0.8041 | `mortal_step_000200000.pth` |
| R2 | 同窗 + **白名单 top400**（480,414 文件） | **grp_h128** h128/l3 (60k) | 200k→424,847 | 6.71 it/s / 9.5 h | **+2.86 pt**（SE 0.59，t=4.83）✅ | **0.8122**（峰值） | `round2/mortal.pth` + step_250k…400k |
| R3 | **与 R2 完全相同**，仅 `num_epochs=2` | 同 grp_h128 | 424k→650,000 | 6.7 it/s / 10.4 h | **−0.03 pt**（SE 0.595，t=−0.05，95%CI[−1.19,+1.14]）❌ | 0.8091（下降） | `round3/mortal.pth` + step_450k…650k |
| R4 | R2 同窗 + **白名单 v2**（492,810 文件）+ **augmentation** | 同 grp_h128 | 424k→**645,000**（中途 NaN 回滚到 500k 重跑，fp32 4.24 it/s / 累计 18.8 h） | **δ=+0.10 pt**（SE 0.593，t=0.16，95%CI[−1.07,+1.26]；bootstrap SE 0.508、CI[−0.90,+1.07]）❌ 无增益 | 0.8112@600k（12 批小样本口径，别与 24 批序列混比） | `round4/mortal.pth`(645k) + step_450k/500k/550k/600k，`best.pth`=500k |
| R5 | R2/B 同窗 + **白名单 v3 top200**（293,416 文件、席位 344,068）+ names 修复补丁树，augmentation=true | grp_h128；奖励 **B 口径 pts=[2,1,0,-3]** | B/756k→**870,000**（fp32 4.28 it/s / ≈7.8 h） | **δ=−0.61 pt**（SE 0.594，t=−1.02，95%CI[−1.77,+0.56]；bootstrap SE 0.508、CI[−1.61,+0.36]）❌ 无结论、方向偏负 | 未测 holdout（本轮不跑） | `round5/mortal.pth`(870k) + step_800k/850k，`best.pth`=850k |
| **IQL-B(v1)** | **`control.objective`: cql_mc→iql**（next_obs bootstrap + expectile V + AWR，CQL 关） | grp_h128；B 口径 pts=[2,1,0,-3] | B/756k→**867,000**（fp32 **2.69 it/s** / 11.5 h，3× encoder 前向） | δ=−31.04 pt（bootstrap CI[−32.13,−29.98]）**——但属无效实验：两处实现偏差（r_t 重复计入 16× + L_V 误用全体动作）** | 0.6917（比父代低 0.12）+ logsumexp_q −176.1 ← 尺度异常即缺陷本身 | `checkpoints/iql_b/`(867k) **封存不续用**；重跑走新命名空间 |
| IQL-A | **与 B 组完全同数据/同配方**，唯一变量 `env.credit_mode`: kyoku→**game**（跨局折扣回报后缀和） | grp_h128；B 口径 pts=[2,1,0,-3] | B/756k→**867,000**（fp32 **4.52 it/s** / 7.25 h，后缀和零 GPU 成本） | **δ=−1.03 pt**（SE 0.595，t=−1.73，解析 CI[−2.20,+0.14]；**bootstrap SE 0.517、CI[−2.06,−0.02]**、ρ≈+0.19）❌ 无增益、方向偏负 | 0.8118（新口径不可横比：R² −0.373、var_q/var_t 0.92/2.42） | `iql_a/mortal.pth`(867k) + `best.pth`(800k) + step_800k/850k |
| **IQL-B(v2)** | 同 B；`objective=iql`（next_obs bootstrap + expectile V **τ=0.7** + AWR，CQL 关） | grp_h128；B 口径 pts=[2,1,0,-3] | B/756k→**867,000**（fp32 2.7 it/s / 11.4 h，3× encoder 前向） | **δ=−3.416 pt**（SE 0.592，t=−5.77，bootstrap CI[−4.46,−2.36]）⚠ **不作路线判决**：expectile 不对称棘轮致价值指数漂移（q_target p50 +0.6→+60.8，倍增 ~32k 步，§14-39） | 0.8069 + logsumexp_q +45（尺度异常即缺陷本身） | `checkpoints/iql_b_v2/`(867k)；种子块 200000/210000 |
| **IQL-B(v3)** | 同 B；`objective=iql`（**τ=0.5**，探针唯一验证的稳定修复；其余=v2） | grp_h128；B 口径 pts=[2,1,0,-3] | B/756k→**867,000**（fp32 ~2.74 it/s / 11.25 h，均速） | **δ=+0.646 pt**（SE 0.592，t=+1.09；**bootstrap SE 0.509、CI[−0.35,+1.63]**）❌ 无结论（CI 跨 0） | **0.8167**（历轮最高）+ lse +1.596 | `checkpoints/iql_b_v3/`(867k)+best+step_800k/850k；种子块 220000/230000、bseed 20261016 |
| **S6a** | 同 B；`control.version` 4→**5**（+6 行紧迫度特征：点差剖面 d1..d4 + overtake/lead gap；首层手术零初始化） | grp_h128；B 口径 pts=[2,1,0,-3] | B/756k→**867,000**（fp32 **4.5 batch/s** / 7.8 h） | **δ=+0.182 pt**（SE 0.596，t=+0.31，解析 CI[−0.99,+1.35]；**bootstrap SE 0.513、CI[−0.82,+1.20]**）❌ 无结论 | 0.8160 + lse +0.467 | `s6a_isolated/code/checkpoints/s6a/`(867k)+best+step_800k/850k；种子块 240000/250000、bseed 20261017 |

→ **三次“同分布层面挤数据”尝试全被否 + 收紧名单也被否**：R3（同池重复一个 epoch）δ=−0.03、R4（席位视角增强，+145k 步新样本）δ=+0.10、**R5（top400→top200 收紧名单）δ=−0.61**，都落在 ±1.2 pt 噪声带内（R5 还是方向偏负）；**目前唯一有效的增益仍只有 R2 的“换更强数据 + 换更好奖励”（δ=+2.86，t=4.83）**。现役最强 = `checkpoints/round4/mortal.pth`（645k）；下一轮应离开数据 reshuffle、转 **IQL/Bellman（带 next-state + target network）**。

→ **新增（10-14）：目标形式改造的第一子路线也已否**——**IQL-A（跨局信用分配）δ=−1.03，bootstrap CI[−2.06,−0.02]**。即“让早期动作看到整场回报”不带来强度；瓶颈不在信用分配视界，而在**无价值传播与政策改进机制**（Q 无法给未访问动作定价）。下一轮 = **Stage B 真 IQL**（next_obs + VHead + EMA target net + τ-expectile + AWR，CQL 关）；若仍无增益 → 扩窗 2019+ + 重训 GRP。**至今八次对照（R3/R4/R5/消融 A/hp×3/Stage A）+ 基线 B 全部无统计增益，仅 R2 有效。**

→ **新增（10-15）：Stage B 首轮 δ=−31.04 不作数**——根因是实现缺陷（r_t 被重复计入 16×，见 §16.26），**真 IQL 路线尚未被有效检验**。重跑（v2，新命名空间 + 种子 200000/210000）才是首次有效试验；若 v2 仍无增益 → ③ 扩窗 2019+。有效增益仍只有 R2 的 +2.86。

→ **新增（10-16~17）：Stage B 三轮走完，IQL 路线关闭**——v1 无效（r_t 双计 16×）、v2 不作判决（τ=0.7 棘轮漂移，δ=−3.42）、**v3（τ=0.5）干净条件首次有效检验 δ=+0.646、bootstrap CI[−0.35,+1.63] 无结论**（会话 28/§16.27）。连同 Stage A（δ=−1.03 显著负）：**目标形式改造（信用分配视界 / TD 价值传播）两条子路线均无可采纳增益 → 路线关闭**；与 R3/R4/R5/augmentation/hp_ablation 的 null 合并，指向瓶颈在数据分布的信息量而非目标函数。**九次对照（R3/R4/R5/hp×3/Stage A/Stage B v3）均无统计增益，唯一有效增益仍只有 R2 的 +2.86**；现役最强仍 round4/645k，下一步 ③ 扩窗 2019+ + 重训 GRP + 按年份审计（读牌模型路径 C 注入为在途新方向，会话 27/28）。

→ **新增（10-17~18）：S6a 紧迫度特征 = 无结论，“零新信息”归因**——δ=+0.182、解析 CI[−0.99,+1.35]、bootstrap CI[−0.82,+1.20]（会话 29/§16.28）。6 个特征全部是 v4 obs 已有分数通道（行 7–14）的确定性函数 → 网络本就能算，预计算不带来归纳偏置。**该 null 只否定“重排已知信息”变体，不否定 S6 方向**；S6c 经分析同样已被 obs 显式编码（SP 表/向听/待牌），纯重算版不单烧轮（会话 30）。**十次对照（R3/R4/R5/hp×3/Stage A/Stage B v3/S6a）中仅 R2 有效**；S6 下一子步 = S6b（读牌后验 = 真新信息），种子块下一块 260000/270000。

### 17.3 可直接引用的量化事实

| 主题 | 数值 |
|---|---|
| 数据 | 全树 2,630,640 局（200902~202610，16 GB）；训练窗 813,145；白名单 top400 → 480,414 文件 / 704,231 席位 |
| 单局量 | 10.6 局/半庄；全座位 678 样本/文件；仅白名单座位 234.8 样本/文件 |
| epoch 换算 | 白名单池 1 epoch ≈ **220,268 步**（R2 实测 224,847 = 102%，验证了估算链） |
| 吞吐 | GPU 上限 8.88 it/s；管道 nw2/rayon4=3.12 → nw8/rayon8=**7.48** it/s（瓶颈在 CPU） |
| 吞吐（fp32） | `enable_amp=false` 稳态 **4.24 it/s**（比 AMP -36%）、显存 5.2 GB、GPU 利用率 91–94% → 瓶颈已从 CPU 管道移到 GPU |
| 判定精度 | 单半庄 std≈88 → 500 sets SE(δ)≈±1.9；5000 sets 解析 SE≈±0.59、**block bootstrap SE≈±0.51**（ρ≈+0.18，解析法偏保守）；需 `sets ≥ 当前 × (2SE/δ)²` |
| 评测代价 | 500 sets（2000 局）≈ 14 分/卡；5 卡两波 5000 sets ≈ **29 分** |
| GRP | val loss 平台在 30k 步；h128/l3 把 nll_last 1.2780→1.2361，但 **delta_pt std 恒≈0.70** → 受 7 维输入的信息量限制 |
| 行为对照 | 与人类凤凰卓同口径：立直 0.1794 vs 0.1769、副露 0.4566 vs 0.4826、**流局 10.9% vs 15.3%（弃和不足）** |
| **S6a（紧迫度特征）** | δ=**+0.182 pt/半庄**（SE 0.596，t=+0.31，解析 CI[−0.99,+1.35]；bootstrap SE 0.513、CI[−0.82,+1.20]）→ **无结论**；holdout top1 0.8160/lse +0.467；训练 4.5 batch/s / 7.8 h；若真实效应 +0.18 需 ~213,885 sets（42.8×） |
| **自家速度信息的 obs 覆盖** | shanten **6 槽 one-hot**（`obs_repr.rs` L395）、自家待牌 one-hot（L382）、SP 表每候选 `required_tiles` 2×34 + 逐巡 tenpai/win/EV 3×17（L576–691）、`doras_owned`（L286）；SP 仅算 shanten≤3（`SHANTEN_THRES=3`）→ **盲区 shanten≥4 占决策点 8.72%**（3,622 点实测：0:14.8% / 1:25.4% / 2:31.6% / 3:19.6% / ≥4:8.72%） |
| 硬件 | 56 线程 / 251 GB 内存 / 4×P100 + 1×RTX 8000 / 磁盘剩 3.5 TB |

### 17.4 代码与配置已做的修改（汇总）

| 文件 | 修改 | 状态 |
|---|---|---|
| `mortal/train.py` | ① `max_train_steps` 预算自停；② `snapshot_every` 编号快照；③ 索引构建改 `start.get('names', [])` 防 KeyError；④ **`best_perf` 重置机制**：新增 `state['best_perf_baseline']` 记录当时的卫冕者，加载时若（手动 `control.reset_best_perf` / 旧 state 无记录 / 卫冕者已更换）则重置为哨兵 `{4.0, −135.0}`；⑤ **10-12 `file_index` manifest**（`FILE_INDEX_VERSION=2`，比对 globs/名单 sha256/data_version，不一致自动重建）；⑥ **10-12 奖励口径**（读 `env.reward_mode`/`reward_scale`/`dataset.always_include_kan_select`）+ **续训目标守卫**（`env_reward_diff` 告警） | ①②③ 已跑过三轮；④ **10-11 四路测试通过**；⑤⑥ **10-12 回归通过**（`test_file_index_manifest.sh` U+E1~E6、`test_reward_modes.py` D 段） |
| `mortal/dataloader.py` | **10-12**：新增 `reward_mode`（`rank_pt`/`delta_points`）、`reward_scale`、`always_include_kan_select` 三个参数与 `reward_desc()`；`delta_points` 复用此前从未接线的 `RewardCalculator.calc_delta_points` | ✅ `test_reward_modes.py` 全过（独立重算误差 7.3e-12） |
| `mortal/holdout_metrics.py` | **10-12**：与训练同奖励口径（新增三个参数的透传）并在表头打印 env 口径 | ✅ |
| **信用分配口径（10-14，Stage A）** | `mortal/dataloader.py`：`FileDatasetsIter` 新增 `credit_mode`（`'kyoku'`/`'game'`）+ `gamma`，`populate_buffer` 沿局轴做折扣后缀和（`returns[k]=Σ_{j≥k}γ^(j−k)·r_j`），`reward_desc()` 带口径标识；`mortal/train.py`：`credit_mode` 进 `_ENV_REWARD_KEYS` 守卫（默认 `'kyoku'`，旧 ckpt 缺键不误报），game 模式 q_target **不再乘 γ^steps_to_done**；`mortal/holdout_metrics.py`：透传 credit_mode/gamma | ✅ `test_credit_modes.py` 五层 ALL PASS + 冒烟 7/7 + 全轮跑完（无 non-finite、吞吐与 B 持平） |
| **IQL 目标族（10-15，Stage B）** | `mortal/model.py`：新增 `VHead`（1024→256 Mish→1）；`mortal/dataloader.py`：`objective`（`'cql_mc'`/`'iql'`），iql 条目 7 元组 `[obs, actions, masks, next_obs, terminal, r_t, ranks]`，**r_t 仅局边界（`dones[i]`）非零**；`mortal/train.py`：`control.objective`/`iql_tau`/`iql_beta`/`ema_decay`，三损失 L_Q/L_V/L_AWR（CQL 关），EMA target（参数 polyak + BN buffer 跟随），state 新增 `v_net/ema_mortal/ema_dqn`，NaN 守卫扩展；`holdout_metrics.py`：iql ckpt 的 q_target=None → 只出 top1/agari/lse | ✅ `test_iql_data.py` 八层 ALL PASS；⚠ 首版 r_t 双重计入缺陷已修（§16.26）；cql_mc 数学路径未变更（回归已验） |
| **IQL 反漂移与监控（10-16，会话 25/26/28）** | `mortal/train.py`：新增 `iql_v_target_decay`（L_Q bootstrap 的 V 改用 EMA target，探针臂 B）、`iql_value_clip`（L_Q 目标钳位 ±8，探针臂 E）、**每存差点 `iql scale:` 分位数日志**（all_q/all_q_target 的 p50/p99 + q_pred + v_ema 状态，供链脚本门禁与事后取证，§14-40）；`config.iql_b_v3.toml` = v2+τ=0.5 定稿（`tools/training/prepare_iql_b_v3.py --fix tau05` 一键生成）；`run_iql_b_v3.sh` 收尾新增价值尺度放行门禁（max q_target p50 ≤ ~5，v2 同点 +60.8） | ✅ v3 全程 103 存点 max p50=1.206；探针 5 臂（`tools/training/prepare_iql_probe.py` + `run_iql_probe.sh`，各 30k 步）判定 τ=0.5 为唯一稳定修复 |
| **S6a v5 obs（10-17，隔离）** | `libriichi/src/consts.rs`：`MAX_VERSION=5`、`obs_shape(5)=(1018,34)`、oracle 5；`state/obs_repr.rs`：v5 复用 v4 全部编码 + 尾部 6 行 `encode_urgency`（点差剖面 + overtake/lead gap；`state.scores` 确定性导出，值域 [0,1]；所有 `match version` 分支补 5）；`mortal/{model,engine,env_check}.py`：`case 5` + `obs_shape(5)` 预检；**工作区隔离** `s6a_isolated/`（code 副本 + 独立 `libriichi.so` + 本地 index/ckpt/runs/logs + 输入软链 + `patches/v5_obs_features.patch`）；共享树已还原 v4 并重建 `.so` | ✅ 回归 17/17 ALL PASS（v4 前 1012 行逐位一致、独立复核 3.7e-8、手术前向等价 0）；冒烟 ALL PASS；正式轮无 non-finite；⚠ 共享树不再含 v5 源码（补丁在隔离区） |
| `libriichi/src/dataset/gameplay.rs` | **10-12**：`add_entry` 的 `apply_gamma` 注释改为实测口径（打牌+立直折扣；杠/鸣牌/和了/pass 不折扣；视界只在当前局内）——**仅注释，无行为变更，无需重编 .so** | ✅ 见 §14-31 |
| `mortal/train_grp.py` | `grp.control.max_train_steps`、`grp_best.pth` 按 val loss 取优、固定验证抽样种子、`est. %` 日志 | ✅ 已用（grp_h128 即此产物） |
| `mortal/player.py` | `enable_amp` / `enable_rule_based_agari_guard` 从 `[test_play]` 读；固定评测种子块 | ✅（固定块现在只用于同轮趋势，见 §17.7-坑 2） |
| `mortal/ladder.py` | 第 5 个参数 `log_dir`（并发必传）；**10-12**：执行参数改为 `CLI > [1v3.ladder] > 历史默认`、双侧强制一致、与 `[1v3.*]` 不一致时告警、输出 `exec_cfg`/两侧 ckpt sha256+steps 并写 `run_meta.json` | ✅ 已用于 10-10/10-11/10-12 三次判定；10-12 2-set 冒烟通过 |
| 配置 | `config.offline.toml`（R1）→ `config.round2.toml` → `config.round3.toml`（由 `make_round3_cfg.sh` 派生） | 共存，R1 产物不变 |
| `.gitignore` | 补 `dataset/`、`mortal/{runs,logs,indexes}/` 等；**故意不忽略 `memory/`、`tools/`**（它们是本项目唯一文档） | ✅ |

### 17.5 工具索引（一条命令能干一件事）

| 工具 | 用途 |
|---|---|
| `mortal/env_check.py` | 开跑前自检（目录遮蔽/环境变量/选卡/config/globs/目录） |
| `mortal/holdout_metrics.py` | 未见月 top1 一致率、Q 校准、R²、和了 top1（秒级，适合频繁监控） |
| `mortal/ladder.py` + `mortal/run_ladder_r3.sh` | 单对/批量双向同牌山对抗 |
| `tools/training/agg_ladder.py` | 汇总分块 → δ / SE / t / 95%CI / “还需多少 sets”；`--logs-root` 可加做按 set 的 block bootstrap（不假设局间独立） |
| `tools/mjai_score.py` | **计分口径的唯一权威实现**（与 `stat.rs`/`rankings.rs` 对齐）：三个工具与 bootstrap 共用 |
| `mortal/bench_pipe.py` | 数据管道吞吐基准（定 num_workers/RAYON） |
| `tools/player_stats/build_whitelist.sh` | 重建玩家强度榜与 top-N 名单（813k 文件 77 秒） |
| `tools/player_stats/whitelist_coverage.py` | 选 top-N 前算席位/样本/步数覆盖度 |
| `tools/training/grp_quality_probe.py` | GRP 分位置真实质量（hit@1/3/6/9/末、nll） |
| `mortal/run_grp_capacity.sh` + `grp_quality_probe.py --state/--hidden/--layers` | GRP 容量实验与同口径定档（注意 `grp_best.pth` 会被不同尺寸实验互盖，历史存档认各自 `state_file`） |
| `tools/training/dump_curves.py` / `check_selection_stats.py` / `paired_selection_diff.py` | 曲线汇总 / 解析交叉校验 / 同牌山配对差 |
| `tools/training/make_round3_cfg.sh`、`make_round4_cfg.sh` | 从上一轮 config 派生下一轮（只改预算/路径/变量） |
| `mortal/run_round4.sh` | **端到端链**：env_check → 暖启 → 训练 → 双向判定 → 汇总（单 tmux 会话） |
| `tools/training/smoke_round4_augmentation.py` | 启前冒烟：验白名单生效 + 增强样本确实不同（样本数守恒） |
| `mortal/launch_round3.sh` / `run_r2_ladder.sh` / `run_ladder_r3.sh` |  tmux 长跑入口，**内建完整 env 重置** |
| `tools/tenhou_data/repair_names.py` | **10-12 新增**：从原始 mjlog 恢复 2026 缺失的 `start_game.names`（自动 unquote 百分号编码），写独立补丁树；报告落 `dataset/index/` |
| `tools/tenhou_data/verify_names.py` | **10-12 新增**：names 覆盖率 / 席位 / 白名单与榜单池**命中率**验收（防"结构合法但语义全错"，见 §14-30） |
| `tools/training/test_file_index_manifest.{py,sh}` | **10-12 新增**：file_index manifest 单测（17 项）+ 端到端回归（E1~E6：构建/复用/换 data_version/旧格式/损坏/续训改奖励口径告警） |
| `tools/training/test_reward_modes.py` | **10-12 新增**：奖励口径验证——独立重算 `Grp::final_scores` 与每局真实分数差核对 `delta_points`、回归 `rank_pt`、检查 `reward_scale` 缩放与非法模式拦截、`env_reward_diff` 单测 |
| `tools/training/inspect_gamma_semantics.py` | **10-12 新增**：在真实牌谱上验证 `apply_gamma`/`dones`/`steps_to_done` 语义（折扣判据、局边界归零、折扣视界=当前局内），并打印动作×折扣分布与局内折扣示例 |
| `tools/player_stats/build_whitelist.sh` | **10-12 更新为 v3 口径**：glob 改为「HF 2022-2025 + `houou4_names/20260[1-8]`」，产出 `_v3` 榜单与 top400/top200 名单（v2 产物保留） |
| `tools/training/test_credit_modes.py` | **10-14 新增**：信用分配口径验证——γ=1 闭式恒等（后缀和 == `pts[final]−E_grp[pts\|局k]`）、GRP 终局 one-hot、telescope 终点、kyoku 回归等价多重集、γ<1 折扣生效、非法 credit_mode 拦截、`env_reward_diff` 守卫 3 例 |
| `mortal/config.iql_a.toml` + `mortal/run_iql_a.sh` | **10-14 新增**：Stage A 端到端链（env_check → B/756k 暖启 → 867k 训练 → holdout → 双向 ladder 160000/170000 → block bootstrap），全 env 重置内建于脚本；`config.iql_a_smoke.toml` + `run_iql_a_smoke.sh` 为同名 2k 步冒烟（**max_train_steps=758000**，见 §14-33） |
| `tools/training/test_iql_data.py` | **10-15 新增**：IQL 数据管线八层验证——next_obs 哈希多重集 == 独立重算 `obs[i+1]`、terminal 每 game 恰 1、**r_t 非零个数 == 局边界样本数（验“每局只计一次”的设计意图，见 §14-35）**、r_t 总和 == 逐局奖励总和、max|r_t|≤pts 极差（10-15 晚修正上界错误，见 §14-38）、cql_mc 默认路径回归（6 元组 + steps_to_done）、三例守卫 |
| `tools/training/diag_iql_reward_double_count.py` | **10-15 新增**：定位 r_t 重复计入的诊断（对比 dataset 非零 r_t 个数与“每局只计一次”应有值，并报出放大倍数 = 每局平均决策点数） |
| `mortal/config.iql_b{,_smoke}.toml` + `mortal/run_iql_b{,_smoke}.sh` | **10-15 新增**：Stage B 链（objective=iql，γ=0.997/τ=0.7/β=3.0/ema=0.995）；冒烟链已加**行为门禁**⑥ avg_rank ≤ 2.60、⑦ holdout top1 ≥ 0.78 且 \|logsumexp_q\| ≤ 10（§14-36）；正式轮判定种子块 180000/190000（v1 已消费）/ **200000/210000（v2 在用）** |
| `mortal/config.iql_b_v2.toml` + `mortal/run_iql_b_v2.sh` | **10-15 新增（会话 24）**：Stage B 正式轮 v2——全新命名空间 `iql_b_v2`（v1/v2 冒烟产物封存）、reset 令牌 `iql-b-v2`、种子块 200000/210000、索引复用 `indexes/iql_b_r4_v2.pth`；16:24 UTC 启动 |
| `tools/read_model/`（dataset/model/model_v2/structure/constraints/train/evaluate/export_onnx/test_pipeline/compare_v1_v2/compare_bootstrap/diag_by_form/probe_tedashi/run_v1.sh/run_v2.sh/run_v2s.sh/README.md） | **10-16~17 新增（会话 27/27b/27c）**：读牌模型 v1+v2+v2s（手牌暗牌预测，监督学习路线）——零侵入实现；`test_pipeline.py` 八层 39 项回归（枚举器 vs Rust calc_all 交叉验证、时序 token 独立重算、GameplayLoader obs 逐位一致、136 牌守恒、tiles_seen 通道比对、增广等变、学习门禁、**结构头专项门禁：软向听公式核对/v2s 前向/梯度流/v2s 小训练**）；`compare_bootstrap.py` = CI 判决对决；`run_v1.sh`/`run_v2.sh`/`run_v2s.sh` = 15000 步训练（813k 训练文件 / 202609 固定验证）；v2s = 结构头专项（向听分布入瓶颈 + 向听直接监督 + 分解-向听一致性约束 + 类均衡 + tenpai 一致性），对决判决向听 0.257→0.526、AUC +0.054、wait hit@1 −0.014 |
| `tools/training/prepare_iql_probe.py` + `mortal/run_iql_probe.sh` | **10-16 新增（会话 25/26）**：5 臂 ×30k 步单旋钮探针（对照/τ=0.5/γ=0.99/V-EMA/clip8）——定位慢漂移根因的最便宜手段（≈4.5 h vs 烧一轮 11.5 h 正式轮）；产出 `iql_probe_chain.log` 与各臂 `iql scale:` 曲线 |
| `tools/training/prepare_iql_b_v3.py` + `mortal/config.iql_b_v3.toml` + `mortal/run_iql_b_v3.sh` | **10-16 新增（会话 28）**：τ=0.5 稳定版正式轮（`--fix tau05` 一键生成配置）；链脚本 = env_check → B/756k 暖启 → 867k 训练（含 `iql scale:` 全程监控）→ holdout → 双向 ladder 220000/230000 → block bootstrap（bseed 20261016）→ 收尾守则自动判读 |
| `s6a_isolated/`（`tools/expand_ckpt_v5.py`、`tools/test_urgency_features.py`、`run_s6a{,_smoke}.sh`、`code/`、`patches/v5_obs_features.patch`、`README.md`） | **10-17 新增（会话 29）**：S6a **完全隔离工作区**——① `expand_ckpt_v5.py`：v4→v5 首层通道手术（新通道零初始化，前向等价）；② `test_urgency_features.py` 六层回归（obs 前 1012 行逐位不变/特征独立复核/手术等价/E2E 前向，`S6A_CODE` 指向隔离 code）；③ 冒烟链（回归+env_check+2k 步+行为门禁）；④ 正式链（回归+手术+**GPU2 等待**+867k+holdout+双向 ladder 240000/250000+bootstrap bseed 20261017） |

### 17.6 数据与产物台账

| 路径 | 内容 | 体量 |
|---|---|---|
| `dataset/mjai/houou4/` | 2,630,640 局 mjai（训练直接读）；⚠ **2026 部分仍无 names**，白名单模式下 0 席位，勿直接用 | 16 GB |
| `dataset/mjai/houou4_names/` | **10-12 新增**：2026（202601~202610，130,407 文件）补 names 的**补丁树**；训练需与旧树并列 glob（见下） | 607.6 MiB |
| `dataset/tenhou_hf/` | 18 个 HF 年度原包 | 17 GB |
| `dataset/raw/`、`dataset/index/` | 2026 自建 mjlog；清单 CSV + **强度榜 CSV/top400** | 1.2 GB / 197 MB |
| `mortal/checkpoints/` | R1 快照、R2、R3、**R4**、**R5(870k)**、**reward_ablation/{A,B,C}**、**hp_ablation/×3**、**iql_a(867k)**、**iql_b_v2(867k，漂移轮不作基线)**、**iql_b_v3(867k+best+step_800k/850k)**、grp×3、随机 baseline | ≈ 5 GB+ |
| `mortal/logs/ladder_r2`、`ladder_r3`、`ladder_r4`；`mortal/{reward_ablation,hp_ablation,iql_a}/ladder*/` | 各轮 20,000 局判定牌谱（可事后重算/bootstrap）；**Stage A 判定牌谱在 `mortal/iql_a/ladder/`**（A_160000..162000 / B_170000..172000，含 `run_meta.json`）；**Stage B v3 在 `mortal/iql_b_v3/ladder/`**（A_220000..222000 / B_230000..232000）、v2 在 `mortal/iql_b_v2/ladder/` | 每轮 ≈ 0.8 GB |
| `s6a_isolated/code/{checkpoints/s6a,s6a,logs}` | **10-17 新增（隔离区，不在 `mortal/` 下）**：S6a 产物——`checkpoints/s6a/`（mortal.pth 867k + best + step_800k/850k）、`s6a/`（train.log / holdout_202609.log / ladder_summary.log）、判定牌谱 `s6a/ladder/`（A_240000–242000 / B_250000–252000，20,000 局）、TB `runs/s6a`、链日志 `logs/s6a_chain.log`；本地索引 `indexes/s6a_r4_v2.pth`（data_version=s6a-r4-v2） | ≈ 1 GB |
| `mortal/logs/tb_archive/` | TB 坏段事件文件取证归档 | 72 KB |
| `mortal/indexes/` | `main_r2_whitelist_v1.pth`（480,414）等文件表 | 248 MB |
| 本地 git | 分支 `main`，基线 `33e63bc`、`5a1867f`，tag `round1-complete`；**无远端** | 4.4 MB |

### 17.7 坑位索引（按主题，详情在 §14 与 §16）

| 主题 | 条目 |
|---|---|
| 评测与判定 | 随机 baseline 使 test_play 饱和（§14-15）；**固定种子块不能跨轮比较**（§16.15，虚高 4.5 pt）；小样本只会低估（§16.13）；`best_perf` 继承会停写 best.pth（已修）；自写解析与 Rust Stat 平分 tie-break 不同（±1.4 pt） |
| 数据 | `file_index` 换名单/换窗必须改名；2026 部分无 `names`；exFAT 簇浪费（已迁服务器）；大文件传输必须打包 |
| 训练 | 离线只跑一个 epoch 就静默退出（§14-22）；state_file 存在即续训；GRP 必须先行且尺寸与 checkpoint 必须同步；`grp_best.pth` 会被不同容量实验互盖；**暖启冒烟的 `max_train_steps` 是全局绝对值，必须写“父代步数+冒烟步数”否则秒退（§14-33）**；**Stage B 四连坑：r_t 双重计入（§14-35）、冒烟缺行为门禁（§14-36）、L_V 对全体动作 expectile（§14-37）、expectile 棘轮慢漂移（§14-39）+ 幅度断言上界须推导（§14-38）** |
| 环境 | `MORTAL_CFG` 必需（默认 config.toml 不存在）；`CUDA_DEVICE_ORDER=PCI_BUS_ID` 否则选卡错（§14-19）；`import libriichi` 必须在 `mortal/` 下（§14-20）；`LD_LIBRARY_PATH` 只有跑 Rust CLI 才必需（§14-21）；RAYON 必须启动前设好；**tmux 重定向目录必须预先 mkdir，否则会话秒退且零日志（§14-34）** |
| 工程 | 长任务一律 tmux + 日志落 `mortal/logs/`（按 §14bis 归置）；脚本内用绝对路径解释器 `$PY`；`pgrep -f` 要锚定模式；并发 ladder 必传独立 log_dir；**并行实验下禁止就地改共享代码/替换共享 `.so`——新实验一律建隔离工作区（§14-41；范例 `s6a_isolated/`）** |

### 17.8 未决与下一步

| 类型 | 事项 |
|---|---|
| 已判定 | **第 4 轮 augmentation = 无增益**（δ=+0.10±0.59）；**奖励消融：C 显著变弱（−5.61）、B 方向正向未显著（+0.18，已采纳为新基线）**（会话 18）；**第 5 轮 top200 收紧名单 = 无结论、方向偏负**（δ=−0.61±0.59，bootstrap CI[−1.61,+0.36]，会话 20）。→ “同分布层面挤数据”（R3/R4/消融 A）+“换信息源的收紧名单”（R5）共 5 次否掉，**数据 reshuffle 路线到头**；**hp_ablation 便宜前置（10-14，cql/aux 小扫）也无增益（均 CI 跨 0）→ 排除“只是超参没扫对”，确认瓶颈在目标形式**；**IQL Stage A（跨局信用分配，10-14）也无增益、方向偏负（δ=−1.03，bootstrap CI[−2.06,−0.02]，会话 22/§16.25）→ “整场回报回传”这条 MC/控制变量路线关闭**；**IQL Stage B 首轮（10-15，δ=−31.04）属无效实验（两处实现偏差：r_t 双重计入 16× + L_V 误用全体动作取 expectile，均已修），不作为 IQL 路线的判决**（会话 23/§16.26）；v2（10-16 凌晨，δ=−3.42 显著变弱）属目标实现不稳定（expectile 棘轮漂移）亦不作判决（§14-39）；**v3（10-16，τ=0.5 干净条件，δ=+0.646、bootstrap CI[−0.35,+1.63]）= 真 IQL 首次有效检验，无统计结论 → 按预注册守则 IQL 路线关闭**（会话 28/§16.27）。至此目标形式改造两子路线（Stage A 跨局 MC / Stage B TD 价值传播）均关闭；**S6a 紧迫度特征（10-17~18，δ=+0.182、bootstrap CI[−0.82,+1.20]）无结论，归因“零新信息”（特征是现有分数通道的确定性函数）**（会话 29/§16.28）；**S6c 经设计分析确认同样已被 v4 obs 显式编码（shanten/待牌/SP 表 3×17），纯重算版不单烧轮**（会话 30） |
| **下一步（10-17~18 更新：IQL 关闭 + S6a 无结论）** | ~~Stage B 正式轮~~ 已关闭（δ=+0.646 无结论）；~~S6a 紧迫度特征~~ 已判定无结论（δ=+0.182，零新信息）。**信息侧当前唯一推进方向 = S6b：读牌后验注入（模型算不出的信息）**——建议与 S6c 的 SP 盲区 acceptance/牌型潜力**合并成一次 v6 注入**（v5 基础上：读牌 tenpai 概率 + 放铳 EV + 盲区 acceptance；一次手术一次训练），30k 探针先行（~2h，门禁 holdout/行为画像），通过后再开正式轮；判定种子块从 **260000/270000** 起。**数据侧并行主线 = ③ 扩窗 2019+ + 重训 GRP + 按年份审计**。父代仍 B/756k、每轮 +111k 预算、bseed 每轮换新。**现役最强仍 `round4/mortal.pth`（645k），基线口径仍 B/756k** |
| **新方向（10-16~17，会话 27/27b/27c/27d）** | **读牌模型 v1+v2+v2s(+v2x 消融)**（`tools/read_model/`，监督学习路线）：bootstrap CI 判决 v2 显著更优（AUC +0.058）、v2s 结构头专项成功（向听 0.257→0.526、AUC 0.686→0.739、一致率 0.28→0.93）；**A1 瓶颈归因已判决**：向听学习全靠损失（瓶颈非必要）、AUC 增益全靠瓶颈馈通（损失单独反伤 AUC −0.019）、wait 代价 55/45 混合 → **保留 46 维，v2s 架构定妆**；**下一步**：A2 v2s@50k 正式轮（定注入基线）→ A3 规则法基线 → B 路径 C 限域注入 + ladder 双向（唯一强度标准，事前门禁）。详见上方「读牌模型」节内计划块 |
| **已完成（10-12）** | 数据侧 P0 两条（分析文档 §2.1/§2.2，详见 §12 会话 15）：① 2026 缺失 names 已修复到**补丁树** `dataset/mjai/houou4_names/`（13 万文件；validate_logs 全绿；白名单命中 20.7% 座位 / 60.6% 文件），强度榜重建为 **v3**（top400 席位 789,454 / 262k 步每 epoch；top200 席位 344,068 / 114k 步）；② `file_index` 升级为带 **manifest 校验**（`FILE_INDEX_VERSION=2` + 可选 `dataset.data_version`），回归测试 E1~E6 全过。**模型侧 P0 三条**（详见 §12 会话 16）：③ 奖励口径可切换（`env.reward_mode` = `rank_pt`/`delta_points` + `reward_scale`）并加**续训目标守卫**；④ gamma 语义实测更正（`gamma<1` **不是**跨局信用分配，见 §14-31）；⑤ `ladder.py` 执行参数改为可配置+显式记录（`[1v3.ladder]`/CLI、`run_meta.json`、ckpt sha256） |
| 已完成（10-13） | **第 5 轮已判定（top200_v3 收紧名单 = 无结论、方向偏负，δ=−0.61）→ 详见 §12 会话 20 / §16.23** |
| 已完成（10-15） | **Stage B 真 IQL 首轮实施 + 跑完 + 当日定位并修复实现缺陷**：① 代码接线（VHead / objective 双分支 / 三损失 / EMA target / state 新键 / holdout 适配）；② `test_iql_data.py` 首次 ALL PASS（但当时断言了实现现状，未拦住 r_t 双重计入，§14-35）；③ 正式链 03:09→15:06 UTC 跑完（867k、11.5 h、2.69 batch/s）→ **δ=−31.04**；④ 用 `diag_iql_reward_double_count.py` 定位为 **r_t 重复计入 16×**（非零 2304 vs 应有 141）→ 首轮作废；⑤ 已修第一处（`dones[i]` 限定）+ 测试改验设计意图（八层 ALL PASS）+ 冒烟加行为门禁⑥⑦（§14-36）；⑥ **新门禁当场 FAIL 了 v2 冒烟，抓出第二处偏差（L_V 误对全体合法动作取 expectile，§14-37）并修正**；⑦ **v3 冒烟 10/10 全过**（avg_rank 2.527、top1 0.8134 > 父代 0.8112、logsumexp_q +1.343）→ 真 IQL 进入可检验状态；⑧ 产物写入 §12 会话 23 / §16.26 / §17.1-17.8。**状态：两处缺陷已修，待开正式轮（种子块 200000/210000）**——后续：v2/v3 均已跑完并判定，**IQL 路线关闭**，终局见会话 28/§16.27 |
| 已完成（10-14） | ① **hp_ablation 终局判定**：cql=1/10、aux=0 三分支 vs 父代全 CI 跨 0 → 超参不是瓶颈，直接支持目标形式改造（§12 会话 21 / §16.24）；② **IQL Stage A 实施完毕**：dataloader/train/holdout 三处 `credit_mode` 接线 + 守卫键扩展（`_ENV_REWARD_KEYS`），验证脚本 `tools/training/test_credit_modes.py` ALL PASS（γ=1 闭式恒等 8.9e-16、GRP 终局 one-hot、kyoku 回归等价多重集、γ=0.5 折扣生效、守卫 3 例）；③ **冒烟链 7/7 全过**（守卫告警恰 1 条、索引 manifest 复用、无 non-finite、758k 预算、快照、test_play 正常、吞吐 4.6 it/s）；④ 正式链 09:32 启动 → **16:47 训完 867k**（exit=0、无 non-finite、7.25 h）；⑤ **Stage A 终局判定（17:16 ALL DONE）：δ=−1.03、bootstrap CI[−2.06,−0.02] → 无增益、方向偏负，“整场回报回传”路线关闭**（§12 会话 22 / §16.25） |
| 10-14 实测数值 | Stage A 吞吐 **4.52 batch/s**（与 B 持平 → 后缀和零 GPU 成本）；test_play 固定种子 800k avg_rank 2.4815/avg_pt 2.5425、850k 2.47/2.295（**仅趋势，不入强度结论**；种子块已被多轮消费）；holdout top1 0.8118 |
| 可做未做 | akochan 外部锚点（需 JRE，本机无）；上游 `git diff` 校准补丁清单（**服务器无外网**，curl 超时）；git 无远端→推不出去（10-14 复核 `git remote -v` 为空）；checkpoint 无异地备份（10-14 复核 `backups/` 仅本机同盘打包）；两套固定 holdout 的“同分布口径”未建（时间外推口径 202609 已常态化，见分析文档 §4/S1）；带 agari-guard 的 holdout 未做（§3.5）；kan-select 训/部署口径回放对照未做（§3.3）。**已解决并移出（原条目）：**~~固定种子块轮换~~（ladder.py CLI 每轮换块，**已消费记录**：R2=60000-、R3=70000-、R4=80000-、消融=90000-/100000+110000（B）、R5=120000/130000、hp_ablation=140000/150000、**Stage A=160000/170000**、**Stage B v1=180000/190000（无效实验，块仍已消费）**；**Stage B v2=200000/210000**、**Stage B v3=220000/230000（IQL 路线已关闭，会话 28）**、**S6a=240000/250000（无结论，会话 29）**；下一轮（S6b/S6c 合并或扩窗 2019+）从 **260000/270000** 起）；~~GRP 真实点数差监督~~（C 组 10-13 测定 δ=−5.61 显著变弱，路线关闭）；~~`agg_ladder` block bootstrap 复核~~（已成每轮判定标配，bseed 记录在案） |
| 文档对账 | §16.11 四项（尤其本机手册的 §4.x 代码修改章节与本节 §17.4 的逐条比对）；本地 `.md` 仍无法从服务器读取（无盘映射），需上传后才能真合并 |

### 17.9 新会话接手三步

1. 读本节 §17.2/§17.8 确认“现在在哪、下一步选什么”；环境细节看 §16.2 与 `ENV_SETUP_P100.md §10`。
2. `tmux ls` + `ps -eo pid,etime,args | grep python` 看有无残留（习惯：长任务只用 tmux，日志进 `mortal/logs/`）。
3. 开跑前必做：`cd mortal && python env_check.py --cfg <本轮 config>`；训练完后只认 **ladder 双向 ≥5000 sets 且用全新种子块**（`run_ladder_r3.sh` 改三行即可复用）。
