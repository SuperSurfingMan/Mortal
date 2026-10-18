# Mortal 项目上下文记忆文档

> **用途**：沉淀项目历史、关键结论、可复现配置与坑位，供新会话快速接手。
> **本版**：2026-10-18 结构化重构。按「快速接手 → 系统与数据 → 实验史 → 现状 → 工程规范 → 索引」重排，合并了旧版 §12/§16/§17 中重复的会话叙述；关键结论、数字、路径、根因均保留；旧版全文：`git show HEAD:memory/MEMORY.md`（重构前 2136 行）。
> **可靠性标注**：✅ 已实测/已读源码　🔶 官方文档或作者声明　⬜ 未验证

**目录**：0 快速接手 · 1 项目与系统 · 2 数据与评测 · 3 训练与实验史 · 4 现状与下一步 · 5 工程规范与坑位 · 6 工具/文件/产物 · 7 附录

---

## 0. 快速接手（TL;DR）

### 0.1 当前坐标

| 项 | 值 |
|---|---|
| 项目 | Mortal（凡夫），开源日本麻将 AI；Rust 规则引擎 + Python/PyTorch 训练 |
| 代码/运行目录 | `/mnt/data8/mortal/Mortal`；训练必须在 `mortal/` 下执行 |
| 现役最强 | `mortal/checkpoints/round4/mortal.pth`（645k，fp32） |
| 判定基线 | B/756k：`reward_ablation/B_rank_pt_aligned/mortal.pth`；top400_v2 + `pts=[2,1,0,-3]` + augmentation |
| 当前阶段 | IQL 路线已关闭；S6a 已判定；读牌模型 A2b 已判决（AUC 0.946，**定为注入基线**）；读牌 **Bayes 上界已估计**（§3.14：数牌信息≈无听牌信息，参考生成式读牌器总体已达 A2b，未测出大块剩余空间；A2b 的 P(听牌) 过标定 6.3×）；**A3 规则法基线已判决**（§3.12：vs A2b 十项指标 8 项 net 优 2 项打平，tenpai_auc 0.9468/riichi_hit1 0.1121 统计打平，规则已拿 wait_hit1 增益的 74%——**读侧 modest 增量、近天花板，注入期望应转向「后验用法」而非读牌精度**）；下一步 路径 C 注入（注入前重标定 P(听牌)）/ 立直专项（新地板 hit@1 0.100、CE 3.35）/ 扩窗 2019+ |
| 数据 | 全树 2,630,640 局（200902~202610，16 GB）；训练窗 813,145 局；holdout = 202609+ |
| 硬件 | Ubuntu 18.04 容器，56 线程，251 GB 内存，4×P100-16GB + 1×RTX 8000-48GB |

### 0.2 一句话结论

- **唯一统计显著的增益**：R2「top400 白名单 + GRP h128」= **+2.86 ± 0.59 pt/半庄**（t=4.83）。
- **已关闭的路线**：同分布数据 reshuffle（R3/R4/R5/消融 A/hp_ablation）、真实分数差奖励 C（−5.61）、IQL Stage A 跨局信用（−1.03）、IQL Stage B 真 IQL（v3 +0.646，CI 跨 0）、S6a 紧迫度特征（+0.182，CI 跨 0）。
- **当前主线**：数据侧扩窗 2019+ 并重训 GRP；信息侧 S6b（读牌后验）与 S6c（SP 盲区/牌型潜力）合并成一次 v6 注入；读牌模型走路径 C 注入 + ladder 在线验收。
- **纪律**：每轮单变量；判定只认「全新种子块 + 双向同牌山 ≥5000 sets + block bootstrap CI」；CI 下界 >0 才算赢。

### 0.3 三条命令

```bash
cd /mnt/data8/mortal/Mortal/mortal
python3 env_check.py --cfg config.round2.toml          # 开跑前自检
python3 -u train.py                                    # 训练（MORTAL_CFG 指向本轮配置）
python3 ladder.py <challenger> <champion> <seed> 500 <log_dir>   # 单卡判定
```

### 0.4 阅读索引

| 想知道 | 去 |
|---|---|
| 系统怎么跑、环境变量、选卡 | §1.6–§1.8 |
| 数据从哪来、白名单怎么选、评测怎么做 | §2 |
| 每轮训练做了什么、结果如何 | §3 |
| 现在该做什么、哪些路线已死 | §4 |
| 踩过的坑、日志规范 | §5 |
| 工具、文件、产物在哪 | §6 |

---

## 1. 项目与系统

### 1.1 项目定位

| 项 | 内容 |
|---|---|
| 项目名 | Mortal（凡夫） |
| 定位 | 开源四人立直麻将 AI，深度强化学习驱动 |
| 作者/仓库 | Equim-chan；https://github.com/Equim-chan/Mortal |
| 文档/在线服务 | https://mortal.ekyu.moe ；https://mjai.ekyu.moe |
| 许可证 | 代码 AGPL-3.0-or-later；Logo/素材 CC BY-SA 4.0 |
| 本项目路径 | `/mnt/data8/mortal/Mortal`（历史文档中的 `c:\Maj\Mortal` 为早期 Windows 本机路径） |
| 权重 | 作者不公开预训练权重（2022-08-19 gist 最终结论）；本项目权重均为自训 |

### 1.2 整体架构

```
Python 层（mortal/）
  train.py 训练循环 · model.py 网络 · dataloader.py 奖励与样本
  server.py/client.py 在线自博弈 · one_vs_three.py / ladder.py 评估
        ↕ PyO3
Rust 核心（libriichi/）
  state 状态机与 obs 编码 · dataset 牌谱加载 · arena 自博弈模拟器
  mjai 事件协议 · algo 向听/和牌/符点/SP 计算器 · stat 牌谱统计
        ↑
训练数据：天凤凤凰卓 mjai 牌谱（*.json.gz）
```

核心设计：Rust 负责性能敏感的解析/状态/编码/模拟；Python 负责网络与训练；**奖励不在 Rust 侧计算**，由 Python 按最终顺位/GRP 生成。

### 1.3 libriichi（Rust 核心）

- workspace 成员：`libriichi`、`exe-wrapper`；release 启用 `lto`、`codegen-units=1`。
- crate 名 `riichi`，产物 `cdylib+rlib`；features：`pymod`、`mimalloc`、`abi3`、`sp_reproduce_cpp_ver`。
- 关键依赖：`pyo3 0.25`、`numpy/ndarray`、`rayon`、`flate2`、`serde_json`、`rand_chacha+sha3`、`crossbeam/parking_lot`。

| 模块 | 职责 |
|---|---|
| `state/` | `PlayerState`：手牌/牌河/副露/分数/顺位/向听/合法动作；`obs_repr.rs` 输出 `(obs,mask)` |
| `dataset/` | `GameplayLoader` → 每步 obs/actions/masks/at_kyoku/dones/apply_gamma/at_turns/shantens；每局 grp/player_id/name |
| `arena/` | 可复现牌山（SHA3→ChaCha12）、`BatchGame`、1v3/2v2；顺位点 `[90,45,0,-135]` |
| `algo/` | 向听（tomohxx 表）、和牌（boomphf）、符点、SP 单人期望打点 |
| `mjai/` | 事件协议与 `Bot` 在线推理封装 |
| `agent/` | `MortalBatchAgent`：`react_batch()` 回调 Python；quick_eval、杠二次推理、agari guard |
| `bin/` | `stat`、`validate_logs`（链接 libpython，运行时需 `LD_LIBRARY_PATH`） |

**obs 版本与常量**

| 版本 | obs [通道,34] | 说明 |
|---|---|---|
| v1 | (938,34) | VAE 风格 |
| v2 | (942,34) | |
| v3 | (934,34) | |
| **v4（现役）** | **(1012,34)** | +SP 期望打点/进张表；共享树当前为 v4 |
| v5（仅 S6a 隔离区） | (1018,34) | v4 + 6 行紧迫度特征；`s6a_isolated/` 内使用 |

`ACTION_SPACE=46`（0–36 打牌/选杠、37 立直、38–40 吃、41 碰、42 杠、43 自摸和、44 九种九牌、45 pass）；`GRP_SIZE=7`；`MAX_VERSION` 共享树 4、S6a 补丁为 5。

### 1.4 Python 训练层

| 组件 | 结构 |
|---|---|
| `Brain` | 1D ResNet 沿 34 种牌卷积；默认 `conv_channels=192`、`num_blocks=40`；Mish + BatchNorm1d；输出 1024 维 φ |
| `DQN` | Dueling：`q=v+a−mean(a)`，mask 填 −inf；v4 为 `Linear(1024,1+ACTION_SPACE)` |
| `GRP` | GRU（7→hidden 64/128，2/3 层）+ FC → 24 维（4! 排列）；float64；`calc_matrix()` 归约成 `(N,4,4)` |
| `AuxNet` | `Linear(1024,4)` 预测下一局顺位 |
| `VHead`（IQL） | `1024→256 Mish→1`（262,657 参数）；Stage B 使用 |

**原始训练目标（cql_mc）**

```python
q_target_mc = gamma ** steps_to_done * kyoku_rewards      # 当前局 MC 回报
q = dqn(phi, masks)[range(bs), actions]
loss = 0.5*mse(q, q_target_mc)                            # L_DQN
     + cql_min_weight * (q_out.logsumexp(-1).mean() - q.mean())
     + next_rank_weight * ce(aux_net(phi), player_ranks)
```

优化器 AdamW；`LinearWarmUpCosineAnnealingLR`；AMP `GradScaler`；每 `save_every` 存 `state_file`；每 `test_every` 跑 `test_play`；离线模式只跑一个 epoch 就 `break`（与 `max_train_steps` 无关）。在线模式有已知卡死 bug，源码用子进程 + `sys.exit(0)` 规避，勿“修复”掉。

**奖励构造（关键）**

- `RewardCalculator`：默认 `pts=[3,1,-1,-3]`；`calc_grp()` 把每局前缀喂 GRP → 玩家×顺位概率；`calc_delta_pt()` 取相邻局期望点差 `exp_pts[1:]−exp_pts[:-1]` 作为奖励。
- `FileDatasetsIter`：在 `build_iter()` 中构造 GRP 并载入 `[grp].state_file`（不能放 `__init__`，Windows 下失效）；`populate_buffer()` 逐局算 `kyoku_rewards`、`player_ranks`、`steps_to_done`。
- 样本字段（cql_mc）：`[obs, (invisible_obs), action, mask, steps_to_done, kyoku_rewards[at_kyoku], player_ranks[at_kyoku+1]]`。
- IQL 样本（Stage B）：`[obs, actions, masks, next_obs, terminal, r_t, ranks]`；`r_t` **仅局边界 `dones[i]` 非零**。

### 1.5 配置字段速查（`config.example.toml` / 各轮 toml）

| 节 | 关键字段 | 说明 |
|---|---|---|
| `[control]` | `version`、`online`、`state_file`、`best_state_file`、`tensorboard_dir`、`device`、`enable_amp`、`enable_compile`、`batch_size`、`opt_step_every`、`save_every`、`test_every`、`snapshot_every`、`max_train_steps`、`seed`、`reset_optimizer`、`reset_best_perf`、`objective`、`iql_tau`、`iql_beta`、`ema_decay`、`iql_v_target_decay`、`iql_value_clip` | 训练主控；`save/test/snapshot/max_steps` 必须满足整除断言 |
| `[dataset]` | `globs`、`file_index`、`file_batch_size`、`reserve_ratio`、`num_workers`、`player_names_files`、`num_epochs`、`enable_augmentation`、`augmented_first`、`data_version`、`always_include_kan_select` | 数据选择与索引 manifest |
| `[env]` | `gamma`、`pts`、`reward_mode`（`rank_pt`/`delta_points`）、`reward_scale`、`credit_mode`（`kyoku`/`game`） | 奖励口径；续训时不一致会告警 |
| `[resnet]` | `conv_channels=192`、`num_blocks=40` | 显存不足再下调 |
| `[cql]` / `[aux]` | `min_q_weight=5` / `next_rank_weight=0.2` | 原始目标正则；IQL 时 CQL 关闭 |
| `[grp]` | `state_file`、`[grp.network] hidden_size/num_layers`、`[grp.control]`、`[grp.dataset]`、`[grp.optim]` | GRP 权重必须先存在；尺寸与 ckpt 必须匹配 |
| `[optim]` | `eps/betas/weight_decay/max_grad_norm` | AdamW；`max_grad_norm=0` 表示不裁剪 |
| `[optim.scheduler]` | `peak/final/warm_up_steps/max_steps` | 常见 `peak=final=1e-4` |
| `[baseline.train/test]`、`[1v3.*]`、`[1v3.ladder]` | `device/state_file/enable_amp/guard/seed_key/games_per_iter` | 1v3 评估；ladder 参数优先级 CLI > `[1v3.ladder]` > 历史默认 |
| `[test_play]` | `games`、`log_dir` | 固定种子趋势评估，不作跨轮强度结论 |
| `[online*]` | `history_window`、`host/port`、`buffer_dir/drain_dir/capacity` | 在线自博弈（暂未启用） |

### 1.6 构建与运行

```bash
# 1) conda（服务器必须 defaults 源，glibc 2.27；禁 conda-forge）
conda env create -f environment.yml && conda activate mortal
# 2) PyTorch（P100 用 cu118；勿换 cu126）
pip install torch --index-url https://download.pytorch.org/whl/cu118
# 3) 编译 libriichi
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
# 4) 自检
cd mortal && python3 env_check.py --cfg config.round2.toml
# 5) 训练/推理
python3 -u train.py          # MORTAL_CFG 指向本轮 toml
python3 mortal.py            # 单文件推理入口（mjai Bot）
```

Docker 多阶段构建仅用于推理（Arch 阶段编译 `libriichi.so`，运行阶段装 Python 依赖），官方明确不用于训练。

### 1.7 服务器环境与选卡

| 项 | 实测值 |
|---|---|
| OS/CPU/内存 | Ubuntu 18.04.2 容器（glibc 2.27），2×Xeon E5-2690 v4 / 56 线程，251 GB |
| GPU | 4×Tesla P100-16GB + 1×Quadro RTX 8000-48GB；驱动 580.173.02 |
| conda | `mortal`：Python 3.12.11、torch 2.6.0+cu118（sm_60/sm_75） |
| Rust/libriichi | rustc/cargo 1.99.0；`libriichi.so` 0.1.0/release；`obs_shape(4)=(1012,34)`、`ACTION_SPACE=46` 已断言 |
| 磁盘 | `/mnt/data8` 7.3 TB（可用约 3.5 TB） |

**启动环境变量（每次长任务前重新完整设置，不能依赖上次会话残留）**

```bash
cd /mnt/data8/mortal/Mortal/mortal          # ★ 必须在 mortal/，仓库根会被 libriichi/ 目录遮蔽
export MORTAL_CFG="$PWD/config.round2.toml"  # ★ 必需；默认 config.toml 不存在
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # ★ 必需；否则序号与 nvidia-smi 不一致
export CUDA_VISIBLE_DEVICES=2                # 主训 RTX 8000（配 PCI_BUS_ID）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # 把 CPU 让给特征编码
export RAYON_NUM_THREADS=8                   # 必须启动前设；rayon 池只初始化一次
# LD_LIBRARY_PATH 仅跑 validate_logs/stat 等 Rust CLI 时需要
```

**CUDA 选卡映射（实测）**

| `CUDA_VISIBLE_DEVICES` | 默认算力序 | 加 `PCI_BUS_ID` 后 |
|---|---|---|
| 0 | RTX 8000 | P100 (nvidia 0) |
| 1 | P100 | P100 |
| 2 | **P100** | **RTX 8000** |
| 3/4 | P100 | P100 |

主训固定 GPU2（RTX 8000）；P100 留给并行判定/读牌模型。启动后检查日志 `device: cuda:0 (型号)`。fp32 下 RTX 8000 约 4.2–4.5 batch/s，P100 约 2.5 batch/s（差 1.76×）。

### 1.8 Git、备份与权重

- **远端**：`origin = ssh://git@github.com/SuperSurfingMan/Mortal.git`；本地/远端主分支 `master`；`~/.ssh/config` 将 github.com 重写为 `ssh.github.com:443`，密钥 `~/.ssh/id_ed25519` 已登记。
- **当前状态**：`master` 领先 `origin/master`（最近远端 `eb4f660`；本地已到 `8dee98e`，MEMORY 重构后还会增加）。推送前先 `git status -sb`；SSH 主机键缺失会导致 push 静默挂起（§5）。
- **忽略规则**：`dataset/`、`mortal/checkpoints|indexes/*.pth`、`mortal/libriichi.so`、`target/`、`backups/`、`mortal/runs/`、`mortal/logs/`、`tools/tenhou_data/logs/`、`*.bak/*.so/*.dll/*.exe/*.tar`；**保留 `memory/` 与 `tools/`**（唯一文档）。
- **基线**：首提交 `33e63bc`（460 文件/4.0 MB），tag `round1-complete`；文档同步提交 `5a1867f`；防泄漏校验 `dataset` 跟踪数 0、敏感扩展名 0；git 2.17.1（不支持 `init -b`/`switch`）。
- **备份**：`backups/ckpt_round1/`（8 个 R1 ckpt，592 MB）；`backups/code_docs_20261009.tar.gz`（32 MB）；全量权重打包 `Mortal_valuable_with_ckpts_20261013.zip`（6.83 GiB，697 entries，含 R1–R5/消融/hp_ablation/源码/README）。
- **权重现状**：`mortal/checkpoints/` 下 R1 12 文件、R2 6、R3 7、R4 6、R5 4、reward_ablation 15、hp_ablation 7、IQL A/B v2/v3、S6a 隔离区等；`.pth` 约 130 MB（fp32）。

---

## 2. 数据与评测

### 2.1 天凤数据源（2026-10-01 实测）

| 接口/来源 | 结论 |
|---|---|
| `https://tenhou.net/sc/raw/list.cgi` | 返回 JS 数组；近 7 天小时窗口；新行可能插在中间，必须按 log_id 去重 |
| 文件类型 | `scc` = 鳳凰卓（含牌谱，2009-02-01 起）；`scb` 仅结果；`sca` 个室；`scd` 雀荘；`sce` 技能战/琥珀；`scf` HTML 汇总 |
| scc 格式 | gzip HTML，html-escaped；先 `html.unescape`；正则 `log=([0-9]{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8})` |
| 单局下载 | `https://tenhou.net/0/log/?<log_id>` → gzip XML（约 15.7 KB）；按 `1f 8b` 魔数判断是否已压缩 |
| 年度 zip | `scrawYYYY.zip` 当前全部 404（2009–2025 无入口） |
| 滚动日归档 | `list.cgi?old` + `dat/YYYY/sccYYYYMMDD.html.gz`；覆盖当年 1/1 至约 7 天前；2026-10-01 实测 266 个日文件 8.4 MB |
| HF 数据集 | `NikkeTryHard/tenhou-to-mjai`：2009–2025 全量约 250 万局、19 GB tar；`.mjai.json.zst` 与 Mortal 兼容；hf-mirror 可下；2026 仅 1 月（12,188 局） |
| 官方红线 | 不得用于竞品/非对局服务；不得做公开下载服务；串行下载、每请求 ≥2 秒；数据仅本地研究训练，不得二次分发 |

**数据落地**：HF 18 个年度包（16.86 GiB）在服务器 `dataset/tenhou_hf/`；merge-hf 展开 + prune --apply 后 `dataset/mjai/houou4/` = **2,630,640 局 / 213 个月 / 16 GB**（200902~202610）。2026 自建 130,407 局（`dataset/raw/` 1.2 GB）。

### 2.2 过滤规则与目录

| 过滤项 | 规则 |
|---|---|
| 数据池 | 只取 `scc` 凤凰卓（琥珀可选） |
| 人数/长度/规则 | `四`（排除三麻）、`南`（半庄）、`喰赤`、排除 `速` |
| 去重/完整性 | 按 log_id；转换后 `validate_logs` + 回放计分校验 |
| 目标行 | `四鳳南喰赤－` |

```
dataset/
├── mjai/houou4/YYYYMM/<log_id>.json.gz      # 全树，训练直接读
├── mjai/houou4_names/20260[1-8]/...        # 2026 names 补丁树（607.6 MiB）
├── raw/                                     # 2026 原始 mjlog（1.2 GB）
├── tenhou_hf/                               # 18 个 HF 年度 tar（17 GB）
└── index/                                   # CSV、强度榜、topN 名单、manifest
```

### 2.3 names 修复与白名单

2026 自建数据（jansou `dump_mjai`）缺少 `start_game.names`；Rust `#[serde(default)]` 会静默变 4 个空串，导致白名单模式 0 席位且不报错。修复：

- `tools/tenhou_data/repair_names.py` 从原始 mjlog `<UN>` 的 `n0~n3` 恢复；**必须 URL-unquote**（直接写百分号编码会“结构合法但语义全错”，白名单命中 0%）。
- 写独立补丁树 `dataset/mjai/houou4_names/`，不动旧树；130,407 文件 / 1.92 min / 607.6 MiB；`validate_logs` 16 s 无告警。
- `tools/tenhou_data/verify_names.py` 验收：names 覆盖、席位/文件白名单命中、`%XX` 残留。全量复核：names 100%、白名单命中 130,415 席（25.00%）/ 89,317 文件（68.49%）、榜单池命中 99.36%；旧树 0%。原树 2026 有 **113,430 文件无 names**，HF 2009–2025 有 **699,715 文件带 names**。

**强度榜**：天凤无段位字段，用长期成绩自证（按最终得分排顺位，累计局数/平均顺位/平均点，天凤点 `[90,45,0,-135]`）。`tools/player_stats/scan_player_ratings.py` 扫 813k 文件约 77 秒（28 workers）。

| 名单版本 | 阈值/规模 | 覆盖（席位/样本/步每 epoch） | 用途 |
|---|---|---|---|
| top400 v1 | ≥400 局且 avg_pt≥+2.41；480,414 文件 | 704,231 席 / 120M / **220,268 步** | R2 |
| top400 v2 | 修供托计分后重建；492,810 文件 | 735,249 席 / 134M / **244k 步** | R4、B 基线 |
| top400 v3 | 含 names 修复后的 2026；全池 813,145 文件，门槛 +2.57 | 789,454 席 / 134M / **262k 步** | 榜单/覆盖率 |
| top200 v3 | 门槛 +3.98；293,416 文件 | 344,068 席 / 57M / **114k 步** | R5 |
| top100 v3 | — | 119,452 席 / 40k 步 | 备选 |

v3 重建后玩家池 4,883→**5,356**，合格（≥400 局且 avg_pt≥2）472→**501**；top400 门槛 +2.45→**+2.57**（69 人进出），top200 门槛 **+3.98**。

强度榜自检：全体加权 avg_pt=0.0000、avg_rank=2.5000（每半庄顺位点排列和为 0）。

### 2.4 评测方法论（硬纪律）

1. **训练内 `test_play` 不作跨轮强度结论**：它写死 `seed_start=(10000,0x2000)`，同一批 500 sets 被多轮选模型反复使用，会牌山过拟合。第 3 轮实测固定块 Δ=+4.52±2.07（看似显著），全新种子块 ladder 仅 δ=−0.03±0.59——**4.5 pt 全是虚高**。固定块只用于同轮趋势。
2. **样本量按 SE 倒推**：`sets ≥ 当前 × (2·SE/δ)²`；500 sets 时 SE≈±1.9 pt，曾把 +2.86 真实增益误判为无增益；小样本还会低估效应。
3. **判定用双向同牌山配对**：A=新当挑战者、B=旧当挑战者，各 5×500 sets（10,000 半庄/方向），`δ=(ptA−ptB)/2`；两向同号才可信。
4. **统计口径**：解析 SE 按局独立，偏保守；`agg_ladder.py --logs-root` 做按 set 的 block bootstrap（同牌山 4 局 ρ≈+0.18，bootstrap SE 0.515 < 解析 0.595）。结论只认 bootstrap 95%CI。
5. **tie-break**：arena `end_game` 不带 scores，自写脚本累加 hora/ryukyoku deltas 推名次，平分 tie-break 与 Rust `Stat` 不同（偏差 0.006–0.023 名次≈1.4 pt）——**一律以 Rust Stat / ladder.py 输出为准**。
6. **多卡并行**：每个 ladder 任务必须传独立 `log_dir`（`ladder.py` 开头 `rmtree`），否则互相删牌谱。

**已消费种子块（不得复用）**

| 用途 | 种子块 |
|---|---|
| R2/R3/R4 判定 | 60000/70000/80000/90000（按轮次使用） |
| 奖励消融 B | 100000/110000 |
| R5 | 120000/130000 |
| hp_ablation | 140000/150000 |
| IQL Stage A | 160000/170000 |
| IQL Stage B v1（无效） | 180000/190000 |
| IQL Stage B v2（漂移） | 200000/210000 |
| IQL Stage B v3 | 220000/230000 |
| S6a | 240000/250000 |
| **下一块** | **260000/270000** |

### 2.5 关键量化事实

| 主题 | 数值 |
|---|---|
| 数据 | 全树 2,630,640 局；训练窗 813,145；分年 2022/23/24/25/26 = 167,627/168,777/184,424/178,887/130,407 |
| 单局量 | 10.6 局/半庄；全座位 678 样本/文件；仅白名单座位 234.8 样本/文件 |
| epoch | top400 v1 1 epoch≈220,268 步（R2 实测 224,847=102%） |
| 吞吐 | GPU 上限 8.88 it/s；管道 nw2/rayon4=3.12 → nw8/rayon8=7.48 it/s；fp32 RTX 8000≈4.24 it/s |
| 判定 | 单半庄 std≈88 → 500 sets SE≈±1.9；5000 sets 解析 SE≈±0.59、bootstrap≈±0.51 |
| 评测代价 | 500 sets（2000 局）≈14 分/卡；5 卡两波 5000 sets ≈29 分 |
| GRP | 30k 步到平台；h128/l3 nll_last 1.2780→1.2361，hit@3 0.379 / hit@6 0.483；`delta_pt` std 恒≈0.70（7 维输入信息量上限） |
| 行为对照 | 和了 0.2228 vs 人类 0.2117；立直 0.1794 vs 0.1769；副露 0.4566 vs 0.4826；放铳 0.1405 vs 0.1255；流局 **10.9% vs 15.3%**（弃和不足） |

---

## 3. 训练与实验史

### 3.1 总表

| 轮次/实验 | 关键变量 | 数据/父代 | 步数 | 吞吐 | δ vs 父代（pt/半庄） | holdout top1 | 判定 |
|---|---|---|---|---|---|---|---|
| R1 | 基线（cql_mc + GRP h64/l2） | 813,145 全体 | 0→200k | 2.56 it/s / 22.9h | —（基准） | 0.8041 | 基准；test_play 饱和，强度持平 |
| R2 | top400 v1 + GRP h128/l3 | 480,414 文件 | 200k→424,847 | 6.71 it/s / 9.5h | **+2.86±0.59**（t=4.83） | **0.8122** | ✅ 唯一有效增益 |
| R3 | 同池再跑一个 epoch | 同 R2，`num_epochs=2` | 424k→650k | 6.7 it/s / 10.4h | −0.03±0.59（t=−0.05） | 0.8091 | ❌ 无增益 |
| R4 | 白名单 v2 + augmentation | 492,810 文件 | 424k→645k（NaN 回滚 500k 重跑） | fp32 4.24 it/s / 18.8h | +0.10±0.59（t=0.16） | 0.8112@600k | ❌ 无增益 |
| R5 | top200 v3 收紧名单 | 293,416 文件，父代 B/756k | B/756k→870k | 4.28 batch/s / 7.8h | −0.61±0.59（CI[−1.61,+0.36]） | 未跑 | ❌ 无结论、方向偏负 |
| 消融 A | 奖励 rank_pt `[6,4,2,0]`（控制） | 父代 R4/645k | →756k | — | −0.04（CI[−1.03,+1.02]） | 0.8179 | 无结论（与父代等价） |
| 消融 B | 奖励 rank_pt `[2,1,0,−3]` | 同上 | →756k | — | +0.18（CI[−0.85,+1.19]） | 0.8158 | 无结论、方向正向；**采纳为基线** |
| 消融 C | 奖励 delta_points（scale 1e-3） | 同上 | →756k | — | **−5.61**（CI[−6.67,−4.55]） | 0.7960（R²+0.01） | ❌ 显著变弱，路线关闭 |
| hp_ablation | cql 1/10、aux 0 | 父代 R4/645k | →756k | — | +0.05 / −0.07 / +0.59（均 CI 跨 0） | 0.812–0.817 | 超参非瓶颈 |
| IQL-A | `credit_mode: kyoku→game` | 父代 B/756k | →867k | 4.52 batch/s / 7.25h | **−1.03**（bootstrap CI[−2.06,−0.02]） | 0.8118 | ❌ 无增益、方向偏负 |
| IQL-B v1 | 真 IQL，τ=0.7；r_t 双计 16× | 父代 B/756k | →867k | 2.69 batch/s / 11.5h | −31.04（无效实验） | 0.6917 / lse −176.1 | ❌ 实现缺陷，不作判决 |
| IQL-B v2 | 修 r_t/L_V，τ=0.7 | 父代 B/756k | →867k | 2.7 batch/s / 11.4h | −3.416（CI[−4.46,−2.36]） | 0.8069 / lse +45 | ❌ 价值漂移污染，不作判决 |
| IQL-B v3 | v2 + τ=0.5 | 父代 B/756k | →867k | 2.74 batch/s / 11.25h | **+0.646**（CI[−0.35,+1.63]） | **0.8167** / lse +1.596 | ❌ 无结论 → IQL 路线关闭 |
| S6a | obs v5：+6 行紧迫度 | 父代 B/756k | →867k | 4.5 batch/s / 7.8h | +0.182（CI[−0.82,+1.20]） | 0.8160 / lse +0.467 | ❌ 无结论（零新信息） |

### 3.2 第 1 轮（R1，2026-10-08~09）

- 配方：v4 192×40；batch 512；`save_every=1000`、`test_every=50000`、`max_train_steps=200k`；AMP=true；`gamma=1`、`pts=[6,4,2,0]`、cql=5、aux=0.2；数据 813,145 局，`num_workers=2`、`file_batch_size=8`。
- 结果：200k 步 / 22.9h / 2.56 it/s；dqn_loss 0.467→0.392，cql 1.261→0.506，next_rank 1.324→0.756；只看了训练池约 18.6%。
- **评测失效**：baseline 是随机初始化权重，几乎不会和牌 → 模型 2000 局和牌 10,729 次 vs 对手 126 次，avg_rank 饱和到 1.000、avg_pt 90；`best.pth` 永久停在 50k。该组指标只能证明“训练没崩”。
- 可用信号：holdout 202609 top1 随步数单调升至 0.7952（50k 0.7812→200k 0.7952），但 R²=−0.791（Q 幅度过估）；同牌山 600 局 ladder：200k vs 50k −5.18、150k vs 50k +4.20，方向相冲且均在 1.5 SE 内 → **50k→200k 强度持平**。
- 结论：dqn_loss 在降、人类一致率在涨，但强度不涨 → 模型在“更像人”而非“更能赢”；必须换信号源（GRP 质量/数据选择/跨局视界）或在线自博弈，单纯堆步数无效。

### 3.3 第 2 轮（R2，2026-10-09~10）

- 唯一变量：数据选择 + 奖励模型升级——`top_players_400_v1`（480,414 文件 / 704,231 席 / 1 epoch≈220k 步）+ `grp_h128`（h128/l3/60k，nll_last 1.2780→1.2361，val 2.44599）；吞吐 nw=8/fbs=16/RAYON=8。
- 执行：暖启 R1/200k；预算 450k，实际 **424,847** 步因离线 `num_epochs=1` 正常退出（不是故障）。
- 判定：A 方向 +3.299（SE 0.820，5 块全正），B 方向 −2.430（SE 0.856，5 块全负）→ **δ=+2.86 pt，SE 0.59，t=+4.83，95%CI[+1.70,+4.03]**；20,000 局仅 29 分钟（5 卡并行）。holdout 202609 top1 0.8122（峰值）。
- 经验：500 sets 时 SE≈±1.9，完全定不了 ±1–3 pt 的胜负，且小样本会低估效应（+0.70→+3.30）；`agg_ladder.py` 会自动算所需 sets。
- 资产：`checkpoints/round2/mortal.pth`(424,847)+step_250k…400k；判定牌谱 `logs/ladder_r2/`（806 MB）；`run_r2_ladder.sh`。

### 3.4 第 3 轮（R3，2026-10-10~11）

- 设计：与 R2 完全一致，唯一变量 `num_epochs=2`；暖启 R2/424k；预算 650k；6.7 it/s / 10.4h。
- 判定：A 650k 挑战 424k = −0.059（SE 0.845），B 反向 = −0.004（SE 0.838）→ **δ=−0.03，SE 0.595，t=−0.05，95%CI[−1.19,+1.14]** → 无增益（不是“测不出”，是测到接近 0，增益上界仅 +1.1）。
- 旁证：holdout 202609 top1 424k 0.8122→650k 0.8091；dqn 0.374→0.363、cql 0.458→0.452、next_rank 0.741→0.725；行为指标无方向性变化。
- 两个状态残留：① `best_perf` 继承 R1 随机对手时代的 `{1.0,90.0}`，导致 R2/R3 从未写 best.pth（后来已修：换代/暖启时重置为哨兵 `{4.0,−135.0}`，并加 `best_perf_baseline` 与一次性令牌）；② 自写解析与 Rust Stat 的平分 tie-break 不同（0.006–0.023 名次≈1.4 pt），判定以 ladder 为准。
- 方法论警告：固定种子块 `test_play` 跨轮配对给出 Δ=+4.52±2.07，而新种子块 ladder 只有 δ=−0.03±0.59——**4.5 pt 全是牌山过拟合**。
- **10-11 代码评审修复 4 个真 bug**：① 三个统计工具只取首局 scores、不扣立直 1000，导致 **11–13% 半庄名次错** → 按 `stat.rs` 重写并交叉验证；② 白名单 v1 因此有偏（阈值附近 53/400 人进出）→ 重建 v2（492,810 文件 / 735,249 席 / 244k 步）；③ round2/3 的 best.pth 从未生成、config 引用悬空 → 各 cp 终态快照；④ `holdout_metrics.py` 未传 `worker_init_fn`、`--seed` 形同虚设 → 已修。另实现 block bootstrap、`reset_best_perf` 一次性令牌，计分收敛到 `tools/mjai_score.py`。
- 资产：`checkpoints/round3/mortal.pth`(650k)+step_450k…650k；`logs/ladder_r3/` 815 MB。

### 3.5 第 4 轮（R4，2026-10-11~12）

- 设计：唯一变量 `enable_augmentation=true` + `augmented_first=true`（席位视角增强；同一文件增强前后样本数不变、obs 不同）；白名单换 v2（492,810 文件）；暖启 R2/424k；预算 645k（较父代 +221k，其中增强新增约 +145k 步）；新种子 80000+/90000+。
- **NaN 事故**：重启后从 541k 续训，542k 起 loss 全 NaN；根因是 AMP fp16 偶发溢出 + **BatchNorm running stats 在前向即被污染**（GradScaler 救不了 buffer）→ 前向永久 NaN，随后污染权重与 Adam 状态（Brain 572 个非有限张量含 162 个 BN buffer、优化器 820 个）；`add_histogram` 遇全 NaN 抛 ValueError 直接打死进程；链脚本未校验退出码，带 NaN 模型继续判定，跑出 avg_rank 3.92/avg_pt −125。
- 三重防护：① 存盘前对 state_dict 与 loss 累加做 `torch.isfinite`，非有限则 `SystemExit(1)` 且绝不覆盖 ckpt；② histogram 先过滤非有限值；③ 链脚本 `train_code=$?` 非 0 即跳过下游。回滚到 `best.pth`(500k) 并改 fp32 重跑。
- 判定：δ=+0.097，SE 0.593，t=0.16，解析 CI[−1.07,+1.26]；bootstrap SE 0.508、CI[−0.90,+1.07] → **无增益**；holdout top1 0.8112@600k（小样本，仅方向）。
- 运维：fp32 稳态 4.24 it/s（比 AMP 6.7 慢 36%），显存 5.2 GB、GPU 利用率 91–94%；三重防护全程未再触发。
- 资产：`checkpoints/round4/mortal.pth`(645k，现役最强)+best(500k)+step_450/500/550/600k；`logs/ladder_r4/` 807 MB；TB 坏段取证 `logs/tb_archive/main_r4_nan_and_ghost/` 72 KB。

### 3.6 奖励口径消融（2026-10-12~13）

- 设计：严格固定 R4 v2 数据池；父代 `round4/645k`；三组均 +111k→756k、fp32、同模型/GRP/CQL/aux；仅奖励不同：A=`rank_pt` `[6,4,2,0]`（控制），B=`rank_pt` `[2,1,0,−3]`（对齐天凤），C=`delta_points` + `reward_scale=1e-3`；新种子 100000/110000。
- 结果：A −0.04（CI[−1.03,+1.02]）；B +0.18（CI[−0.85,+1.19]，方向正向未显著）；C **−5.61**（CI[−6.67,−4.55]，t=−9.38，确证变弱）。C 的 A 方向 5 块全负、B 方向 5 块全正，典型反对称形态。
- 终态 holdout：A 0.8179/R²−0.65；B 0.8158/R²−1.01；C 0.7960/R²+0.01。C 的 R² 首次转正但强度崩——**“Q 能拟合分数差”与“会打牌”是两回事**。
- 解释：分数差方差约为顺位奖励 40 倍（var_t 22.2 vs 0.54），CQL 权重被稀释；每局分数差被对手行为主导，噪声大；`scale=1e-3` 后目标范围仍 ±16，是 rank_pt（±3.2）的 5 倍。
- 结论：C 路线关闭；**B 采纳为新基线**（top400_v2 + `[2,1,0,−3]` + augmentation）。资产：`reward_ablation/{A,B,C}_*/`，元数据 `mortal/reward_ablation/experiment.json`。

### 3.7 第 5 轮（R5，2026-10-13）

- 设计：以 B/756k 为父代，唯一变量 top400_v2→**top200_v3**（293,416 文件 / 344,068 席 / 1 epoch≈114k 步）；其余继承 B（`pts=[2,1,0,−3]`、augmentation=true、fp32、同 GRP/CQL/aux、不重置优化器）；预算 870k；新种子 120000/130000。
- 执行：首次误选 GPU0 P100，fp32 仅 2.48 batch/s；换 GPU2 RTX 8000 后 4.36 batch/s，训练从 12.8h 降到 7.3h。**主训固定 GPU2**。
- 判定：δ=−0.605，解析 SE 0.594、t=−1.02、CI[−1.77,+0.56]；bootstrap SE 0.508、CI[−1.61,+0.36]，ρ≈+0.185 → **无结论、方向偏负**（未出现 C 组式干净反对称，不能称确证变弱）。
- 结论：top200 席位只有 top400 的 ~45%，“少而精”未兑现；连同 R3/R4/消融 A，“同分布层面挪数据”已连续 5 次否掉 → **数据 reshuffle 路线到头**。现役最强仍 round4/645k。
- 资产：`checkpoints/round5/mortal.pth`(870k)+best(850k)+step_800/850k；`logs/ladder_r5/`；`logs/round5_chain.log`。

### 3.8 hp_ablation（2026-10-13~14）

- 设计：三条分支从 `round4/645k` 暖启、`reset_optimizer`、top400_v2/B 配方，+111k→756k；只改一个旋钮：`cql_min1`（cql=1）、`cql_min10`（cql=10）、`aux_next0`（aux=0）；控制组 = B 本身（cql5/aux0.2）；新种子 140000/150000。
- 结果：aux0 +0.59（CI[−0.42,+1.61]）、cql1 +0.05（CI[−0.97,+1.06]）、cql10 −0.07（CI[−1.09,+0.99]）→ 全部无结论；holdout 0.812–0.817，R² 全负；var_q/dqn_loss 随 cql 升高（0.29→0.44→0.51）但不转化为强度。
- 插曲：GPU1 热降频至 607 MHz（同温 GPU0 1240 MHz），cql_min10 从 4.3→1.2 batch/s；kill 原链后原地续训到凉卡（cql_min10→GPU2 RTX，cql_min1→GPU4 P100），reset 令牌已消费不会二次触发。
- 结论：**超参不是瓶颈**，排除“只是超参没扫对” → 直接支持目标形式改造。资产：`mortal/hp_ablation/<tag>/`、`logs/hp_resume_chain.log`、`prepare_hp_ablation.py`、`run_hp_ablation.sh`、`resume_and_judge_hp.sh`。

### 3.9 IQL Stage A：跨局信用分配（2026-10-14）

- 设计：唯一实质变量 `env.credit_mode: kyoku→game`；Q 目标改为全程回报折扣后缀和 `returns[k]=Σ_{j≥k}γ^(j−k)r_j`；γ=1 闭式 = `pts[final_rank] − E_grp[pts|第 k 局开始]`，GRP 中途期望自动成控制变量、telescope 中间项抵消、终局无需特判。数据/名单/B 口径/CQL/aux/GRP 与 B 锁死。
- 代码：`dataloader.py` 加 `credit_mode/gamma` 与后缀和；`train.py` 把 `credit_mode` 进 `_ENV_REWARD_KEYS` 守卫，game 模式不再乘 `γ^steps_to_done`；`holdout_metrics.py` 透传；`test_credit_modes.py` 五层 ALL PASS（闭式恒等 8.9e-16、GRP 终局 one-hot、telescope、kyoku 回归等价、γ=0.5 生效、守卫）。冒烟 7/7。
- 执行：B/756k 暖启→867k，7.25h，4.52 batch/s（后缀和零 GPU 成本）；holdout top1 0.8118、R²−0.3734、var_q/var_t 0.92/2.42。
- 判定：A 方向 −1.435（SE 0.847），B 方向 +0.621（SE 0.837）→ δ=−1.028，解析 t=−1.73；解析 CI[−2.20,+0.14] 跨 0；bootstrap SE 0.517、CI[−2.06,−0.02] 不跨 0，工具判“变弱”，但上界仅 −0.02 属**边际显著**。
- 结论：不采纳；产物 `checkpoints/iql_a`(867k) 归档、不替代基线；代码不回滚（默认 kyoku 与历史逐位等价）。**“把整场回报回传给早期动作”的 MC/控制变量路线关闭**；瓶颈不在信用分配视界，而在没有价值传播/策略改进。资产：`mortal/iql_a/`、`logs/iql_a_chain.log`、`config.iql_a{,_smoke}.toml`、`run_iql_a{,_smoke}.sh`。

### 3.10 IQL Stage B：真 IQL（2026-10-15~16）

**机制**：每样本附 `next_obs`（末条 terminal 自身占位）+ `r_t`（仅局边界非零）；三损失 `L_Q=0.5·MSE(Q(s,a), r+γV(s'))`、`L_V=τ-expectile(Q̄(s,a_data)−V(s))`、`L_AWR=优势加权 CE（β=3.0）`；CQL 关闭；`γ=0.997`/决策步；EMA target（mortal+dqn，decay 0.995，参数 polyak + BN buffer 跟随）。模型加 `VHead(1024→256 Mish→1)`。

| 轮次 | 关键差异 | 结果 | 有效性 |
|---|---|---|---|
| v1 | τ=0.7；`r_t` 每样本重复计入 16× | δ=−31.04、top1 0.6917、lse −176.1 | 无效（实现缺陷） |
| v2 | 修 r_t/L_V；τ=0.7 → 价值指数漂移 | δ=−3.416、CI[−4.46,−2.36]、top1 0.8069、lse +45 | 不作判决（目标不稳定） |
| v3 | v2+τ=0.5（唯一验证稳定修复） | **δ=+0.646，bootstrap CI[−0.35,+1.63]、top1 0.8167、lse +1.596** | **首份干净判决：无结论 → 路线关闭** |

v3 冒烟 10/10：avg_rank 2.527、top1 **0.8134**（>父代 0.8112）、lse **+1.343**；三代 avg_rank/lse = v1 2.789/−176.1 → v2 2.840/−33.6 → v3 2.527/+1.343。

**两处实现偏差（v1 作废根因）**：① `dataloader.py` 把 `kyoku_rewards[at_kyoku[i]]` 发给该局每个样本，导致 Bellman 沿链重复累加；诊断脚本实测非零 r_t 2304 条 vs 应有 141 条，放大倍数恰=每局平均决策点 **16.00×**；修为 `r_t = kyoku_rewards[at_kyoku[i]] if dones[i] else 0.0`。② `L_V` 的 expectile 误对全部合法动作（~30 个未受监督 Q 值）取上分位，V 被外推拖走并经 V(s') 自激；修为只对 `q_bar_all[range(bs), actions]` 回归。测试随之改为断言设计意图（非零个数==局边界数、总和==逐局总和），并给冒烟加数值型行为门禁（avg_rank≤2.60、top1≥0.78、abs(lse)≤10）；门禁当场抓出第二处缺陷。

**v2 价值漂移（τ=0.7）**：q_target p50 从 757k +0.58 指数增长到 861k +60.8（倍增 ~32k 步），超理论界 [−5,+5] 约 12 倍；q_loss/v_loss 同步上升。根因是 expectile 不对称的**棘轮效应**：增长 regime 中 V 持续落后于上升目标，残差为正，τ=0.7 对正残差强上拉（0.7）、负残差弱下拉（0.3），每轮净向上蠕动，速率∝当前尺度 → 纯指数。5 臂 30k 步探针：对照 +2.99、V-EMA +2.19、γ=0.99 +2.59、**τ=0.5 +0.20（唯一稳定）**、clip8 待出；γ 与 target lag 只影响传播速度、不影响注入，故无效。

**v3 稳定性**：τ=0.5，`iql scale:` 每存盘点监控；103 个存点 max q_target p50=1.206（v2 同点 +60.8），p99 +2.04，q_pred 紧贴；δ=+0.646，t=+1.09。按预注册守则 CI 下界>0 判进，否则关闭；v3 落“无结论”格 → 与 Stage A 合并，**目标形式改造两子路线全部关闭**。

**归因（基于实测的推断）**：① B 基线每局平均 16 个决策点、γ=0.997 → 局内折扣 0.997^16≈0.95，`kyoku_rewards` 本就近全权重回传，A/B 真正改的只有跨局视界；② 终局 pts 共同分量 std≈1.2 被 ~190 个决策分享，单决策因果效应 ~±0.1 → SNR≈1:10，跨局信号噪声太大；③ AWR 在近最优专家数据上权重≈均匀（top1 0.8167，与专家分歧只剩 ~18%，多为合理方差）；④ 与 R3/R4/R5/hp_ablation 的 null 一致：**瓶颈是数据分布的信息天花板，不在目标函数/优化维度**。

**资产**：`checkpoints/iql_b_v3/`(867k+best+step_800/850k)；`iql_b_v3/{train,holdout_202609,ladder_summary}.log`；判定牌谱 `iql_b_v3/ladder/`；`logs/iql_b_v3_chain.log`；`config.iql_b_v3.toml`+`run_iql_b_v3.sh`；v2 产物 `checkpoints/iql_b_v2/`（漂移轮，不作基线）。

### 3.11 S6a / S6c：特征注入（2026-10-17~18）

**S6a（obs v5，紧迫度）**：v5=v4+6 行——与排序后 1/2/3/4 名的点差 `d1..d4` + `overtake_gap` + `lead_gap`；全部由 `state.scores` 确定性导出、归一化 [0,1]，不做剩余局数假设。首层 Conv1d 手术（新通道零初始化，前向与 v4 逐位等价）；**完全隔离**在 `s6a_isolated/`（独立 code/`libriichi.so`/index/ckpt/runs/logs，输入软链只读共享树；共享树已还原 v4）。回归 17/17、冒烟 ALL PASS；正式轮 867k / 7.8h / 4.5 batch/s；holdout 0.8160、lse +0.467、R²−0.995；δ=+0.182、解析 CI[−0.99,+1.35]、bootstrap CI[−0.82,+1.20] → **无结论**。归因：6 个特征全是 v4 已有分数通道的确定性函数 → **零新信息**，ResNet 本就能算；该 null 只否定“重排已知信息”，不否定 S6 方向。种子块 240000/250000 已消费。

**S6c（自家速度类）分析**：acceptance/期望巡目/牌型大多已被 v4 obs 显式编码——向听 6 槽 one-hot、自家待牌 34 维 one-hot、SP 每候选 `required_tiles` 2×34、逐巡 tenpai/win/EV 3×17、`doras_owned`/`tiles_seen`。SP 仅算 shanten≤3（`SHANTEN_THRES=3`），盲区 shanten≥4 占决策点 **8.72%**（3,622 点实测：0:14.8%/1:25.4%/2:31.6%/3:19.6%/≥4:8.72%）。结论：**纯重算版不单烧轮**；优先级 S6b（读牌后验，真新信息）> S6c（盲区补全）> S6d（GRP 扩展）；建议 S6b+S6c 合并成一次 **v6** 注入，先 30k 步探针（~2h，门禁 holdout/行为画像），下一块种子 260000/270000。

### 3.12 读牌模型（2026-10-16~18，`tools/read_model/`）

**定位**：监督学习预测三家暗牌后验与听牌分布，输出校准概率 + 硬约束层；对 `mortal/` 与 `libriichi/` 零改动，尚未注入 Mortal。强度结论只认路径 C 注入 + ladder 在线验收，离线指标仅诊断。

**数据管线**：4 个 `PlayerState` 同步回放；观察者 obs 用 v4 `encode_obs(4)`，三家标签取 `tehai`/`waits`；公开信息独立台账（`tiles_seen_A=自家tehai+c_total`，c_total=四家打牌+鸣牌+暗杠+加杠+宝牌指示，逐局重置）；快照=观察者决策点+局内公告；课程阶段过滤前置。关键修正：听牌标签=有效荣和集 `waits & ~river & expressible`（河牌对所有目标家硬掩码）；加杠净贡献 0 → `hand=13−3*(吃+碰+大明杠+暗杠)`；`[u8;34]` getter 返回 bytes、`[bool;34]` 返回 list；万↔筒增广必须保持 aka。

| 版本 | 结构 | 关键结果 |
|---|---|---|
| v1 | 单帧 1D ResNet（1012→96，12 块）+ H1 计数分配/ H2 听牌/ H3 辅助 + 硬约束（IPF）1.13M | alloc CE 0.6705 vs 二项 0.7271；ECE 0.0038；top1 0.566；wait hit@1 0.139、AUC 0.628、立直 hit@1 0.124 |
| v2 | 三流：空间 + 手切时序 Transformer（43 维 token×≤26，2 层 d=64/4 头，手写注意力便于 ONNX）+ 结构估计流；1.36M | vs v1：alloc 0.5658→0.5830、立直 hit@1 0.1239→0.1447、wait hit@1 0.1464、**AUC 0.6276→0.6855**；结构头向听 25.7%<28.4% 基线 |
| v2s | 六改：向听分布入瓶颈（38→46 维）、向听直接 CE、分解-向听一致性（可微软向听+形制门控 normal）、tenpai 一致性、aux 向听×3、类均衡 | 向听 0.2574→**0.5263**；AUC 0.6855→**0.7392**；头↔分解一致率 0.28→0.93；代价 wait hit@1 −0.014、立直 −0.017、alloc −0.007 |
| v2x | 38 维瓶颈 + v2s 新损失（A1 归因） | wait 0.1386、AUC 0.6670、向听 0.5257、alloc 0.5719、立直 0.1295 → 向听学习全靠损失；AUC 增益全靠瓶颈馈通（损失单独反伤 AUC −0.019）→ **保留 46 维，v2s 定妆** |
| v2s@50k（A2） | 同配置扩预算 | wait Δ(vs v2)=−0.0112 显著未回补；AUC 0.6671（−0.0184）、alloc −0.0051、立直 −0.0282；但 wait_hit5 +0.0473、deal_pure 0.0422→0.0286、riichi_deal_pure 0.0641→0.0428、向听 +0.2747 → **不按原判据作注入基线** |
| A2b | 输出侧 tenpai BCE（`wait[34]` 拟合 `1−tenpai`，全目标，权重 0.2） | **已判决（best@43k, val 2.0924）**：AUC 0.6671→**0.9461**（+0.279，CI[+.273,+.285]）；wait_hit1 0.1343（Δ vs v2 −0.0121，达 −0.015 护栏点估计；CI 下界 −0.0207 略越界）；wait_hit5 0.5497、deal_pure 0.0245、向听 0.5330、一致率 0.9520。vs v2：deal_pure/riichi_deal_pure 显著更安全、wait_hit5 +0.0596、向听 +0.2756；代价 riichi_hit1 −0.0330、alloc −0.0067 → **预注册 (a)(b)(c) 全满足，定为注入基线** |
| A3 | 规则法基线（纯 numpy 零学习：现物/筋/壁 + 立直枚举 + 见逃排除；A2b 的「vs 人类规则」参照） | **已判决（2026-10-18，同 20000 固定评估集 + bootstrap 200 CI，`compare_a3.py`）**：alloc_ce 0.6996（均匀二项 0.7271 / A2b 0.6617）、alloc_top1 0.5204、wait_hit1 0.1166、hit@5 0.4857、wait_ce 3.6582、**tenpai_auc 0.9468**、**riichi_hit1 0.1121**、deal_pure 0.0574（不过滤口径 deal_pure_all 0.0091）、随机弃牌点炮 0.0709。判定：A2b 全指标优或统计打平，规则法达 A2b 的 87%~96%（hit1 87% / hit5 88% / alloc CE 质量 96%）；**tenpai_auc 与 riichi_hit1 统计打平**（CI 跨 0）——巡目+摸切尾逻辑回归追平神经网络的「该不该防」，筋/壁/宝牌+枚举支撑集追平「点名立直家听牌」；A2b 真实优势在 hit@5（−0.064）、wait CE（−0.758）与防守选牌（deal_pure_all 0.0041 vs 0.0091）。**标定证伪的经典先验**（202508 非评估月，`calibrate_a3.py`）：字牌危险只成立于初盘（turn≤3 字牌等待占比 17.3% → turn≥8 仅 1.2%）；「留幺九」不适用于持有倾向（字牌持有率仅中张 45%）；晚切筋无效（条件于听牌后归 1，+2.3 表观效应全是巡目混杂）；壁对持有率强（×0.57）对等待只是弱效应（×0.75）。**「modest 增量」逐指标分析与注入期望校准见下方专段** |

**A2 事故（10-17）**：A2 首跑两次在 step~7500 **确定性**崩溃（DataLoader worker Bus error/Segfault，与 worker 数/墙钟无关）。根因 = 共享树 `libriichi` 被 s6a 实验改版（v5 obs 补丁）+ `mortal/libriichi.so` 未随还原重建；还原并重编后 `obs_shape(4)=(1012,34)`、39 项回归 ALL PASS。处置：作废被污染的 8000 步 checkpoint（隔离 `mortal/read_model_v2s_50k.contaminated_so/`），从头重跑。教训：libriichi 实验必须完全隔离（s6a_isolated 模式）、还原后必须重编 `.so` 并跑回归门禁；`except Exception` 捕获不了原生段错误。

**A2b 实现与机制**：`--wait-tenpai-weight 0.2`（`l_wait_tenpai = BCE(wait_logits[...,34], 1−tenpai)`，全目标）；梯度验证：8 样本无听牌目标批次下 h2_out 梯度 0→0.456（复现并修复 A2 缺陷）。A2 机制 = 非听牌目标（85.5%）对 `wait[34]` 无直接监督，长训练 P(未听) 集体塌缩（非听牌均值 0.0122→0.0080、中位 0.0030→0.0013）→ AUC 0.739→0.667；A2b 后非听牌 P(未听) 回到 0.058–0.082、AUC 0.946。预注册判据（已满足）：AUC≥0.72、deal_pure≤0.0408、wait_hit5≥0.50、wait_hit1 Δ≥−0.015。

**P1 分层诊断（10-18，`strat_metrics.py`，A2b@20000 固定集）**：头部 AUC 0.946 被"简单局"抬高——按公开信息分层：无立直 0.898、无立直且无鸣牌 **0.833**、**无立直无鸣牌且 turn≥8 仅 0.683**（该子集 hit1 0.073、hit5 0.461、alloc 信息增益 0.163 最高）；called=1 AUC **0.775**（开手判别更难）vs called=0 0.966；**riichi=1 的 hit1 0.112 反而低于 riichi=0 的 0.155**（立直后本应最准 → 明显改进点）、riichi deal 3.56% vs 非立直 0.96%。hit1 随机基线 0.049（模型 2.75×）；turn0-3 的 hit1≈随机（该段听牌率仅 0.6%）；**alloc top-1 0.715≈众数 0.707**（该指标信息量极低，应看 CE 0.662/信息增益 0.115/MAE 0.422）。**结论：进步空间集中在 ① 困难局听牌判别、② 精确点名（尤其立直后）、③ 开手（called=1）判别**；全量 AUC 已近饱和。产物 `strat_report.json`。

**A3 对决判决与「modest 增量」分析（10-18，`compare_a3.py`，同 20000 固定评估集 + bootstrap 200 CI；Δ=rule−net，正值=规则更差）**

**判决：A2b 相对人类规则法（零学习手写规则）的增量真实但 modest——10 项指标 8 项 net 优、2 项统计打平，无一指标规则反超；但「优」的绝对幅度普遍很小，且两项最决策相关的指标与规则读者无法区分。**

| 指标 | 均匀二项 | A2b | A3-rule | Δ(rule−net) [95% CI] | 规则已拿「均匀→A2b」增益的 | 判定 |
|---|---|---|---|---|---|---|
| tenpai_auc | 0.5000 | 0.9461 | 0.9468 | +0.0007 [−0.0019,+0.0033] | —（点估计规则反超） | **打平** |
| riichi_hit1 | 0.0875 | 0.1116 | 0.1121 | +0.0005 [−0.0132,+0.0121] | —（点估计规则反超） | **打平** |
| wait_hit1 | 0.0659 | 0.1343 | 0.1166 | −0.0177 [−0.0285,−0.0074] | **74%** | net 优但增量小 |
| wait_hit5 | 0.3262 | 0.5497 | 0.4857 | −0.0640 [−0.0773,−0.0536] | 71% | net 优（最大绝对差） |
| wait_ce_active | 5.0285 | 2.8999 | 3.6582 | +0.7584 [+0.7388,+0.7784] | 64% | net 优（相对差距最大） |
| alloc_ce | 0.7271 | 0.6617 | 0.6996 | +0.0379 [+0.0374,+0.0384] | **42%（规则最弱项）** | net 优 |
| alloc_top1 | 0.4501 | 0.5763 | 0.5204 | −0.0559 [−0.0606,−0.0512] | 56% | net 优 |
| deal_pure_all | 0.0144 | 0.0041 | 0.0091 | +0.0050 [+0.0031,+0.0073] | 51% | net 优但绝对量级小 |
| riichi_deal_pure | 0.1152 | 0.0356 | 0.0679 | +0.0323 [+0.0160,+0.0512] | 59% | net 优 |

**「modest」的四条证据**（这是本项目对读牌模型价值的关键校准，勿忘）：
① **两项决策核心指标统计打平**：「该不该防」（tenpai_auc）——巡目+摸切尾+鸣牌的简单逻辑回归即追平神经网络（0.9468 vs 0.9461，CI 跨 0）；「点名立直家听牌」（riichi_hit1）——筋/壁/宝牌加权 + 立直枚举支撑集即追平（0.1121 vs 0.1116，点估计规则反超 +0.0005）。
② **规则法已吃掉大部分增益**：wait_hit1 上规则拿到「均匀→A2b」总增益的 **74%**（A2b 净剩 +0.0177，相对规则仅 +15%）；hit@5 拿 71%（净剩 −0.064）；wait_ce 拿 64%（净剩 +0.758 nats）。除 alloc 外，A2b 的净增量多在 0.02~0.06 绝对量级。
③ **A2b 的两个实质优势**：(a) wait_ce_active +0.758 nats——规则把概率质量摊在规则/枚举支撑的 ~20 个候选上（支撑集均值 23.5 张），A2b 分布锐利，这是最大的相对差距；(b) alloc_ce +0.038 nats——规则只拿 42%，是其最弱项：乘性先验-二项无法表达「具体搭子」级的持有结构（中张/字牌/宝牌粗粒度倾向 + 河牌/筋覆盖不了），此处神经网络 genuinely 增量最大。
④ **防守指标必须看口径**：deal_pure 项目口径 0.0245 vs 0.0574（差 2.3×）有误导性——该口径过滤「所选牌概率>0」的场合，系统性滤掉规则法的枚举证明安全牌/genbutsu（规则纯子集 n=1394，A2b 几乎全量）；换不过滤口径 deal_pure_all 后仅差 0.0050（0.41% vs 0.91%），统计显著但实战量级小。

**三方汇合（一致结论：读侧已近公共信息天花板，剩余空间集中在困难子集）**：A3（自下而上：手写规则 = A2b 的 87%~96%）+ §3.14 Bayes 上界（自上而下：参考生成式读牌器总体已达 A2b，未测出大块剩余空间）+ P1 分层（剩余增益集中在 无立直无鸣牌 turn≥8 的 AUC 0.683、called=1 的 0.775、riichi hit1 0.112 < 非立直 0.155）——三个独立证据链同指：全量读牌指标已近饱和。

**对 B1 注入的期望校准（据此调整投入）**：① 注入增益的主要来源**不应**是「读得更准」——读侧剩余空间有限且集中在困难子集，全量读牌指标 modest 增量意味着 ladder 收益可能主要来自决策融合而非读牌精度；② 应来自「后验的用法」：防守门控标定（§3.14：A2b 的 P(听牌) 过标定 6.3×，均值 0.921 vs 实际 0.147，注入前必须重标定否则门控全时开）、攻/守分档、与 SP/牌型潜力融合；③ 若继续投读侧，瞄准分层诊断的三个困难子集（困难局听牌判别、立直后精确点名、called=1 开手判别）而非全量 AUC；④ ladder 门禁判读仍按 §2.6 铁律（CI 下界>0 采纳、跨 0 无结论、<0 回滚），但先验期望应放低——读牌精度已不是瓶颈，决策用法才是。

**工程资产**：回归 39 项 ALL PASS（枚举器 vs Rust `calc_all` 20.3 万手牌零不一致、时序 token 独立重算、与 GameplayLoader obs 逐位一致、136 牌守恒、tiles_seen、增广等变、学习门禁、结构头专项）+ A3 专属 40+ 项 ALL PASS（`test_rule_baseline.py`：alloc 不变式与 flat 不动点、wait 硬/软规则方向、**立直枚举暴力重算**——独立「精确划分」判据 vs 被测「子集搜索」语义，6 随机小池+构造案例+5 张同牌容量边界+暗杠参数化+见逃排除、dora 映射、真实数据门禁）；`compare_bootstrap.py` 做 CI 判决；`calibrate_a3.py` 权重标定（只用非评估月，常数冻结）；`run_v1.sh/run_v2.sh/run_v2s.sh/run_v2s_a2b.sh/run_a3.sh`。dataset 增补 `meld_counts`/`dora_indicators` 两个公开信息样本字段（39 项回归复跑 ALL PASS，训练语义零变化）。下一步：B0 立直振听/吞吐（`with_sp_tables`）→ B1 限域注入（读牌冻结，wait_probs 35 维 + exp_counts + 向听分布作额外特征；防守/全决策两档；**期望校准：读侧 modest 增量已证，注入收益重点押「后验用法」= 防守门控标定 + 攻守分档 + SP/牌型融合，而非读牌精度**）→ B2 ladder 门禁（CI 下界>0 采纳，跨 0 无结论，<0 回滚）。

### 3.13 项目准备期（2026-10-01~10-09，压缩）

- 10-01：完成 Rust+Python 架构分析；确认作者不公开权重、官方 Training 文档全为空占位；实测天凤 `list.cgi`/scc/`0/log`，年度 zip 404，滚动日归档可用。
- 10-02：发现 HF `tenhou-to-mjai` 2009–2025 全量；自建 2026 管线；实测 HF 2009/2026 包合并成功；D 盘 exFAT 簇 256 KiB 导致小文件膨胀 64×（13 万文件逻辑 521 MiB 实占 33 GiB，全量需 ~620 GiB）→ 用户拍板只保留 18 个年度 tar（16.86 GiB），不转换。
- 10-06：服务器环境落地（conda mortal/py3.12.11/torch 2.6.0+cu118/rust 1.99，`libriichi.so`+CLI 编译通过）。
- 10-07：HF 18 包 merge-hf + prune --apply → 2,630,640 局；GRP 30k 步（21 min）；R1 200k 步（22.9h）。
- 10-09：首次复盘——评测饱和、best.pth 陷阱、吞吐瓶颈、代码补丁清单；建本地 git 基线、玩家强度榜、holdout_metrics/ladder/bench_pipe；R2 启动。
- 采集预算存档（2026-10-01 精算）：2026 YTD 实测 129,475 局、raw+mjai≈1.53 GiB；历史 2009–2025 约 340–680 万局，仅粗估 40–80 GiB（未落地；最终走 HF 年度包）。

### 3.14 读牌 Bayes 上界（2026-10-18，`tools/read_model/bayes_bound.py`）

**目的**：A2b 的分层诊断（§3.12）只说明「未饱和」，不能区分 **(a) 模型没学到** 与
**(b) 可观测信息本身只有这么多**。本实验构造 Bayes 可达性能估计来分开两者。

**三个信息集（tier）**

| tier | 条件化信息 | 求解 |
|---|---|---|
| `counts` | 观察者手牌 + 全部可见牌**张数** | ✅ 精确：未知牌在「三家暗牌（张数=13−3·副露）+ 牌山」上均匀（洗牌⇒超几何），无任何行为假设 |
| `hard` | `counts` + **立直⇒听牌** | ✅ 立直家从 tenpai 手牌流形抽样（完整和牌形−1 张，按 1/#待牌 纠偏） |
| `behave` | `hard` + 舍牌行为似然 | MH（非立直）/ 重要性采样（立直） |

**行为似然**：`L(H)=Π_k P(d_k|H+d_k)`，`P(舍 x|手牌)∝exp(−β·value(x))`，
`value` = 对子/刻子 + 同色 ±1/±2 邻接（ukeire 代理）。语义 = 「打掉的都是废牌」。

**两条被实测否掉的设计（重要，勿重走）**
1. **弃后向听当主信号**：14 张手牌向听对几乎每张舍牌都不退化，无法区分「留住有用
   摸牌」与「打了无用摸牌」，且贵 34×。
2. **完整轨迹滤波**：前向粒子滤波因「手切⇒该牌当时在手里」逐巡死亡（每巡存活
   ~15%，连续 10 次手切 ~1e-8，实测必然全灭）；反向轨迹滤波虽不死，但「被留住的
   摸牌」是隐变量，随机抽样会把 tenpai 手牌组合打散（实测 tenpai 反而 −6 nats）。
   改静态似然后 tenpai vs 随机手牌平均对数似然 **+8 nats（β=1）/ +17 nats（β=3）**。
3. β 在 **202608（非评估月）** 标定：β=1 时后验均值 P(听)≈经验听牌率（1.3×）。

**结果（202609，2500 快照 / 7500 目标；`mortal/read_model_bayes/`）**

| 分层 | 指标 | A2b | counts | hard | behave |
|---|---|---|---|---|---|
| all (n=7500, 听14.7%) | 听牌 AUC | **0.946** | 0.640 | 0.856 | 0.942 |
| | wait hit@1 / hit@5 | **0.137/0.542** | 0.057/0.266 | 0.079/0.314 | 0.120/0.442 |
| | alloc CE / MAE | **0.661/0.422** | 0.735/0.467 | 0.737/0.465 | 0.755/0.425 |
| | mean P(听) vs 实际 | **0.916 vs 0.146**（全量 20000） | 0.001 | 0.069 | 0.141 |
| riichi=1 (n=508) | hit@1 / hit@5 | **0.120/0.518** | 0.077/0.311 | 0.100/0.404 | 0.108/0.419 |
| | wait CE | **2.98** | 27.1 | 3.35 | 3.89 |
| called=1 (n=1305) | 听牌 AUC | **0.778** | 0.651 | 0.646 | 0.721 |
| riichi=0&called=0 (n=5693) | 听牌 AUC | **0.840** | 0.540 | 0.530 | 0.818 |
| 干净困难局 turn≥8 (n=1744) | 听牌 AUC | **0.690** | 0.515 | 0.500 | 0.581 |
| 干净困难局 full-set（n=13779，`counts_full/`） | 听牌 AUC | 0.683 | **0.519** | — | — |

**结论（回答「还能提取多少」）**
1. **数牌信息几乎不含听牌信息**：干净局 `counts` AUC 0.52（≈无信息），全量 0.640 的
   来源只有手牌张数/副露数。⇒ A2b 的 0.683（干净局）/0.946（全量）是**真实的行为
   读牌能力**，不是计数/规则占的便宜。alloc 侧同理：`counts` CE 0.735 vs A2b 0.661，
   **行为信息带来 0.073 nats/列（10%）**。
2. **参考生成式读牌器（counts+立直+价值策略）总体已达 A2b 水平**：全量 AUC 0.942 vs
   0.946、alloc MAE 0.425 vs 0.422、`called=0` hit@1 0.103 vs 0.101。⇒ **没有测出
   「模型还有一大块可挖」的证据**；离注入前的「容量探针」应当按「已接近参考上限」
   预期，而不是「4× 参数就能涨」。
3. **三个仍落后于 A2b 的分层**（参考模型更差 ⇒ 模型确有额外信息，但参考不构成上界）：
   干净困难局 AUC 0.581 vs 0.690、`called=1` 0.721 vs 0.778、wait hit@1 0.120 vs 0.137。
   ⇒ 这三处「上界未咬住」，需更强的人类策略模型才能定论；不能用本实验声称还有多少。
4. **立直家的问题被量化**：只要知道「立直⇒听牌」+ 数牌先验（零学习），hit@1 就有
   **0.100 / CE 3.35 / hit@5 0.404**；A2b 是 0.120/2.98/0.518。⇒ A2b 在立直家上
   **只比「听牌先验」多 0.02 hit@1**，手牌特异性信息极少。立直专项的目标应当以
   `hard` 的 0.100/3.35 为**新地板**，而不是以 0.112 的随机对比自我安慰。
5. **A2b 的 P(听牌) 严重过标定**：全量均值 0.916 vs 实际 0.146（6.3×），turn0-3
   为 0.760 vs 0.006（125×）。排序（AUC）不受影响，但**作为概率直接注入/设阈值会
   错**（作为特征喂给主模型可自行重标定）。⇒ 注入前必须做 Platt/isotonic 重标定，
   或只注入排序分位。参考模型 `behave` 的均值 0.141 ≈ 实际 0.147，可作校准参照。
6. **`called=1` 的 AUC 瓶颈更像信息上限**：model 0.778 vs 参考 0.721，两者都远低于
   `called=0`（0.96/0.96）。「0.775→0.85」的目标缺乏参考支持，先做特征/辅助任务
   前应先确认不是信息上限。

**方法论**：`tenpai_mi_nats`（= H_emp − E[H(p)]）在该实验里**不可用作信息量**——
它奖励「过度自信且错误」的后验（`counts` 的 p≈0.001 使 E[H]≈0 而 MI 反而最高）。
判读一律用 AUC（序不变）与 CE/MAE（水平），并同时看 meanP vs 实际率。

**自检**：`--leak-test` PASS（重排非观察者配牌后后验逐位不变）；`shanten_fast` 与
`structure.structure_labels` 交叉验证 4000 手牌零不一致；张数守恒（136=13+1+39+83）
由构造保证；`hard` 在非立直目标上严格退化为 `counts`（CE 0.7355/0.7373）。

**资产**：`tools/read_model/bayes_bound.py`、`tools/read_model/shanten_fast.py`（numpy
向量化的 tomohxx 查表，4096 手牌 3.7 ms）、`mortal/read_model_bayes/{report,tables.txt}`、
`counts_full/report.json`（全 20000 快照的精确 counts 口径）、日志
`mortal/logs/read_model/bayes_bound.log`。成本：2500 快照 3 tier ≈ 23 min（192 链 ×
100 步 MH）。

---

## 4. 现状与下一步

### 4.1 路线状态

| 路线 | 结论 | 证据 |
|---|---|---|
| 数据选择（top400+GRP h128） | ✅ 唯一有效 | R2 δ=+2.86±0.59 |
| 同分布增量（epoch/增强/收紧名单/消融优化器） | ❌ 关闭 | R3 −0.03、R4 +0.10、R5 −0.61、消融 A −0.04 |
| 奖励口径对齐（B `[2,1,0,−3]`） | 🔶 采纳为基线，未显著 | +0.18，CI 跨 0 |
| 真实分数差奖励（C） | ❌ 关闭 | −5.61，CI[−6.67,−4.55] |
| 超参（cql/aux） | ❌ 非瓶颈 | 三分支均 CI 跨 0 |
| IQL Stage A 跨局信用 | ❌ 关闭 | −1.03，bootstrap CI[−2.06,−0.02] |
| IQL Stage B 真 IQL | ❌ 关闭（干净否证） | v3 +0.646，CI[−0.35,+1.63] |
| S6a 紧迫度特征 | ❌ 无结论（零新信息） | +0.182，CI[−0.82,+1.20] |
| 读牌 Bayes 上界（MCMC） | ✅ 已判定（§3.14） | 数牌信息≈无听牌信息（干净局 AUC 0.519）；参考生成式读牌器总体已达 A2b（0.942 vs 0.946）；**未测出大块剩余空间** |
| 读牌 P(听牌) 概率标定 | ❌ 严重过标定 | A2b 均值 0.921 vs 实际 0.147（6.3×）；作概率注入前必须重标定 |
| S6b 读牌后验 | ⏳ 当前信息侧主线 | 尚未注入；A2b 已定为基线（§3.12），但注入特征需重标定 |
| 扩窗 2019+ + 重训 GRP | ⏳ 数据侧主线 | 成本最高，IQL 关闭后接棒 |

### 4.2 下一步优先级

| 优先级 | 事项 | 要点 |
|---|---|---|
| **P0** | S6b+S6c 合并 v6 注入 | 读牌后验 + SP 盲区 acceptance/牌型潜力；先 30k 步探针（~2h，门禁 holdout/行为画像）；种子块 260000/270000；通过后再开正式轮。**注入前必须先重标定 P(听牌)**（§3.14：均值 0.921 vs 实际 0.147），否则防守门控会全时开 |
| **P0** | 读牌模型 A2b 收尾 | **已完成**：AUC 0.9461、wait_hit5 0.5497、deal_pure 0.0245、wait_hit1 Δ −0.0121 → 预注册全满足，A2b 定为注入基线。**A3 规则法基线已完成**（§3.12：A2b 全指标优/平，tenpai_auc 与 riichi_hit1 统计打平，规则法达 87%~96%）→ 下一步路径 C 注入（B0 立直振听/吞吐 → B1 限域注入 → B2 ladder 门禁） |
| **P0** | 立直专项（基线已换） | §3.14 给出新地板：counts 0.077 / hard（听牌先验）**0.100 hit@1、CE 3.35** / A2b 0.120。目标改为「显著高于 0.100 且 CE < 3.35」，而不是对照 0.112 |
| **P1** | 数据扩窗 2019+ | 与 GRP 重训、按年份审计同步做；主模型 holdout 仍用 202609+ |
| **P1** | 读牌路径 C 注入 | B0 立直振听/吞吐（`with_sp_tables`）→ B1 限域注入（防守/全决策两档）→ B2 ladder 双向 ≥5000 sets 判定 |
| **P2** | 外部锚点 akochan | 需 JRE + `AKOCHAN_DIR`/`AKOCHAN_TACTICS`；人类牌谱不能当 arena 对手（无 seed） |
| **P2** | 多卡并行对照 | 5 张卡各跑一组单变量配方，再用 ladder 选优；注意 CPU 争用会让双方慢 20–30% |
| **P2** | 工程债 | git push（本地 ahead）、异地备份、上游 `git diff` 校准、两套 holdout 同分布口径、agari-guard holdout、kan-select 回放对照、文档对账（`TRAINING_MANUAL.md` §4.x/健康判据） |
| **P3** | 在线自博弈（M5） | 仅在评测可信后开；`online=true` + `freeze_bn.mortal=true`；子进程规避卡死；P100×4 可当 client 集群 |
| **P3** | 防守行为专项 | 流局 10.9% vs 人类 15.3%、放铳 0.1405 vs 0.1255 → 弃和不足；试 cql 权重/奖励平滑/oracle 特征 |
| **P3** | 更强行为先验（若要继续压上界） | §3.14 的参考策略弱于 A2b；要真正回答「还剩多少」需 ukeire/安全度策略或学习型 policy（如用 A2b 之外的人类舍牌模型），并保留 `hard` 作为下界 |

### 4.3 当前资产与基线

- 现役最强：`mortal/checkpoints/round4/mortal.pth`（645k，fp32，0 非有限张量）。
- 判定基线：B/756k（`reward_ablation/B_rank_pt_aligned/mortal.pth`，top400_v2 + `[2,1,0,−3]` + aug）。
- 已封存不续用：`iql_b`(v1)、`iql_b_smoke`、`iql_b_smoke_v2`、`iql_b_v2`（漂移轮）。
- 可用但非基线：`round5/870k`、`iql_a/867k`、`iql_b_v3/867k`、`s6a/867k`。
- 判定牌谱均保留在对应 `ladder/` 或 `logs/ladder_r*/`，可事后重算/bootstrap。

### 4.4 新会话接手三步

1. 读 §0/§4.1，确认“现在在哪、下一步选什么”；环境看 §1.7。
2. `tmux ls` + `ps -eo pid,etime,args | grep python` 查残留；长任务只用 tmux，日志进 `mortal/logs/` 对应子目录。
3. 开跑前 `cd mortal && python3 env_check.py --cfg <本轮 config>`；训练完只认 ladder 双向 ≥5000 sets + 全新种子块 + block bootstrap CI。

---

## 5. 工程规范与坑位

### 5.1 强制规范

1. **长任务一律 tmux 承载**，不前台直跑、不 nohup；日志写 `mortal/logs/` 对应子目录（§5.3）。
2. **每次任务前重置全部环境变量**（§1.7），不依赖上次会话残留。
3. **冒烟必须含数值型行为门禁**：avg_rank、top1、logsumexp_q 等；结构性检查（无 NaN、到预算）不够。
4. **判定纪律**：全新种子块、双向 ≥5×500 sets、block bootstrap CI；固定 `test_play` 只作趋势。
5. **并行实验隔离**：改变 obs/目标/数据口径的实验必须复制代码+独立 `.so`+本地 index/ckpt，禁止就地改共享树（范例 `s6a_isolated/`）。
6. **停机规程**：SIGINT 优雅停 → 清理 DataLoader 孤儿 → 校验 ckpt 可 `torch.load` → 清阶段性产物 → git 提交停机说明。
7. **提交规范**：`git status --porcelain=v1` → 显式 `git add` → 中文提交信息含改动/验证/文档对账；运行产物先加 `.gitignore`。
8. **主模型训练优先用 GPU 2（Quadro RTX 8000）**；GPU 1 易过热降频（实测 SM 607 MHz、速率降至 1.2 batch/s），避免在其上安排长训练。

### 5.2 坑位总表（按主题，编号沿用旧版便于回查）

| # | 主题 | 坑与处置 |
|---|---|---|
| 1 | 数据 | scc 是 html-escaped，先 `html.unescape` 再正则，否则含 `<>&` 的玩家名行错位 |
| 2 | 数据 | 新行可能插在文件中间（长时对局），一切按 log_id 去重 |
| 3 | 数据 | `00a9`=四鳳、`00b9`=三鳳 是观测规律；过滤以行内规则文本为准 |
| 4 | 数据 | 牌谱 URL 响应可能已是 gzip；按 `1f 8b` 魔数判断，勿二次压缩 |
| 5 | 数据 | `list.cgi` 当前窗口没有 scd/sce 属正常（无对局不出现） |
| 6 | 数据 | 2026 年度 zip 未生成；今年数据靠 7 天窗口滚动，必须定期归档 |
| 7 | 训练 | `GameplayLoader` 必须在 `build_iter()` 而非 `__init__` 构造 GRP（Windows 下失效） |
| 8 | 训练 | 训练依赖 GRP 权重先存在（`[grp].state_file`），否则 dataloader 失败 |
| 9 | 训练 | CQL 损失仅离线模式生效；在线模式只有 DQN+aux |
| 10 | 训练 | 在线模式有已知卡死 bug，源码用子进程 + `sys.exit(0)` 规避，勿“修复” |
| 11 | 训练 | `save_every % opt_step_every == 0`、`test_every % save_every == 0` 等断言必须满足 |
| 12 | 训练 | `avg_pt([90,45,0,−135])` 仅显示/最优判定，从不参与训练 |
| 13 | 数据 | 旧本机 D 盘是 exFAT 簇 256 KiB，海量小文件膨胀 64×；历史数据只留年度 tar；服务器 NTFS 无此问题，但本机→服务器传输必须先打包 |
| 14 | 工程 | 当前已有 origin（SSH over 443）；仍需注意：临时身份 `mortal-trainer`、上传旧代码前先 `git status`、单机单副本/异地备份未完全解决 |
| 15 | 评测 | 随机 baseline 使 `test_play` 饱和（avg_rank 1.0、avg_pt 90），指标无区分度；`best.pth` 只记第一个点 |
| 16 | 评测 | `test_play` 写死 `seed_start=(10000,0x2000)` 且每次 `rmtree(log_dir)`；跨轮比较会牌山过拟合，历史牌谱也丢失 |
| 17 | 性能 | 吞吐瓶颈默认在 CPU 数据管道：nw2/rayon4=3.12 it/s vs GPU 8.88；nw4~8 + RAYON=8 可到 7.48 |
| 18 | 评测 | `best_perf` 初始值/继承与“更强 ckpt”无关；选模型用 ladder，不信任 best.pth |
| 19 | 环境 | `CUDA_VISIBLE_DEVICES` 默认按算力序，与 nvidia-smi 的 PCI 序不一致；必须加 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，启动后核对 `device: cuda:0 (型号)` |
| 20 | 环境 | `import libriichi` 必须在 `mortal/` 下；仓库根同名 Rust crate 目录会遮蔽成空命名空间包，`PYTHONPATH` 也无效 |
| 21 | 评测 | `ladder.py` 并发必须传独立 `log_dir`（开头 `rmtree`），否则互相删牌谱 |
| 22 | 训练 | 离线模式一个 epoch 就静默退出，与 `max_train_steps` 无关；日志末尾 `ResourceWarning` 是正常 GC 噪声；要跑更长提 `num_epochs` |
| 23 | 环境 | `LD_LIBRARY_PATH` 只对 `validate_logs`/`stat` 等 Rust CLI 必需（缺失退出码 127）；训练进程不需要 |
| 24 | 工程 | 运行中的 bash 脚本不能编辑（按字节偏移增量读取，会错位/重跑命令）；先停会话或换新脚本文件 |
| 25 | 工程 | `pkill -f "<模式>"` 会匹配执行它的 shell 自身；用 `pgrep -f "[t]rain.py"` 拿 PID 再 kill |
| 26 | 工程 | SIGINT 停训后 DataLoader worker 会变孤儿（PPID=1，继续占 CPU/句柄）；必须按 PID 补杀并 `pgrep` 归零 |
| 27 | 训练 | AMP fp16 溢出会永久污染 BatchNorm running stats（GradScaler 救不了 buffer）；坏 state 实测 Brain 572 个非有限张量含 162 BN buffer、优化器 820 个；兜底 `enable_amp=false` |
| 28 | 训练 | TB `add_histogram` 遇全 NaN 抛 ValueError 打死进程；链脚本不校验退出码会带 NaN 模型继续判定。已加 histogram 过滤 + `train_code=$?` 守卫 |
| 29 | 训练 | 同一 TB run 目录会混入多段事件文件（重启/崩溃/暖启），同 step 多点；坏段移到 `logs/tb_archive/`，判读时记住 step 跨轮延续 |
| 30 | 数据 | “结构合法但语义全错”最危险：`<UN>` 的 `n0~n3` 是 URL 百分号编码；直接写入后 validate_logs 全绿但白名单命中 0%。必须 `unquote` + `verify_names.py` 命中率验收 |
| 31 | 训练 | `gamma<1` **不是**跨局信用分配：`steps_to_done` 局边界归零，折扣只在当前局内；实测只有打牌(0–36)/立直(37) 参与折扣，杠/鸣牌/和了/pass 不折扣。真跨局需 next-state + target net |
| 32 | 性能 | 主训必须 GPU2 RTX 8000（P100 2.48 vs RTX 4.36 batch/s，差 1.76×）；脚本固定 `CUDA_VISIBLE_DEVICES=2` + `CHECK_GPU_NAME='RTX 8000'` |
| 33 | 训练 | 暖启冒烟的 `max_train_steps` 是全局绝对值，必须写“父代步数+冒烟步数”（如 758000），否则启动即 `budget already reached` 秒退；同步缩小 `test_every/snapshot_every` 并满足整除 |
| 34 | 工程 | `tmux new -d -s x 'cmd > /path/log'` 的重定向目录必须预先存在；否则 shell 秒退、零日志、`tmux new` 仍返回 0。先 `mkdir -p`，存活看脚本内 tee 日志 |
| 35 | 测试 | 回归测试若照“实现现状”写断言，等于给 bug 盖章；必须从设计要求逐条翻译（如“每局奖励只求和一次”→ 非零个数==局边界数、总和==Σr） |
| 36 | 测试 | 冒烟只查“结构活着”不查“模型没被打坏”会白烧一轮；必须加数值型行为门禁（avg_rank≤2.60、top1≥0.78、abs(lse)≤10），FAIL 禁开正式轮 |
| 37 | IQL | `L_V` 的 expectile 只能对数据里实际执行的动作 `a~πβ` 回归，不能对全部合法动作（~30 个未监督 Q 值）取分位；否则 V 被外推拖走并经 V(s') 自激。masked 统计量先 `where(mask,x,0)` 再乘方，避免 inf×0=NaN |
| 38 | 测试 | 幅度断言上界要数学推导：r_t = E[pts 在 k+1 局] − E[pts 在 k 局] 的上界是 pts 极差 `max−min`（B 口径 5.0），不是 `max(abs(pts))`（3.0）；双重计入的确定性判据是计数/多重集/总和 |
| 39 | IQL | expectile + γ≈1 + 边界稀疏奖励会价值尺度指数自激：τ=0.7 时 q_target p50 +0.6→+60.8（倍增 ~32k 步）；τ=0.5 是唯一验证稳定修复。慢漂移时间尺度 ~3 万步，2k 冒烟完全无感 |
| 40 | IQL | 慢漂移必须靠每存盘点 `iql scale:` 分位数日志做正式轮全程监控（间隔 ≪ 现象时间尺度 1/10），链脚本把 max p50 做成放行门禁（≤~5） |
| 41 | 工程 | 并行实验禁止就地改共享代码/替换共享 `.so`：已启动进程的 DataLoader/holdout 子进程会重新 import，`cp` 覆盖 `.so` 还可能被读到半写文件；一律隔离工作区（`s6a_isolated/`） |

### 5.3 日志归置规范（强制执行）

> 所有 `.log` 禁止散落 `mortal/logs/` 顶层；按文件名前缀/语义入子目录。启动前 `mkdir -p`（§5.2-34）。

| 目录 | 内容 | 示例 |
|---|---|---|
| `train/` | 主模型训练/轮次链 | `main_r{2..5}_train`、`round{4,5}_chain`、`round4_resume` |
| `grp/` | GRP 训练/探针 | `grp_train`、`grp_h128_train`、`grp_capacity` |
| `read_model/` | 读牌模型训练/守护 | `read_model_v{1,2,2s,2x}_{train,tmux}` |
| `iql/` | IQL 全链路 | `iql_{a,b,b_v2,b_v3,probe}_chain`、`status_iql_*`、`diag_iql_reward` |
| `eval/` | ladder/holdout/bootstrap | `ladder_*`、`holdout_r{3,4}`、`bootstrap_r{3,4}` |
| `smoke/` | 冒烟 | `smoke_*`、`smoke_v{2,3}_*` |
| `check/` | 代码/配置/环境检查 | `cfg_*_check`、`gamma_check_*`、`envcheck_selftest` |
| `test/` | 功能/行为测试 | `test_credit_modes`、`bp_test*` |
| `env/` | 环境/全链路 | `env_map`、`env_probe`、`pre4_envcheck` |
| `git/` | git 留痕 | `git_*`（init/fix/push） |
| `ablation/` | 奖励/超参消融 | `hp_ablation_chain`、`hp_resume_chain` |
| `data/` | 榜单/白名单/扫描 | `whitelist_*`、`toplist`、`refac_scan` |
| `monitor/` | 训练监控/估算 | `progress_*`、`r2_monitor` |
| `misc/` | 其它 | `pack_zip`、`tensorboard` |
| `s6a_isolated/code/logs/` | S6a 隔离区（不在 `mortal/logs/`） | `s6a_chain.log`、`s6a/{train,holdout,ladder_summary}.log` |

例外：`ladder_r2/`～`ladder_r5/`、`_agtest/`、`r2_evaldone/`、`tb_archive/` 内的 `.log` 保持原位（与同目录牌谱配套，移动会断证据链）。非 `.log` 文件不强制归置。

---

## 6. 工具、文件与产物索引

### 6.1 工具表（一条命令干一件事）

| 工具 | 用途 |
|---|---|
| `mortal/env_check.py` | 开跑前自检：目录遮蔽/环境变量/选卡/config/globs |
| `mortal/holdout_metrics.py` | 未见月 top1、Q 校准、R²、和了 top1 |
| `mortal/ladder.py` + `mortal/run_ladder_r3.sh` | 单对/批量双向同牌山对抗；并发必传 log_dir |
| `tools/training/agg_ladder.py` | 分块 δ/SE/t/95%CI/所需 sets；`--logs-root` 做 block bootstrap |
| `tools/mjai_score.py` | 计分口径唯一权威（与 `stat.rs`/`rankings.rs` 对齐） |
| `mortal/bench_pipe.py` | 数据管道吞吐基准（定 nw/RAYON） |
| `tools/player_stats/build_whitelist.sh` | 重建强度榜与 topN（813k 文件约 77 s） |
| `tools/player_stats/whitelist_coverage.py` | 选 topN 前算席位/样本/步数覆盖 |
| `tools/training/grp_quality_probe.py` | GRP 分位置 hit@1/3/6/9/末、mae、nll |
| `mortal/run_grp_capacity.sh` | GRP 容量实验（注意 `grp_best.pth` 会被不同尺寸互盖） |
| `tools/training/dump_curves.py` / `check_selection_stats.py` / `paired_selection_diff.py` | 曲线/解析交叉校验/同牌山配对差 |
| `tools/training/make_round{3,4,5}_cfg.sh` | 从上一轮配置派生下一轮 |
| `mortal/run_round4.sh` / `run_round5.sh` | 端到端链：env_check→暖启→训练→双向判定→汇总 |
| `tools/training/smoke_round4_augmentation.py` | 启前验白名单+增强样本守恒 |
| `mortal/launch_round3.sh` / `run_r2_ladder.sh` / `run_ladder_r3.sh` | tmux 长跑入口（内建 env 重置） |
| `tools/tenhou_data/repair_names.py` | 从 mjlog 恢复 2026 names（unquote）写补丁树 |
| `tools/tenhou_data/verify_names.py` | names 覆盖/席位/白名单命中率验收 |
| `tools/training/test_file_index_manifest.{py,sh}` | manifest 单测 + E1~E6 回归 |
| `tools/training/test_reward_modes.py` | 奖励口径/缩放/守卫验证 |
| `tools/training/inspect_gamma_semantics.py` | 折扣/dones/局边界实测 |
| `tools/training/test_credit_modes.py` | Stage A 口径五层验证 |
| `mortal/config.iql_a.toml` + `run_iql_a.sh` | Stage A 端到端链（种子 160000/170000） |
| `tools/training/test_iql_data.py` | IQL 数据八层验证（next_obs/r_t/terminal/幅度） |
| `tools/training/diag_iql_reward_double_count.py` | r_t 重复计入诊断（报放大倍数） |
| `mortal/config.iql_b{,_smoke}.toml` + `run_iql_b{,_smoke}.sh` | Stage B 链 + 行为门禁 |
| `mortal/config.iql_b_v2.toml` + `run_iql_b_v2.sh` | Stage B v2（漂移轮，已封存） |
| `mortal/config.iql_b_v3.toml` + `run_iql_b_v3.sh` | Stage B v3（τ=0.5，干净轮） |
| `tools/training/prepare_iql_probe.py` + `run_iql_probe.sh` | 5 臂×30k 步单旋钮探针 |
| `tools/training/prepare_iql_b_v3.py` | `--fix tau05` 一键生成 v3 配置 |
| `tools/training/prepare_hp_ablation.py` / `run_hp_ablation.sh` / `resume_and_judge_hp.sh` | 超参消融/挪卡续训 |
| `tools/read_model/` | 读牌模型 v1/v2/v2s/v2x/A2b + 39 项回归 + ONNX 导出 + A3 规则法基线（rule_baseline/a3_eval/compare_a3/calibrate_a3/test_rule_baseline/run_a3） |
| `tools/read_model/bayes_bound.py` | 读牌 Bayes 上界（§3.14）：counts（精确超几何）/hard（立直⇒听牌）/behave（MH + 静态逆规划似然）三档 + `--leak-test`/`--calibrate` |
| `tools/read_model/shanten_fast.py` | numpy 向量化的 tomohxx 向听查表（与 Rust 逐位同口径，4096 手牌 3.7 ms；MCMC/上界专用） |
| `s6a_isolated/`（`tools/expand_ckpt_v5.py`、`tools/test_urgency_features.py`、`run_s6a{,_smoke}.sh`、`patches/v5_obs_features.patch`） | S6a 隔离工作区：v4→v5 手术、回归、冒烟、正式链 |

### 6.2 关键文件（Linux 路径）

| 类别 | 文件 |
|---|---|
| Python 训练/模型 | `mortal/model.py`（Brain/DQN/GRP/Aux/VHead）、`train.py`、`dataloader.py`、`reward_calculator.py`、`holdout_metrics.py`、`ladder.py`、`env_check.py`、`player.py` |
| Python 入口 | `mortal/mortal.py`（推理）、`server.py`/`client.py`（在线，暂未用）、`train_grp.py` |
| Rust 核心 | `libriichi/src/{lib.rs,consts.rs,state/,dataset/,arena/,algo/,mjai/,agent/,bin/}` |
| 配置 | `mortal/config.example.toml`、`config.round{2,3,4,5}.toml`、`config.iql_a*.toml`、`config.iql_b*.toml` |
| 脚本 | `mortal/run_round{4,5}.sh`、`run_iql_a*.sh`、`run_iql_b*.sh`、`run_hp_ablation.sh`、`resume_and_judge_hp.sh`、`run_ladder_r3.sh` |
| 文档 | `memory/MEMORY.md`（本文）、`memory/NEXT_STEPS_ANALYSIS_20261012.md`、`tools/training/TRAINING_MANUAL.md`、`tools/training/ENV_SETUP_P100.md`、`tools/tenhou_data/TASKLIST.md` |
| 数据索引 | `dataset/index/{player_ratings_*.csv,top_players_*.txt,repair_names_report.json}`；`mortal/indexes/*.pth` |

**代码改动台账（相对上游）**

| 文件 | 主要改动 |
|---|---|
| `mortal/train.py` | `max_train_steps` 预算自停、`snapshot_every` 编号快照、索引读 `start.get('names',[])`、`best_perf` 重置+一次性令牌、file_index manifest、奖励口径守卫、`credit_mode`、IQL 三损失/EMA/`iql scale:`/`iql_v_target_decay`/`iql_value_clip` |
| `mortal/dataloader.py` | `reward_mode/reward_scale/always_include_kan_select`；`credit_mode/gamma` 后缀和；`objective='iql'` 7 元组；`r_t` 仅 `dones[i]` 非零 |
| `mortal/model.py` | 新增 `VHead(1024→256 Mish→1)`；v5 `case 5`（隔离区） |
| `mortal/holdout_metrics.py` | 奖励/信用/IQL 口径透传；`worker_init_fn` + 真随机种子；iql ckpt 只出 top1/agari/lse |
| `mortal/player.py` | `enable_amp`/`agari_guard` 从 `[test_play]` 读；固定评测种子块（仅趋势） |
| `mortal/ladder.py` | 第 5 个参数 `log_dir`；执行参数 `CLI>[1v3.ladder]>默认`、双侧一致、`run_meta.json`、ckpt sha256/steps |
| `mortal/train_grp.py` | `grp.control.max_train_steps`、`grp_best.pth` 按 val loss、固定验证种子、`est. %` 日志 |
| `libriichi/src/dataset/gameplay.rs` | `apply_gamma` 注释改为实测口径（仅注释，无行为变更） |
| `libriichi/src/consts.rs` / `state/obs_repr.rs` | S6a v5（仅隔离区）：`MAX_VERSION=5`、`obs_shape(5)=(1018,34)`、6 行 `encode_urgency` |
| 配置/脚本 | `config.round{2,3,4,5}.toml`、`config.iql_{a,b,b_v2,b_v3}.toml`、各 `run_*.sh`、`prepare_*.py`、`test_*.py` |
| 读牌模型 | `tools/read_model/`：dataset/model/model_v2/model_v2s/structure/constraints/train/evaluate/export_onnx/compare_bootstrap 等；A3 规则法基线（rule_baseline.py + a3_eval/compare_a3/calibrate_a3/test_rule_baseline/run_a3.sh），dataset 增补 meld_counts/dora_indicators 公开字段 |

### 6.3 数据与产物台账

| 路径 | 内容 | 体量 |
|---|---|---|
| `dataset/mjai/houou4/` | 2,630,640 局（200902~202610）；⚠ 2026 部分无 names，白名单模式 0 席位 | 16 GB |
| `dataset/mjai/houou4_names/` | 2026 names 补丁树（130,407 文件） | 607.6 MiB |
| `dataset/tenhou_hf/` | 18 个 HF 年度 tar | 17 GB |
| `dataset/raw/`、`dataset/index/` | 2026 原始 mjlog；清单/榜单/manifest | 1.2 GB / 197 MB |
| `mortal/checkpoints/` | R1–R5、reward_ablation A/B/C、hp_ablation×3、iql_a、iql_b_v2/v3、grp×3、baseline | ≈5 GB+ |
| `mortal/read_model_bayes/` | §3.14 Bayes 上界报告（2500 快照三档分层表）+ `counts_full/report.json`（20000 快照精确 counts 口径） | 104 KB |
| `mortal/logs/ladder_r{2,3,4,5}/`、`mortal/{reward_ablation,hp_ablation,iql_a,iql_b_v2,iql_b_v3}/ladder*/` | 各轮判定牌谱（每轮约 20,000 局） | 每轮 ≈0.8 GB |
| `s6a_isolated/code/{checkpoints/s6a,s6a,logs,runs}` | S6a 隔离产物（867k+best+step_800/850k、判定牌谱、TB、链日志） | ≈1 GB |
| `mortal/logs/tb_archive/` | TB 坏段取证 | 72 KB |
| `mortal/indexes/` | 各轮文件索引（`main_r2_whitelist_v1.pth` 等） | 248 MB |
| `backups/ckpt_round1/`、`backups/code_docs_20261009.tar.gz` | R1 ckpt 副本 8 个；代码/文档备份 | 592 MB / 32 MB |
| `Mortal_valuable_with_ckpts_20261013.zip` | 全量打包（697 entries，含 README） | 6.83 GiB |

### 6.4 外部链接

- 作者权重说明：https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- 官方文档站：https://mortal.ekyu.moe
- 在线服务：https://mjai.ekyu.moe
- 天凤日志一覧（含 ToS）：https://tenhou.net/sc/raw/
- HF 数据集：`NikkeTryHard/tenhou-to-mjai`（hf-mirror 可下）

---

## 7. 附录：早期本机环境与术语

### 7.1 早期 Windows 本机（仅历史参考，当前训练不在本机）

| 项 | 值 |
|---|---|
| 路径 | `c:\Maj\Mortal` |
| Python/git | 3.13.5（Anaconda）/ 2.45.1.windows.1；**cargo 未安装** |
| GPU | RTX 3050 Ti Laptop 4GB（仅数据管线，不宜训练） |
| 用途 | 2026 天凤牌谱采集、mjlog→mjai 转换、文档；训练已全部迁到 Linux 服务器 |

### 7.2 术语

| 术语 | 含义 |
|---|---|
| GRP | 每局顺位预测器；输入 7 维（大局数/本场/供托/四家分数），输出 24 排列概率；奖励=相邻局期望点差 |
| CQL | Conservative Q-Learning 正则；离线训练时压 Q 值 |
| MC | Monte Carlo，当前局最终奖励回传 |
| IQL | Implicit Q-Learning；L_Q + expectile L_V + AWR |
| AWR | Advantage-Weighted Regression，优势加权策略提取 |
| ladder | 双向同牌山 1v3 配对对抗判定；δ=(ptA−ptB)/2 |
| holdout | 未见月 202609+，只用于诊断，不作强度结论 |
| bseed | block bootstrap 随机种子 |
| S6a/b/c/d | 特征注入子阶段：a 紧迫度、b 读牌后验、c 自家速度/牌型、d GRP 扩展+攻防重规划 |

---

> **维护约定**：新增实验先写 §3 的对应小节与 §3.1 总表；新坑写 §5.2 并按主题编号；新工具写 §6.1；新数据/产物写 §6.3。不要在多个章节重复同一结论，引用小节号即可。
