# Mortal 项目上下文记忆文档

> **用途**：沉淀项目历史、关键结论、可复现配置与坑位，供新会话快速接手。
> **本版**：2026-11-02 数据质量加权探针结案（第五版）。相较 11-01 第四版：新增 §3.19（局内 4 人强度加权 100k 探针：top1 −0.26pp 在带内、dqn_loss +5.6% 超门禁 ⇒ 不烧 76k 臂，关闭该轴）；§3.1 总表补一行；§0.1/§0.2、§4.1/§4.2/§4.3、§6.3 同步（数据侧机制全部关闭，下一步 anchor3 + M5 smoke）。
> **上一版全文**：`git show 9ff2453:memory/MEMORY.md`（2026-11-02 编辑前）。
> **可靠性标注**：✅ 已实测/已读源码　🔶 官方文档或作者声明　⬜ 未验证

**目录**：0 快速接手 · 1 项目与系统 · 2 数据与评测 · 3 训练与实验史 · 4 现状与下一步 · 5 工程规范与坑位 · 6 工具/文件/产物 · 7 附录

---

## 0. 快速接手（TL;DR）

### 0.1 当前坐标

| 项 | 值 |
|---|---|
| 项目 | Mortal（凡夫），开源日本麻将 AI；Rust 规则引擎 + Python/PyTorch 训练 |
| 代码/运行目录 | `/mnt/data8/mortal/Mortal`；训练必须在 `mortal/` 下执行（§1.7） |
| 现役最强 | **S6c/956k + 守卫 λ=10**（2026-10-28 换代；v4 obs + 2 行速度特征，obs 1014）；历史称法 round4/645k 已退役 |
| 判定基线 | B/756k：`reward_ablation/B_rank_pt_aligned/mortal.pth`（top400_v2 + `pts=[2,1,0,-3]` + augmentation） |
| 已采纳增益 | ① R2 数据选择 **+2.86±0.59**（§3.3）；② 推理期守卫 G3 **λ=10，+0.78**（§3.13；S6c 上复验 +0.348、8/8 正号）；③ S6c 换代 **+0.422**（77k sets 臂，§4.2）。其余方向全部关闭（§4.1）；档 A 自博弈 2026-11-01 结案（§3.18）、数据质量加权 2026-11-02 结案（§3.19） |
| 当前主线 | 数据侧机制全部关闭（含 11-02 数据质量加权探针，§3.19）；下一步 = anchor3 绝对强度（P0）+ M5 在线自博弈 smoke（P1）；推理期线已封顶（§3.17） |
| 在跑任务 | **无**。数据质量加权探针 11-02 10:36 UTC 完成（100k 步，门禁未过，§3.19）；GPU 全部空闲 |
| 数据 | 全树 2,630,640 局（200902~202610，16 GB）；训练窗 813,145 局；holdout = 202609+（§2.5） |
| 硬件 | Ubuntu 18.04 容器，56 线程，251 GB 内存，4×P100-16GB + 1×RTX 8000-48GB |

### 0.2 三句话结论

- **三个已采纳的增益**：R2「top400 + GRP h128」（§3.3）、推理期守卫 G3 λ=10（§3.13）、S6c 表示/换代（+0.422，§3.11/§4.2）；其余数据 reshuffle、奖励改造、超参、IQL、S6a/S6b、读牌注入、数据扩窗、**档 A 自博弈（两轮全零效应，§3.18）**、**数据质量加权（100k 探针 dqn_loss +5.6%，§3.19）** 均为 null 或负——数据侧机制至此全部关闭。
- **读牌侧已冻结**：A2c 作为归档资产保留，不再注入主模型；S6b 四臂证明「同源信息注入」无行为效应，代码已回退（§3.12）。
- **测量基建（已落地）**：5000 sets 的 CI ±0.6~1.0 看不见 +0.3 的改进，且「我们到底多强」一直没有答案 → ladder 已扩到 16000 sets 标准臂，akochan 绝对锚点 408 局已完成（§3.16）。
- **纪律**：单变量；全新种子块；双向同牌山；block bootstrap CI 下界 >0 才算赢（§2.4、§5.1）。

### 0.3 三条命令

```bash
cd /mnt/data8/mortal/Mortal/mortal
python3 env_check.py --cfg config.round2.toml          # 开跑前自检
python3 -u train.py                                    # 训练（MORTAL_CFG 指向本轮配置）
python3 ladder.py <challenger> <champion> <seed> 500 <log_dir>   # 单卡判定
```

标准判定臂改用 `bash ladder_arm.sh <challenger> <champion> <out> --blocks 16`（§3.16-A）。

### 0.4 阅读索引

| 想知道 | 去 |
|---|---|
| 系统怎么跑、环境变量、选卡 | §1.6–§1.8 |
| 数据从哪来、白名单怎么选、评测怎么做 | §2 |
| 每轮训练做了什么、结果如何 | §3.2–§3.11 |
| 守卫 G3 / EV 重排 / 数据扩窗 / 测量基建 | §3.13–§3.16 |
| 档 A 自博弈两轮终局（为何关闭） | §3.18 |
| 数据质量加权探针（为何关闭） | §3.19 |
| 现在该做什么、哪些路线已死 | §4 |
| 踩过的坑、日志规范 | §5 |
| 工具、文件、产物在哪 | §6 |
| 种子块台账（下一块用什么） | §2.4 |
| **读牌模型（暗牌/听牌/待牌预测）的一切** | **`tools/read_model/docs/README.md`**（本文仅 §3.12 摘要） |
| 局部形状线终结论（**有条件成立**）+ cond 口径修正 | `tools/read_model/local_shape/README.md`、`memory/DESIGN_local_shape_cond_20261103.md` |

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
| 本项目路径 | `/mnt/data8/mortal/Mortal`（历史文档中的 `c:\Maj\Mortal` 为早期 Windows 本机路径，见 §7.2） |
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
| `agent/` | `MortalBatchAgent`：`react_batch()` 回调 Python；quick_eval、杠二次推理、agari guard；`akochan.rs` 外部锚点 pipe 客户端（§3.16-B） |
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
| `Brain` | 1D ResNet 沿 34 种牌卷积；默认 `conv_channels=192`、`num_blocks=40`；Mish + BatchNorm1d；输出 1024 维 φ；可选 `extra_channels`（S6c=2，默认 0） |
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
- `FileDatasetsIter`：在 `build_iter()` 中构造 GRP 并载入 `[grp].state_file`（不能放 `__init__`）；`populate_buffer()` 逐局算 `kyoku_rewards`、`player_ranks`、`steps_to_done`。
- 样本字段（cql_mc）：`[obs, (invisible_obs), action, mask, steps_to_done, kyoku_rewards[at_kyoku], player_ranks[at_kyoku+1]]`。
- IQL 样本（Stage B）：`[obs, actions, masks, next_obs, terminal, r_t, ranks]`；`r_t` **仅局边界 `dones[i]` 非零**。

### 1.5 配置字段速查（`config.example.toml` / 各轮 toml）

| 节 | 关键字段 | 说明 |
|---|---|---|
| `[control]` | `version`、`online`、`state_file`、`best_state_file`、`tensorboard_dir`、`device`、`enable_amp`、`enable_compile`、`batch_size`、`opt_step_every`、`save_every`、`test_every`、`snapshot_every`、`max_train_steps`、`seed`、`reset_optimizer`、`reset_best_perf`、`objective`、`iql_tau`、`iql_beta`、`ema_decay`、`iql_v_target_decay`、`iql_value_clip`、`extra_features` | 训练主控；`save/test/snapshot/max_steps` 必须满足整除断言；`extra_features='s6c'` 才注入 2 行特征 |
| `[dataset]` | `globs`、`file_index`、`file_batch_size`、`reserve_ratio`、`num_workers`、`player_names_files`、`num_epochs`、`enable_augmentation`、`augmented_first`、`data_version`、`always_include_kan_select` | 数据选择与索引 manifest |
| `[env]` | `gamma`、`pts`、`reward_mode`（`rank_pt`/`delta_points`）、`reward_scale`、`credit_mode`（`kyoku`/`game`） | 奖励口径；续训时不一致会告警 |
| `[resnet]` | `conv_channels=192`、`num_blocks=40` | 显存不足再下调 |
| `[cql]` / `[aux]` | `min_q_weight=5` / `next_rank_weight=0.2` | 原始目标正则；IQL 时 CQL 关闭 |
| `[grp]` | `state_file`、`[grp.network] hidden_size/num_layers`、`[grp.control]`、`[grp.dataset]`、`[grp.optim]` | GRP 权重必须先存在；尺寸与 ckpt 必须匹配 |
| `[optim]` | `eps/betas/weight_decay/max_grad_norm` | AdamW；`max_grad_norm=0` 表示不裁剪 |
| `[optim.scheduler]` | `peak/final/warm_up_steps/max_steps` | 常见 `peak=final=1e-4` |
| `[baseline.train/test]`、`[1v3.*]`、`[1v3.ladder]`、`[1v3.akochan]` | `device/state_file/enable_amp/guard/seed_key/games_per_iter/iters/mortal_vs_ako/dir` | 1v3 评估；ladder 参数优先级 CLI > `[1v3.ladder]` > 历史默认 |
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
- **当前状态（2026-10-26）**：`master` 领先 `origin/master` **10 个提交**（`3d615c3..90f8660`：D2 判决、EV 重排、测量基建、akochan 四项修复 + 进程数修复、gitignore）。推送前先 `git status -sb`；SSH 主机键缺失会让 push 静默挂起（§5.2-14，用 `timeout` 包裹）。
- **忽略规则**：`dataset/`、`mortal/checkpoints|indexes/*.pth`、`mortal/libriichi.so`、`target/`、`backups/`、`mortal/runs/`、`mortal/logs/`、`tools/tenhou_data/logs/`、`*.bak/*.so/*.dll/*.exe/*.tar`；**保留 `memory/` 与 `tools/`**（唯一文档）。
- **基线**：首提交 `33e63bc`（460 文件/4.0 MB），tag `round1-complete`；文档同步提交 `5a1867f`；防泄漏校验 `dataset` 跟踪数 0、敏感扩展名 0；git 2.17.1（不支持 `init -b`/`switch`）。
- **备份**：`backups/ckpt_round1/`（8 个 R1 ckpt）、`backups/code_docs_20261009.tar.gz`、`backups/libriichi_20261025/`（akochan 改动前源码 + `.so` + 回滚说明）、`Mortal_valuable_with_ckpts_20261013.zip`（全量含 ckpt）；体量见 §6.3。
- **权重**：`mortal/checkpoints/` 下 R1–R5、reward_ablation、hp_ablation、IQL、S6a 隔离区、`s6c`、`mix_v1`(D1)、`d2_2026`、`inject_s6b` 等；`.pth` 约 130 MB（fp32）。**逐目录路径与体量见 §6.3**（不在此重复）。

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

**落地**：HF 18 个年度包（16.86 GiB）在 `dataset/tenhou_hf/`；merge-hf 展开 + prune --apply 后 `dataset/mjai/houou4/` 为全量树（200902~202610）；2026 自建数据在 `dataset/raw/`。**局数见 §2.5，路径与体量见 §6.3**。

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

1. **训练内 `test_play` 不作跨轮强度结论**：它写死 `seed_start=(10000,0x2000)`，同一批 500 sets 被多轮选模型反复使用，会牌山过拟合；固定块只用于同轮趋势（R3 的实测对照见 §3.4）。
2. **样本量按 SE 倒推**：`sets ≥ 当前 × (2·SE/δ)²`；小样本既测不出也低估效应（R2 的案例见 §3.3）。标准臂用 16000 sets（§3.16-A）。
3. **判定用双向同牌山配对**：A=新当挑战者、B=旧当挑战者，各 5×500 sets（10,000 半庄/方向）为历史口径，`δ=(ptA−ptB)/2`；两向同号才可信。
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
| 守卫 G3 剂量曲线（λ=5~20 每向 8000 sets；λ=50 每向 2000 sets） | 260000/270000 |
| S6c（每向 5×500 sets） | 280000/300000 |
| D1 2019-2021 混入 | 320000/340000 |
| D2 2026H1 补量 | 360000~362000 / 380000~382000 |
| S6c 77k 复判（76 块/向） | 400000–437500 / 420000–457500 |
| 守卫 λ=10 on S6c 复验（8 副本×8000 sets） | 440000~580000 |
| 档 A 第一轮臂（76 块/向） | 800000–837500 / 820000–857500 |
| 档 A 第二轮臂（76 块/向） | 910000–947500 / 930000–967500 |
| 守卫 risk 口径修正 λ 重扫（4 臂×16 块/向 + 空臂） | 1000000/1020000 |
| **下一块** | **1040000/1060000** |

> `ladder_arm.sh` 注释里的「台账见 MEMORY §4.2 备注」即指本表。

### 2.5 关键量化事实

| 主题 | 数值 |
|---|---|
| 数据 | 全树 2,630,640 局；训练窗 813,145；分年 2022/23/24/25/26 = 167,627/168,777/184,424/178,887/130,407 |
| 单局量 | 10.6 局/半庄；全座位 678 样本/文件；仅白名单座位 234.8 样本/文件 |
| epoch | top400 v1 1 epoch≈220,268 步（R2 实测 224,847=102%） |
| 吞吐 | GPU 上限 8.88 it/s；管道 nw2/rayon4=3.12 → nw8/rayon8=7.48 it/s；fp32 RTX 8000≈4.24 it/s |
| 判定 | 单半庄 std≈88 → 500 sets SE≈±1.9；5000 sets 解析 SE≈±0.59、bootstrap≈±0.51；双向共 16000 sets（8000/向）解析 SE≈0.33 |
| 评测代价 | 500 sets（2000 局）≈14 分/卡；5 卡两波 5000 sets ≈29 分 |
| GRP | 30k 步到平台；h128/l3 nll_last 1.2780→1.2361，hit@3 0.379 / hit@6 0.483；`delta_pt` std 恒≈0.70（7 维输入信息量上限） |
| 行为对照 | 和了 0.2228 vs 人类 0.2117；立直 0.1794 vs 0.1769；副露 0.4566 vs 0.4826；放铳 0.1405 vs 0.1255；流局 **10.9% vs 15.3%**（弃和不足） ⚠️ **2026-11-07 口径修正，见下** |

> ⚠️ **行为对照行已部分证伪（2026-11-07）**：上行的「副露 0.4566 vs 0.4826」是**错误口径**。
> 用 `libriichi/src/stat.rs` 的同一定义（`fuuro_rate` = **有 chi/pon/daiminkan 的局占比**，
> 不含暗杠；round = 每人每局）实测人类凤凰场（202[2-5] 抽 3,000 文件 / 126,288 人·局）：
> **和了 21.14% / 放铳 12.65% / 副露局占比 32.63% / 立直 18.25% / 流局 16.00%**。
> S6c/956k 对比：**+0.53 / −0.38 / −1.54 / +0.30 pp** ⇒ **四项均在 ±1.6pp 内**。
> ⇒ **「弃和不足 / 太保守」不成立**（它是拿 S6c 的 31.09% 比了一个错误口径的 48.26%）。
> B/756k 的副露 −4.37pp 才是真实偏离，而 S6c 已把它修到 −1.54pp。
> 唯一未结：流局率（Mortal 侧未记录 `ryukyoku_rate`），但和了+放铳已解释绝大部分局的结束方式。
> 详见 [`BEHAVIOR_PROFILE_20261107.md`](BEHAVIOR_PROFILE_20261107.md) 与 `tools/behavior_profile.py`。

---

## 3. 训练与实验史

> 阅读约定：§3.2–§3.10 按时间（R1→IQL-B）；§3.11 起按主题（表示/读牌 → 推理期 → 数据侧 → 测量）。**所有实验数字只在本节出现**，§4 只做方向聚合。

### 3.1 主模型实验总表（训练轮次与探针）

> 推理期实验（守卫 G3、EV 重排）不在本表，见 §3.13 / §3.14。

| 轮次/实验 | 关键变量 | 数据/父代 | 步数 | 吞吐 | δ vs 父代（pt/半庄） | holdout top1 | 判定 |
|---|---|---|---|---|---|---|---|
| R1 | 基线（cql_mc + GRP h64/l2） | 813,145 全体 | 0→200k | 2.56 it/s / 22.9h | —（基准） | 0.8041 | 基准；test_play 饱和，强度持平 |
| R2 | top400 v1 + GRP h128/l3 | 480,414 文件 | 200k→424,847 | 6.71 it/s / 9.5h | **+2.86±0.59**（t=4.83） | **0.8122** | ✅ 训练侧唯一有效增益 |
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
| S6b | 注入 2 行读牌危险度（obs 1014） | 父代 B/756k + top400_v2 | 756k→776k（20k 四臂探针） | 1.6–2.6 batch/s | A1≈A2（安慰剂，全部 <1σ）；`deal_top1` 0.0120/0.0121/0.0118/0.0123 | 0.8135/0.8143/0.8156/0.8136 | ❌ 信息效应为零，关闭并回退 |
| S6c | 注入 2 行自家速度（向听/有效牌） | 父代 B/756k | 756k→956k（20k 探针→180k） | 4.15–4.26 batch/s / 13.1h | **+0.268**（解析 CI[−0.90,+1.44]；bootstrap CI[−0.76,+1.27]） | 0.8167（+0.40pp；dqn_loss +3.3%） | ✅ 转正（77k 臂 +0.422、CI[+0.15,+0.69]，2026-10-28 换代） |
| D1 | 混入 2019-2021 20% | B/756k + 70,588 文件（+14.3%） | 756k→856k（100k） | 4.27→2.64 batch/s | **−1.402**（解析 CI[−2.57,−0.23]；bootstrap CI[−2.39,−0.44]） | 0.8105（−0.22pp；dqn_loss +6.4%） | ❌ 负迁移 |
| D2 | 补 2026H1（571,043 文件 / +15.87%） | 父代 B/756k | 756k→856k（100k） | — | **−0.283**（解析 CI[−1.45,+0.88]；bootstrap CI[−1.26,+0.74]） | 0.8124（−0.03pp） | ❌ 无结论 |
| tierA（档 A 第 1 轮） | 自博弈 8,000 半庄×2（占 3.1%） | 父代 S6c/956k | 956k→1,140k（200k） | 4.0 it/s / 13.9h | **−0.024**（t=−0.16；bootstrap CI[−0.29,+0.25]） | 0.8132（−0.35pp） | ❌ 零效应（76k sets 双向臂，§3.18） |
| tierA2（档 A 第 2 轮） | 自博弈 35,200 半庄×2（占 12.5%） | 父代 S6c/956k | 956k→1,356k（400k） | 4.2 it/s / 27.4h | **+0.072**（t=+0.47；bootstrap CI[−0.17,+0.33]） | 未测（只跑臂） | ❌ 零效应 ⇒ 档 A 关闭（§3.18） |
| strength（数据质量加权） | 局内最弱一席 avg_pt 分档重复 ×1/×2/×3（mean 1.35） | 父代 S6c/956k | 956k→1,056k（100k） | 4.41 batch/s / 6.85h | 未跑臂（探针门禁未过） | 0.8141（−0.26pp；dqn_loss +5.6%） | ❌ 探针 dqn_loss 超门禁，关闭（§3.19） |

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
- 两个状态残留：① `best_perf` 继承 R1 随机对手时代的 `{1.0,90.0}`，导致 R2/R3 从未写 best.pth（修法与遗留说明见坑 18）；② 自写解析与 Rust Stat 的平分 tie-break 不同（见 §2.4-5），判定以 ladder 为准。
- 方法论警告（数字是 §2.4-1 的原始出处）：固定种子块 `test_play` 跨轮配对给出 Δ=+4.52±2.07，而新种子块 ladder 只有 δ=−0.03±0.59——**4.5 pt 全是牌山过拟合**。
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

### 3.11 S6 特征注入：S6a（紧迫度）与 S6c（自家速度）

**S6a（obs v5，紧迫度）**：v5=v4+6 行——与排序后 1/2/3/4 名的点差 `d1..d4` + `overtake_gap` + `lead_gap`；全部由 `state.scores` 确定性导出、归一化 [0,1]，不做剩余局数假设。首层 Conv1d 手术（新通道零初始化，前向与 v4 逐位等价）；**完全隔离**在 `s6a_isolated/`（独立 code/`libriichi.so`/index/ckpt/runs/logs，输入软链只读共享树；共享树已还原 v4）。回归 17/17、冒烟 ALL PASS；正式轮 867k / 7.8h / 4.5 batch/s；holdout 0.8160、lse +0.467、R²−0.995；δ=+0.182、解析 CI[−0.99,+1.35]、bootstrap CI[−0.82,+1.20] → **无结论**。归因：6 个特征全是 v4 已有分数通道的确定性函数 → **零新信息**，ResNet 本就能算；该 null 只否定“重排已知信息”，不否定 S6 方向。

**S6c（obs 1014，自家速度）——由来**：v4 的 SP 前瞻只算 `shanten≤3`（`SHANTEN_THRES=3`），shanten≥4 占决策点 **8.72%**（3,622 点实测：0:14.8% / 1:25.4% / 2:31.6% / 3:19.6% / ≥4:8.72%），这些点完全没有有效牌信息；同时 v4 向听 one-hot（第 **862..868** 行）**在打牌后才更新**（`state/update.rs` 的 `update_shanten` 挂在 `dahai`），3n+2（自摸后 14 张）状态只有 74.5% 与真值相等、失配恒为 `obs = 真值 + 1`，且 `state.shanten` 被 clamp 到 6。⇒ 「当前向听 + 有效牌数」不是纯冗余，值得单独注入。

- **特征**（`mortal/s6c_features.py`：libriichi 同款 tomohxx 向听表的 numpy 向量化，与 Rust 单测 **19/19** 逐位一致）：ch0 = `clamp(最优向听,0,8)/8`；ch1 = `max_候选(有效牌剩余枚数)/136`，剩余 = `4 − tiles_seen`（第 **835** 行，实战口径），候选 = 打牌后达到最优向听的每一张（3n+2）或当前手牌（3n+1），有效 = 摸进后向听前进 1；`best ≤ −1`（已和了）两通道置 0。34 列广播，obs 1012→1014。
- **成本**：**0.148 ms/样本**（2579 样本批，按文件批量；小批更贵），8 worker 下约占单核 2%。
- **管线**：复用 S6b 的 `extra_channels` 形态，配置键 `[control].extra_features='s6c'`；`model.Brain(extra_channels)` + dataloader 按文件批量算并拼 + engine 推理时拼（守卫仍吃原始 1012 行）；train/holdout/ladder/player/one_vs_three/mortal.py/client.py 全部由各自 ckpt 的 `[control]` 经 `s6c_features.resolve` 解析（旧 ckpt → 0/''）；手术 `tools/expand_ckpt_v6.py`（零初始化，**phi max|diff|=0**）。
- **透明性（2026-10-23 12:22）**：`config.s6c_base.toml`（关）与 `config.s6c.toml`（开）在同一 holdout（202609 / seed 20261009 / 24×512）上 `holdout_sample_digest=9c5362bf99d60661` 与 `top1=0.8127 / dqn_loss=0.3496 / agari 149/149 / lse +0.522` **逐位相同** ⇒ 手术 ckpt 与 B/756k 等价（该 digest 下的基线就是 0.8127，与 §3.12 记的 0.8223 是不同抽样口径，不可横比）。
- **20k 探针（10-23 13:22~15:43，2.74~2.84 batch/s / 141 min）**：top1 **0.8115 vs 0.8127**（−0.12pp≈0.34σ）、dqn_loss 0.3479 vs 0.3496、R² −1.2063 vs −1.2166、lse +0.505 vs +0.522、agari 149/149；新通道 ΔL2 **62.4% / 16.8% 分位（均 ≥ p10）** ⇒ 门禁 PASS，启动 200k。
- **正式轮（10-23 15:48 ~ 10-24 05:59；956k / 180k 步 / 13.1 h）**：全程 4.15~4.26 batch/s、**0 条非有限/异常**；每 50k 行为画像 agari .2240/.2214/.2177/.2167、houjuu .1299/.1254/.1200/.1227、fuuro .3074/.2950/.2829/.3109、riichi .1933/.1957/.1950/.1855（放铳略降、其余在噪声内）。holdout（同 digest）756k 0.8127 / 776k 0.8115 / **956k 0.8167（+0.40pp≈1.1σ）**，但 `dqn_loss` 0.3496→**0.3613（+3.3%）**、R² −1.2911、`var_q/var_t` 0.472（Q 尺度漂移）、lse +0.525、agari 149/149；通道使用度 ΔL2 **9.786e-1 / 4.106e-1** → 分位 **100% / 58.1%** ⇒ 特征确实被模型用上。
- **ladder（λ=0，新种子 280000/300000，双向各 5×500 sets = 20000 半庄）**：A +0.301（SE 0.845）、B −0.234（SE 0.843）→ **δ=+0.268**，解析 SE 0.597、t=0.45、CI[−0.90,+1.44]；block bootstrap（2500 sets/向，ICC 0.186/0.175）SE 0.509、**CI[−0.76,+1.27]**。两个独立量都为正但都不显著 ⇒ **判无结论、不采纳（underpowered，未否证）**；主模型保留 B/756k，956k 归档。要分辨 +0.27 需 ~95k sets。
- **77k sets 复判（2026-10-27，✅ 转正）**：`ladder_arm.sh` 双向 **76 块×500 sets = 76,000 sets / 152,000 半庄**，种子块 **400000–437500 / 420000–457500**（已消费），GPU2 RTX 8000 4 并发，历时 **10.6 h**；λ=0、`MORTAL_CFG=config.s6c.toml`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`（坑：不加时 cuda:2 = P100）。结果：**A +0.447±0.216 / B −0.398±0.216 → δ=+0.422**，解析 SE 0.153、t=+2.76、CI[+0.12,+0.72]；**block bootstrap（ICC 0.184/0.186）SE 0.133、CI[+0.15,+0.69] ⇒ WIN（下界>0）**。比 5000 sets 时的 +0.268 更强（那时是欠采样低估）。产物 `logs/s6c77k/`；种子块台账已更新。
- **产物与复核**：`checkpoints/s6c/{mortal(956k),best(900k),B756k_s6c_init,mortal_step_*}`、`logs/s6c/{SUMMARY.md,ladder_boot.log,ladder/}`；代码已入库但默认关闭（`extra_features=''`），复核 `STAGE=ladder bash run_s6c.sh`（SEED_A/SEED_B 可换新块）。探针工具两个 bug 已修：① `grep -ao 'top1='` 会误中 `agari_top1=`（曾把 top1 读成 149，门禁假过）；② 严格「≥ 基线」在 12,288 样本被噪声卡死（SE≈0.35pp），改为 `TOP1_TOL=0.005`。**脚本运行中不要改 `run_s6c.sh`**（坑 24）。

### 3.12 读牌模型（完整记录在 `tools/read_model/docs/README.md`）

读牌模型的实验记录、判决依据、复核修正、改进路线（RM1~RM16）、坑位与产物台账自 2026-10-19 起统一维护在 **`tools/read_model/docs/README.md`**；诊断脚本与原始数字在 `tools/read_model/diagnostics/`。旧章节号（§3.12/§3.14/§3.15.x）对照见该 README **§16**。这里只留四件事：

1. **基线是 A2c**（v2s + RM4/RM5，best@153k）：官方 20000 固定集（seed 20261016）全项通过——tenpai AUC **0.9698**（≥0.9685）、标定 **1.0364**（∈[0.95,1.05]）、cond_ce **2.7532**（≤2.896）、hit@1 **0.1993**（≥0.16）、立直 **0.1724**（≥0.135）、val_score **2.0524**（< A2b 2.0924）。A2c 取代 A2b 成为注入基线，特征口径冻结；A2b 归档作对照。对 A3 规则法：alloc/cond_ce/tenpai_AUC/hit@1 全面显著胜，立直 hit@1 **0.1724 vs 0.1121**，nlt 由 rule 胜转为打平。**A2c 模型本身从未注入 `mortal/`**（S6b 只注入其 2 行后验且已回退），强度只认「注入 + ladder」在线验收，离线指标仅诊断。
2. **两条旧判决被推翻**（零训练成本）：① 「P(听牌) 过标定 6.3×」实为读错输出——BCE 标定的是 logit 而项目读 softmax，换 `1−sigmoid(wait_logits[34])` 后标定比值 **0.998**、AUC **0.9461→0.9685**（显著胜 A3 规则法 0.9468）；② wait_ce 的「+0.758 nats 优势」**71% 是听牌因子伪影**，真实待牌条件分布优势仅 **+0.217** ⇒「读侧已近天花板」证据不足；实测空间在输出层参数化、H2 缺逐牌列特征（冻结骨干探针 hit@1 相对 +35%）与训练规模（只用了全量数据的 **4.05%**）。P0 零成本项落地后回归 **52 项 ALL PASS**；口径纪律：报 `wait_ce_active` 必须同时给 `−logP(t)` 与 `cond_ce`（README §11.2）。⚠️ **2026-11-04 更正**：上述三项里“训练规模”已探完且为负结论——从 best@153k 续训四组配置（init 1.2e-4 跑 86k 步 / init 8e-5 / init 4e-5 / resume 4e-5）**全部退化 +0.008~+0.018**，resume（保 optimizer 动量）不改变结论 ⇒ **A2c 的 2.0524 是 cosine 末端噪声低谷，真实收敛水平 ≈2.060**；“val 单调降”是 LR 退火效应而非数据不足。立直层温度校准也仅能回收 ≈16%（0.011/0.065 nats，见 `local_shape/README.md` §5f）。**加训练/加校准两条线均关闭**，详见 `tools/read_model/docs/README.md` 文首两段。
3. **集合级充分、顺序级非立直层约 8 成、立直层仍缺口**：集合级局部推断 A2b 已学 **75~79%**（A3 仅 13~19%）；真顺序盲地板（`rule_predict(order_blind=True)`）下 A2b 扣地板复现率 66.6% / 非立直 81.9% / 立直 +31.0%，A2c 验收后重测为全样本 **85%**、非立直 **92.7%**、立直 **+66.2%**；区域级（同色 ±2）all +0.0062（CI 跨 0）/ 非立直 +0.0155（CI>0）已转正，但**立直窗口仍显著为负（−0.0656，CI[−0.1134,−0.0142]）**、反事实斜率仅 **b=0.0093** ⇒ 「逐牌列通路被激活、时序注意力仍空转」。P1b 两版（RM12 v1 b=0.0131、RM14 next-tedashi aux b=0.0038）20k 均 FAIL，不启动 200k；原计划的显式 order 特征/输入消融（README §13.5~§13.7）未再启动，读牌侧随后冻结（第 4 点）。
4. **注入 P0 GO → P1 四臂 20k FAIL，读牌侧冻结**：P0-A 冻结主模型 `phi` 探针 5-fold by file——linear alloc_ce **0.7761±0.0012** / hit@1 **0.0785±0.0089** / AUC **0.7273±0.0078**；mlp128 **0.7588±0.0014 / 0.0775±0.0182 / 0.7993±0.0049**，A2c 同批 **0.6517 / 0.2003 / 0.9701** ⇒ 主模型表示缺读牌（探针 train hit@1 0.42 vs val 0.08 ⇒ 只在训练分布内可解码）；P0-B 主模型 top-1 点炮 **1.32%** ≈ 人类 **1.40%**、A2c 最安全 **0.27%**，与 A2c 最安全牌一致率 **28.5%**，`fixable` **1.20%（239/20000，30/30 文件，turn≥8 2.28% / 有立直家 2.59% / turn<8 0.49%）**；P0-C 20k `deal_pure` **0.0108**（≈16 个事件）vs 100k **0.0147**，**差 36%** ⇒ deal 类判据一律 ≥100k；P0-D 零 `tl_tokens` 只掉 2.5% ⇒ 可复用主模型 obs 预计算。**注入形态 = 2 行 obs**（1012 = 三家 union 危险度 `Σ_j P(听_j)P(待牌_j=t)`，1013 = `P(至少一家听牌)`），不是 102 维后验。P1 四臂（A0 基线 / A1 真特征 / A2 同批 shuffle 安慰剂 / A3 turn<8 置零）同数据同种子 20261022，756k→776k：`deal_top1` 0.0120/0.0121/0.0118/0.0123、holdout top1 0.8135/0.8143/0.8156/0.8136，**A1≈A2（全部 <1σ）**；新通道 `abs(w)` 均值 A1 **0.0158** vs A2 **0.0062** ⇒ 权重长起来但行为不变，**信息效应为零** ⇒ S6b 关闭、不再跑 100k/200k（完整记录 README §13.14.8）。`mortal/{model,dataloader,train,holdout_metrics,ladder}.py` 已恢复为注入前 `3a793a1`（commit `7bb5b7d`），`config.s6b_*` 删除；B/756k holdout top1=0.8223 / dqn_loss=0.3881 / agari 29/29 与注入前一致。读牌侧自此**冻结为归档资产**。

### 3.13 推理期防守守卫 G3（2026-10-22~23；2026-11-03 起默认关闭）

> ⚠️ **2026-11-03 更正**：本节的 +0.778 是 risk **双乘 P(听)** 旧口径的结论。
> 口径修正后，在全新种子 1000000/1020000 上重扫 legacy λ=10 与 fixed
> λ=4.5/6/7.5，四臂均 **NO-CONCLUSION**；legacy λ=10 本轮 −0.184、CI[−0.61,+0.27]。
> 线上默认已改为 `MORTAL_GUARD_LAMBDA=0`。历史数字保留，详见
> `tools/read_model/docs/GUARD_RISK_FIX_20261102.md`。

**动机**：行为对照显示主模型放铳偏高、流局偏低（弃和不足，数字见 §2.5）。G1（非筋固定罚）把不同牌的危险度抹平且只覆盖立直家；G2（硬 mask）在无现物/筋时退化、强行放弃高价值手牌。G3 用连续 risk 覆盖全部对手（含默听/副露），是「Q 减去期望放铳损失」的贝叶斯形式：

```
Q'(t) = Q(t) − λ · max(risk(t) − τ, 0),      t ∈ [0, 34) 打牌
risk(t) = Σ_j P(听_j) · P(待牌_j = t)          # A2c 后验，三家 union
```

λ 固定、不随牌值自适应（主模型 Q 已含手牌价值，再乘牌值等于重复计算）；只改打牌动作，34..45 的立直/鸣牌/荣和/pass 不动。τ（risk_floor）是 λ=20 线性版之后的修正：线性版把低 risk 区（0.001~0.01，噪声大）也按比例扣分，导致「houjuu −21%、agari −7.4%、avg_pt −0.40（CI[−1.69,+0.89]）」——用和了换放铳净不划算；τ 只拦 risk 明显高于背景的牌。

- **实施**：`mortal/defense_guard.py`（A2c `tl=0`，连续、全对手）+ `engine.py`/`ladder.py` 钩子，commit `6fab9ce`；`mortal.py` 默认 **λ=10**（`MORTAL_GUARD_LAMBDA` 覆盖，设 0 关闭；`MORTAL_GUARD_CKPT` 可换后验）。
- **单步离线（20000 样本）**：λ=20 令 `deal_top1` 0.0132→**0.0081（−39%）**，改选动作 **100% 更安全**。
- **剂量曲线（每臂双向 64000 半庄；λ=50 为 16000）**：

| λ | δ（pt/半庄） | 95%CI | 判定 |
|---|---|---|---|
| 5 | +0.550 | [−0.10,+1.20] | CI 跨 0 |
| 7.5 | +0.712 | [+0.07,+1.36] | 显著，备选 |
| **10** | **+0.778** | **[+0.13,+1.42]** | **峰值，采纳** |
| 20 | −0.496 | [−1.14,+0.15] | 过强 |
| 50（τ=0.03） | −2.663 | —（16000 半庄） | 崩 |

- **λ=10 行为**：agari 0.217→0.207（−4.6%）、houjuu 0.130→0.111（−14.6%）；两方向同号（+1.226 / +0.330）。
- **结论（历史）**：当时判 GO，λ=10 为默认、7.5 为保守备选；这是 R2 之后**当时唯一被采纳的新增益**（2026-11-03 起默认关闭，见下条）。产物：剂量牌谱在 `logs/guard_pipeline/`，聚合脚本 `tools/read_model/agg_guard_ladder.py`。
- **2026-11-02/03 口径修正与 λ 重扫（终局 λ=0）**：`factorized_wait_probs` 的 `wp[:34]` 已是 `P(听)·P(待|听)`，旧守卫又乘 `p_tenpai`，实际是 `P(听)^2·P(待|听)`；修正后 risk 均值约 1.55~1.66×，20k 上 fixed λ=6 与 legacy λ=10 工作点重合。全新种子 1000000/1020000、每臂 64k 半庄：legacy λ=10 −0.184 [−0.61,+0.27]；fixed λ=6 −0.124 [−0.56,+0.35]；fixed λ=4.5 +0.047 [−0.37,+0.46]；fixed λ=7.5 −0.103 [−0.56,+0.39]，全部 NO-CONCLUSION。配对 Δδ（fixed−legacy）+0.059/+0.231/+0.080，均不劣但无正收益。旧 +0.778 本身是两次同向运行的 `avg_pt` 均值（非标准 δ、未 A/B 换边），按 δ 粗算约 +0.448 且 CI 跨 0。按预注册规则默认 **λ=0**，不跑 fresh-seed（1040000/1060000 未消费），不保留 legacy 默认路径；修复代码与 `tools/test_guard_risk.py` 保留。

### 3.14 推理期 EV 重排（2026-10-25，离线否证冗余，未进 ladder）

**思路**：把守卫的「单纯风险惩罚」升级为「收益–风险同尺度比较」——`score(t) = Q(t) + β·ev_norm(t) − λ·max(risk(t)−τ,0)`，`ev_norm` 取自 v4 obs 里**已有的 SP 前瞻表**（行 961..977 tenpai / 978..994 win / 995..1011 EV），`risk` 与守卫同源（A2c）；可选 `margin` 只在 Q 近邻内重排。

**实现**：`mortal/ev_rerank.py`（obs 布局常量 + `sp_features`/`ev_neutral`/`rerank_scores`/`EVRerank`，接口与 `DefenseGuard` 一致）+ `tools/test_ev_rerank.py`（**ALL PASS**：889/890 常量行、max_ev 两路一致、SP 覆盖 ⊆ 自己手牌/候选行、`sh≥5` 零覆盖、均值填充、β=λ=0 不动 / 大 β 选高 EV / margin 限域）+ `tools/read_model/eval_ev_rerank.py`（20k 单步诊断）。**未接入默认推理路径**。

**离线诊断（202609 固定 20000 样本，A2c tl=0，P0-B 口径）**：

| 配置 | 变化率 | deal_top1 | ev_norm | p_win |
|---|---|---|---|---|
| Q only | — | 0.0132 | 0.6149 | 0.0953 |
| guard λ=10（线上） | 0 | **0.0094** | 0.6140 | 0.0943 |
| β=0.8 λ=10 | 0.8% | 0.0094 | 0.6167 (+0.0027) | 0.0947 |
| β=10 λ=10 | 5.6% | 0.0100 (+0.0006) | 0.6279 (+0.0139) | 0.0965 (+0.0022) |
| β=10 λ=0 | 12.7% | 0.0138 (+0.0044) | 0.6284 | 0.0973 |
| β=0.4 λ=10 margin=0.3 | 1.9% | 0.0105 (+0.0011) | 0.6156 | 0.0949 |

**结论：按这个形式没有头部空间（结构冗余）**——SP 的 tenpai/win/EV 三张表本来就是 v4 obs 的输入通道（行 961..1011），`β·ev` 只是对已知输入做线性重加权（与 S6a 同一失效模式）；实测 Q 与 SP-EV 的排序在 **94~99%** 的决策上一致，β 拉大只是沿「放铳↔速度」同一 frontier 移动（与调 λ 同构），不是新信息。⇒ **不为此烧 ladder**（16000 sets 的臂约 **2~4 h**；旧估 6.5 h 偏保守，但预期效应就在被测 frontier 上，仍不值得）。若还要在推理期找增益，只能引入 obs 里**没有**的量（如按对手手牌价值/立直状态加权的 risk，或攻守两套 EV 的显式比较），且必须先有更细的分辨率。

### 3.15 数据扩窗：D1 旧年代混入与 D2 同年代补量（2026-10-24~25）

同分布 reshuffle 到头后，分布扩展是唯一没试过的新信息源。**年度审计**（`tools/data_audit/year_style_audit.py`，每年随机 4000 文件，单位：每席位每半庄）：

| 年 | 立直 | 副露 | 和了 | 自摸占比 | 流局/局 | 和了收入 | 局时长 | v2名单席位 |
|---|---|---|---|---|---|---|---|---|
| 2013 | 1.700 | 5.924 | 2.284 | .402 | .1622 | 6424 | 47.97 | 4.2% |
| 2016 | 1.746 | 5.590 | 2.263 | .399 | .1606 | 6548 | 47.96 | 6.0% |
| 2019 | 1.831 | 5.380 | 2.242 | .400 | .1623 | 6792 | 48.21 | 9.5% |
| 2020 | 1.880 | 5.293 | 2.241 | .405 | .1651 | 6831 | 48.29 | 13.2% |
| 2021 | 1.889 | 5.303 | 2.248 | .404 | .1644 | 6793 | 48.40 | 16.8% |
| 2022 | 1.891 | 5.277 | 2.230 | .410 | .1662 | 6808 | 48.35 | 23.7% |
| 2025 | 1.883 | 5.314 | 2.243 | .404 | .1564 | 6769 | 47.95 | 28.3% |

**审计结论**：① 牌风是**单调但温和**的漂移——2019-2021 与 2022 只差 ~3%（立直 +3%、副露 −2%），2013-2016 才差 ~10%/12%；② **玩家池换代严重**：用 2022+ 名单打旧年只覆盖 9.5~16.8% 席位（文件 33~44%），必须按年代自建名单（`top_players_400_2019_2021.txt`：扫 577,757 文件 / 4,762 人 / ≥300 局 574 人 / top400 阈值 avg_pt≥2.90，与 v2 只重叠 **18.8%**）——用年代名单后旧年文件覆盖升到 **57.4% / 58.7% / 66.6%**。

**D1（2019-2021 混入 20%）**：数据池 = B/756k 的 492,810 文件（2022-2025，top400_v2；**注意无 2026**）+ 旧年 577,757 候选 → 名单保留 354,564 → 按文件名 sha1 抽 20% = **70,588 文件**，合计 **563,398（+14.3%）**；旧年占比 12.5%（2019 19,811 / 2020 25,819 / 2021 24,958）。设计：B/756k 暖启，不动 obs/奖励/超参，100k 步（756k→856k）；判据 holdout（同 digest）+ ladder 5000 sets（λ=0，新种子 **320000/340000**）；旧年局沿用 `grp_h128.pth`（探针可接受，若有效再重训 GRP）。

| 测量 | B/756k | D1/856k | 差 |
|---|---|---|---|
| holdout top1（同 digest） | 0.8127 | 0.8105 | −0.22pp（噪声内） |
| holdout dqn_loss | 0.3496 | 0.3721 | **+6.4%（变差）**，R² −1.217→−1.359 |
| ladder A（新当挑战者，10000 半庄） | — | **−1.224** ± 0.852 | — |
| ladder B（旧当挑战者，10000 半庄） | — | **+1.579** ± 0.834 | — |
| **δ（新−旧）** | — | **−1.402** | 解析 SE 0.596、t=−2.35、CI[−2.57,−0.23]；**bootstrap SE 0.497、CI[−2.39,−0.44]** |

- 行为：agari .2178→.2144、houjuu .1266→**.1192**、fuuro .2922→.2890、riichi .1823→.1852——**不是“学了旧年代爱鸣牌的风格”（副露反而降），更像价值估计变差导致决策变差**。
- 可能成因（未拆开）：① 年代牌风/强度差异；② 旧年名单阈值 avg_pt≥2.90 比 v2 宽松（强手密度低）；③ **GRP 奖励年代错配**（GRP 只见过 2022-2025）。三条都指向“旧年代不适合当增量数据”。
- **它不否定**：同年代数据补齐（D2）、更强过滤（旧年 top200）、数据质量加权。另提醒：D1 的 −1.40 比 S6c 的 +0.27 更接近“可判”，说明**数据侧变量的效应量比特征侧大一个量级**。
- 运行备注：ladder 期间其他用户的 4 个 3D 检测任务占满 5 张卡，训练速率 4.27→2.64 batch/s；每 job 内两引擎同卡受影响对称，不引入胜负偏。

**D2（补 2026H1 训练窗）**：B 的索引里**一个 2026 文件都没有**（names 修复前被名单滤掉），而修复树 `houou4_names/20260[1-8]` 有 113,430 文件、names 100%、v3 名单覆盖 69.3% 文件（席位 25.3%）⇒ 加 **78,233** 文件，索引 571,043（**+15.87%**）。单变量设计同 D1，仅换数据段；新种子 **360000~362000 / 380000~382000**。

| 测量 | B/756k | D2/856k | 差 |
|---|---|---|---|
| holdout top1（同 digest） | 0.8127 | **0.8124** | **−0.03pp（几乎完全不变）** |
| holdout dqn_loss | 0.3496 | 0.3500 | +0.1%（R² −1.217→−1.219，lse +0.522→+0.422） |
| ladder A（新当挑战者） | — | −0.072 ± 0.836 | — |
| ladder B（旧当挑战者） | — | +0.495 ± 0.845 | — |
| **δ（新−旧）** | — | **−0.283** | 解析 SE 0.594、t=−0.48、CI[−1.45,+0.88]；**bootstrap SE 0.509、CI[−1.26,+0.74]** |

- 行为：agari .2192→.2188、houjuu .1286→.1257、fuuro .2826→.2905、riichi .2005→.1889；800k 的 test_play avg_pt **+4.995**（B 的 best 是 +3.06），但单次 2000 半庄噪声 ±1。
- **解读**：同年代的纯量增（+15.87% 文件 / 100k 步）对 holdout 拟合几乎零影响、ladder 点估计略负 ⇒ 「数据量不是瓶颈」得到**第三次确认**（R3/R4/R5 同分布 reshuffle → D1 旧年代负迁 → D2 同年代补量 null）。
- **数据侧总结（2026-10-25）**：现有 2022-2026 人类牌谱池上「多给数据」已无收益；若继续投入，只剩「按局内 4 人强度加权」与「换监督信号（轨迹/辅助头）」两条。

### 3.16 测量基建：ladder 标准臂与 akochan 绝对锚点（2026-10-25~26）

**为什么先做**：数据侧（R3/R4/R5/D1/D2）与表示侧（S6a/S6b）全部归零、S6c 未决后，候选改进的先验都在 +0.2~0.5；5000 sets 的分辨率只有 ±1.2 ⇒ **不先买分辨率，后续实验只能得到 null**。另有一个更基础的空洞：`test_play` 的对手是自家老模型、ladder 全是内部对照 ⇒ **「我们到底多强」一直没有答案**。

**A. ladder 标准臂（已落地，`mortal/ladder_arm.sh`）**

- 一条命令跑完一个双向臂：`bash ladder_arm.sh <challenger> <champion> <out> --blocks 16 --sets 500 --seed-a S --seed-b T --gpus 0,1,2,4 --guard-lambda 0 [--tag 名]`。
- 默认 **16 块×500 sets/向 = 8000 sets/向 = 32000 半庄/向（双向共 16000 sets / 64000 半庄）** ⇒ 解析 SE≈**0.33**、可分辨 |δ|≈**0.66**（标定基准：历史 5000 sets = 2500/向，SE 0.594）；预算不足时明确警告。
- ⚠️ **脚本单位笔误（待修）**：`ladder_arm.sh` 头部注释与打印把「双向总量」当成了「每向」（会打印 `16000 sets/向`、`SE≈0.47` 并误告警「低于 16000」）；按 1 set=4 局的真实口径以上一条为准，下次改脚本时一并修正。
- GPU 轮转 4 并行、块级 `wait`，产物 + `summary.log`。
- 末尾直接给判定行（正则已用 D2 真实 summary 验证：`δ=-0.283 SE=0.594 t=-0.48 CI=[-1.26,+0.74] → NO-CONCLUSION`）。
- **空臂自检（B vs B，2×100 sets，λ=0）**：A 向两块 `avg_rank 2.5000 / avg_pt 0.000`（完全对称）。

**B. akochan 绝对锚点（通路已通，408 局规模化在跑）**

**构建**：akochan（Equim-chan fork）默认 `NPROCS=核数` 会让 `selector.o` 的 `.bss` 到 **9.4 GB**、链接报 `R_X86_64_PC32 ... relocation truncated`；解法 = `NPROCS=2`（.bss≈0.34 GB）+ 把 `ai_src/obj/*.o` 以 `-no-pie` 静态链进 `system.exe`（不产 `libai.so`）。只编库用 `cargo build --release --lib`（整包会在 `validate_logs` bin 上因 pyo3 符号失败）。

**四个确定性缺陷（全部修完，`libriichi/src/agent/akochan.rs` + 垫片）**：

| # | 缺陷 | 修法 |
|---|---|---|
| 1 | `start_game` 缺 `kyoku_first`/`aka_flag` → `assert_with_out` abort | 垫片补字段（`tools/akochan_pipe_shim.py`，生效位 `tools/akochan_src/ako/system.exe`） |
| 2 | OpenMP 线程数 > 编译期 `NPROCS=2` → SIGSEGV（无输出、exit −11） | 垫片强制 `OMP_NUM_THREADS=2` |
| 3 | 非触发点也读子进程（3 个 akochan 只 1 个回话）→ 死锁 | Rust：非触发点直接返回 `Event::None`、不读 |
| 4 | `set_scene` 用同一份 log 重复调用仍去读 → 卡死 | Rust：`sent_new` 判定 |

- **第一次成功对局**（`ako_vs_py`，1 akochan vs 3 Mortal）：573 次决策 0 卡死 0 illegal action，`challenger rankings [0 1 2 1] (3.0, −22.5pt)`，1 set（4 局并行）**14.7 min**。
- **方向修正**：`ako_vs_py` 测的是 akochan，要测「Mortal vs 3 akochan」必须用 **`py_vs_ako`**；已在 `one_vs_three.py` 加 `[1v3] mortal_vs_ako = true`，此时 `challenger rankings` 才是 Mortal 自己的名次/uma pt。
- **进程数暴涨 → 已修**：`run_anchor.sh` 起初 6 进程 × 68 局时 akochan 进程涨到 **1224**（≈17 sets × 12 席 × 6 进程）、负载 20+，全部 kill；根因是 `py_vs_ako` 的 champion 分组有 12 个 player_id 且每个 set 都重建 agent（`Drop` 不及时）。commit `e557a87` 改用 `iters=N, games_per_iter=4`（每 iter 重建并 drop）→ **1224→72**（6 进程 × 12）。
- **✅ anchor1 已完成（2026-10-27 01:32 UTC，历时 23.7 h）**：6 进程 × 68 局 = **408 局**（102 sets，各进程 RC=0）。结果：名次分布(1..4) **[85, 117, 119, 87]**（均匀期望 102）、**avg_rank 2.510**、**avg_pt +2.87**（uma 90/45/0/−135，3×akochan 人均 −0.96）；朴素 SE 4.06 / CI **[−5.1, +10.8]**，按 ICC≈0.19 修正后 SE≈5.1 / CI≈**[−7.1, +12.9]**。⇒ 点估计略强于 3×akochan，但 **CI 跨 0（不显著）**；与人类标尺（均值 −5.44 / 中位 −2.61 / p95 +9.29）的关系不唯一。要分辨「+3 vs 0」需 ~1600 局（SE≈2，约 4 天）。汇总：`python agg_anchor.py --dir logs/anchor1`。
- **人类场标尺（2026-11-07 修正口径）**：`player_ratings_2022_2026H1_v3.csv` 的玩家级 `avg_pt` 与 `one_vs_three.py` 的 uma（90/45/0/−135）同尺度，但**必须分组看**：

  | 口径 | 人数 | 均值 | 中位 | p95 | SD |
  |---|---|---:|---:|---:|---:|
  | 全部（含几局游） | 5,356 | −5.44 | −2.61 | +9.29 | 19.6 |
  | ≥100 局 | 3,658 | −2.17 | −1.56 | +5.62 | 5.33 |
  | **≥500 局（常客）** | **1,717** | **+0.08** | **0.00** | **+5.38** | **3.18** |

  旧口径拿「全部 5,356 人」的 −5.44 当基准，会把 Mortal 相对抬高 5.5 pt（多数玩家只打几十局、长期均值为负）。**常客基准是 +0.08 / p95 +5.38**。
  另：`houou_games.csv` 的**真实**每局 pt（含段位补正）零和、SD **36.4**、中位 −6.7 —— 与 uma 的理论 SD 84 不是一回事，勿混。
- **✅ anchor3 已完成（2026-11-05 02:12 → 11-07 09:17，约 55 h，5 卡 × 10 进程）**：
  - phase1（S6c + λ=10）**1,588 局**：名次 [376,387,453,372]、avg_rank **2.517**、avg_pt **+0.65**、SE 2.06、CI **[−3.4,+4.7]**
  - phase2（S6c 无守卫）**800 局**：名次 [170,213,225,192]、avg_rank **2.549**、avg_pt **−1.29**、SE 2.90、CI **[−7.0,+4.4]**
  - 两臂差 Δ=**+1.94**、SE 3.56、CI [−5.04,+8.92]、t=0.54 ⇒ **守卫无显著增益**，与 risk 口径修正后 ladder 四臂 NO-CONCLUSION 独立一致
  - **pooled（2,388 局，同 ckpt）avg_pt = +0.00 ± 1.68，CI [−3.3,+3.3]** ⇒ **Mortal(S6c) vs 3×akochan 完全均势**；对照常客基准 +0.08 可知 **Mortal ≈ 凤凰场常客平均**，未达 p95（+5.38）
  - 执行事故：p1 在第 38 批遇 akochan C++ `std::bad_alloc` 早退（RC=1），少 12 局（0.75%，不影响结论）
  - 产物：`mortal/logs/anchor3/summary.txt`、`mortal/logs/anchor3_noguard/summary.txt`、`mortal/start_anchor3.sh`（预检+两阶段+自动汇总）
- **产物**：`mortal/run_anchor.sh`、`mortal/agg_anchor.py`、`mortal/config.anchor_ako.toml`、`tools/akochan_build_notes.md`、`backups/libriichi_20261025/`（改前源码 + `.so` + 回滚说明）。

### 3.17 Guard-P2：状态依赖押し引き（2026-10-27，离线否证，不烧 ladder）

**动机**：守卫 λ=10 是唯一被证明有效的机制（+0.778），但剂量曲线显示线性风险惩罚已饱和（λ=20 → −0.496），
且代价集中在和了率（−4.6%）——说明「无差别扣分」在不需要弃和的局面里也在扣。设计：
`Q' = Q − λ0·w_state(s)·max(risk_w−τ,0)`，`w_state = w_opp(立直家数) × w_turn(早/中/晚) × w_score(点差)`，
`risk_w = Σ_j P(听_j)P(待_j=t)·v_j`（v_j：默听 1.0 / 立直 1.3 / 副露 0.7）。
**只调风险价格、不动进攻价值**；默认 `mode='fixed'` 保持线上行为逐位不变。
设计文档：`memory/DESIGN_guard_statewise_20261027.md`（含 M1~M4 工时与终止条件）。

**M1（已实现，回归 ALL PASS）**：`mortal/guard_state.py` + `tools/test_guard_state.py`。
两个关键实测：
1. **v4 obs 的分数已旋转到本家视角**：`rank==0 ⟺ scores[:,0] 为最大`（100%，n=1256）；
   行号锁定 = 分数 `7..14`（4 家×2 行）/ 名次 one-hot `15..18` / 局数 `19..22` / `tiles_seen` `835`；
2. 分数和 = `100000 − 1000×供託`（立直棒不计入 scores），且 **8.6% 样本存在平分** ⇒
   `lead/deficit` 用「对最高其他人」口径，避免名次索引在平分处不自洽。

**M2（20k 固定集，seed 20261016，基线 = 固定 λ=10，A2c `tl=0`）**：

| 配置 | 变化率 | deal_top1 | p_win |
|---|---|---|---|
| fixed λ=10 | 0 | 0.0094 | 0.0943 |
| risk_w（v=1.3） | 0.26% | 0.0093 | 0.0941 |
| w_opp | 0.19% | 0.0093 | 0.0942 |
| w_turn | 0.50% | 0.0091 | 0.0942 |
| **w_opp×turn** | 0.70% | **0.0089（−5.3%）** | 0.0941（−0.2%） |
| w_opp×turn×score | 0.56% | 0.0092 | 0.0941 |
| w_score only | 0.34% | 0.0095 | 0.0943 |

**结论：无一条满足准入**（`deal ≤ fixed+0.3pp` 且 `p_win ≥ fixed+0.10pp`；所有配置 p_win 都不升）。
最好的 `w_opp×turn` 交换率 **≈25:1**，与固定守卫自身的 **≈27:1** 几乎相同
⇒ **状态重分配只是沿同一条 frontier 移动，没有更好的交易**。
可读作：**Q 已经按局面给风险定价了**（rank-pt 奖励 + obs 本就含点差/立直信息），
外挂 `w_state` 与「把 λ 调大一点」等价——S6a 教训的再次复现。
**决定：不跑 16k 臂（省 5~8 h），Guard-P2 记 neg；`guard_state.py`/回归/`eval_guard_state.py` 保留为基建。**

**3.17b 加性期望损失 `P(deal)×E[loss_pt]`（2026-10-27，仍无支配）**

动机：Guard-P2 的 `v_j` 是乘法权重（等价于调 λ）；这次改成**加性期望损失**：
`loss_j = 3000 + 4000·立直_j + 250·max(0,progress−10)`（点），
`Q' = Q − c·Σ_j risk_j·loss_j`，`c` 一次性标定到「平均惩罚 = λ=10」（实得 `c=0.00198`）。
`tools/read_model/eval_loss_pt.py`（20k 固定集，seed 20261016，A2c `tl=0`）：

| 缩放 | deal_top1 | 相对 fixed | p_win | 相对 fixed | 交换率(Δdeal%/Δpwin%) |
|---|---|---|---|---|---|
| fixed λ=10 | 0.00940 | — | 0.09432 | — | 自身 vs 无守卫 ≈ **27.6** |
| ×1 | 0.00935 | −0.53% | 0.09415 | **−0.18%** | 2.9 |
| ×2 | 0.00780 | −17.0% | 0.09295 | −1.46% | 11.7 |
| ×3 | 0.00705 | −25.0% | 0.09204 | −2.42% | 10.3 |
| ×5 | 0.00625 | −33.5% | 0.09048 | −4.07% | 8.2 |

分层（×1，相对 fixed）：`riichi0`（n=16054）**Δdeal +0.12pp / Δpwin +0.16pp**（不立直的局少弃一点、多和一点）；
`riichi1`（n=3639）−0.55pp / −1.48pp；`riichi2`（n=304）−3.29pp / −1.52pp。
⇒ **机制按设计生效**（把弃和从「无立直家」搬到「有立直家」），但**总交换率比平 λ 更差**
（×3 折到与 fixed 同等安全时约 10:1 vs 27.6:1）⇒ 不满足支配，**不上 ladder**（与 Guard-P2 同结论）。

⚠️ 口径限制：`p_win` 是 SP 前瞻代理，**不含「放铳给立直家 vs 默听」的打点差**，
所以本实验只能否证「在同一条 SP frontier 上更优」，不能否证「按 rank-pt 计更优」；
若要做实，需要把 `p_win` 换成按打点分布积分后的期望顺位损失（成本高一个量级）。

**守卫线至此三条全部探完**：λ 剂量曲线（10 最优）、EV 重排（冗余）、状态依赖（同 frontier）。
要继续只剩两条真正不同的路：① 把 risk 换成 `P(deal)×E[loss_pt]` 并换算回 rank-pt（唯一未否证的推理期方向）；
② 回训练侧做结构性改动（容量探针 / v5 obs），但受分辨率所限。

### 3.18 档 A：自博弈数据（tierA → tierA2，2026-10-29~11-01，两轮零效应 → 关闭）

**动机与设计**：搜索/EI（档 B/C）与全知铳损两条线均已否证；档 A 不依赖任何 teacher，只把模型自己的对局按比例混入人类数据（`train.py` 新增 `dataset.extra_logs/extra_repeat`，不写 `file_index` manifest，不触发索引重建），再用双向标准臂判强度。设计见 `memory/DESIGN_tierA_selfplay_20261029.md`（启动时计划 16k sets、种子 700000/720000，实际执行见下）。

**第一轮（tierA：3.1% / 200k）**

- 数据：`logs/selfplay/s6c_{700000,700500,701000,701500}` 各 2,000 半庄 = 8,000 × `repeat=2`，占混合后 **3.1%**；配置 `mortal/config.tierA.toml`。
- 训练：S6c/956k 暖启，956k→1,140k（200k 步），13.9 h（≈4.0 it/s）。
- holdout（202609，12,288 样本，digest `9c5362bf99d60661`）：top1 0.8167→**0.8132（−0.35pp，容忍带内）**；dqn_loss **−4.7%**（0.3613→0.3442）；R2 −1.2911→−1.1824；var_q/var_t 0.4722→0.4396（警戒 0.38 未破）；agari 149/149 ⇒ 未训崩、价值拟合改善、轻度保守化。
- 行为画像（test_play 末次）：agari 0.2163 / houjuu 0.1244 / fuuro 0.3137 / riichi 0.1912（S6c 末次 0.2167/0.1227/0.3109/0.1855）。
- **臂**：2026-10-30 02:37→15:04 UTC（12.4 h），challenger=tierA/1,140k vs S6c/956k，λ=0，双向 76 块×500 = **38,000 sets/向（76,000 sets / 304,000 半庄）**，种子 800000–837500 / 820000–857500。
- **结果**：A avg_pt −0.244、B −0.196（各 SE 0.216）；**δ=−0.024**，SE 0.153，t=−0.16，解析 CI[−0.32,+0.28]；block bootstrap SE 0.136、95%CI **[−0.29,+0.25]**（ρ≈0.18）⇒ **NO-CONCLUSION（点估计≈0）**。
- 口径提醒：中途汇总曾读出 +0.053（t=0.37，旧读数），跑满 76k sets 后为 −0.024 ⇒ **只认跑满值**，旧读数不要引用。

**第二轮（tierA2：12.5% / 400k，决定性检验）**

- 数据：8 个目录 `s6c_{700000,700500,701000,701500,710000,711700,713400,715100}` 共 **35,200 半庄 × repeat=2**，占混合后 **12.5%**（563,210 文件）；配置 `mortal/config.tierA2.toml`。
- 训练：S6c/956k 暖启，956k→1,356k（400k 步），2026-10-31 02:27→11-01 05:53 UTC（**27.4 h**，≈4.2 it/s），RC=0；判定臂用最后一个 20k 快照 **1,340k**（末次落盘 1,356k 未单独判，差 16k 步）。
- holdout：**未测**（`tierA2_health.log` 只有 S6c/956k 与 tierA/1,140k 两行；本轮直接以臂为准，离线画像缺失是流程缺口）。
- 行为画像（test_play 末次）：agari 0.2210 / houjuu 0.1200 / fuuro 0.2801 / riichi 0.1899（相对 tierA 末次：和了 +0.5pp、放铳 −0.4pp、副露 **−3.4pp**、立直 −0.1pp；test_play 噪声大，仅作画像）。
- **臂**：2026-11-01 05:51→15:49 UTC（9.97 h），5 卡（0–4，`CUDA_DEVICE_ORDER=PCI_BUS_ID`），双向 76 块×500 = **38,000 sets/向（76,000 sets / 304,000 半庄）**，种子 910000–947500 / 930000–967500；均速 15.3 块/h，日志无 error/traceback。
- **结果**：A avg_pt −0.016、B −0.159（各 SE 0.216）；**δ=+0.072**，SE 0.153，t=+0.47，解析 CI[−0.23,+0.37]；block bootstrap SE 0.128、95%CI **[−0.17,+0.33]**（ρ≈0.184）⇒ **NO-CONCLUSION**。

**合并判读（结案依据）**

1. **占比无梯度**：3.1%→12.5%（4 倍）+ 步数 200k→400k，δ 仅从 −0.024 到 +0.072，两个 |t|<0.5、CI 全跨 0 ⇒ **“信号被稀释”假设不成立**；
2. **样本量算术**：若真效应 +0.07，要 2σ 需 ≈**1,382,012 sets**（当前 76,000 的 18.2 倍）；若真效应 −0.02 需 1,206 万 sets ⇒ 投入产出不成立；
3. **预注册判据触发**（δ>+0.3 才算成功；|t|<2 ⇒ 自博弈无效）⇒ **档 A 于 2026-11-01 正式关闭**，不再以“加占比 / 加步数”重开；
4. **同质化风险未出现但也没收益**：行为画像未崩、第一轮 holdout 带内，强度与父代等价。

**工程侧沉淀（可复用）**：`extra_logs/extra_repeat` 不触索引、可直接混采；`selfplay_gen.py` 4 席同权重 ≈6 s/半庄（8k 半庄四卡≈3.3 h）；判定臂 5 卡并行 ≈15.3 块/h（76k sets 约 10 h），瓶颈在 CPU 数据管道而非 GPU；块间 `ladder.py` 重启会让 worker 计数瞬时掉到 1~4，属已知误报，看门狗应“连续 2 次采样低值”才告警。

**产物**：`logs/arm_tierA.log` / `logs/arm_tierA/summary.log`、`logs/arm_tierA2.log` / `logs/arm_tierA2/summary.log`、`logs/tierA_train.log`、`logs/tierA2_train.log`、`checkpoints/tierA/`、`checkpoints/tierA2/`、`mortal/config.tierA.toml`、`mortal/config.tierA2.toml`、`mortal/watch_arm2.sh`。

### 3.19 数据质量加权：100k 探针（2026-11-02，门禁未过 → 关闭）

**动机**：数据侧只剩“局内 4 人强度加权”未试；top400 只保证一局至少 1 名强手，另外 3 席可能偏弱，BC 可能模仿弱手坏习惯。

**设计**：每局取 4 名玩家长期 avg_pt 的最小值（每人 ≥100 局）作为质量分；按分位 [0.5,0.75,0.9] 映射重复 [1,1,2,3]（q50 惰性，实际为底部 75% ×1、75–90% ×2、前 10% ×3），mean **1.350**；492,810 文件全部保留，扩到 **665,485** 条；只改 file_list，不改损失/file_index manifest（`train.py` 新增 `[dataset.strength_weight]`，权重表 `indexes/strength_weights_v1.pth`）。

**训练**：S6c/956k 暖启，956k→1,056k（100k 步），2026-11-02 03:45→10:36 UTC（**6h51m**，4.41 batch/s），RC=0；2k 冒烟先过（命中 492,810/492,810、预算到达、无 NaN）。

**holdout（202609，12,288，digest `9c5362bf`）**

| 指标 | 加权 1,056k | S6c/956k | 变化 | 门禁 |
|---|---|---|---|---|
| top1 | 0.8141 | 0.8167 | −0.26pp | ≥0.8117 ✅ |
| dqn_loss | 0.3816 | 0.3613 | **+5.6%** | ≤0.365 ❌ |
| R2 | −1.4199 | −1.2911 | 变差 0.129 | — |
| var_q/var_t | 0.5105/0.3154 | 0.4722/0.3154 | Q 尺度 +8.1% | — |
| agari_top1 | 149/149 | 149/149 | 持平 | — |
| logsumexp_q | +0.471 | +0.525 | −0.054 | — |

**判读**：top1 在噪声内（SE≈0.35pp），行为一致率未坏；但 dqn_loss/R2/Q 尺度三项一致恶化 ⇒ 加权把分布推向四强桌后价值拟合变差。按预注册两项同时满足才算过 ⇒ **不烧 76k 臂（省 ~10 h），关闭该轴**；数据侧机制（reshuffle/补量/混年代/自博弈/质量加权）至此全部走完，无正信号。

**边界**：① 无同预算不加权对照，dqn_loss 上升可能混入续训漂移（但 tierA 200k 是 −4.7%，更可能归因加权）；② 评分表为全窗口静态值，早期牌局有轻微未来信息泄漏；③ 重复条目降低 100k 内的唯一牌谱数（~16 万 vs 父代 220k），本探针只回答“该方案是否值得继续”，不等于“机制必然无效”；④ `best.pth` 因 reset 不可与父代比较。

**产物**：`memory/DESIGN_strength_weight_20261102.md`、`tools/training/build_strength_weights.py`、`mortal/config.strength_probe{,_smoke}.toml`、`mortal/run_strength_probe.sh`、`mortal/indexes/strength_weights_v1.pth`、`mortal/checkpoints/strength_probe/`、`mortal/logs/strength_probe_*.log`。

### 3.20 在线自博弈（M5）：管线跑通 + 卡死修复，效果中性（2026-11-07~08）

**动机**：档 A 的**离线**自博弈（自生成 8k/35k 半庄再混入人类数据）两轮全零效应（§3.18），
根因是“离线生成的样本对当前策略没有增量信息”。在线版把生成与训练耦合成循环，数据始终来自**当前策略**（+ boltzmann 探索）。

**⚠️ 命名澄清（重要）**：`player.py` 的 `TrainPlayer` 实际是
`OneVsThree.py_vs_py(challenger=学习者, champion=固定 baseline)` = **1 个学习者席位 vs 3 个固定卫冕者**。
所以这条线的准确定义是「**on-policy 数据生成**」，不是“四个自己打”的对手共演化自博弈；
真正的自博弈需要对手池，本仓库没有。

**架构（原代码，配置里标“暂未启用”）**：

| 组件 | 职责 |
|---|---|
| `mortal/server.py` | 参数服务器：`get_param` 下发权重 / `submit_replay` 收牌谱 / `submit_param` 收新权重 / `drain` 交付样本 |
| `mortal/client.py` | 对局 worker：拉权重 → `TrainPlayer.train_play` 打 4 局 → 提交 |
| `mortal/train.py` | 训练侧：`drain()` → `train_epoch()` → `submit_param()` |

**修的 7 个问题**（前 6 个是“一开开关就坏”；第 7 个是原作者留了 5 年的卡死）：

| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| 1 | `train.py` | online 时 `max_train_steps` 被 `if not online` 跳过 ⇒ **无终止条件** | 新增 `control.online_max_steps`，按**本次启动的增量**判断 |
| 2 | `train.py` | **每轮（4 局）就推一次权重**，策略极度非平稳 | 新增 `online.submit_round_interval`（缺省 1 = 旧行为） |
| 3 | `train.py` | 外层是 `Popen` **崩溃自动重启循环**，`break` 会导致无限续跑 | 写 `<state_file>.DONE` + `sys.exit(1)` 让父进程一起退 |
| 4 | `client.py` | 权重未变时仍每轮重传+重载 41MB | 独立 `loaded_version` 变量，未变则 `skip load` |
| 5 | `player.py` | **多 client 共用 `log_dir`**，各自启动时 `rmtree` 会删对方牌谱 | `MORTAL_WORKER_ID` 环境变量分流 |
| 6 | `common.py` | `drain()` 空轮询 `sleep(5)`，白等占总墙钟 23% | 改 `sleep(1)`：1.85 → **2.54 步/秒** |

附带：`recv_msg` 的 `MemoryError`（header 错位时 `bytearray(size)` 直接爆，堆栈看不到 header）改为明确的
`protocol desync: header=... size=... peer=...`。

**#7 原作者 5 年未解的 `test_play` 后卡死（2026-11-08 修掉）**：
原代码在 `test_play` 后留了 `if online: sys.exit(0)`，注释写
“BUG: unknown reason ... the process will get stuck here ... this is the reason why main spawns a sub process”。

- **根因（高置信）**：`test_play` 在本进程里初始化了 **CUDA context** 与 **libriichi 的 rayon 全局线程池**
  （`stat.rs:17 use rayon`、`agent/mortal.rs:259 rayon::spawn` + `Arc<Mutex<SyncFields>>` + `WaitGroup`）；
  随后 `DataLoader(num_workers=8, pin_memory=True)` 用 **fork** 起 worker——子进程继承了这些线程的**锁状态**
  但**线程本身不随 fork 复制** ⇒ 子进程可能永远等在某个锁上。Python 3.12 对此有明确告警：
  `DeprecationWarning: This process is multi-threaded, use fork() may lead to deadlocks in the child.`
- **为何 offline 不卡**：只跑一个 epoch 就 `break`，不会在 `test_play` 之后再 fork。
  **为何 online 每 20k 步必卡**：`test_every == save_every == 20000`，每次 save 之后立即踩雷。
- **修法**：online 模式强制 `num_workers = 0`（每轮只 4~16 个牌谱，单进程加载 < 2 s），从源头消除 fork。
- **验证（连续 5.7 小时）**：`loaded:` 从“每 2 小时 1 次”变成 **全程 1 次**；`round` 从“反复回到 5~9”
  变成 **连续 1505**；跨过 1,100,000 的 test 点无重启（`mortal_step_001100000.pth` 已生成）；0 错误。
- **诚实边界**：三次简化复现（单独 rayon / 单独 CUDA / 单独 `pin_memory`）**都不卡** ⇒ 死锁是概率性竞态，
  根因不能 100% 确证；但 `num_workers=0` 消除了 fork 这个**必要条件**，逻辑上充分。若将来又观察到卡死，
  恢复 `if online: sys.exit(0)` 作为临时绕过即可（注释已留在原处）。
- **附带**：终止条件改为 `control.online_target_steps`（**绝对**步数）。原来的增量口径
  `online_max_steps` 在每次重启后重置，永远达不到目标。

**M5-A：200k 步正式跑 → 模型完全坍塌（2026-11-08 13:24 完成）**

跑满 1,156,135 步（200k）、关机路径正常（`.DONE`）、全程 0 重启。但 **holdout top1 0.8167 → 0.2417（−57.5pp）**，
行为画像：agari 21.7%→**0.55%**、houjuu 12.3%→**28.7%**、fuuro 31.1%→**68.4%**、riichi 18.6%→**1.04%**、avg_rank 2.5→3.89。
**且坍埽在第一个 20k 步（~45 min）就发生**。

**根因（三重叠加，主因是 LR）**：
1. **LR 未随 batch 缩放**（主因）：`[optim.scheduler] peak=final=1e-4` 是 S6c 的 **batch-512** 口径，
   而本次 `batch_size=32` ⇒ 等效 LR = 1e-4 × 512/32 = **1.6e-3（16 倍）**。
2. **optimizer/scheduler 未从 ckpt 继承**：`train.py:444` 的 `if not online or state[...]["online"]`
   在 online 时整块跳过 ⇒ LR 不是 S6c 末端的 ~1e-9，而是全新的 1e-4。
3. **纯 on-policy 无人类锚定**：online 的 `train_epoch` 只读 drain，不混人类数据。
   对照：档 A 混了 87.5% 人类数据 ⇒ 零效应而非坍埽。

**诊断特征（有用）**：`dqn_loss` 只涨 3.7%（0.3613→0.3746）但 top1 崩 57pp ⇒ 这是“大 LR 破坏 Q 排序
但不抬高绝对误差”的典型特征；若为数据分布问题，`dqn_loss` 会显著恶化。

**M5-B：LR 修正后的 10k 步验证（2026-11-08 13:28~14:27）→ 中性**

`peak=final=6.25e-6`（= 1e-4 × 32/512）、从 S6c/956,125 重起、跑 10,000 步（171 轮、~2,736 局），
`config.selfplay_lr.toml`、端口 5430、产物 `checkpoints/selfplay_lr/`。

| 指标 | S6c | 10k 步后 | 变化 |
|---|---:|---:|---:|
| agari / houjuu | 21.67% / 12.27% | **21.48% / 12.08%** | −0.19 / −0.19 pp |
| fuuro / riichi | 31.09% / 18.55% | **28.84% / 20.50%** | −2.25 / +1.95 pp |
| avg_rank（test_play） | ~2.45 | **2.4545** | ≈0 |

⇒ **LR 修正后完全不坍埽**，确认 M5-A 的崩是配置而不是方法。

**holdout（必须同 seed 比！三个 seed 各 60 batches）**：

| seed | S6c/956k | M5-LR/960k | 差值 |
|---|---:|---:|---:|
| 20261009 | 0.8365 | 0.8286 | −0.79 pp |
| 11111 | 0.8120 | 0.8047 | −0.73 pp |
| 22222 | 0.8109 | 0.8125 | +0.16 pp |
| **平均** | **0.8198** | **0.8153** | **−0.45 pp** |

`dqn_loss` 系统性下降（三个 seed 平均 **−7.9%**）、`var_q` 从 0.37~0.49 缩到 0.31~0.45、
`logsumexp_q` 从 ~0.52 降到 ~0.365 ⇒ Q 值尺度收缩、拟合更平滑，但 **argmax 略差**。

**⚠️ 方法论教训（2026-11-08 自己踩的）**：首次只用 seed=20261009 测得 M5 top1=0.8286，
与 S6c 的**历史值** 0.8167（另一个 seed）相比得出“+1.19pp 提升”——**跨 seed 比较**。
同 seed 重测后发现 S6c = **0.8365**（比 M5 高 0.79pp）。**任何 holdout 对比都必须同 seed 同 batch 同 cfg**，
否则 1,920 样本的 seed 方差（~±1.5pp）会伪装成“大提升”。这与 P0-a 的“P2c 15/18”和守卫 risk 双乘是同一类错误。

**结论**：在线自博弈（纯 on-policy、无人类混合）**10k 步 = 中性**（行为对齐、holdout −0.45pp 在噪声内）。
未跑 200k，因为：① 10k 步已显示中性；② `dqn_loss` 改善伴 `top1` 微降，提示自博弈数据在**稀释人类先验**；
③ 档 A（离线混合 12.5%、35k 局）已证零效应。**除非改成“人类:自博弈 = 8:1 混合 + batch 512”，否则不再烧卡。**

**两个自己写出来的 bug（已修，留作教训）**：① `client.py` 首版用
`param_version == rsp['param_version']` 判断，但前一行刚给它赋值 ⇒ 永远相等 ⇒ **永久冻结在初始权重**；
② `train.py` 首版用 `Path(...)` 写标记，但 `train()` 里只导入了 `sys`/`os`/`path`，没有 `Path` ⇒ 停止时 NameError。

**吞吐（实测，4 client + 1 trainer）**：

| 项 | 数值 |
|---|---|
| 单 client | 4 局 / 63 秒 |
| trainer | **2.54 步/秒**（≈9,150 步/小时） |
| 每轮 | drain 到 4~16 个牌谱 → 30~152 步 |
| **200k 步 ETA** | **≈22 小时** |
| 产出规模 | ≈21,000 局（与档 A 第二轮 35,200 局同量级） |

**启动（4 client）**：

```bash
cd mortal
# server（不占卡）
CUDA_DEVICE_ORDER=PCI_BUS_ID MORTAL_CFG=config.selfplay_smoke.toml python -u server.py
# trainer（GPU0）
CUDA_VISIBLE_DEVICES=0 ... python -u train.py
# 4 client（GPU1-4）
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i+1)) MORTAL_WORKER_ID=$i ... python -u client.py
done
```

**产物**：`mortal/config.selfplay_smoke.toml`（生产）、`mortal/config.selfplay_test.toml`（150 步冒烟）、
`logs/selfplay_smoke/`（server + 4 client + trainer 日志）、`mortal/checkpoints/selfplay_m5/`。

**判据（跑完才执行）**：holdout top1 ≥ 0.8167（不低于 S6c）+ 16k 双向臂 bootstrap CI 下界 > 0。

---

## 4. 现状与下一步

### 4.1 方向状态（聚合；数字与证据见对应 §3 小节）

| 方向 | 状态 | 依据 |
|---|---|---|
| 数据选择（top400 + GRP h128） | ✅ 训练侧最大增益（+2.86±0.59），已采纳 | §3.3 |
| 数据侧变化（epoch/增强/收紧名单/补量/混年代） | ❌ 全关闭：R3/R4/R5 null、D1 负迁移、D2 null | §3.4、§3.5、§3.7、§3.15 |
| 自博弈数据（档 A：3.1%→12.5%，200k→400k） | ❌ 两轮全部零效应（−0.024 / +0.072，CI 跨 0），2026-11-01 关闭 | §3.18 |
| 数据质量加权（按局内 4 人强度） | ❌ 100k 探针 dqn_loss +5.6%（门禁 ≤+1%）、top1 −0.26pp，2026-11-02 关闭；数据侧机制全部走完 | §3.19 |
| 奖励口径（B `[2,1,0,−3]`） | 🔶 采纳为基线但未显著；C 显著变弱关闭 | §3.6 |
| 超参（cql/aux） | ❌ 非瓶颈 | §3.8 |
| 信用/目标形式（IQL A/B） | ❌ 两子路线全部关闭 | §3.9、§3.10 |
| 表示注入 | S6a/S6b ❌ 关闭；S6c ✅ 已采纳（77k 臂 **+0.422**、CI[+0.15,+0.69]，2026-10-28 换代） | §3.11、§4.2 |
| 读牌模型（注入上游） | 🧊 冻结为归档资产（A2c） | §3.12 |
| 推理期守卫 G3 | ✅ 已采纳 λ=10（R2 后唯一新增增益） | §3.13 |
| 推理期 EV 重排 | ❌ 离线否证（结构冗余），不进 ladder | §3.14 |
| 推理期状态依赖守卫（Guard-P2） | ❌ 离线否证（7 配置全无支配，与固定 λ 同 frontier），不烧 ladder | §3.17 |
| 推理期加性期望损失（`P(deal)×E[loss_pt]`） | ❌ 机制生效但交换率 10:1 < 平 λ 的 27.6:1 ⇒ **推理期线封顶** | §3.17b |
| 容量/宽度（net2wider） | ❌ **前置诊断否证**（train≈holdout，无过拟合/欠拟合余量），不做 | §4.2 |
| 泛化间隙诊断 | ✅ train(202512) top1 **0.8163** vs holdout(202609) **0.8161**（Δ=0.02pp），dqn_loss 0.352/0.343 ⇒ **BC 噪声地板，非容量瓶颈** | §4.2 |
| 测量基建 | ✅ ladder 标准臂已固化（16000 sets/向）；akochan 锚点 **2,388 局完成**（pooled +0.00±1.68） | §3.16 |
| **在线自博弈（M5）** | ⏸️ **2026-11-08 结案：中性**。管线已跑通、修了 7 个问题（含原作者 5 年的卡死）；但 200k 步那次因 LR 未随 batch 缩放而坍塌（−57.5pp），LR 修正后 10k 步 = 中性（行为对齐、holdout −0.45pp 在噪声内） | §3.20 |

### 4.2 下一步（只列未完成事项；已关闭项见 §4.1）

| 优先级 | 事项 | 要点/判据 |
|---|---|---|
| **P0** ✅ | akochan 绝对锚点 408 局（**已完成**） | anchor1：**+2.87 pt/半庄**（CI[−7.1,+12.9]，含组内相关修正），名次 [85,117,119,87]、avg_rank 2.510； 人类标尺 −5.44/+9.29 **只是 uma 单位参照，对手场不同，不能读成人类百分位**；作为外部标尺使用，需 ~1600 局才能判 +3 级差异 |
| **P0** 🔜 | **anchor3：部署组合 S6c+λ=10 vs 3×akochan ~1600 局**（同协议补 800 局 S6c 无守卫） | 2~4 天（CPU）；SE≈2，回答“+3 vs 0”；**不用于验证守卫 +0.78**（§3.16） |
| **P0** | 下一轮改进的判定预算 | 用 `ladder_arm.sh --blocks 16`（双向共 16000 sets，SE≈0.33）；**新种子块取 §2.4 的「下一块」**；候选排序见 `memory/NEXT_STEPS_ANALYSIS.md` §4；δ 的 bootstrap CI 下界 >0 才采纳 |
| **P1** ❌ | 推理期**状态依赖**守卫（Guard-P2，**已否证 2026-10-27**） | 20k 上 7 个配置全部 `p_win` 不升（−0.1~−0.2%）、无支配；最好 `w_opp×turn` 的交换率 25:1 ≈ 固定守卫自身 27:1 ⇒ 只是沿同一 frontier 移动（Q 已按局面定价）⇒ **不烧 ladder**；实现与回归保留为基建（§3.17，设计 `memory/DESIGN_guard_statewise_20261027.md`） |
| **P0** 🔄 | ~~S6c 77k 复判（2026-10-27 02:52 进行中，已由下行结论取代）~~ | 臂：`checkpoints/s6c/mortal.pth`(956k) vs B/756k，λ=0，**双向 76 块×500 sets = 76,000 sets / 152,000 半庄**，种子块 **400000–437500 / 420000–457500**（首次使用）；`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `--gpus 2,2,2,2`（4 并发于 GPU2 Quadro RTX 8000，3.7 GB/100%）；解析 SE≈0.22、δ_min≈0.43（比预估好，因为总量是双向 76k）； 产物 `logs/s6c77k/`；ETA ≈ 9~14 h。**⚠️ 坑：不设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`  时 CUDA 索引 ≠ nvidia-smi 索引（cuda:2 = P100 而非 RTX 8000），会静默跑到别的卡上** |
| **P0** ✅ | **守卫 λ=10 在 S6c 上复验（8 副本完成，确认有效）** | **+0.402 / +0.046 / +0.523 / +0.176 / +0.629 / +0.489 / +0.385 / +0.135 → 均值 +0.348、8/8 正号**；理论 SE 0.162 ⇒ **t=2.15、CI[+0.03,+0.67]**；经验跨副本 SD 0.194 ⇒ SE 0.068、t=5.09、CI[+0.21,+0.48]。对照 B/756k 同协议 **+0.78** ⇒ 守卫在 S6c 上仍有效，幅度约为 B 的 **45%**。**按预注册判据（均值≥+0.25 且理论 t≥2）⇒ 部署组合 = `S6c/956k + guard λ=10`**；种子 440000~580000 已消费；产物 `logs/guard_s6c/`（8 个副本 + `wave2.log`） |
| ~~旧记录（第二波进行中）~~ | `run_guard_s6c_wave2.sh`：同协议再加 4 副本（seed 520000/540000 → 560000/580000，两波各 2 并发于 GPU2），凑满 **8 副本 × 8000 sets = 64,000 sets（256,000 局）**，理论 SE≈0.16、经验跨副本 SD 0.187 时 SE≈0.066；ETA ≈ 7~8 h（北京 ~23:30）；产物 `logs/guard_s6c/l10_5*.log` + `wave2.log` |
| **P0** ⚠️ | **守卫 λ=10 在 S6c 上复验（首 4 副本，结论：偏弱/临界）** | **+0.402 / +0.046 / +0.523 / +0.176，4/4 正号，均值 +0.287**；理论 SE 0.229 ⇒ t=1.25（不显著），而跨副本观测 SD 仅 0.187（4 副本 df=3 ⇒ 经验 t≈3.1、p≈0.05）⇒ **临界**。对照 B/756k 同协议 **+0.78** ⇒ 守卫在 S6c 上的边际增益约为 B 的 **1/3**。**不做部署结论**；要定论需 16k sets 标准臂（S6c+guard vs S6c） |
| **P0** ⚠️ | **anchor2：S6c+守卫 vs 3×akochan（89 sets / 356 局，完成 2026-10-28）** | 名次 [82,86,100,88]、avg_rank 2.545、**avg_pt −1.77**（SE≈4.35，CI[−10.3,+6.7]）；对照 anchor1（B 无守卫，408 局）**+2.87**（CI[−7.1,+12.9]）⇒ 点估计 **低 4.6**，但差值的 SE≈5.9、**无统计结论**。**注意**：p1 因 akochan EOF 崩掉（少 13 sets，89/102），且守卫在弱对手场（akochan << 人类）里可能过度弃和 —— anchor 场与 ladder 场强度不同，不可直接互推 |
| ~~旧记录~~ | **已完成两副本：+0.402（seed 440000）/ +0.046（seed 460000）→ 均值 +0.224、SE≈0.32（t≈0.7，不显著）**；对照 B/756k 同协议 +1.226/+0.330（均值 +0.78）⇒ 守卫在 S6c 上的边际增益疑似缩水到 1/3。**2026-10-28 01:55 追加 r3/r4（seed 480000/500000）于 GPU2**，4 副本合计 SE≈0.23，用于分辨「真缩水」还是噪声 |
| ~~旧记录~~ | 沿用剂量曲线协议：challenger=champion=`checkpoints/s6c/mortal.pth`、`--guard-lambda 10`（**只作用于挑战者席**），**两个独立副本**（seed 440000 / 460000，各 8000 sets=32,000 半庄）并行于 GPU2；判据=两副本同号且均值明显 >0（B 上是 +1.226/+0.330→+0.78）；SE≈0.46/副本。汇总看 `logs/guard_s6c/l10_*.log` 的 avg_rank/avg_pt |
| **P0** ✅ | S6c 77k 复判（**转正**） | **δ=+0.422、bootstrap CI[+0.15,+0.69]**（A +0.447 / B −0.398，双向同号）；主模型换代为 **S6c/956k**；**待办：守卫 λ=10 在 S6c 上复验**（λ 是在 B 上扫的）+ 锚点重测（可选） |
| **P0** ❌ | ~~档 A：自博弈 + MC 回报~~ **两轮均零效应，2026-11-01 关闭** | 第一轮（3.1%/200k，76k sets 双向）：δ=**−0.024**、t=−0.16、bootstrap CI[−0.29,+0.25]；第二轮（12.5%/400k，76k sets 双向）：δ=**+0.072**、t=+0.47、CI[−0.17,+0.33] ⇒ 无梯度、CI 全跨 0；追 +0.07 需 ~138 万 sets（18 倍）不划算。详见 §3.18 与 `memory/DESIGN_tierA_selfplay_20261029.md` §7 |
| **P0** ❌ | ~~容量探针（net2wider 192→256）~~ **诊断否证，不做** | 2026-10-27 前置诊断：同口径 30,720 样本下 **train(202512) top1 0.8163 vs holdout(202609) 0.8161**（Δ=0.02pp）、dqn_loss 0.352 vs 0.343 ⇒ 无泛化间隙、已到 BC 噪声地板，加宽无余量；审计另确认：全仓 48 个 toml 均为 192ch/40blocks，唯一容量变动是 GRP h64→h128（与 R2 数据变更混杂） |
| **P0** ❌ | ~~数据质量加权（局内 4 人强度）~~ **100k 探针门禁未过，2026-11-02 关闭** | top1 **0.8141**（−0.26pp，带内）、dqn_loss **0.3816（+5.6% > +1% 门禁）**、R2 变差 0.129、var_q +8.1%、agari 149/149 ⇒ 不烧 76k 臂，关闭该轴；详见 §3.19 |
| **P2** | RM10 轨迹级监督 | 读牌侧唯一新监督；需先解冻读牌模型，并说明它为何不同于 S6b 注入（坑 72：BC 不需要显式读牌） |
| **P2** | 放铳率辅助头 | aux 预测本局放铳；风险是辅助头绕过主通路（坑 63） |
| **P2** | 工程债 | 异地备份、上游 `git diff` 校准、两套 holdout 同分布口径、agari-guard holdout、kan-select 回放对照、optimizer 暖启消融、AMP 重开验证、文档对账 |
| **P1** ⏸️ | 在线自博弈（M5）**已结案：中性**（纯 on-policy、无人类混合）。1 个真实收获：原作者 5 年的 `test_play` 后卡死被修掉（`num_workers=0`）；剩下两个是坑：LR 必须随 batch 缩放、holdout 对比必须同 seed | §3.20 | 若要重试：人类:自博弈 = 8:1 + batch 512 + 强对手；否则不再烧卡 |
| **P3** | 更强行为先验（若要继续压上界） | 读牌 README §9 的参考策略弱于 A2b；要回答「还剩多少」需 ukeire/安全度策略或学习型 policy，并保留 `hard` 下界 |

### 4.3 当前资产与基线

- 现役最强（历史称法）：`round4/645k`（fp32，0 非有限张量）；**实际部署与判定基线是 B/756k**（相对 R4 的消融 B 组，见 §3.6）。
- 判定基线：B/756k（`reward_ablation/B_rank_pt_aligned/mortal.pth`，top400_v2 + `[2,1,0,−3]` + aug）；推理默认加载守卫 λ=10。
- 已封存不续用：`iql_b`(v1)、`iql_b_smoke`、`iql_b_smoke_v2`、`iql_b_v2`（漂移轮）；`inject_s6b/A1~A3` 需 checkout `27632a2` 才能加载。
- 可用但非基线：`round5/870k`、`iql_a/867k`、`iql_b_v3/867k`、`s6a/867k`、`s6c/956k`、`d2_2026/856k`、`mix_v1/856k`（D1）。
- 档 A 产物（均未采纳，保留复核）：`tierA/1,140k`、`tierA2/1,340k`（判定对象；末次落盘 1,356k）。
- 数据质量加权探针（未采纳，保留复核）：`strength_probe/1,056k`（top1 0.8141、dqn_loss 0.3816，§3.19）。
- 判定牌谱均保留在对应 `ladder/` 或 `logs/ladder_r*/`，可事后重算/bootstrap；路径台账见 §6.3。

### 4.4 新会话接手三步

1. 读 §0.1 / §4.1，确认“现在在哪、下一步选什么”；环境看 §1.7。
2. `tmux ls` + `ps -eo pid,etime,args | grep python` 查残留；长任务只用 tmux，日志按 §5.3 归置。
3. 开跑前 `cd mortal && python3 env_check.py --cfg <本轮 config>`；训练完只认 ladder 双向 ≥16000 sets + 全新种子块 + block bootstrap CI（§2.4）。

---

## 5. 工程规范与坑位

### 5.1 强制规范

1. **长任务一律 tmux 承载**，不前台直跑、不 nohup；日志按 §5.3 归置（启动前 `mkdir -p`，坑 34）。
2. **每次任务前重置全部环境变量**（§1.7），不依赖上次会话残留。
3. **冒烟必须含数值型行为门禁**：avg_rank、top1、logsumexp_q 等；结构性检查（无 NaN、到预算）不够（坑 36）。
4. **判定纪律按 §2.4**：全新种子块、双向同牌山、block bootstrap CI；固定 `test_play` 只作趋势。
5. **并行实验隔离**：改变 obs/目标/数据口径的实验必须复制代码 + 独立 `.so` + 本地 index/ckpt，禁止就地改共享树（范例 `s6a_isolated/`）。
6. **停机规程**：SIGINT 优雅停 → 清理 DataLoader 孤儿 → 校验 ckpt 可 `torch.load` → 清阶段性产物 → git 提交停机说明。
7. **提交规范**：`git status --porcelain=v1` → 显式 `git add` → 中文提交信息含改动/验证/文档对账；运行产物先加 `.gitignore`。
8. **主模型训练优先 GPU 2（RTX 8000）**；GPU 1 易过热降频（实测 SM 607 MHz、速率降至 1.2 batch/s），避免长训练。

### 5.2 坑位总表（按主题，编号沿用旧版便于回查）

| # | 主题 | 坑与处置 |
|---|---|---|
| 1 | 数据 | scc 是 html-escaped，先 `html.unescape` 再正则，否则含 `<>&` 的玩家名行错位 |
| 2 | 数据 | 新行可能插在文件中间（长时对局），一切按 log_id 去重 |
| 3 | 数据 | `00a9`=四鳳、`00b9`=三鳳是观测规律；过滤以行内规则文本为准 |
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
| 18 | 评测 | `best_perf` 初始值/继承与“更强 ckpt”无关（换代/暖启时重置为哨兵 `{4.0,−135.0}` + 一次性令牌）；选模型用 ladder，不信任 best.pth |
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
| 42~67 | 读牌 | **读牌专题坑位（P(听) 口径分裂、CE 必须先分解、分桶 support 命名、选型判据须覆盖曾退化指标、cosine 走平≠到顶、CLI 参数静默失效、设计动机与实现对账、必须减行为盲地板、合并跨格引入组成混杂、反事实要配逐样本期望、集合级掩盖顺序级、tile-agnostic 池化承载不了逐牌信息、三家共享同一份 `h`、切序直觉方向可能反、地板/参照系须先验证其对自变量不响应、同一符号只能有一种约定且明细必须落盘、判据段重构后必端到端跑通并做基线数值回归、冗余通路/辅助头可绕过目标通路、显式特征被权重使用≠反事实响应、输入消融先于通路工程、暖启探针不能用 --resume、枚举式 H2 三源不可行）已随读牌记录整体迁至 `tools/read_model/docs/README.md` §14，编号原样保留** |
| 68 | 评测 | **小样本决策指标不能当选型判据**（2026-10-22 P0 实证）：`deal_pure` 在官方 20000 集上 `n_deal=1476`（≈16 个点炮事件），点估计 0.0108；扩到 100k 后 0.0147（n_deal 7507），**差 36%**。处置：任何 deal 类结论必须 ≥100k 或按文件 block bootstrap 报 CI，且不得与 20k 的旧值直接横比 |
| 69 | 工程 | **同名 `model.py` 会静默串味**（2026-10-22）：`tools/read_model/model.py`（读牌 v1 骨干，含 `ReadBackbone`）与 `mortal/model.py`（主模型 `Brain`）同名；`_common.bootstrap_paths()` 把 `mortal/` 插到 `sys.path` 最前（脚本目录已存在时它不会再插 `tools/read_model`），裸 `import model` 会命中主模型版本，`model_v2.py` 报 `ReadBackbone` 缺失。处置：用 `spec_from_file_location` 显式占住名字（读牌=`model`、主模型=`mortal_model`），`train.py` 同理 |
| 70 | 工程 | **首层手术后必须重置 optimizer**（2026-10-22 S6b 冒烟实证）：`Conv1d(192,1012,3)→(192,1014,3)` 的模型已换形，但 ckpt 里的 AdamW 动量仍是 1012 维 ⇒ 第一步 `AdamW.step` 报 `size of tensor a (1012) must match b (1014)`。处置：config 设 `reset_optimizer='<新 token>'` 触发跳过 optimizer/scheduler 加载（新通道零初始化，旧动量本就不适用）；另注意新通道梯度 = 上游×输入，零填充样本对它的梯度为 0，预计算覆盖不全时信号会被稀释 |
| 71 | 工程 | **暖启训练的 `state_file` 绝不能指向基线/共享起手 ckpt**（2026-10-22 S6b 实证）：train.py 每个 `save_every` 会把训练中权重**原地写回** `control.state_file`。A0 臂直接指向 `reward_ablation/B_rank_pt_aligned/mortal.pth`（判定基线），2000 步后基线被覆盖（mtime 04:19、steps 756000→758000）。可精确恢复的唯一原因：`inject_s6b/B756k_v6_init.pth` 是手术前从基线复制的，首层前 1012 通道逐位相同（手术自检 phi diff=0），裁掉新增 2 通道即还原。处置：每个实验臂先复制一份独立起手 ckpt（A0.pth/A1.pth/...），`state_file` 指向副本；基线只作只读父代 |
| 72 | 建模 | **「权重长起来」+「行为不变」= 信息冗余，不是学习不足**（2026-10-22 S6b 四臂实证）：2 行读牌危险度注入后，A1 新通道 `abs(w)` 均值 **0.0158**（A2 安慰剂 0.0062，2.5×），但 `deal_top1` 0.0121 vs 0.0118、holdout top1 0.8143 vs 0.8156，全部 <1σ。合并 S6a（6 行紧迫度 null）与坑 64（显式 order 权重涨但反事实 b 不动）：**当注入特征与 obs 已有信息同源时，权重增长只反映「无害正则」，不构成行为改变的证据**。判据必须是 **A1 vs A2（安慰剂）的配对差**，而不是「权重是否非零」；同信息注入的天花板由「BC 目标是否需要它」决定，不由「表示里有没有它」决定（P0-A 的 0.80 vs 0.97 与 P1 的 null 同时成立） |
| 73 | 训练 | **`dataset.strength_weight.table` 必须与当前 `file_index` 同源**：换索引/白名单/数据版本后不重建权重表，命中率会下降，缺失文件按 1 次处理（只 warning），加权会静默退化为“部分生效”；开跑后核对日志里的 `命中=...` 与重复直方图 |

### 5.3 日志归置规范（强制执行）

> 所有 `.log` 禁止散落 `mortal/logs/` 顶层；按文件名前缀/语义入子目录。启动前 `mkdir -p`（§5.2-34）。

| 目录 | 内容 | 示例 |
|---|---|---|
| `train/` | 主模型训练/轮次链 | `main_r{2..5}_train`、`round{4,5}_chain`、`round4_resume` |
| `grp/` | GRP 训练/探针 | `grp_train`、`grp_h128_train`、`grp_capacity` |
| `read_model/` | 读牌模型训练/守护 | `read_model_v{1,2,2s,2x}_{train,tmux}`、`a2c_accept_20261021/` |
| `iql/` | IQL 全链路（含 `iql_beta/` 探针） | `iql_{a,b,b_v2,b_v3,probe}_chain`、`status_iql_*`、`diag_iql_reward` |
| `eval/` | ladder/holdout/bootstrap | `ladder_*`、`holdout_r{3,4}`、`bootstrap_r{3,4}` |
| `smoke/` | 冒烟 | `smoke_*`、`smoke_v{2,3}_*` |
| `check/` | 代码/配置/环境检查 | `cfg_*_check`、`gamma_check_*`、`envcheck_selftest` |
| `test/` | 功能/行为测试 | `test_credit_modes`、`bp_test*` |
| `env/` | 环境/全链路 | `env_map`、`env_probe`、`pre4_envcheck` |
| `git/` | git 留痕 | `git_*`（init/fix/push） |
| `ablation/` | 奖励/超参消融 | `hp_ablation_chain`、`hp_resume_chain` |
| `data/` | 榜单/白名单/扫描 | `whitelist_*`、`toplist`、`refac_scan` |
| `data_audit/` | 年代审计/混入索引 | `year_style_audit`、`mix_index` |
| `guard_pipeline/` | 守卫剂量与门禁 | `guard_l{5,75,10,20}_x8k_*` |
| `s6c/` | S6c 探针/正式/判定 | `SUMMARY.md`、`holdout_*.log`、`ladder/` |
| `mix_v1/` | D1 数据混入 | `SUMMARY.md`、`ladder/` |
| `d2_2026/` | D2 数据补量 | `SUMMARY.md`、`ladder/` |
| `anchor1/` | akochan 锚点（按 tag） | `p0.log`…`p5.log` |
| `monitor/` | 训练监控/估算 | `progress_*`、`r2_monitor` |
| `misc/` | 其它 | `pack_zip`、`tensorboard` |
| `s6a_isolated/code/logs/` | S6a 隔离区（不在 `mortal/logs/`） | `s6a_chain.log`、`s6a/{train,holdout,ladder_summary}.log` |

例外：`ladder_r2/`~`ladder_r5/`、`ladder_null_smoke/`、`anchor1/`、`_agtest/`、`r2_evaldone/`、`tb_archive/` 内的 `.log` 保持原位（与同目录牌谱配套，移动会断证据链）。顶层历史散落文件（`defense_guard_*`、`ev_rerank_*`、`s6b_*`、`mix_*`、`d2_*`、`anchor_ako_*`）不再搬动；新任务的日志仍按上表入子目录。

---

## 6. 工具、文件与产物索引

### 6.1 工具表（一条命令干一件事）

> **2026-10-28 源码层清理**：已关闭实验的驱动脚本与配置变体（IQL A/B/probe/β 扫、D1/D2、HP 消融、R2/R3 长跑入口、anchor p0~p5 扫描副本）共 **53 个文件**已归档删除（另有 3 个配置因仍被仓库内工具当输入引用而还原）；下面标“（已归档 10-28）”的行仅存于归档包，还原见 `backups/cleanup_20261028/MANIFEST.md`。保留口径：当前主线 s6c（含 `run_s6c.sh`、`run_s6c_chain.sh`、`run_guard_s6c_wave2.sh`）、上一代主链 round4/5、活跃 anchor 工具、基建 `ladder_arm.sh`，以及被本文/工具引用的历史配方（round2/3、offline、grp_h128）。

| 工具 | 用途 |
|---|---|
| `mortal/env_check.py` | 开跑前自检：目录遮蔽/环境变量/选卡/config/globs |
| `mortal/holdout_metrics.py` | 未见月 top1、Q 校准、R²、和了 top1 |
| `mortal/ladder.py` + `mortal/run_ladder_r3.sh` | 单对/批量双向同牌山对抗；并发必传 log_dir（`run_ladder_r3.sh` 已归档 10-28） |
| `mortal/ladder_arm.sh` | **标准双向臂**：16 块×500 sets/向（可换种子/守卫 λ），自动打印分辨率与判定行 |
| `tools/training/agg_ladder.py` | 分块 δ/SE/t/95%CI/所需 sets；`--logs-root` 做 block bootstrap |
| `tools/mjai_score.py` | 计分口径唯一权威（与 `stat.rs`/`rankings.rs` 对齐） |
| `mortal/bench_pipe.py` | 数据管道吞吐基准（定 nw/RAYON） |
| `mortal/run_anchor.sh` + `mortal/agg_anchor.py` | akochan 绝对锚点：多进程起跑 + 累加 rankings 出 avg_rank/avg_pt（人类场对照） |
| `mortal/defense_guard.py` + `tools/read_model/{eval_defense_guard,agg_guard_ladder}.py` | 守卫 G3 推理钩子 + 离线/剂量聚合 |
| `mortal/ev_rerank.py` + `tools/test_ev_rerank.py` + `tools/read_model/eval_ev_rerank.py` | EV 重排实现 + 单测 + 20k 离线诊断 |
| `mortal/s6c_features.py` + `tools/test_s6c_features.py` + `mortal/run_s6c{,_chain}.sh` | S6c 特征/口径回归/端到端链（默认关闭） |
| `tools/data_audit/year_style_audit.py` / `build_mix_index.py` | 年代牌风审计 / D1 混入索引 |
| `mortal/run_mix{,_chain}.sh` / `run_d2{,_chain}.sh` | D1 / D2 端到端链（训练→holdout→ladder→SUMMARY）**（脚本已归档 10-28，配置留存）** |
| `tools/akochan_pipe_shim.py` + `tools/akochan_build_notes.md` | akochan pipe 垫片与构建/缺陷记录 |
| `tools/player_stats/build_whitelist.sh` | 重建强度榜与 topN（813k 文件约 77 s） |
| `tools/player_stats/whitelist_coverage.py` | 选 topN 前算席位/样本/步数覆盖 |
| `tools/training/grp_quality_probe.py` | GRP 分位置 hit@1/3/6/9/末、mae、nll |
| `mortal/run_grp_capacity.sh` | GRP 容量实验（注意 `grp_best.pth` 会被不同尺寸互盖）**（已归档 10-28）** |
| `tools/training/dump_curves.py` / `check_selection_stats.py` / `paired_selection_diff.py` | 曲线/解析交叉校验/同牌山配对差 |
| `tools/training/make_round{3,4,5}_cfg.sh` | 从上一轮配置派生下一轮 |
| `mortal/run_round4.sh` / `run_round5.sh` | 端到端链：env_check→暖启→训练→双向判定→汇总 |
| `tools/training/smoke_round4_augmentation.py` | 启前验白名单+增强样本守恒 |
| `mortal/launch_round3.sh` / `run_r2_ladder.sh` / `run_ladder_r3.sh` | tmux 长跑入口（内建 env 重置）**（均已归档 10-28）** |
| `tools/tenhou_data/repair_names.py` | 从 mjlog 恢复 2026 names（unquote）写补丁树 |
| `tools/tenhou_data/verify_names.py` | names 覆盖/席位/白名单命中率验收 |
| `tools/training/test_file_index_manifest.{py,sh}` | manifest 单测 + E1~E6 回归 |
| `tools/training/test_reward_modes.py` | 奖励口径/缩放/守卫验证 |
| `tools/training/inspect_gamma_semantics.py` | 折扣/dones/局边界实测 |
| `tools/training/test_credit_modes.py` | Stage A 口径五层验证 |
| `mortal/config.iql_a.toml` + `run_iql_a.sh` | Stage A 端到端链（种子 160000/170000）**（均已归档 10-28）** |
| `tools/training/test_iql_data.py` | IQL 数据八层验证（next_obs/r_t/terminal/幅度） |
| `tools/training/diag_iql_reward_double_count.py` | r_t 重复计入诊断（报放大倍数） |
| `mortal/config.iql_b{,_smoke}.toml` + `run_iql_b{,_smoke}.sh` | Stage B 链 + 行为门禁**（均已归档 10-28）** |
| `mortal/config.iql_b_v2.toml` + `run_iql_b_v2.sh` | Stage B v2（漂移轮，已封存）；**配置留存（3 个 prepare/diag 工具当输入），脚本已归档 10-28** |
| `mortal/config.iql_b_v3.toml` + `run_iql_b_v3.sh` | Stage B v3（τ=0.5，干净轮）；**配置留存（TECHNICAL_REPORT 与 prepare_iql_beta_probe 引用），脚本已归档 10-28** |
| `tools/training/prepare_iql_probe.py` + `run_iql_probe.sh` | 5 臂×30k 步单旋钮探针（脚本已归档 10-28） |
| `tools/training/prepare_iql_b_v3.py` | `--fix tau05` 一键生成 v3 配置（生成物已归档，工具保留） |
| `tools/training/prepare_hp_ablation.py` / `run_hp_ablation.sh` / `resume_and_judge_hp.sh` | 超参消融/挪卡续训**（两个 run 脚本已归档 10-28）** |
| `tools/read_model/` | **读牌模型全部工具**：v1/v2/v2s/v2x/A2b/**A2c(200k 结项 + 官方验收全通过)**/**A2d 20k 探针（RM12 v1 / RM14 aux，均 FAIL）**/A3 + 86 项回归 + ONNX 导出 + A3 规则法（`rule_baseline`/`a3_eval`/`compare_a3`/`calibrate_a3`）+ Bayes 上界（`bayes_bound`/`shanten_fast`）+ 注入探针（`inject_p0`/`precompute_danger`/`eval_s6b_arms`）+ 验收链 `run_a2c_acceptance.sh`（四任务并行）+ 诊断脚本 `diagnostics/`。逐条清单与用法见 `tools/read_model/docs/README.md` §15 与 `diagnostics/README.md` |
| `s6a_isolated/`（`tools/expand_ckpt_v5.py`、`tools/test_urgency_features.py`、`run_s6a{,_smoke}.sh`、`patches/v5_obs_features.patch`） | S6a 隔离工作区：v4→v5 手术、回归、冒烟、正式链 |

### 6.2 关键文件（Linux 路径）

| 类别 | 文件 |
|---|---|
| Python 训练/模型 | `mortal/model.py`（Brain/DQN/GRP/Aux/VHead）、`train.py`、`dataloader.py`、`reward_calculator.py`、`holdout_metrics.py`、`ladder.py`、`env_check.py`、`player.py` |
| 推理/实验钩子 | `mortal/mortal.py`（推理入口 + 守卫默认 λ=10）、`defense_guard.py`、`ev_rerank.py`、`s6c_features.py`、`one_vs_three.py`（含 `py_vs_ako`）、`agg_anchor.py` |
| Python 入口 | `mortal/server.py`/`client.py`（在线，暂未用）、`train_grp.py` |
| Rust 核心 | `libriichi/src/{lib.rs,consts.rs,state/,dataset/,arena/,algo/,mjai/,agent/,bin/}`（`agent/akochan.rs` 为锚点客户端） |
| 配置（现存 15） | `mortal/config.example.toml`、`config.round{2,3,4,5}.toml`、`config.offline.toml`、`config.grp_h128.toml`、`config.s6c{,_base,_probe,_smoke}.toml`、`config.anchor_ako.toml`、`config.iql_b_v2.toml`、`config.iql_b_v3.toml`、`config.mix_v1.toml`（留存原因：仍被仓库内工具/文档当输入引用；+ 生成物 `config.style_*.toml` 5 个） |
| 脚本（现存 7） | `mortal/run_round{4,5}.sh`、`run_s6c.sh`、`run_s6c_chain.sh`、`run_guard_s6c_wave2.sh`、`ladder_arm.sh`、`run_anchor.sh` |
| 文档 | `memory/MEMORY.md`（本文）、`memory/NEXT_STEPS_ANALYSIS.md`、`tools/training/TRAINING_MANUAL.md`、`tools/training/ENV_SETUP_P100.md`、`tools/tenhou_data/TASKLIST.md`、`tools/read_model/docs/README.md`、`tools/akochan_build_notes.md` |
| 数据索引 | `dataset/index/{player_ratings_*.csv,top_players_*.txt,repair_names_report.json}`；`mortal/indexes/*.pth` |

**代码改动台账（相对上游）**

| 文件 | 主要改动 |
|---|---|
| `mortal/train.py` | `max_train_steps` 预算自停、`snapshot_every` 编号快照、索引读 `start.get('names',[])`、`best_perf` 重置+一次性令牌、file_index manifest、奖励口径守卫、`credit_mode`、IQL 三损失/EMA/`iql scale:`/`iql_v_target_decay`/`iql_value_clip`、`extra_features` 解析 |
| `mortal/dataloader.py` | `reward_mode/reward_scale/always_include_kan_select`；`credit_mode/gamma` 后缀和；`objective='iql'` 7 元组；`r_t` 仅 `dones[i]` 非零；S6c `extra_features` 批内拼行 |
| `mortal/model.py` | 新增 `VHead(1024→256 Mish→1)`；`Brain(extra_channels)`（S6c=2）；v5 `case 5`（隔离区） |
| `mortal/defense_guard.py` | 新增：G3 守卫 `Q'=Q−λ·max(risk−τ,0)`，A2c 后验，仅改打牌 0..33 |
| `mortal/ev_rerank.py` | 新增：SP 前瞻 + risk 的 EV 重排（离线已否证，未接入默认路径） |
| `mortal/s6c_features.py` | 新增：向听/有效牌 2 行特征 + ckpt 自解析（`resolve`） |
| `mortal/holdout_metrics.py` | 奖励/信用/IQL 口径透传；`worker_init_fn` + 真随机种子；iql ckpt 只出 top1/agari/lse；`extra_features` 透传 |
| `mortal/player.py` | `enable_amp`/`agari_guard` 从 `[test_play]` 读；固定评测种子块（仅趋势） |
| `mortal/ladder.py` | 第 5 个参数 `log_dir`；执行参数 `CLI>[1v3.ladder]>默认`、双侧一致、`run_meta.json`、ckpt sha256/steps；守卫与 `extra_features` 钩子 |
| `mortal/one_vs_three.py` | `[1v3] mortal_vs_ako` 开关（`py_vs_ako`），供绝对锚点用 |
| `mortal/train_grp.py` | `grp.control.max_train_steps`、`grp_best.pth` 按 val loss、固定验证种子、`est. %` 日志 |
| `libriichi/src/agent/akochan.rs` | 新增：akochan pipe 客户端（`sent_new` 判定、非触发点不读） |
| `libriichi/src/dataset/gameplay.rs` | `apply_gamma` 注释改为实测口径（仅注释，无行为变更） |
| `libriichi/src/consts.rs` / `state/obs_repr.rs` | S6a v5（仅隔离区）：`MAX_VERSION=5`、`obs_shape(5)=(1018,34)`、6 行 `encode_urgency` |
| 配置/脚本 | `config.round{2,3,4,5}.toml`、`config.iql_{a,b,b_v2,b_v3}.toml`、`config.s6c*.toml`、各 `run_*.sh`、`prepare_*.py`、`test_*.py`；S6b 注入代码曾短暂入 main，10-22 已回退（`7bb5b7d`，5 个 py 恢复 `3a793a1`）；此后 S6c/守卫又各自改动，当前文件不等于 `3a793a1` |
| 读牌模型 | 全部资产在 `tools/read_model/`（对 `mortal/`、`libriichi/` **零改动**）；代码改动台账见该目录 `README.md` §15.2 |

### 6.3 数据与产物台账

| 路径 | 内容 | 体量 |
|---|---|---|
| `dataset/mjai/houou4/` | 2,630,640 局（200902~202610）；⚠ 2026 部分无 names，白名单模式 0 席位 | 16 GB |
| `dataset/mjai/houou4_names/` | 2026 names 补丁树（130,407 文件） | 607.6 MiB |
| `dataset/tenhou_hf/` | 18 个 HF 年度 tar | 17 GB |
| `dataset/raw/`、`dataset/index/` | 2026 原始 mjlog；清单/榜单/manifest | 1.2 GB / 197 MB |
| `mortal/checkpoints/` | R1–R5、reward_ablation A/B/C、hp_ablation×3、iql_a、iql_b_v2/v3、grp×3、baseline | ≈5 GB+ |
| `mortal/checkpoints/{s6c,mix_v1,d2_2026}` | S6c 956k+best+init；D1 856k；D2 856k+best（各含 step 快照） | 1.6 GB / 874 MB / 874 MB |
| `mortal/checkpoints/inject_s6b/` | S6b 四臂 A0~A3（A1~A3 需 checkout `27632a2`）+ `B756k_v6_init.pth` | 见目录 |
| `mortal/read_model_bayes/`、`mortal/read_model_v2s_a2b/`、`mortal/read_model_v2s_a2c/`、`mortal/logs/read_model/a2c_accept_20261021/`、`tools/read_model/diagnostics/results/` | 读牌全部产物（Bayes 上界 / A2b 对照 / **A2c 注入基线 best(153k)+last(200k)+train_log.jsonl+三份验收报告** / 验收日志与 A2b 回归 / 诊断 json） | 台账见 README §15.3；json 272 KB |
| `mortal/read_model_v2s_a2d_probe20k/`、`mortal/read_model_v2s_a2d_nt_probe20k/`、`tools/read_model/diagnostics/results/rm12_*_20261021.json` | P1b 两版 20k 探针（RM12 v1 / RM14 aux）ckpt + train_log + order_metrics | 见 README §13.5~§13.7 |
| `mortal/logs/ladder_r{2,3,4,5}/`、`mortal/{reward_ablation,hp_ablation,iql_a,iql_b_v2,iql_b_v3}/ladder*/` | 各轮判定牌谱（每轮约 20,000 局） | 每轮 ≈0.8 GB |
| `mortal/logs/{s6c,mix_v1,d2_2026}/` | 各探针 SUMMARY、holdout、ladder 牌谱与 bootstrap 日志 | ≈0.83 GB 每个 |
| `mortal/logs/guard_pipeline/` | 守卫剂量曲线各 λ 的双向牌谱/汇总 | 见目录 |
| `mortal/checkpoints/{tierA,tierA2}/`、`mortal/logs/selfplay/` | 档 A 两轮 ckpt（tierA 1,140k；tierA2 1,356k+20k 快照）；自博弈数据 8 目录共 35,200 半庄（生成种子 700000~715100） | 1.5 GB / 2.7 GB / 1.5 GB |
| `mortal/checkpoints/strength_probe/`、`mortal/indexes/strength_weights_v1.pth` | 数据质量加权探针 ckpt（1,056k + 20k 快照 + best）与权重表（492,810 文件；min avg_pt 分档 ×1/×2/×3，mean 1.350） | 874 MB / 50 MB |
| `mortal/logs/anchor1/` | akochan 锚点 408 局日志（已完成） | 28 KB |
| `mortal/logs/anchor_ako_*.log` | 早期锚点冒烟/排障日志（t9~t11、full） | 顶层历史，见 §5.3 例外 |
| `s6a_isolated/code/{checkpoints/s6a,s6a,logs,runs}` | S6a 隔离产物（867k+best+step_800/850k、判定牌谱、TB、链日志） | ≈1 GB |
| `mortal/logs/tb_archive/` | TB 坏段取证 | 72 KB |
| `mortal/indexes/` | 各轮文件索引（`main_r2_whitelist_v1.pth` 等） | 248 MB |
| `backups/ckpt_round1/`、`backups/code_docs_20261009.tar.gz`、`backups/libriichi_20261025/` | R1 ckpt 副本 8 个；代码/文档备份；akochan 改动前源码+.so | 592 MB / 32 MB / 24 MB |
| `tools/akochan_src/` | akochan 源码与 `system.exe`（`NPROCS=2` 构建） | 124 MB |
| `Mortal_valuable_with_ckpts_20261013.zip` | 全量打包（697 entries，含 README） | 6.83 GiB |

### 6.4 外部链接

- 作者权重说明：https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- 官方文档站：https://mortal.ekyu.moe
- 在线服务：https://mjai.ekyu.moe
- 天凤日志一覧（含 ToS）：https://tenhou.net/sc/raw/
- HF 数据集：`NikkeTryHard/tenhou-to-mjai`（hf-mirror 可下）

---

## 7. 附录

### 7.1 项目准备期（2026-10-01~10-09，压缩）

- 10-01：完成 Rust+Python 架构分析；确认作者不公开权重、官方 Training 文档全为空占位；实测天凤 `list.cgi`/scc/`0/log`，年度 zip 404，滚动日归档可用。
- 10-02：发现 HF `tenhou-to-mjai` 2009–2025 全量；自建 2026 管线；实测 HF 2009/2026 包合并成功；D 盘 exFAT 簇 256 KiB 导致小文件膨胀 64×（坑 13）→ 用户拍板只保留 18 个年度 tar，不转换。
- 10-06：服务器环境落地（conda mortal/py3.12.11/torch 2.6.0+cu118/rust 1.99，`libriichi.so`+CLI 编译通过）。
- 10-07：HF 18 包 merge-hf + prune --apply 完成；GRP 30k 步（21 min）；R1 启动（结果见 §3.2）。
- 10-09：首次复盘——评测饱和、best.pth 陷阱、吞吐瓶颈、代码补丁清单；建本地 git 基线、玩家强度榜、holdout_metrics/ladder/bench_pipe；R2 启动。
- 采集预算存档（2026-10-01 精算）：2026 YTD 实测 129,475 局、raw+mjai≈1.53 GiB；历史 2009–2025 约 340–680 万局，仅粗估 40–80 GiB（未落地；最终走 HF 年度包）。

### 7.2 早期 Windows 本机（仅历史参考，当前训练不在本机）

| 项 | 值 |
|---|---|
| 路径 | `c:\Maj\Mortal` |
| Python/git | 3.13.5（Anaconda）/ 2.45.1.windows.1；**cargo 未安装** |
| GPU | RTX 3050 Ti Laptop 4GB（仅数据管线，不宜训练） |
| 用途 | 2026 天凤牌谱采集、mjlog→mjai 转换、文档；训练已全部迁到 Linux 服务器 |

### 7.3 术语

| 术语 | 含义 |
|---|---|
| GRP | 每局顺位预测器；输入 7 维（大局数/本场/供托/四家分数），输出 24 排列概率；奖励=相邻局期望点差 |
| CQL | Conservative Q-Learning 正则；离线训练时压 Q 值 |
| MC | Monte Carlo，当前局最终奖励回传 |
| IQL | Implicit Q-Learning；L_Q + expectile L_V + AWR |
| AWR | Advantage-Weighted Regression，优势加权策略提取 |
| ladder | 双向同牌山 1v3 配对对抗判定；δ=(ptA−ptB)/2 |
| arm | ladder 标准双向臂：16 块×500 sets/向（16000 sets，可分辨 δ≈0.66） |
| holdout | 未见月 202609+，只用于诊断，不作强度结论 |
| bseed | block bootstrap 随机种子 |
| S6a/b/c/d | 特征注入子阶段：a 紧迫度、b 读牌后验、c 自家速度/牌型、d GRP 扩展+攻防重规划 |
| G3 | 推理期连续风险守卫：`Q'=Q−λ·max(risk−τ,0)`，risk 来自 A2c 后验 |
| D1 / D2 | 数据扩窗探针：D1=2019-2021 混入；D2=2026H1 补量 |
| anchor | akochan 绝对锚点：1 Mortal vs 3 akochan，uma 与人类场同尺度（§3.16-B） |

---

> **维护约定**：
> 1. **一处只讲一件事**：新实验先在 §3.1 加一行，再在 §3.x 写一节；**该实验的数字只写在那里**，§0/§4 只引用小节号，不复述数字。
> 2. 新方向状态写 §4.1（聚合行）；新待办写 §4.2；已关闭的实验从 §4.2 删除，只在 §4.1 留一行“已关闭 + 依据”。
> 3. 新坑写 §5.2 并按主题编号；新工具写 §6.1；新数据/产物写 §6.3；新种子块写 §2.4。
> 4. 读牌模型的一切细节写 `tools/read_model/docs/README.md`，本文只保留 §3.12 摘要；训练操作细节写 `tools/training/TRAINING_MANUAL.md`。
