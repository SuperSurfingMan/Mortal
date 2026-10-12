# Mortal 训练执行手册（从零到可对战模型）

> **手册版本**：2026-10-03（**远程实跑部分已于 2026-10-09 追加 §14；§10 的时间估算已被实测推翻，请以 §14.5 为准**）
> **适用对象**：已按 `tools/tenhou_data/TASKLIST.md` 备好牌谱，准备在 **远程 Linux 服务器（4×Tesla P100-16GB + 1×Quadro RTX 8000-48GB）** 上完整跑通 Mortal 训练闭环的操作者
> **数据基线**：本机 `C:\Maj\Mortal\dataset\mjai\houou4` 为 **130,407 局**（2026 YTD）；**服务器已是全量** `200902~202610` 共 **2,630,640 局 / 16 GB**，本轮训练用窗 2022-01~2026-08 = **813,145 局**（详见 [MEMORY.md §16.3](../../memory/MEMORY.md)）
> **本手册与旧文档的关系**：旧版 `TRAINING_PLAN.md` 的配方骨架仍然有效，但其数据基线为 42,587 局、路径为 2009+2026 混合，**已过时**。本手册为最新、可直接照做的版本；旧文档保留仅作历史参考。
> **配套文档**：[TASKLIST.md](../tenhou_data/TASKLIST.md)（数据管线与最终校验）、[build.md](../../docs/src/user/build.md)（官方构建说明）

---

## 目录

0. [开始之前：先读懂 5 个事实](#0-开始之前先读懂-5-个事实)
1. [环境要求](#1-环境要求)
2. [服务器环境搭建（M0-a）](#2-服务器环境搭建m0-a)
3. [数据格式与数据上传（M0-b）](#3-数据格式与数据上传m0-b)
4. [训练原理与数据流（理解这张图再动手）](#4-训练原理与数据流)
5. [训练配方：config.toml 全文](#5-训练配方configtoml-全文)
6. [bootstrap.py：随机 baseline](#6-bootstrappy随机-baseline)
7. [执行顺序：M0 → M5 逐步操作](#7-执行顺序m0--m5-逐步操作)
8. [监控指标与判停标准](#8-监控指标与判停标准)
9. [关键坑位与故障处置](#9-关键坑位与故障处置)
10. [时间与资源估算](#10-时间与资源估算)
11. [里程碑验收清单](#11-里程碑验收清单)
12. [附录 A：命令速查表](#附录-a命令速查表)
13. [附录 B：数据更新后如何重训](#附录-b数据更新后如何重训)
14. [远程实跑记录与修正（2026-10-07~10-09）](#14-远程实跑记录与修正2026-10-0710-09)
15. [第 2 轮就绪状态（10-09 已落地）](#15-第-2-轮就绪状态10-09-已落地)
16. [第 2 轮判定结果与第 3 轮（10-10）](#16-第-2-轮判定结果与第-3-轮10-10)

---

## 0. 开始之前：先读懂 5 个事实

在动手前请先接受以下 5 条，它们决定了后面每一个决策：

1. **训练机是远程 Linux，数据在本机 Windows**。本机（`C:\Maj\Mortal`）只负责采集/整理/打包；服务器负责编译、训练、评测。两者通过 `scp` 传输。
2. **`libriichi` 是硬前置**。它是 Rust 编译出的 Python 扩展（`libriichi.so`），提供 `GameplayLoader`（读牌谱→张量）、`Grp`（GRP 训练数据）、`OneVsThree`（1v3 评测）、`obs_shape` 等。**没有它，一行训练代码都跑不起来**。而编译它需要 Rust 工具链（`cargo`）。
3. **GRP 必须最先训**。离线训练每一步都要用 GRP 把牌谱转成"每局奖励"，GRP 权重（`grp.pth`）不存在时，数据加载器会在 `build_iter()` 里直接崩。顺序不可颠倒。
4. **必须先造一个随机 baseline**。`test_play` 每次评测都要一个"陪练"（`baseline.pth`）。首次训练时它还不存在，所以要先跑 `bootstrap.py` 生成一个随机权重的占位 baseline（M2 之前）。
5. **离线模式只跑一个 epoch 就退出**。这是源码的有意设计（`if not online: break`），不是 bug。想接着练就再启动一次，`state_file` 会自动续训。

---

## 1. 环境要求

### 1.1 硬件

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **NVIDIA Tesla P100-PCIE-16GB** | 本手册按此校准。其他卡也能跑，但 `enable_amp` / 显存参数需按 §9.4 调整 |
| 显存 | ≥ 16 GB | `batch_size = 512` 在此显存下无压力，不需要梯度累积 |
| CPU | ≥ 8 核（建议） | **CPU 才是真正的瓶颈**：特征编码（Rust/rayon）+ GRP float64 推理都在 CPU |
| 内存 | ≥ 32 GB | 数据加载器按 `file_batch_size` 成批解压，内存不足会 OOM |
| 磁盘 | ≥ 50 GB 空闲 | 数据 0.6 GiB + checkpoint（约 0.15 GB × 2）+ TensorBoard event（数 GB）+ 评测日志 |
| 网络 | 能访问 PyPI / crates.io / rustup | 只在本机与服务器之间传数据；训练本身**不需要联网** |

### 1.2 软件栈

| 组件 | 版本 | 备注 |
|---|---|---|
| OS | Linux x86_64 | Windows 训练见 §9.5 差异说明 |
| conda/miniconda | 任意 recent | 用于隔离环境 |
| Python | **3.12** | 与 `environment.yml` 一致 |
| **PyTorch** | **2.6.0+cu118** | ⚠ **必须是 cu118 通道**：其 wheel 含 `sm_60`（Pascal），CUDA 12 系 wheel 不保证支持 sm_60。**不要用 cu126** |
| CUDA 驱动 | ≥ 450.80.02 | `nvidia-smi` 能看到 P100 即可，**不需要**单独装 CUDA Toolkit |
| Rust | ≥ 1.85（stable） | `libriichi` 使用 Rust edition 2024，旧编译器会直接报错 |
| 纯 Python 依赖 | `tqdm` `toml` `tensorboard` | 就这三个，见 `environment.yml` |

### 1.3 目录规划（本手册全程使用这套约定）

```text
$MORTAL_ROOT/                 # 代码仓库，例如 ~/Mortal
├── mortal/                   # ← 运行目录：config.toml / checkpoints/ / tb_log/ 都放这
│   ├── libriichi.so          #   cargo 编译产物，复制到这里
│   ├── config.toml           #   训练配置（§5 生成）
│   ├── bootstrap.py          #   随机 baseline 脚本（§6 生成）
│   ├── checkpoints/          #   mortal.pth / best.pth / baseline.pth / grp.pth
│   ├── tb_log/               #   离线训练 TensorBoard
│   ├── tb_log_grp/           #   GRP 训练 TensorBoard
│   └── 1v3/ test_play/ ...    #   评测与对局日志
└── target/release/           # cargo 编译产物

$DATA_ROOT/                   # 训练数据，例如 ~/mortal_data
└── dataset/
    └── mjai/houou4/YYYYMM/*.json.gz   # 130,407 个 mjai 牌谱
```

**下面所有命令，除特别说明外，工作目录都是 `$MORTAL_ROOT/mortal`**（因为 `config.py` 默认读当前目录的 `config.toml`，且所有相对路径基于此）。

---

## 2. 服务器环境搭建（M0-a）

登录服务器后依次执行，每步都有期望输出。

### 步骤 2.1：拿到代码

```bash
git clone https://github.com/Equim-chan/Mortal.git ~/Mortal
cd ~/Mortal
export MORTAL_ROOT=~/Mortal          # 建议写进 ~/.bashrc
```

### 步骤 2.2：创建并激活 conda 环境

```bash
cd $MORTAL_ROOT
conda env create -f environment.yml   # 生成名为 mortal 的环境
conda activate mortal
```

### 步骤 2.3：安装 PyTorch（cu118）

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

验证（**这一步必须通过，否则后面全是徒劳**）：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.6.0+cu118 True Tesla P100-PCIE-16GB
```

若不通过，排查顺序：驱动版本（`nvidia-smi`）→ 是否是 cu118 通道 → `torch.cuda.is_available()` 是否 False（多为驱动过旧或装成了 CPU 版）。

### 步骤 2.4：安装 Rust 与 MSVC/GCC 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
rustc --version      # 需 ≥ 1.85（edition 2024 要求）
```

### 步骤 2.5：编译 libriichi 并安装

```bash
cd $MORTAL_ROOT
# ① 编译 Python 扩展（release + LTO，首次较慢，5~15 分钟）
cargo build -p libriichi --lib --release
cp target/release/libriichi.so $MORTAL_ROOT/mortal/libriichi.so

# ② 编译命令行校验器（用于数据闭环校验，不含 pymod）
cargo build -p libriichi --bins --no-default-features --release
```

验证：

```bash
cd $MORTAL_ROOT/mortal
python -c "import libriichi; help(libriichi)"
# 期望: 打印出 libriichi 模块的文档（含 dataset / arena / consts 等子模块）
```

> **说明**：`flate2` 是 `libriichi` 的直接依赖（未被 feature 门控），所以 `--no-default-features` 编译出的 `validate_logs` 依然能读 `.json.gz`，无需额外加 feature。
> 若 `import libriichi` 报架构/未定义符号，几乎一定是：① 用了非 cu118 的 torch；② `libriichi.so` 没复制到运行目录。

### 步骤 2.6：确认三个纯 Python 依赖

```bash
python -c "import tqdm, toml, tensorboard; print('deps ok')"
# 若缺: pip install tqdm toml tensorboard
```

至此 M0-a 完成。**验收标准**：`import libriichi` 成功 + `torch.cuda.is_available()` 为 True。

---

## 3. 数据格式与数据上传（M0-b）

### 3.1 训练数据的确切格式要求

Mortal 的加载器（`libriichi/src/dataset/gameplay.rs::load_gz_log_files`）只接受一种格式：

- **文件类型**：`.json.gz`（gzip 压缩的 JSON Lines）
- **目录匹配**：通过 `glob` 匹配 `**/*.json.gz`（递归）
- **内容结构**：解压后**每一行是一个独立的 JSON 事件对象**（mjai 协议），**不是**一个 JSON 数组、不是缩进美化过的 JSON
- **必需事件**（缺任一，结构校验会失败）：
  - 首行 `{"type":"start_game", ...}`
  - `{"type":"start_kyoku", ..., "scores":[...]}`（必须带 `scores`）
  - 末行 `{"type":"end_game", ...}`
  - `hora` / `ryukyoku` 事件必须带 `deltas`
  - `hora` 事件必须带 `ura_markers`（⚠ 这是本项目的关键修正点：jansou 输出的是 `uradora_markers`，必须改名）
- **不支持**：zstd、tar、未压缩 `.json`、整个目录打成一个包

也就是说：**不要试图直接把 D 盘的 HF 年度 `.tar.zst` 包丢给训练器**，它读不了（详见 `TASKLIST.md` §5.7）。

### 3.2 本机（Windows）数据现状

| 项 | 值 |
|---|---|
| 路径 | `C:\Maj\Mortal\dataset\mjai\houou4\YYYYMM\` |
| 文件数 | **130,407**（已逐文件校验通过） |
| 体积 | **595.5 MiB** |
| 月份分布 | 202601 16,179｜202602 14,352｜202603 14,294｜202604 13,443｜202605 14,698｜202606 13,463｜202607 13,174｜202608 13,827｜202609 15,892｜202610 1,085 |

**抽样自检（本机 PowerShell）**：

```powershell
# 计数
(Get-ChildItem -Path 'C:\Maj\Mortal\dataset\mjai\houou4' -Recurse -Filter '*.json.gz').Count   # 应为 130407
```

### 3.3 打包上传（Windows → Linux）

**本机 PowerShell：**

```powershell
cd C:\Maj\Mortal\dataset
# 只打包训练必需的 mjai（raw 牌谱训练用不到，不必上传）
tar -czf mjai_20261003.tar.gz mjai
# 生成约 590 MiB 的包

# 传到服务器（把 user@SERVER 换成实际地址）
scp mjai_20261003.tar.gz user@SERVER:~/mortal_data/
```

> **注意**：130,407 个小文件逐个 `scp` 会极慢，**必须先 tar 再传**。
> 若 `tar` 因盘符 `C:` 报错，改用：`tar -czf mjai_20261003.tar.gz -C C:\Maj\Mortal\dataset mjai`。

**服务器：**

```bash
mkdir -p ~/mortal_data/dataset && cd ~/mortal_data/dataset
tar -xzf ~/mortal_data/mjai_20261003.tar.gz

# ① 计数必须为 130407
find mjai -name '*.json.gz' | wc -l

# ② 抽查首行必须是 start_game
zcat mjai/houou4/202601/$(ls mjai/houou4/202601 | head -1) | head -1
# 期望: {"type":"start_game","names":[...],...}

# ③ 目录层级抽样
ls mjai/houou4/          # 应看到 202601 ... 202610
```

### 3.4 服务器端数据闭环校验（强烈建议，M0 收尾）

这是第一次真正用上 `libriichi`，也是发现数据问题的最后一道闸门。**服务器端一律用 Rust 校验器**：

```bash
cd $MORTAL_ROOT
./target/release/validate_logs ~/mortal_data/dataset/mjai
```

> 备选：若你的服务器上也同步了本项目的 `tools/tenhou_data/` 与 `jansou`，还可以跑更细的语义校验（结构 + 回环解析 + 回放计分），它比 `validate_logs` 多检查"和牌复核"。此命令**依赖本机采集管线的产物，官方仓库克隆默认没有**，不必强求：
>
> ```bash
> cd $MORTAL_ROOT
> python tools/tenhou_data/tenhou_pipeline.py check --sample 1000
> ```
>
> 该命令在数据管线机器（本机）已跑过 `--sample 1000`，结果：结构 1000/1000、回环解析 1000/1000、和牌复核 8932/8932（详见 TASKLIST §8.2）。

**期望输出**：扫描进度条走完，末尾没有 `ERROR` 行，进程退出码 0。它会逐文件检查：
牌型合法性、`chi` 必须来自上家、`hora` 必须带 `ura_markers` 与 `deltas`、回放一致性等。

**验收标准**：`validate_logs` 0 错误。若报 `missing field ura_markers` → 说明转换时少了 §3.1 的改名修正，需回到本机重转。

---

## 4. 训练原理与数据流

先建立心智模型，后面配置每一项才知道在调什么。

```text
                    ┌──────────────────────────────┐
   牌谱 *.json.gz ──► │ libriichi GameplayLoader      │ ──► (obs, actions, masks, ...)
                    └──────────────────────────────┘           每行 = 一个决策步
                                                                      │
                                                          ┌───────────┴───────────┐
                                                          ▼                       ▼
                                              ┌────────────────────┐   ┌──────────────────┐
                                              │  GRP (float64 GRU) │   │  仅用于算奖励     │
                                              │  预测四人最终顺位   │   │  不参与梯度       │
                                              └────────────────────┘   └──────────────────┘
                                                          │
                                                       每局奖励 kyoku_rewards
                                                          │
                                                          ▼
   ┌─────────┐   φ    ┌───────────────┐   q    ┌────────────────────────────┐
   │ Brain   │ ─────► │ DQN (Dueling) │ ─────► │ 离线 CQL 损失 + next_rank  │
   │ ResNet  │        │  + CQL        │        │ 辅助损失                   │
   └─────────┘        └───────────────┘        └────────────────────────────┘
        └──────────────────► AuxNet (4 类顺位预测, 辅助任务)
```

三个网络的职责：

| 网络 | 文件 | 作用 | 训练阶段 |
|---|---|---|---|
| **GRP** | `checkpoints/grp.pth` | 预测"给定局面，各玩家最终顺位"的概率。离线训练的奖励由它计算 | **M1 单独预训练**，之后冻结 |
| **Brain** | `checkpoints/mortal.pth`（`mortal` 键） | ResNet + ChannelAttention，把观测编码成 φ | M2 与 DQN 一起训 |
| **DQN** | `checkpoints/mortal.pth`（`current_dqn` 键） | Dueling + CQL，从 φ 输出动作 Q 值 | M2 与 Brain 一起训 |
| **AuxNet** | `checkpoints/mortal.pth`（`aux_net` 键） | 顺位预测辅助任务（`next_rank_weight`） | M2 与 DQN 一起训 |

**关键顺序依赖**：

1. `bootstrap.py` → 生成 `baseline.pth`（随机权重）
2. `train_grp.py` → 生成 `grp.pth`
3. `train.py` → 读 `grp.pth` 算奖励；读 `baseline.pth` 做 `test_play` 评测 → 生成 `mortal.pth` / `best.pth`
4. `one_vs_three.py` → 用 `best.pth` 打 `baseline.pth`，量化强度

---

## 5. 训练配方：config.toml 全文

把下面内容保存为 **`$MORTAL_ROOT/mortal/config.toml`**，并把所有 `YOUR_USER` 替换为服务器用户名（或用绝对路径的 `~` 展开结果）。

```toml
[control]
version = 4
online = false

state_file = 'checkpoints/mortal.pth'
best_state_file = 'checkpoints/best.pth'
tensorboard_dir = 'tb_log'

device = 'cuda:0'
enable_cudnn_benchmark = true
enable_amp = true            # P100 无 Tensor Core；若 q 值/损失出现 NaN 或异常抖动 → 改 false
enable_compile = false       # Pascal 上无明显收益，保持 false

batch_size = 512             # 16GB 显存标准值，不需要梯度累积
opt_step_every = 1

save_every = 400             # 每 400 步存一次 checkpoint
test_every = 20000           # 每 20000 步评测一次
submit_every = 400           # 仅在线模式有意义
# ⚠ 整除约束（源码 assert，违反会直接报错退出）:
#   save_every % opt_step_every == 0
#   test_every % save_every   == 0

[train_play.default]         # 仅在线模式使用
games = 800
log_dir = 'train_play'
boltzmann_epsilon = 0.005
boltzmann_temp = 0.05
top_p = 1.0
repeats = 1

[test_play]
games = 3000                 # 每次评测的基准局数（内部按 //4 拆种子）
log_dir = 'test_play'

[dataset]
globs = ['/home/YOUR_USER/mortal_data/dataset/mjai/**/*.json.gz']
file_index = 'file_index.pth'    # ⚠ 数据更新后必须删除此缓存重建（见 §9.1）
file_batch_size = 15
reserve_ratio = 0.0
num_workers = 2              # Linux 下 2~4；CPU ≥8 核可试 4（worker 会自动切分文件表）
player_names_files = []
num_epochs = 1
enable_augmentation = false  # 第一轮 false；第二轮迭代起可改 true（样本数 ×2）
augmented_first = false

[env]
gamma = 1
pts = [6.0, 4.0, 2.0, 0.0]   # 天凤顺位点（1位+6 / 2位+4 / 3位+2 / 4位0）；仅在线模式参与梯度

[resnet]
conv_channels = 192
num_blocks = 40

[cql]
min_q_weight = 5             # 离线时 CQL 损失权重；q 值爆炸就降到 1~2.5

[aux]
next_rank_weight = 0.2

[freeze_bn]
mortal = false               # ⚠ 在线模式必须改 true；离线保持 false

[optim]
eps = 1e-8
betas = [0.9, 0.999]
weight_decay = 0.1
max_grad_norm = 0            # 0 = 不做梯度裁剪

[optim.scheduler]
peak = 1e-4
final = 1e-4
warm_up_steps = 0
max_steps = 0                # 0 = 恒定学习率（无衰减）

[baseline.train]
device = 'cuda:0'
enable_compile = false
state_file = 'checkpoints/baseline.pth'

[baseline.test]
device = 'cuda:0'
enable_compile = false
state_file = 'checkpoints/baseline.pth'

[online]
history_window = 50
enable_compile = false

[online.remote]
host = '127.0.0.1'
port = 5000

[online.server]
buffer_dir = 'buffer'
drain_dir = 'drain'
sample_reuse_rate = 0
sample_reuse_threshold = 0
capacity = 1600
force_sequential = false

[1v3]
seed_key = -1                # -1 = 每次随机；想复现固定某个整数
games_per_iter = 2000
iters = 500
log_dir = '1v3'

[1v3.challenger]             # 挑战者 = 最新 best.pth
device = 'cuda:0'
name = 'mortal'
state_file = 'checkpoints/best.pth'
stochastic_latent = false
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true

[1v3.champion]               # 卫冕者 = 上一代 baseline.pth
device = 'cuda:0'
name = 'baseline'
state_file = 'checkpoints/baseline.pth'
stochastic_latent = false
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true

[1v3.akochan]                # 用 akochan 当对手时才开
enabled = false
dir = '/path/to/akochan'
tactics = '/path/to/tactics.json'

[grp]
state_file = 'checkpoints/grp.pth'

[grp.network]
hidden_size = 64
num_layers = 2

[grp.control]
device = 'cuda:0'
enable_cudnn_benchmark = true
tensorboard_dir = 'tb_log_grp'
batch_size = 512
save_every = 2000
val_steps = 400

[grp.dataset]
# 时间序切分：用较早的月份训练，用最近的月份验证（避免信息泄漏）
train_globs = [
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202601/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202602/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202603/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202604/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202605/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202606/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202607/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202608/**/*.json.gz',
]
val_globs = [
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202609/**/*.json.gz',
    '/home/YOUR_USER/mortal_data/dataset/mjai/houou4/202610/**/*.json.gz',
]
file_index = 'grp_file_index.pth'    # ⚠ 数据更新后必须删除重建
file_batch_size = 50

[grp.optim]
lr = 1e-5
```

### 5.1 为什么是这些关键值

| 配置 | 取值理由 |
|---|---|
| `version = 4` | 官方当前主流版本；决定观测张量形状（`obs_shape(4)`）与模型结构。**不要改**，否则与已发布的权重不兼容 |
| `conv_channels = 192, num_blocks = 40` | 官方参考规模，约 10.8M 参数，P100 16GB 可承载 |
| `enable_amp = true` | P100 无 Tensor Core，加速有限，但能省显存；副作用是可能引入数值不稳，故留了"改 false"的后路 |
| `enable_cudnn_benchmark = true` | 输入形状固定，benchmark 能挑到更快的卷积算法 |
| `optim.scheduler` 恒 1e-4 | 官方默认；数据量大时不建议一开始就上余弦衰减 |
| `grp.dataset` 时间序切分 | GRP 是奖励预测器，验证集必须"未来"于训练集，否则 val 指标虚高 |
| `file_batch_size = 15/50` | 每批解压的文件数；太小则 CPU 空转，太大则内存峰值高 |
| `num_workers = 2` | Windows 下必须 1（见 §9.5）；Linux 下可 2~4 |

### 5.2 需要按数据规模微调的两个旋钮（可选）

- **`save_every` / `test_every`**：默认 `400/20000`。本数据量下每 epoch 约 168,000 步，意味着每 epoch 会存 ~420 次 checkpoint、评测 ~8 次（每次数分钟到十余分钟）。如果觉得评测太占时间，可改成 `save_every = 2000`、`test_every = 40000`（仍满足整除约束：`40000 % 2000 == 0`、`2000 % 1 == 0`）。
- **`enable_augmentation`**：第二轮迭代起设为 `true`，等效样本数翻倍（步数 ×2），代价是每 epoch 时间翻倍。

---

## 6. bootstrap.py：随机 baseline

保存为 **`$MORTAL_ROOT/mortal/bootstrap.py`**。用途：在还没有任何模型时，造一个随机权重的 `baseline.pth`，让 `test_play` 有陪练可用。

```python
import torch
from datetime import datetime
from os import makedirs
from torch import optim, nn
from torch.amp import GradScaler
from model import Brain, DQN, AuxNet
from lr_scheduler import LinearWarmUpCosineAnnealingLR
from config import config

out = 'checkpoints/baseline.pth'
makedirs('checkpoints', exist_ok=True)

version = config['control']['version']
device = torch.device('cpu')
mortal = Brain(version=version, **config['resnet'])
dqn = DQN(version=version)
aux_net = AuxNet((4,))

all_models = (mortal, dqn, aux_net)
decay_params, no_decay_params = [], []
for model in all_models:
    params_dict, to_decay = {}, set()
    for mod_name, mod in model.named_modules():
        for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
            params_dict[name] = param
            if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                to_decay.add(name)
    decay_params.extend(params_dict[n] for n in sorted(to_decay))
    no_decay_params.extend(params_dict[n] for n in sorted(params_dict.keys() - to_decay))
param_groups = [
    {'params': decay_params, 'weight_decay': config['optim']['weight_decay']},
    {'params': no_decay_params},
]
optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0,
                        betas=config['optim']['betas'], eps=config['optim']['eps'])
scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
scaler = GradScaler(device.type, enabled=False)

state = {
    'mortal': mortal.state_dict(),
    'current_dqn': dqn.state_dict(),
    'aux_net': aux_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),
    'steps': 0,
    'timestamp': datetime.now().timestamp(),
    'best_perf': {'avg_rank': 4., 'avg_pt': -135.},
    'config': config,
}
torch.save(state, out)
print(f'{out} created (random weights)')
```

运行：

```bash
cd $MORTAL_ROOT/mortal
python bootstrap.py
# 期望: checkpoints/baseline.pth created (random weights)
```

> **注意**：`bootstrap.py` 里的 `'config': config` 会被 `one_vs_three.py` / `TestPlayer` 读出来重建网络结构，所以**它记录的是运行时的 `config.toml`**——先写好 §5 的配置，再跑这一步。

---

## 7. 执行顺序：M0 → M5 逐步操作

**全程在 `$MORTAL_ROOT/mortal` 目录、`conda activate mortal` 环境下执行。长任务请用 `tmux` 或 `nohup` 防断线。**

```bash
conda activate mortal
cd $MORTAL_ROOT/mortal
tmux new -s mortal_train      # 建议：所有长任务放进 tmux，断线不丢
```

### M0 —— 环境与数据就绪

| 步骤 | 命令 | 验收 |
|---|---|---|
| M0-a 环境 | §2 全部步骤 | `import libriichi` 成功；`torch.cuda.is_available()` True |
| M0-b 数据 | §3 上传 + 解压 | `find ... \| wc -l` = 130407 |
| M0-c 闭环校验 | `./target/release/validate_logs ~/mortal_data/dataset/mjai` | 0 错误 |

> `validate_logs` 全量 13 万局会跑一段时间（Rust + rayon 多核，通常几十分钟量级）。若想先快速确认可用，可先只校验一个月：`./target/release/validate_logs ~/mortal_data/dataset/mjai/houou4/202610`。

### M1 —— GRP 预训练（必须最先）

```bash
python bootstrap.py     # 造随机 baseline（M1/M2 都需要，先跑）
python train_grp.py     # 训练 GRP
```

- **产物**：`checkpoints/grp.pth`
- **看板**：`tensorboard --logdir tb_log_grp --bind_all`
- **判停**：`acc/val` 到平台期（见 §8）。想停就 `Ctrl+C`，`grp.pth` 会自动存盘；重跑同命令会**自动续训**。
- 本数据量下每 epoch 约 2,500 步（按每局 ~10 个 kyoku 前缀序列估算），通常跑 1~2 个 epoch 即收敛。

### M2 —— 离线 CQL 第一轮

```bash
python train.py
```

- 首次运行会**构建文件索引**（glob 13 万文件，秒级），写入 `file_index.pth`
- **产物**：`checkpoints/mortal.pth`（每 400 步覆盖）+ `checkpoints/best.pth`（破纪录时复制）
- **看板**：`tensorboard --logdir tb_log --bind_all`
- **判停**：`test_play/avg_rank` < 2.5 并持续下降即健康（见 §8）
- **注意**：离线模式跑完**一个 epoch 就退出**（`if not online: break`），这是设计行为。想继续就再跑一次 `python train.py`（自动续训）。

### M3 —— 1v3 评测

```bash
python one_vs_three.py
```

- 挑战者 = `checkpoints/best.pth`，卫冕者 = `checkpoints/baseline.pth`
- **产物**：`1v3/` 下的对局日志 + 屏幕打印的 `avg_rank` / `avg_pt`
- 单次 2000 局（`games_per_iter`），麻将方差大，**下结论至少要 4000 局以上**，可多跑几轮取平均

### M4 —— 轮次迭代（性能不再涨就停）

```bash
cp checkpoints/best.pth  checkpoints/round1_best.pth      # 归档本轮最佳
cp checkpoints/best.pth  checkpoints/baseline.pth         # 卫冕者换代：新 baseline
mv checkpoints/mortal.pth checkpoints/round1_state.pth    # ⚠ 挪走才会从头训（否则续训旧状态）

# 可选：本轮开数据增强（样本 ×2，耗时 ×2）
#   编辑 config.toml: dataset.enable_augmentation = true

python train.py           # 第二轮
python one_vs_three.py    # 对比：新 best vs 上一代 baseline
```

**何时停止迭代**：相邻两轮 `best.pth` 在 1v3 上（≥4000 局）`avg_rank` / `avg_pt` 无显著改善。

### M5 —— 在线自博弈（可选，进阶）

离线迭代到瓶颈后再考虑。**改动有两处**：

```toml
# config.toml
[control]
online = true                # ← 改
[freeze_bn]
mortal = true                # ← 必须改
```

```bash
# 三个终端（或 tmux 三个窗格）
python server.py     # 终端1：自博弈数据服务（buffer/drain）
python train.py      # 终端2：在线训练主进程
python client.py     # 终端3：对局客户端；可多开增加吞吐
```

> 源码注释明确指出：在线模式训练到评测点会**卡死**（unknown reason），因此 `train.py::main()` 特意用子进程包裹来规避。**不要试图"修复"这个 bug**，也不要绕过 `main()` 直接调 `train()`。

---

## 8. 监控指标与判停标准

TensorBoard 启动：

```bash
tensorboard --logdir tb_log --bind_all --port 6006
```

远程查看：本机 `ssh -L 6006:localhost:6006 user@SERVER`，然后浏览器开 `http://localhost:6006`。

### 8.1 GRP 阶段（`tb_log_grp`）

| 指标 | 健康表现 | 异常处置 |
|---|---|---|
| `acc/val` | 快速上升到平台期（顺位预测优于随机 25%） | 不涨 → 检查 `grp.dataset` glob 是否匹配到文件、val 月份是否有数据 |
| `loss/train`、`loss/val` | 同步下降后趋平 | val 明显反弹 → 过拟合，提前停 |

**判停**：`acc/val` 连续 2 个 `save_every` 周期无提升，即可 `Ctrl+C`。

### 8.2 离线 CQL 阶段（`tb_log`）

| 指标 | 健康表现 | 异常处置 |
|---|---|---|
| `loss/dqn_loss` | 稳步下降后趋平 | 完全不降 → 查 `dataset.globs` 是否匹配、`grp.pth` 是否存在 |
| `loss/cql_loss` | 存在且非负 | 爆炸式增大 → 把 `cql.min_q_weight` 降到 1~2.5 |
| `loss/next_rank_loss` | 下降到 ~0.8 附近 | — |
| `hparam/lr` | 恒为 1e-4 | — |
| `q_predicted` vs `q_target` 直方图 | 量级接近、同向移动 | q 值飞天 → CQL 权重问题，先降 `min_q_weight` |
| **`test_play/avg_rank`** | ⚠️ **只有当 baseline 是“真的会打麻将”的模型时才有意义**（见 §14.4） | **baseline 为随机权重时会直接饱和在 1.000 / avg_pt=90**，失去一切区分度（实跑已验证） |
| `test_play/avg_pt` | 同步上升 | — |
| `test_play/behavior/*` | `agari`/`riichi`/`fuuro` 比例合理，`houjuu` 不异常高 | `houjuu` 飙升 → 弃和倾向异常，可能奖励计算有问题 |

> `avg_rank` 的基准：4 人随机 = 2.5。**< 2.5 才意味着强于随机**；训练早期在 2.5 附近徘徊是正常的。
>
> ⚠️ **2026-10-09 实测修正**：用 §6 的随机 `baseline.pth` 做陪练时，`avg_rank` 会从第一次评测（50k 步）就顶到 **1.0000**（1 位率 99.9%），因为随机对手几乎不会宣布和牌（实测 2000 局：mortal 和了 10,729 次 vs baseline 126 次）。此时该指标、`behavior/*`、`agari_point/*` 均**不能用于判断训练进展**，且 `best.pth` 只会记录第一个评测点（后续 89.955 < 90.0 永远“不破记”）。可用监控量见 §14.4。

### 8.3 1v3 评测

看 `challenger rankings`（形如 `[a b c d]`）与 `avg_rank` / `avg_pt`：

- 4 人均衡（各 ~25%）→ 挑战者与卫冕者同水平
- 挑战者 1 位率明显高于 25% → 有实质提升

**方差提醒**：麻将单局方差极大，2000 局的 `avg_rank` 误差仍在 ±0.02 量级，**不要凭一轮定胜负**。

---

## 9. 关键坑位与故障处置

### 9.1 文件索引缓存 `file_index.pth` / `grp_file_index.pth`（最高频）

这两个文件缓存了 glob 出来的文件列表。**每次新增/删除数据后，必须删除它们再重跑**，否则训练器会继续用旧列表（可能包含已删除文件→报错，或漏掉新文件→白跑）。

```bash
rm -f file_index.pth grp_file_index.pth
rm -f checkpoints/mortal.pth        # 若想让离线训练从头开始，也要删/挪 state_file
```

### 9.2 `state_file` 自动续训

`train.py` / `train_grp.py` 启动时若发现 `state_file` 存在，会自动加载并续训（含 optimizer / scheduler / steps）。想从头训必须先移走或删除它（见 M4）。

### 9.3 整除断言（源码硬性要求）

```text
assert save_every % opt_step_every == 0
assert test_every % save_every == 0
```

改 `save_every` / `test_every` 时必须保持整除，否则启动即崩。

### 9.4 AMP 数值不稳（Pascal 特有）

P100 无 Tensor Core，`enable_amp = true` 加速有限。**若 loss / q 值出现 NaN 或异常抖动，第一反应就是把 `enable_amp = false`**。另外 Pascal 上 `enable_compile = false`，不要开 `torch.compile`。

### 9.5 Windows 训练差异（若改在本机 Windows 跑）

| 项 | Linux（本手册主线） | Windows |
|---|---|---|
| libriichi 产物 | `target/release/libriichi.so` → `mortal/libriichi.so` | `target/release/riichi.dll` → `mortal/libriichi.pyd`（MSYS2 环境编译） |
| `num_workers` | 2~4 | **必须 1**（`FileDatasetsIter.build_iter()` 的 GRP 构造在 Windows 子进程下不可用，源码注释已明确） |
| 路径 | `/home/...` | `C:/...` 或 `C:\\...`（toml 里建议用正斜杠） |

### 9.6 其他已知坑

| 现象 | 原因 | 处置 |
|---|---|---|
| `missing field ura_markers` | 数据转换时未把 `uradora_markers` 改名 | 回本机用 `tools/tenhou_data` 重转（见 TASKLIST §5.5） |
| `FileNotFoundError: grp.pth` | 没先训 GRP | 先跑 M1 |
| `FileNotFoundError: baseline.pth` | 没先跑 bootstrap | 先跑 M1 的 `bootstrap.py` |
| `test_every % save_every` 断言失败 | 配置不整除 | 按 §9.3 修正 |
| checkpoint 写入很慢 | `save_every` 太小 + 数据量大 | 见 §5.2，调大 `save_every` |
| 在线模式训练卡死 | 源码已知 bug | 已在 `main()` 用子进程规避，**保持现状** |
| 显存 OOM | `batch_size` 过大或其他进程占用 | 先查 `nvidia-smi`；仍不足则降 `batch_size`（并保持整除约束） |

### 9.7 合规红线

天凤牌谱受其 ToS 约束：**数据只在训练机本地使用，不要上传到 git 仓库、不要对外提供下载**。本仓库的 `.gitignore` 不应包含 `dataset/`，请自行确认。

---

## 10. 时间与资源估算

### 10.1 估算依据（按 130,407 局重算）

| 量 | 值 | 来源 |
|---|---|---|
| 局数 | 130,407 | 实测 |
| 单局决策步 | ≈ 660 | `dataloader.py` 注释 `around 660 instances per file` |
| 单 epoch 样本数 | ≈ 8,607 万 | 130,407 × 660 |
| `batch_size` | 512 | 配置 |
| **单 epoch 步数** | **≈ 168,000 步** | 8,607 万 ÷ 512 |
| 开增强后 | ≈ 336,000 步 | ×2 |
| 单步算力 | ≈ 1 TFLOP（batch 512，含反向） | 模型 ~10.8M 参数，前向 ≈0.64 GFLOPs/步 |
| P100 实效 | 4~6 TFLOPS | 理论 4~8 it/s（开 AMP），数据管道拖累时 3~5 it/s |
| GRP 单 epoch | ≈ 2,500 步 | 130,407 × ~10 kyoku ÷ 512 |

> **实测校准（强烈建议）**：训练启动 1 小时后，从 TensorBoard 读实际 `it/s`，用下面公式替换下表的推算值：
> ```text
> 每 epoch 耗时(h) ≈ 168,000 ÷ (it/s × 3600)
> ```

### 10.2 分阶段估算（标"估"为推算值）

| 阶段 | 内容 | 预计耗时 | 人工/挂机 |
|---|---|---|---|
| M0-a | rustup + libriichi 编译（LTO 全量）+ 依赖 | 0.5~1 h | 人工 |
| M0-b | 打包上传 590 MiB（含 tar 解压） | 10~30 min | 人工 |
| M0-c | `validate_logs` 全量 13 万局 | 数十分钟（Rust 多核） | 挂机 |
| M1 | bootstrap（<5 min）+ GRP 预训练 1~2 epoch | **0.5~2 h** | 挂机 + 偶尔看板 |
| M2 | 离线第一轮：168,000 步 + ~8 次 test_play | **10~18 h**（5 it/s ≈ 9.3h；4 it/s ≈ 11.7h；3 it/s ≈ 15.6h，另加评测 1~2h） | 挂机 |
| M3 | 1v3 评测 2000 局/次 | 10~20 min/次 | 挂机 |
| M4 | 每轮同 M2；通常 2~4 轮；第二轮起开增强（耗时 ×2） | **每轮 1~2 天** | 挂机 |
| M5 | 在线自博弈（可选） | 流程验证数小时；有效增强需 5~10 万自博弈局 → 数天 | 挂机 |

### 10.3 汇总

| 目标 | 墙钟时间 | 人工投入 |
|---|---|---|
| **首个可用模型（M0 → M3）** | **≈ 1~2 天** | 3~5 h |
| 迭代至性能平台（M4） | 再 +2~6 天 | 数小时（每轮检查归档） |
| 在线自博弈增强（M5，可选） | 再 +数天 | 数小时 |

### 10.4 何时需要更多数据/算力

- **130,407 局**够"跑通闭环 + 出第一个能打的模型"，并支撑 **2~3 轮**有效迭代；
- 社区/作者级别的强度（凤凰卓上位）通常需要 **50 万 ~ 250 万局**量级；
- 历史 2009–2025（约 240 万局）可作为强度跃升来源，但要以散装 `.json.gz` 落到 **NTFS**（簇 4 KiB）卷上——⚠ **绝不能放 exFAT 卷**（簇 256 KiB，每文件白占 256 KiB，13 万文件就膨胀 33 GiB，详见 TASKLIST §5.7/§7.10）；
- 数据量上升到数百万局后，单 epoch 步数线性放大，P100 单卡将进入**数百小时/轮**，届时建议按年分批训练或改用云 GPU / 多卡。

### 10.5 GPU 资源占用参考

| 阶段 | 显存 | GPU 利用率 | 说明 |
|---|---|---|---|
| GRP 预训练 | 低 | 高（FP64 强项） | P100 FP64 = 1/2 FP32，float64 GRU 跑得飞快 |
| 离线 CQL 训练 | ~10 GB（batch 512） | 中~高 | 瓶颈常在 CPU 数据管道 |
| test_play / 1v3 | 低（推理） | 中 | 速度约 3~8 局/秒 |

---

## 11. 里程碑验收清单

逐条打勾，任何一条不满足都不要进入下一阶段。

| # | 里程碑 | 通过标准 | 命令/位置 |
|---|---|---|---|
| M0-a | 环境就绪 | `import libriichi` 成功；`torch.cuda.is_available()` 为 True | §2.5 / §2.3 |
| M0-b | 数据就位 | `find mjai -name '*.json.gz' \| wc -l` = 130407 | §3.3 |
| M0-c | 数据闭环 | `validate_logs` 0 错误 | §3.4 |
| M1 | GRP 完成 | `acc/val` 进入平台期；`checkpoints/grp.pth` 存在 | `tb_log_grp` |
| M2 | 离线第一轮 | `dqn_loss` 下降；`test_play/avg_rank` < 2.5 且下降 | `tb_log` |
| M3 | 第一轮收敛 | `avg_rank`/`avg_pt` 趋平；`best.pth` 已归档 | `1v3/` |
| M4 | 迭代增益枯竭 | 相邻两轮 1v3（≥4000 局）无显著差异 | `1v3/` |
| M5 | （可选）在线增强 | 自博弈轨迹持续回流；1v3 仍能涨 | `buffer/` `drain/` |

**当前进度（2026-10-09 更新）**：M0-a✅（rust 1.99.0 + libriichi.so/bins）｜M0-b✅（**全量 2,630,640 局已在服务器**）｜M0-c✅（`validate_logs` + `prune --apply`）｜M1✅（GRP 30k 步，但 **val loss 未收敛**）｜M2✅（离线 CQL 200k 步，覆盖一个 epoch 的 19%）｜M3⚠️（评测指标饱和，结论：强度持平，见 §14.4）｜M4/M5 未开始。具体数据与诊断见 **§14**。

---

## 附录 A：命令速查表

```bash
# ── 环境 ──────────────────────────────────────────
conda activate mortal
cd $MORTAL_ROOT/mortal
export MORTAL_CFG=config.toml          # 想用别的配置文件名时

# ── 数据（本机 Windows）──────────────────────────
cd C:\Maj\Mortal\dataset
tar -czf mjai_20261003.tar.gz mjai
scp mjai_20261003.tar.gz user@SERVER:~/mortal_data/

# ── 数据（服务器）────────────────────────────────
mkdir -p ~/mortal_data/dataset && cd ~/mortal_data/dataset
tar -xzf ~/mortal_data/mjai_20261003.tar.gz
find mjai -name '*.json.gz' | wc -l                      # 130407
$MORTAL_ROOT/target/release/validate_logs ~/mortal_data/dataset/mjai

# ── 训练主流程 ───────────────────────────────────
cd $MORTAL_ROOT/mortal
python bootstrap.py            # 随机 baseline（一次性）
python train_grp.py            # M1  GRP 预训练
python train.py                # M2  离线 CQL
python one_vs_three.py         # M3  1v3 评测

# ── 看板 ─────────────────────────────────────────
tensorboard --logdir tb_log     --bind_all --port 6006
tensorboard --logdir tb_log_grp --bind_all --port 6006

# ── 数据更新后（重要）────────────────────────────
rm -f file_index.pth grp_file_index.pth                  # 删缓存
rm -f checkpoints/mortal.pth                             # 想从头训才删

# ── 轮次归档（M4）────────────────────────────────
cp checkpoints/best.pth checkpoints/round1_best.pth
cp checkpoints/best.pth checkpoints/baseline.pth
mv checkpoints/mortal.pth checkpoints/round1_state.pth
```

---

## 附录 B：数据更新后如何重训

当 `tools/tenhou_data` 又采到新对局（例如 2026-10 后续整月）后：

1. **本机**：重跑管线增量补齐，再重新打包上传

   ```powershell
   python tools/tenhou_data/tenhou_pipeline.py all      # 增量：索引→清单→下载→转换→校验
   cd C:\Maj\Mortal\dataset; tar -czf mjai_<date>.tar.gz mjai
   scp mjai_<date>.tar.gz user@SERVER:~/mortal_data/
   ```

2. **服务器**：解压覆盖，并**删除两个索引缓存**（否则训练器看不到新文件）

   ```bash
   cd ~/mortal_data/dataset && tar -xzf ~/mortal_data/mjai_<date>.tar.gz
   cd $MORTAL_ROOT/mortal
   rm -f file_index.pth grp_file_index.pth
   ```

3. **是否重训 GRP**：
   - 新增数据与训练集同分布且量不大（< 10%）→ 可沿用旧 `grp.pth`，只重训 Brain/DQN；
   - 新增数据量大或分布变化明显 → 建议用新数据重训 GRP（把新月份加进 `grp.dataset.train_globs`）。

4. **Brain/DQN**：默认**续训**（保留 `checkpoints/mortal.pth` 会自动接着练）；若数据规模翻倍以上，建议归档旧 checkpoint 后从头训一轮。

5. **重跑顺序**永远不变：`(GRP) → train.py → one_vs_three.py`。

---

## 14. 远程实跑记录与修正（2026-10-07~10-09）

> 本节全部为 **实测结果**（日志 / TensorBoard / checkpoint / 牌谱重算），用于把§10 的推算换成真数，并把“能照做的正确命令”固定下来。
> 完整现状叙述见 [MEMORY.md §16](../../memory/MEMORY.md)。

### 14.1 服务器实际目录与启动方式（与 §1.3 约定的差异）

```bash
export MORTAL_ROOT=/mnt/data8/mortal/Mortal
export MORTAL_CFG="$MORTAL_ROOT/mortal/config.offline.toml"     # ★ 不是 config.toml
export CUDA_DEVICE_ORDER=PCI_BUS_ID                            # ★ 必加：否则序号与 nvidia-smi 不一致（见 ENV_SETUP_P100.md §10）
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # 训练可省；跑 validate_logs/stat 必需
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8        # ★ §14.5：原用 4，实测偏慢，新轮请直接用 8
cd $MORTAL_ROOT/mortal && mkdir -p checkpoints runs indexes logs     # ★ 必须在这个目录（仓库根目录会被 libriichi/ 遮蔽导入）
CUDA_VISIBLE_DEVICES=1 tmux new -s train_grp      # GRP → P100
tmux new -d -s train_main                          # 主训 → 在窗内先 export CUDA_VISIBLE_DEVICES=2（RTX 8000）
tensorboard --logdir runs --host 127.0.0.1 --port 6006   # 本机 ssh -L 6006:localhost:6006 后看 localhost:6006
```

> ⚠️ 上面的 `CUDA_DEVICE_ORDER` 是 2026-10-09 补上的：不加时 `CUDA_VISIBLE_DEVICES=2` 实际会落到 **P100**（RTX 8000 在默认算力序里是 0），第 1 轮启动命令就踩了这个坑。启动后一定看日志里的 `device: cuda:0 (型号)` 那一行确认真的卡。

产物布局（实际）：`checkpoints/`（含 `mortal_step_*.pth` 快照）、`runs/{grp,main,selection,ladder}`、`indexes/{grp_recent_v1,main_recent_v1}.pth`（即 `file_index`）、`logs/{grp_train,main_train,ladder_probe}.log`。

### 14.2 代码补丁清单（上游没有、本轮训练依赖）

| 文件 | 能力 | 配置开关 |
|---|---|---|
| `mortal/train.py` | 训练预算到点自停；按周期另存编号快照 | `control.max_train_steps`、`control.snapshot_every` |
| `mortal/train_grp.py` | 预算自停；`grp_best.pth`（按 val loss 取优）；固定验证抽样种子；日志加 `est. %` | `grp.control.max_train_steps` |
| `mortal/player.py` | `TestPlayer` 的 AMP / agari guard 改从配置读；评测固定种子块 | `test_play.enable_amp`、`test_play.enable_rule_based_agari_guard` |

⚠️ 三处均靠 `.get(key, default)` 兼容上游配置。**10-09 已建本地 git 仓库**（基线 `33e63bc` / `main` / tag `round1-complete`，见 §15.5），补丁不再只存一份；但**仍无远端**，单机器风险未根除。本机文档若已把补丁写成 §4.1/§4.2/§4.3，请逐条与本表对校。

### 14.3 实跑配方与结果（第一轮）

**GRP（M1）**：`grp.control` device=P100、batch 512、lr **1e-4**（§5 手册写的 1e-5）、save_every 2000、val_steps 200、max_train_steps **30000**。训 21 分钟（=1.89 epoch；单 epoch ≈ 15,882 步）。val loss 2.543→2.455 **仍在降**（未收敛），acc(全排列) 0.218→0.229。
→ 不要拿 `acc` 当绝对水平：它是对每个“局前缀”的全排列 argmax 命中（含大量不可预测的早期前缀）。用 `tools/training/grp_quality_probe.py` **按位置**评估才是真水平（3 个月 ×150 局）：
  grp.pth(30k)：hit@1 0.249 / **hit@3 0.368 / hit@6 0.467** / hit@9 0.575 / hit@末局 0.767；mae@3 0.833、mae@6 0.703；nll@末局 1.2780；`delta_pt` std 0.700、absmax 4.80
  grp_v2.pth(42k)：hit@3 0.379 / hit@6 0.476 / hit@9 0.597；nll 1.2591 → 中段略有提升但已入平台期
  **判读**：hit@1≈随机、hit@末局高只是因为分数已定死顺位；决定奖励质量的是中段的 hit@3/hit@6（现在只有 37%/48%）。本节早期版本写的“局末命中 ≈50–56%”是探针把开局前缀混进末局统计导致的错数，已修正。

**主模型（M2）**：v4 192×40（10.79M 参）、RTX 8000 + AMP、batch 512、save_every 1000、test_every/snapshot_every 50000、max_train_steps 200000、`gamma=1`、`pts=[6,4,2,0]`、`cql.min_q_weight=5`、`max_grad_norm=5.0`、scheduler warm_up 2000 / peak=final=1e-4。墙钟 22.9 h 训练 + 4×21 min 评测。dqn_loss 0.467→0.392，cql 1.261→0.506，next_rank 1.324→0.756。
→ 200k 步 = 1.024 亿样本；实测 678 样本/文件 → 单 epoch ≈ **1,076,000 步** → 本轮只覆盖训练池的 **18.6%**（≈ 15 万局）。

### 14.4 评测协议（修正版，今后照此执行）

随机 baseline 下的 `test_play` 已证实不可用（§8.2 修正说明）。三类可用信号：

**① 未见月离线指标（最便宜，秒级，适合每 5k–10k 步监控）**

用 `202609+` 做 holdout（主模型只训到 202608），看：
- **与人类动作 top-1 一致率**（本轮实测：随机 0.248 → 50k 0.7812 → 100k 0.7905 → 150k 0.7932 → 200k **0.7952**）
- **Q 校准**：`0.5·MSE` vs “只预测常数”的 loss（本轮 0.419 vs 0.234），以及 var_q/var_t（0.49/0.47 → 幅度过高估，R² 为负）
- 和了动作（索引 43）是否永远排第一（本轮 161/161，agari guard 正常）

**② 同牌山配对对抗（选模型、定胜负用这个）**

```bash
cd $MORTAL_ROOT/mortal
python ladder.py <challenger.pth> <champion.pth> [seed_start=30000] [sets=500]
# sets 组×4 半庄（四座位轮换）；500 sets = 2000 局 ≈ 21 min（RTX 8000）
```

本轮实测（150 sets = 600 局，SE ≈ ±3.6 pt）：

| 挑战者 | 卫冕者 | avg_rank | avg_pt | 结论 |
|---|---|---|---|---|
| 200k | 50k | 2.538 | −5.18 | 均在噪声内 |
| 150k | 50k | 2.440 | +4.20 | 均在噪声内 |

→ **本轮离线训练 50k→200k 强度持平**；下轮定结论至少用 500–1000 sets，并固定不同 `seed_start` 防牌山过拟合（§8.3 的“≥ 4000 局”仍适用）。

**③ 行为合理性（免模拟，与人类牌谱同口径对照）**

| 每局每家 | 人类凤凰卓（4000 局） | 模型自博弈（600 局） |
|---|---|---|
| 和了 | 0.2117（ron 0.1255） | 0.2228（ron 0.1405） |
| 立直 | 0.1769 | 0.1794 |
| 副露 call | 0.4826 | 0.4566 |
| 流局率 | 15.3% | **10.9%**（→ 弃和不足） |

### 14.5 时间/吞吐量实测（推翻 §10 估算）

| 量 | §10 推算 | **10-09 实测** |
|---|---|---|
| 数据管道（num_workers=2, RAYON=4） | 未区分 | **3.12 it/s**（1,596 样本/s）← 瓶颈 |
| GPU 单步上限（RTX 8000, bs512, AMP） | 4–8 it/s（按 P100 估） | **8.88 it/s**（112.6 ms/步） |
| 实得训练速度 | 5 it/s（估） | **2.56 it/s**（50k 步 = 5.5 h） |
| 管道（nw=4, RAYON=8） | — | 6.97 it/s |
| 管道（nw=8, RAYON=8） | — | **7.48 it/s（≈ 2.4×）** |
| 一个全 epoch（813k 局） | ≈ 10 h（错） | 现配置 ≈ **4.8 天** → 调优后 ≈ **2 天** |
| 单次 1v3 评测（2000 局） | 10–20 min | **21 min**（1.6 局/s） |
| GRP 30k 步 | 0.5–2 h | **21 min**（P100） |

复现/调优命令：`python bench_pipe.py <num_workers> <RAYON>`（只跑数据管道，不占 GPU）。
注意：`RAYON_NUM_THREADS` 必须在启动前设好（rayon 池进程内只建一次）；`num_workers` 抬高后内存代价≈ 0.7 GB/worker @ `file_batch_size=8`，本机 251 GB 可支持 nw=8 且 `file_batch_size` 可提到 15–20。

### 14.6 下一轮（M4 候选）执行清单

前提：**先修评测，再动训练**；先把代码与 checkpoint 备份。

| # | 动作 | 命令/配置要点 |
|---|---|---|
| 1 | ✅ 备份与版本化（10-09 已完成） | `tools/git_init_repo.sh` 已建本地 git（基线 `33e63bc`）；`.gitignore` 已补 `dataset/`、`mortal/runs/`、`mortal/logs/`、`backups/`；checkpoint 副本与代码包已入 `backups/`。待做：有网时加私有远端并 push |
| 2 | 吞吐调优 | `num_workers=4~8`、`RAYON_NUM_THREADS=8`、`file_batch_size=15~20`；先 `bench_pipe.py` 跑一遍定参 |
| 3 | ~~把 GRP 训收敛~~ → **加 GRP 容量** | 续训已证伪“未收敛”（30k→42k 仅 nll −0.019、val loss 在噪声带内）；改跑 `mortal/run_grp_capacity.sh`（h128/l3、从零 60k 步），用 `grp_quality_probe.py` 的 **hit@3/hit@6/nll_last** 判胜负，胜出才换奖励模型（详见 §15.3） |
| 4 | 开新一轮离线（换信息源，不是只加步数） | 新目录/新 `state_file`（否则自动续训旧步数）；候选变量（一次只动 1–2 个）：<br>a) `enable_augmentation=true`（样本×2）<br>b) `player_names_files` 高手白名单（把“模仿全体凤凰卓”变成“模仿顶尖”）<br>c) `gamma` 1→0.99/0.995（跨局信用分配）<br>d) 扩大时间窗（全量 2.63 M 或 2019+）**且同步重训 GRP**<br>e) scheduler 改长程衰减（现 warm_up=2000/peak=final=1e-4） |
| 5 | 选模型与定胜负 | `ladder.py` 新 vs 旧 best（≥500 sets），胜者才能当下一轮 `baseline.pth`/卫冕者；别再信 `best.pth` |
| 6 | 外部锚点（可选） | akochan：装 JRE → `AKOCHAN_DIR`/`AKOCHAN_TACTICS` → `[1v3.akochan] enabled=true` → `python one_vs_three.py` |
| 7 | 在线自博弈（最后才动） | `online=true` + `freeze_bn.mortal=true`；`server.py` + `train.py` + `client.py`×多（P100 当 client 集群）；不要绕过 `main()` 的子进程规避 |

**重要认知更新**（§0 “三个事实”之外的第 6 条）：本轮已经实测到“**loss 在降、人类一致率在涨，但对局强度持平**”。在 `gamma=1` + CQL 锚定行为策略的设定下，离线阶段本质上是“把人类行为拟合得更准”，而不是“超越人类”。想拿到真增益，必须换信号源（GRP 质量/数据选择/跨局视界）或进入在线自博弈。**因此“把 200k 改成 1M 同配方续训”是低优先选项。**（→ 10-11 已由 §16.5 实证：同池第 2 个 epoch 的 δ=−0.03±0.59，确实无增益）

> ⚠️ 本小节里“GRP 未收敛、先续训到 100k~150k”的判断已由 10-09 的续训实验**推翻**（平台期在 30k 就到），结论以 §15 为准。

---

## 15. 第 2 轮就绪状态（10-09 已落地）

按选定路线「**先修评测 + 吞吐 → GRP → 换信号源离线**」，以下资产已就绪并跑过验证；全部**非破坏性**（第 1 轮的 `config.offline.toml`、`grp.pth`、`mortal*.pth` 一律不动）。

### 15.1 新增工具

| 工具 | 作用 | 验证状态 |
|---|---|---|
| `tools/player_stats/scan_player_ratings.py` | 扫牌谱自建**玩家强度榜**（牌谱无段位字段 → 用长期 平均顺位/顺位点 自证） | ✅ 813,145 文件 / 28 进程 **77 秒**；全体加权 avg_pt=0.0000、avg_rank=2.5000（数学自洽） |
| `tools/player_stats/whitelist_coverage.py` | 定 top-N 名单的**席位/样本/epoch 步数**覆盖度 | ✅ 见 §15.2 表 |
| `tools/player_stats/build_whitelist.sh` | 上面两步一键跑（tmux 承载） | ✅ 产出 CSV + `top_players_400.txt` |
| `mortal/holdout_metrics.py` | 未见月离线指标（top1 一致率、Q 校准、R²、和了 top1），一次比多个 checkpoint | ✅ baseline top1=0.2567 vs 200k=0.8187 |
| `mortal/ladder.py` | checkpoint 间同牌山 1v3 配对对抗 | ✅ 已用于 §14.4 的两组对照 |
| `mortal/bench_pipe.py` | 数据管道吞吐基准（不占 GPU） | ✅ §14.5 的数来自参数扫描 |
| `tools/training/grp_quality_probe.py` | GRP 真实质量探针：hit_last / mae_rank / nll_last / rew std | ✅ 与 §14.3 的局末顺位口径一致 |

### 15.2 玩家白名单（本轮“换信号源”的具体内容）

实测语义（`libriichi/src/dataset/gameplay.rs::load_events`）：**只有名单内玩家自己座位的决策会成为样本** → 决定数据量的是席位，不是命中文件数。

窗口 2022-01~2026-08（813,145 文件）：有 names 的 699,715（自建 2026 的 113,430 **无 names**）；玩家 4,883 位，其中 ≥400 局的 1,784 位；阈值 top100 +5.08 / top200 +3.97 / **top400 +2.41** / top600 +1.23。

> ⚠ **下表是 v1 口径（供托计分修正前）存档**。v2 重算（10-11，`build_whitelist.sh` 输出 `_v2` 文件）阈值 top100 +5.2 / top200 +3.9 / **top400 +2.45** / top600 +1.2，且 top400 与 v1 相比 **53/400 人进出**：
>
> | 名单（v2） | 命中文件 | 席位 | 样本≈ | 1 epoch 步数 |
> |---|---|---|---|---|
> | top100 | 121,506 (14.9%) | 130,206 | 22 M | 43 k |
> | **top200** | 279,713 (34.4%) | **334,907** | 57 M | **111 k** |
> | **top400（R4 在用）** | **492,810 (60.6%)** | **735,249** | 125 M | **244 k** |
> | top600 | 601,258 (73.9%) | 1,089,419 | 185 M | 362 k |
> | top900 | 671,884 (82.6%) | 1,563,682 | 266 M | 519 k |

| 名单 | 席位 | 样本≈ | 1 epoch 步数 | 备注 |
|---|---|---|---|---|
| top200 | 324,484 | 55 M | 108 k | 过采风险（一个预算就会跑 >2 epoch） |
| **top400** | **704,231** | **120 M** | **234 k** | **已选**：+250k 步预算 ≈ 1.07 epoch |
| top600 | 1,098,750 | 187 M | 365 k | 若想把第 2 轮拉到 1.5 epoch 可换这个 |
| top900 | 1,527,953 | 260 M | 507 k | 已接近“不过滤”的数据量，筛选效应下降 |

> 注意两个坑：① `file_index` **必须换新名**（旧的 `indexes/main_recent_v1.pth` 是无白名单时缓存的文件表，沿用会静默跳过过滤）；② `train.py` 原本对无 names 的文件会 `KeyError`，已修为 `start.get('names', [])`。

### 15.3 GRP：实验已完成，奖励模型已定档为 grp_h128

**两个假设都跑过了（同口径探针，3 个月 ×150 局 = 450 局，固定 seed）**：

| 模型 | 结构 | 步数 | val loss | hit@3 | hit@6 | hit@末局 | mae@末局 | **nll@末局** | rew std |
|---|---|---|---|---|---|---|---|---|---|
| `grp.pth`（第 1 轮在用） | h64/l2 | 30k | 2.45495 | 0.3675 | 0.4665 | 0.7667 | 0.3812 | 1.2780 | 0.6996 |
| `grp_v2.pth`（加步数） | h64/l2 | 42k | 2.45207 | 0.3786 | 0.4758 | 0.7644 | 0.3741 | 1.2591 | 0.7041 |
| **`grp_h128.pth`（加容量）** | h128/l3 | 60k | **2.44599** | 0.3786 | **0.4827** | **0.7689** | **0.3736** | **1.2361** | 0.6955 |

结论：

1. **“加步数”无效**：30k→42k 的 val loss 仅在 2.4530~2.4566 噪声带内波动（−0.003），acc_val 恒定 ≈0.229 → 30k 步就已是平台期，所以提前终止了 `grp_v2`（推翻本节之前“未收敛、续训到 150k”的判定）。
2. **“加容量”有效但有限**：nll_last 1.2780 → 1.2361（−0.042，即校准改善约 3.3%），hit@6 +1.6 pp；但 argmax 类的 hit@3 几乎没动、**`delta_pt` 的 std 仍≈ 0.70**（奖励尺度与噪声幅度基本不变）。
3. **根因是信息量，不是拟合能力**：GRP 的输入只有 7 维/局（大局数、本场、供托、四家分数），中段（第 3/6 局）的真实顺位本身就高度不确定 → hit@3 停在 0.38、mae@3 停在 0.83 个名次。想要更好的奖励，得改**目标形式**（如用真实点数差监督、或把 GRP 换成期望点数回归）而不是继续改尺寸/步数。
4. **定档**：`config.round2.toml` 已改为 `state_file='checkpoints/grp_h128.pth'` + `hidden_size=128, num_layers=3`（两者必须同步，`dataloader.py` 按 `[grp.network]` 重建网络）。已做冒烟验证：`MORTAL_CFG=config.round2.toml` 下 GRP 重建 + 奖励计算跑通，输出 obs (512,1012,34)、reward std 0.643（日志 `mortal/logs/smoke_round2.log`）。第 1 轮的 `grp.pth` 保持不动，可随时回退。

> 第 2 轮的主要变量因此仍然是**数据选择（白名单）**，而不是奖励模型；对 GRP 不要抱太大期待。

### 15.4 第 2 轮开跑命令（前置项已全部验证，只待执行）

已完成的预检（均用 tmux，日志在 `mortal/logs/`）：

| 项 | 结果 | 日志 |
|---|---|---|
| 强度榜构建 | ✅ 813,145 文件 / 77 秒 / 4,883 位玩家（≥400 局的 1,784 位） | `whitelist_build.log` |
| top400 名单与覆盖度 | ✅ 保留 59.1% 文件（≈480,365）、席位 704,231 → ≈234k 步/epoch；无 names 的自建 2026 部分占 13.4% | `cfg_round2_check2.log` |
| 索引构建代价 | ✅ 串行 10,574 文件/s → 全量约 **1.3 分钟**（不是十几分钟） | 同上 |
| config.round2.toml 加载 + GRP 重建 + 奖励计算 | ✅ obs (512,1012,34)、reward std 0.643 | `smoke_round2.log` |
| 奖励模型定档 | ✅ grp_h128（见 §15.3） | `grp_capacity.log` |

**上述三步已于 10-09 14:17 执行（用户在 tmux 会话 `main_r2` 里亲手拉）**，实测与预检预测完全对得上：

| 预检预测 | 实跑确认 |
|---|---|
| 白名单保留 59.1% ≈ 480,365 文件 | 日志 `loaded 400 players` → `file list size: **480,414**`（误差 0.01%） |
| 吞吐 6~7 it/s | **6.71 it/s**（第 1 轮 2.56 → **2.6×**），瞬时 7.2 batch/s |
| 暖启自第 1 轮 200k | `loaded: 2026-10-09 04:42:28` → `total steps: 200,000`；新目录 `checkpoints/round2/` 不覆盖旧文件 |
| 奖励模型 grp_h128 | obs (1012,34)、9 个进程（nw=8）、device=RTX 8000 ✓ |
| 预算与 ETA | 247k 步 ÷ 6.71 ≈ 10.2 h + 5 次评测 1.8 h → **≈ 12 h**（10-10 02:30 左右自停于 `training budget reached`） |

已挂上**自动验收监控**（tmux `r2_watch` → `mortal/run_r2_monitor.sh`，日志 `mortal/logs/r2_monitor.log`）：

- 每出现一个新的 `checkpoints/round2/mortal_step_*.pth`（每 50k 步）→ 自动跑 `holdout_metrics.py`（与第 1 轮 200k 同一批未见月样本、同一口径比 top1/dqn_loss/R²）；
- 训练进程一退出 → 自动跑**双向 ladder**：新 vs 旧（seed 40000）与旧 vs 新（seed 41000），各 500 sets = 2000 局；两方向吻合才能算真增益（消除牌山与座位偏置）；
- 评测全部放空闲 P100（`CUDA_VISIBLE_DEVICES=1`）+ `nice -n 10` + `RAYON_NUM_THREADS=4` + `--workers 2`，避免抢训练的 CPU（已冒烟验证：约 2.5 min 跑完，训练瞬时吞吐从 7.2 降到 6.0 后即恢复）。

> 写这类监控的坑：用 `pgrep -f "python -u train.py"` 判断训练是否还在跑会**永不出循环** —— tmux 里的 `bash -c "… python -u train.py …"` 包装进程 cmdline 含同样字符串；必须锚定：`pgrep -c -f '^python -u train\.py$'`（实测正确匹到主进程+8 个 worker = 9）。

> 备份与版本化已完成：14:28 做了 `backups/code_docs_20261009.tar.gz`（32 MB）+ `backups/ckpt_round1/`（592 MB，含 grp / grp_h128 / grp_best / baseline / 四个 mortal_step 副本），14:42 建了本地 git（§15.5）。**副作用提醒**：容量实验覆盖了 `checkpoints/grp_best.pth`（`train_grp.py` 硬编码把最优盘写到 `dirname(state_file)/grp_best.pth`），现在它是 h128/60k 的那份，与 `grp_h128.pth` 等价，旧份与保留的 `grp.pth` 完全相同 → 无信息丢失，但**跨尺寸 GRP 实验共用这个名字会互盖**。

下面三个步骤的执行命令（**已于 14:17~14:28 全部执行完毕**，保留在此供后续轮次复用）：

```bash
# 1) 备份代码与文档（backups/ 已在 .gitignore 里，不会入库）
cd /mnt/data8/mortal/Mortal && tar czf /mnt/data8/mortal/code_docs_$(date +%Y%m%d).tar.gz \
    --exclude=dataset --exclude=target --exclude='mortal/runs' --exclude='mortal/logs' \
    --exclude='mortal/checkpoints' --exclude='mortal/__pycache__' .

# 2) 暖启：把第 1 轮终态拷到新目录（不覆盖任何旧文件；round2 的 state_file 指向这里）
mkdir -p mortal/checkpoints/round2 && cp mortal/checkpoints/mortal.pth mortal/checkpoints/round2/mortal.pth

# 3) 开跑（RTX 8000；RAYON 与 CUDA_DEVICE_ORDER 必须在启动前设好）
tmux new -d -s main_r2 "cd /mnt/data8/mortal/Mortal/mortal && \
  export MORTAL_CFG=\$PWD/config.round2.toml LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib \
         OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=8 \
         CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 && \
  python -u train.py > logs/main_r2_train.log 2>&1"
# 开跑后 30 秒内必验：日志应出现 `device: cuda:0 (Quadro RTX 8000)`
grep -a 'device:' /mnt/data8/mortal/Mortal/mortal/logs/main_r2_train.log | head -1

# 进度查看（不进训练也能看）
tail -f mortal/logs/main_r2_train.log
tensorboard --logdir runs --host 127.0.0.1 --port 6006    # 新增 runs/main_r2 与 runs/grp_h128
python holdout_metrics.py checkpoints/round2/mortal.pth checkpoints/mortal_step_000200000.pth
```

预期：首次会重建白名单索引（≈1.3 分钟，写入 `indexes/main_r2_whitelist_v1.pth`，之后秒级复用）；调优后吞吐目标 ≈ 6~7 it/s → 250k 步 ≈ **10~12 h**（含 5 次 test_play，每次 21 min，现在对手是上一代自己，指标有区分度）。

**判进规则（写死，免得起争议）**：`ladder.py 新快照 vs mortal_step_000200000.pth`（≥500 sets）avg_pt > 0 且超出 ±3.6 pt 噪声 → 本轮有效；否则回到 §14.6 换下一个变量（`enable_augmentation` / `gamma<1` / 扩大时间窗），**不要靠 dqn_loss 下降当作进步证据**。

### 15.5 版本控制（10-09 14:42 已建立）

用 `tools/git_init_repo.sh`（幂等：已有 `.git` 直接退出；**防泄漏 + 必要文件齐全性双重校验**，任一不过则不提交）。结果：

| 项 | 值 |
|---|---|
| 分支 / 基线提交 | `main` / `33e63bc`（本机 git 2.17 不支持 `init -b`，改用 `git symbolic-ref HEAD refs/heads/main`） |
| tag | `round1-complete`（annotated，标第 1 轮完成态，方便 baseline 换代与回退） |
| 入库规模 | 460 个文件、共 4.0 MB；`.git` 4.4 MB |
| 防泄漏校验 | `git ls-files -- dataset` = **0**；`.json.gz/.pth/.so/.dll/.exe/.tar` 被跟踪数 = **0**；`git status --ignored` 确认 `dataset/ target/ backups/ mortal/{checkpoints,indexes,runs,logs,libriichi.so}` 均被忽略 |
| 工作区 | `git status --porcelain` 为空（干净） |

两个约定：

1. **提交身份**：环境里本来没有任何 git 身份，脚本**没改你的配置**，只用临时 `-c user.name='mortal-trainer' -c user.email='mortal@localhost'`。想换成自己的：`git -c user.name='你' -c user.email='你@x' commit --amend --reset-author`（并用同样方式重做 tag）。
2. **不入库、需自行重建的两个前置件**（牌谱按 ToS 红线永不进 git）：`mortal/libriichi.so` ← `cargo build -p libriichi --lib --release`（§2.5）；`dataset/index/top_players_400.txt` ← `bash tools/player_stats/build_whitelist.sh`（77 秒，需牌谱）。

尚未做（需外网）：`git remote add origin <私有仓地址> && git push -u origin main && git push origin --tags`（今天 `curl raw.githubusercontent.com` 超时）。推不动的这段时间，`backups/code_docs_*.tar.gz` 是唯一可带走的副本。

> 坑：`git tag -a`（annotated）**也需要身份**。第一次只给 `commit` 加了 `-c`，结果 commit 成功、tag 报 `fatal: unable to auto-detect email address`；脚本已修正（两条命令共用 `GITID` 数组）。

---

## 16. 第 2 轮判定结果与第 3 轮（10-10）

### 16.1 第 2 轮为何“提前退出”（不是故障）

预算 `max_train_steps = 450000`，实际停在 **424,847 步**：从 200k 暖启算起吃了 **224,847 步**，而同口径实测白名单池（480,414 文件 × 234.8 样本/文件 ÷ 512）= **1 个 epoch ≈ 220,268 步**（± 10,442），即实际消耗为预测的 **102%** → **离线模式跑完一个 epoch 就 `break`**（源码有意设计），与预算无关。日志特征：无 Traceback、无 `training budget reached`，末尾只有一行 `ResourceWarning: Implicitly cleaning up <TemporaryDirectory>`（解释器正常关闭的 GC 噪声，第 1 轮正常退出时也一样），cgroup `oom_kill=0`。

要跑更长只有两条路：提 `num_epochs`（第 3 轮就是这么做的），或降 `max_train_steps` 让它“预算到点自停”（数字游戏，不增加数据）。

### 16.2 判定实验的正规做法（5 卡并行，已实测）

```bash
# 前提：ladder.py 已支持第 5 个参数 log_dir（并发必传，否则互相 rmtree 删牌谱）
bash mortal/run_r2_ladder.sh          # 已写成脚本，内含完整 env 重置
```

| 要点 | 值 |
|---|---|
| 规模 | 双向各 5 块 × 500 sets = 5000 sets = 20,000 半庄 |
| 卡分配 | 每任务一张卡（`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=2/0/1/3/4`），两波串行 |
| 实测耗时 | **每波 ≈ 14 分，总计 29 分**（500 sets = 2000 局） |
| 统计量 | δ = (pt_A − pt_B)/2，SE 按独立样本保守估计；`agg_ladder.py` 输出 t、95%CI 与“还需多少 sets” |
| 牌谱 | 全部保留在 `mortal/logs/ladder_r2/{A,B}_*/`（806 MB），可事后重算行为统计 |

### 16.3 结果：第 2 轮【有效】

| 项 | 值 |
|---|---|
| A（新当挑战者，10,000 局） | avg_pt **+3.299**，SE 0.820；5 块全正（+2.00 ~ +4.43） |
| B（旧当挑战者，10,000 局） | avg_pt **−2.430**，SE 0.856；5 块全负 |
| **强度差** | **δ = +2.86 pt/半庄，SE 0.59，t = +4.83，95%CI [+1.70, +4.03]** |
| 结论 | 白名单 top400 + grp_h128 的组合 **统计显著变强**（相对第 1 轮 200k） |

**重要教训**：上次两组各 500 sets 给出 +0.70 / +2.23，均落在噪声内→差点误判为“无增益”。样本量需按 `sets ≥ 当前 sets × (2·SE/δ)²` 倒推；小样本不只能“判不出”，还会**低估**效应。

### 16.4 第 3 轮（在跑）

| 项 | 值 |
|---|---|
| 变量 | **只有 `num_epochs` 1→2**（同一白名单池再过一遍）；其余与第 2 轮全同 |
| 配置 | `mortal/config.round3.toml`（`tools/training/make_round3_cfg.sh` 从 round2 派生） |
| 暖启 / 预算 | `checkpoints/round3/mortal.pth ← round2/mortal.pth`（424,000）；`max_train_steps = 650000`（再 +226k 步） |
| 启动 | 10-10 05:34，tmux 会话 `train_r3`，日志 `logs/main_r3_train.log`，看板 `runs/main_r3` |
| 实测 | 6.7 it/s（与第 2 轮一致）→ 10-10 15:56 已到预算并正常退出（见 §16.5） |
| 判定基准 | ⚠ ladder 必须 **vs round2 终态 424k**（不是 vs 第 1 轮）；且 epoch 2 属重复数据，holdout top1 上涨不算变强证据 |

**→ 10-11 续章见 §16.10**（恢复后触发 NaN 崩坏，已加三重防护，并从 500k 干净快照以 fp32 重启）
| 启动命令 | `bash mortal/launch_round3.sh`（内含“每次跑前重置全部环境变量”的写法） |

> 坑（本次踩过）：`launch_round3.sh` 里的前置自检用了 base 环境的 `python`（无 torch）→ `set -e` 下脚本在启动前就退出。脚本里**所有 python 调用都必须用绝对路径的解释器** `$PY`，不能依赖已经激活的 conda 环境（tmux 新会话不会继承）。

### 16.5 第 3 轮结果：跑到预算正常退出，判定为【无增益】

10-10 15:56 日志 `training budget reached`（650,000 步，10.4 h，6.7 it/s，无报错）。用与 §16.3 相同的双向配对方法（新种子块 60000+/70000+）判定：

| 方向 | avg_pt | SE |
|---|---|---|
| A：650k 挑战 424k（10,000 局） | −0.059 | 0.845 |
| B：424k 挑战 650k（10,000 局） | −0.004 | 0.838 |
| **δ(650k − 424k)** | **−0.03 pt/半庄** | **SE 0.595，t=−0.05，95%CI [−1.19, +1.14]** |

旁证：未见月 top1 **不升反降** 0.8122(424k) → 0.8091(650k)；dqn_loss 仅 0.374→0.363。

→ **结论：同一白名单池重复过数据（epoch 2）不产生强度增益**。下一轮请换信息源，不要再靠加 epoch/加步数；复跑命令：`bash mortal/run_ladder_r3.sh`（脚本内含每任务完整 env 重置，~29 分钟出结果）。

### 16.6 两个必修的残留

1. **~~`best.pth` 自第 2 轮起再没写过~~（已修，10-11）**：原因—— state 里的 `best_perf` 一路继承第 1 轮在随机对手下的 `{1.0, 90.0}`，`better` 条件永不成立。现 `mortal/train.py` 已实现自动重置：
   - 存盘时多写一个字段 `best_perf_baseline = config['baseline']['test']['state_file']`（当时的卫冕者）；
   - 加载时若命中三者任一→ 重置为哨兵 `{avg_rank:4, avg_pt:-135}` 并记日志：① 手动开关 `control.reset_best_perf = true`；② 旧 state 里没有该字段（来源不明，保守重置，正好修好现有的 round2/round3 盘）；③ 卫冕者路径变了（`abspath` 比较）；
   - 日志样子：`resetting best_perf {...} → {...}, reason: 卫冕者已更换: A → B`。
   - 手动开关 `control.reset_best_perf` 是**一次性令牌**：非 false 值只在「与 state 里记录的上次令牌不同」时生效一次（存盘写入 `best_perf_reset_token`）；想再次强制重置就换一个值（如日期串）。避免忘删开关导致每次重启都重置 best.pth 竞争。
   - 回归测试：`bash tools/training/test_bestperf_reset.sh`（四路：A 旧 state 重置+存盘写入标识 / B 同卫冕者不重置 / C 换卫冕者重置 / D 令牌一次性——D1 生效并写令牌、D2 同值 0 次重置），10-11 全部通过。
   - 效果：换代后首次评测必记为新纪录，`best.pth` 回到“在当前卫冕者面前刷新纪录”的正确语义；但**选模型仍应以 ladder 为准**（`best.pth` 只反映单个固定种子块）。
2. **固定种子块的 `test_play` 不能跨轮比较**：同一批 500 sets 已被多轮选模型使用，逐局配对法给出 Δ=+4.52±2.07，而新种子块 ladder 给出 −0.03±0.59（差 4.5 pt，全是牌山过拟合）。定成败一律用**未参与过选模型的新种子块**。

### 16.7 第 4 轮候选配方（已执行：2 号变量 augmentation 被否，1 号留给第 5 轮）

| 优先 | 变量 | 做法 | 预期与代价 |
|---|---|---|---|
| 1 | 收紧名单 | `player_names_files` 换 top200（长期 avg_pt≥+3.97） | 席位 324,484 → 108k 步/epoch；强度更高但数据更少，建议配 `max_train_steps≈+110k` |
| 2 | 数据增强 | `enable_augmentation = true` | ❌ **已实测无增益**（R4：δ=+0.10±0.59，bootstrap CI[−0.90,+1.07]）→ 第 5 轮改回 false |
| 3 | 跨局视界 | `env.gamma = 0.995` 或 0.99 | 让 Q 覆盖多局，改善信用分配；与第 1 轮 “强度不涨”的推测直接相关 |
| 4 | 扩大时间窗 | globs 改 2019+ 或全量 2.63M，**并同步重训 GRP** | 单 epoch 步数线性变大（全量 ≈ 3.3× 当前），需重算预算 |
| 5 | 加 epoch / 加步数 | ❌ 已证无效（§16.5） | 不推荐 |

选模型的硬规则：每轮结束只认 **ladder 双向 ≥5000 sets 且用新种子块**；胜出才把它的终态当作下一轮的 `baseline.test.state_file`（卫冕者），并同时重置 `best_perf`。

### 16.8 代码评审（10-11）与第 4 轮配方

**评审范围**：全部本地修改（train.py / train_grp.py / player.py / ladder.py / 三份 config / 11 个新工具），含对真实牌谱的回放验证。结果与修复见 [MEMORY.md §16.17](../../memory/MEMORY.md)；对训练侧最重要的三条：

1. **三个自写工具的供托计分错**（只取首局 scores、不扣立直宣言的 1000）→ 每人分数高估 1000×立直次数 → 排名偏高频立直者（实测 **11~13% 半庄名次错**）。已按 `libriichi/src/stat.rs::from_game` 权威口径重写（每局 `start_kyoku` 重置 + `reach_accepted` −1000 + hora/ryukyoku 的 deltas 含其余供托；排名不含终局残供补齐）。**校验方式：用修好的 `check_selection_stats.py` 复算 `runs/selection_r2/r3`，必须与 Rust Stat（TB 里的 avg_rank/avg_pt）**逐位一致**：实测 2.5055/+0.158 与 2.4560/+3.60 ✓**。
2. **白名单 v1 因此有偏** → 已重建 v2（阈值 +2.45；与 v1 相比 **53/400 人进出**）；文件名 `player_ratings_2022_2026H1_v2.csv` / `top_players_400_v2.txt`，旧版存 `_v1` 备查；第 4 轮起用 v2。
3. round2/round3 缺失的 `best.pth` 已补（各自 `mortal.pth` 的副本），配置里的 `[1v3.challenger]` 不再悬空。

> ⚠ 两个口径变更提醒：① `holdout_metrics.py` 已修 worker 分片与 `--seed`（**换 seed 就换抽样**，与修复前的绝对值不可跨版比较）；② `paired_selection_diff.py` / `check_selection_stats.py` 的旧输出（曾报 Δ=+4.52±2.07）作废；三个工具的逻辑已收敛到 `tools/mjai_score.py`，修复后配对法复算 r2/r3 为 Δ=+3.44±2.08（但仍是"被反复使用的选择块"，只作参考）。
>
> 附加：`agg_ladder.py --logs-root` 可对现有牌谱做**按 set 的 block bootstrap** 复核。对 R3 的 20,000 局实测：ρ≈+0.18、bootstrap SE 0.515 < 解析 0.595 → **解析口径偏保守**，δ≈0 结论不变。

### 16.11 第 5 轮配方（top200，**已定稿待批准，未开工**）

唯一变量 = 名单收紧到 top200（v2 口径）；`enable_augmentation` 改回 false（R4 已否）。详细论证与待决选项见 [MEMORY.md §16.22](../../memory/MEMORY.md)。

| 项 | 值 |
|---|---|
| 名单 | `dataset/index/top_players_200_v2.txt`（阈值 ≥+3.9）——**需先用 `build_whitelist.sh --top 200` 生成**（当前只有 top400 v2） |
| 数据量 | 命中 279,713 文件 / 席位 334,907 / 1 epoch ≈ **111k 步**（席位决定样本量） |
| 暖启 | `checkpoints/round4/mortal.pth`（645k，0 非有限张量，现役最强） |
| 预算 | `max_train_steps = 756000`（645k + 111k，恰好一整遍新池）；fp32 ≈ 7.3 h / AMP ≈ 4.6 h |
| 索引 | 新名 `indexes/main_r5_whitelist_top200_v2.pth`（名单变必须换名） |
| 判定 | ladder **vs round4/645k**，双向各 5×500 sets，新种子块 **100000+/110000+**；判进 δ>0 且超 ±0.6 pt |
| 沿用 | env 全套重置、`env_check` 预检、三重防护（finite / histogram / 退出码守卫）、链式脚本单会话跑完 |
| 待拍板 | ① AMP 开关（4.24 vs 6.7 it/s，风险已被守卫框住）② 是否在 4 张空闲 P100 上并行跑 `gamma=0.995` 对照 |

派生与启动（批准后执行，沿用 R3/R4 的成熟做法）：`tools/training/make_round5_cfg.sh`（由 config.round4.toml sed 派生，只改名单/索引/预算/路径/augmentation）+ `mortal/run_round5.sh`（`cp round4/mortal.pth → round5/mortal.pth` 暖启，训练到点后自动判定）。

### 16.10 第 4 轮 NaN 事故与三重防护（10-11）

一句话：**恢复续跑后 1000 步内 AMP+BN 污染导致全状态 NaN**；训练崩在 TB 的 histogram 上，链条又带着 NaN 模型去判定跑了 12 分钟。完整时间线与扫描数据见 [MEMORY.md §16.20](../../memory/MEMORY.md)。

已固化的防护（后续轮次自动继承，因为都改在 train.py / 链脚本模板里）：

| 防护 | 位置 | 行为 |
|---|---|---|
| finite 守卫 | `mortal/train.py` 存盘段 | 参数或本周期 loss 非有限 → `logging.error` + `SystemExit(1)`，**不覆盖 checkpoint**，并提示回滚到 `mortal_step_*.pth` |
| histogram 守卫 | 同上 | 先过滤非有限值再 `add_histogram`，为空则跳过 → NaN 不再打崩训练进程 |
| 退出码守卫 | `mortal/run_round4.sh`（后续轮同型） | 训练非 0 退出即打印回滚指引并 `exit`，**不进判定** |

标准恢复动作（本次实际执行）：

```bash
cd /mnt/data8/mortal/Mortal/mortal
# 1) 找最新干净快照（finite 检查；best.pth 优先，因为它带最新 best_perf 与 baseline 标识）
python -c "
import torch
for f in ['checkpoints/round4/best.pth','checkpoints/round4/mortal_step_000500000.pth']:
    s=torch.load(f,map_location='cpu',weights_only=True)
    bad=sum(1 for m in ('mortal','current_dqn','aux_net') for k,v in s[m].items() if not torch.isfinite(v).all())
    print(f, s['steps'], '非有限张量', bad)
"
# 2) 回滚 + 关 AMP（config 里 enable_amp=false）后重跑同一链脚本
cp checkpoints/round4/best.pth checkpoints/round4/mortal.pth
tmux new -d -s r4_fix 'bash run_round4.sh >> logs/round4_resume.log 2>&1'
```

吞吐代价（fp32 实测）：**4.24 it/s**（AMP 为 6.7，-36%）、GPU 显存 2.7→5.2 GB、利用率 91–94%（瓶颈已从 CPU 管道移到 GPU）。三重防护本轮全程未触发（16:21 之后 0 条 non-finite）。

### 16.9 第 4 轮（已完成）：augmentation —— 判定无增益

唯一实验变量 = `enable_augmentation = true` + `augmented_first = true`（席位视角增强；冒烟验证：同一文件增强前后**样本数不变、obs 张量不同** → ×2 来自两遍遍历）。基础设施：白名单 v2 + 新 `file_index`；暖启自 `round2/mortal.pth`（424k）。

| 项 | 值 |
|---|---|
| 配置 | `mortal/config.round4.toml`（`tools/training/make_round4_cfg.sh` 派生） |
| 预算 | `max_train_steps = 645000`（+221k 步 ≈ 0.91 遍增强样本；v2 池 1 遍 ≈ 244k 步） |
| 链式脚本 | `mortal/run_round4.sh`：env_check → 暖启 → 训练 645k → 双向各 5×500 sets（新种子 80000+/90000+）→ agg_ladder 汇总 |
| 会话/日志 | tmux `r4_chain`；`logs/round4_chain.log`、`logs/main_r4_train.log`、`logs/ladder_r4/` |
| 实际耗时 | 累计 18.8 h 墙钟（AMP 段 6.1 h → NaN 事故 → fp32 段 4.24 it/s 重跑 10h）；训练 10-12 02:11 到预算、判定 02:39 完成 |
| **判进结果** | δ = **+0.10 ± 0.59 pt**（t=0.16，95%CI[−1.07,+1.26]；block bootstrap SE 0.508、CI[−0.90,+1.07]）→ **未过 2σ，augmentation 无增益** |

---

> **手册维护**：数据规模、路径、关键参数发生变化时，请同步更新本文件与 `tools/tenhou_data/TASKLIST.md`，两者互为配套。
