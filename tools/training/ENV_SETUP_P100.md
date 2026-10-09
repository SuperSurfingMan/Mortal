# Mortal 训练/推理环境配置手册（P100 服务器版）

> **创建时间**：2026-10-06
> **适用主机**：`00e5ad44ea95`（Ubuntu 18.04.2 容器，内核 6.8，glibc 2.27）
> **硬件**：2× Xeon E5-2690 v4（28 核 56 线程）· 4× Tesla P100-PCIE-16GB · 1× Quadro RTX 8000 48GB · 驱动 580.173.02
> **前置状态**：miniconda3 已装于 `/mnt/data8/miniconda3`，base 环境可用，root 用户
> **配套文档**：[TRAINING_PLAN.md](TRAINING_PLAN.md)（训练方案与 config）、[MEMORY.md](../../memory/MEMORY.md)

---

## 0. 开始前必读：本机三个关键事实

1. **glibc 2.27（Ubuntu 18.04）是全局约束**：conda-forge 社区源的新包已逐步迁移到 glibc 2.28 基线，装完可能报 `GLIBC_2.28 not found`。**创建环境一律用默认 defaults 源**，不要随手加 `-c conda-forge`。
2. **不需要 nvcc / CUDA Toolkit**：PyTorch 的 cu118 wheel 自带全部 CUDA 运行时库，Rust 编译也用不到 nvcc。驱动 580 兼容所有旧 CUDA 运行时，`nvidia-smi` 显示的 "CUDA Version: 13.0" 只是驱动能力上限，与我们要装的无关。
3. **GPU 分工**（详见 §8）：
   - P100 ×4：FP64 = FP32 的一半（4.7 TFLOPS）→ **GRP 训练必须用 P100**（GRP 是 float64 网络）；
   - RTX 8000（sm_75，Turing）：有 Tensor Core、48GB 显存，FP64 只有 1/32 → **只用于主模型训练/推理**（`enable_amp=true` 时加速显著），严禁跑 GRP。

---

## 1. 基础检查

```bash
# 逐项执行，记录结果
conda --version                     # 应 ≥ 4.x
git --version                       # 无则: conda install -y git
gcc --version; g++ --version        # Rust 链接需要 cc；记录是否存在
nvidia-smi -L                       # 记录 GPU 序号↔型号对应关系
df -h /mnt/data8                    # 确认 ≥ 20 GiB 可用（torch ~3G + cargo target ~3G + 数据 ~2G）
free -g                             # 内存
```

**若 gcc 不存在**，二选一：

```bash
# 方案 A：apt 安装（Ubuntu 18.04 已 EOL，源可能 404，先修复源）
apt-get update
# 若上面报 404/找不到 bionic，执行源切换后重试：
sed -i.bak -E 's|https?://(archive\|security)\.ubuntu\.com|http://old-releases.ubuntu.com|g' /etc/apt/sources.list
apt-get update && apt-get install -y build-essential

# 方案 B：不动 apt，用 conda 装编译器（锁旧版保证 glibc 2.17 兼容）
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17
```

## 2. 创建 conda 环境

```bash
conda create -n mortal python=3.12 -y    # 注意：不加 -c conda-forge（见 §0-1）
conda activate mortal
python --version                          # 期望: Python 3.12.x
which python                              # 应指向 /mnt/data8/miniconda3/envs/mortal/...
```

## 3. 安装 PyTorch cu118 与 Python 依赖

```bash
# PyTorch（cu118 wheel 自带 CUDA 运行时，兼容 P100 sm_60 与 RTX 8000 sm_75）
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Mortal 依赖 + numpy（environment.yml 未列 numpy，但 libriichi 与 Python 间传数组必须要有）
pip install tqdm toml tensorboard numpy
```

> **网络慢的替代**（国内服务器）：
> ```bash
> # torch 用阿里云 wheels 页
> pip install torch==2.6.0+cu118 -f https://mirrors.aliyun.com/pytorch-wheels/cu118/
> # 其余包用清华 pypi 镜像
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm toml tensorboard numpy
> ```

**验证（关键步骤，必须全绿再继续）：**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_arch_list())"
```

期望输出：`2.6.0+cu118 True 5`，且 arch 列表**同时含 `sm_60` 和 `sm_75`**。
若 `device_count` 是 5、`get_arch_list` 含 sm_60/sm_75 → torch 就绪。

## 4. 安装 Rust 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source ~/.cargo/env
rustc --version && cargo --version
```

> **若 rustc 执行时报 GLIBC 版本错误**：说明最新 stable 已抬高 glibc 基线，回退旧版即可（libriichi 用 PyO3 0.25，≥1.75 均可）：
> ```bash
> rustup toolchain install 1.81.0 --profile minimal && rustup default 1.81.0
> ```

> **可选：crates.io 国内加速**（服务器访问 crates.io 慢时）：
> ```bash
> cat > ~/.cargo/config.toml <<'EOF'
> [source.crates-io]
> replace-with = 'rsproxy-sparse'
> [source.rsproxy-sparse]
> registry = "sparse+https://rsproxy.cn/index/"
> EOF
> ```

## 5. 获取代码

```bash
mkdir -p /mnt/data8/mortal && cd /mnt/data8/mortal
git clone https://github.com/Equim-chan/Mortal.git
cd Mortal && git log -1 --oneline        # 记录 commit hash，便于复现
```

> GitHub 慢的替代：在本机 `git clone` 后 `tar czf Mortal.tar.gz Mortal` → `scp` 上传解压。

## 6. 编译 libriichi

两条命令 feature 集不同、产物不同，**都要执行**：

```bash
cd /mnt/data8/mortal/Mortal

# ① Python 扩展模块（cdylib，训练/推理用）——首次编译会拉取 crates，约 5-10 分钟
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so

# ② CLI 工具（validate_logs 牌谱校验 / stat 统计）
cargo build -p libriichi --bins --no-default-features --release
ls -lh target/release/validate_logs target/release/stat
```

> ② 若报 flate2 "no backend" 类错误，改为：
> `cargo build -p libriichi --bins --no-default-features --features flate2/rust_backend --release`

## 7. 全链路验证

```bash
cd /mnt/data8/mortal/Mortal/mortal
conda activate mortal

# 7.1 模块导入 + 版本常量（v4 obs 应为 1012×34，动作空间 46）
python - <<'EOF'
import torch, libriichi
from libriichi.consts import obs_shape, ACTION_SPACE
print('torch      :', torch.__version__, '| cuda:', torch.cuda.is_available())
print('libriichi  :', libriichi.__version__)
print('obs_shape v4 =', obs_shape(4), '| ACTION_SPACE =', ACTION_SPACE)
assert obs_shape(4)[0] == 1012 and ACTION_SPACE == 46, '版本常量异常!'
print('== 基础验证通过 ==')
EOF
```

```bash
# 7.2 数据校验（数据上传后执行；42,587 局并行校验，预计十几分钟内）
../target/release/validate_logs /mnt/data8/mortal/dataset/mjai
# 期望：全部 ok / 0 errors

# 7.3 GPU 冒烟（随机张量过一遍前向，确认当前卡可用）
python - <<'EOF'
import torch
x = torch.randn(512, 1012, 34, device='cuda:0')
w = torch.randn(192, 1012, 3, device='cuda:0')
y = torch.nn.functional.conv1d(x, w, padding=1)
torch.cuda.synchronize()
print('GPU 前向冒烟 OK:', y.shape)
EOF
```

## 8. GPU 选择与分工（多卡用法）

Mortal 脚本都是单卡（config 里 `device='cuda:0'`），多卡靠 `CUDA_VISIBLE_DEVICES` 切换。

> ⚠️ **2026-10-09 实测陷阱：不 `CUDA_DEVICE_ORDER` 时，`CUDA_VISIBLE_DEVICES` 的序号跟 `nvidia-smi` 不一致！**
> 本机的 CUDA 默认枚举按**算力优先**（RTX 8000 排第 0），而 `nvidia-smi` 按 PCI 序（RTX 8000 是 2）。逐个取值实测结果：
>
> | `CUDA_VISIBLE_DEVICES` | 默认（算力序）实际落到 | `CUDA_DEVICE_ORDER=PCI_BUS_ID` 时实际落到 |
> |---|---|---|
> | 0 | **Quadro RTX 8000** | Tesla P100（nvidia 0） |
> | 1 | Tesla P100 | Tesla P100（nvidia 1） |
> | 2 | Tesla P100 | **Quadro RTX 8000** |
> | 3 / 4 | Tesla P100 | Tesla P100 |
>
> **做法：固定加上 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，让序号永远等于 `nvidia-smi` 的序号**（下面示例均按此）。否则你以为在用 RTX 8000，实际在跑 P100，慢一截还看不出来。验证方法：启动后看日志里的 `device: cuda:0 (xxx)` 那一行。

```bash
nvidia-smi -L      # 记下序号；本容器：0/1/3/4 = P100，2 = RTX 8000
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # ★ 必加，使序号与 nvidia-smi 对齐

# GRP 训练 → 指定 P100（如 GPU 0）
CUDA_VISIBLE_DEVICES=0 python train_grp.py

# 主模型训练 → 首选 RTX 8000（AMP 走 Tensor Core，48GB 显存），config 保持 device='cuda:0'
CUDA_VISIBLE_DEVICES=2 python train.py
# 或任一 P100（如 GPU 1）
CUDA_VISIBLE_DEVICES=1 python train.py

# 评测（one_vs_three / test_play 单独跑时）→ 再分一张卡，避免与训练争抢
CUDA_VISIBLE_DEVICES=3 python one_vs_three.py
```

| 任务 | 推荐卡 | 原因 |
|---|---|---|
| GRP 训练 | P100（任一） | float64 网络，P100 FP64=4.7 TFLOPS，RTX 8000 仅 0.5（1/32） |
| 离线 CQL / 在线训练 | **RTX 8000 优先** | Turing Tensor Core + 48GB，`enable_amp=true` 下快于 P100 |
| 自博弈 client / test_play | P100 | 推理以 FP32 为主，P100 足够；与训练分卡并行 |
| 并行实验 | 4×P100 各跑一份 | batch 512 只占约 4–6 GB，一卡一实验互不干扰 |

## 9. 常见问题（FAQ）

| 症状 | 原因与处置 |
|---|---|
| conda/pip 装完导入报 `GLIBC_2.28' not found` | 用了 conda-forge 新包。改回 defaults 源重建环境；或锁定旧版本包 |
| `apt-get update` 404 | 18.04 EOL，按 §1 方案 A 切 old-releases 源 |
| torch 下载龟速 | 用 §3 阿里云 wheels 页 |
| `cargo build` 卡在下载 | 配 §4 rsproxy 镜像 |
| `python -c "import libriichi"` 报 undefined symbol / 找不到模块 | 确认 `mortal/libriichi.so` 是最新编译产物；`python` 必须是 mortal 环境的（abi3 ≥3.8 通用） |
| `rustc: /lib64/... version 'GLIBC_2.xx' not found` | 按 §4 回退 1.81.0 工具链 |
| TensorBoard 打不开 | 远程看板：服务器 `tensorboard --logdir tb_log --bind_all`，本地 `ssh -L 6006:localhost:6006 user@server` 后访问 `http://localhost:6006` |
| 训练中 loss 抖动/NaN（RTX 8000 或 P100） | 先把 config `enable_amp` 改 false 复测；P100 本就无 Tensor Core，关掉无性能压力 |

## 10. 环境变量总表（2026-10-09 逐个实测，完整清单）

> 来源均为代码硬证据：`grep -rn environ mortal/*.py` + `grep -rn env::var libriichi/src exe-wrapper/src`；必要性列为实测结果（用 `env -i PATH=... HOME=...` 跑真实链路，日志见 `mortal/logs/env_probe.log`、`mortal/logs/env_map.log`、`mortal/logs/env_e2.log`）。

### 10.1 必需（不设就跑不起来）

| 变量 | 谁读 | 本项目的值 | 不设会怎样 |
|---|---|---|---|
| `MORTAL_CFG` | `mortal/config.py:4` | **绝对路径**指向本轮配置，如 `/mnt/data8/mortal/Mortal/mortal/config.round2.toml`（第 1 轮用的是 `config.offline.toml`） | 默认找相对路径 `config.toml`，本仓库没有该文件 → `FileNotFoundError: 'config.toml'`（实测） |
| `PATH` | shell + `exe-wrapper/src/main.rs:50` | 至少含 `/usr/bin:/bin`（推理打包时还用于定位 `java`） | 命令找不到 |
| `HOME` | pip/rustup/matplotlib 缓存 | `/root` | 缓存目录报错（一般不致命） |
| `python` 必须是 mortal 环境的解释器 | — | `/mnt/data8/miniconda3/envs/mortal/bin/python`（或先 `conda activate mortal`） | 导入不到 torch/ libriichi |
| **工作目录 = `mortal/`** | `config.py`、`libriichi.so`、各相对路径 | `cd $MORTAL_ROOT/mortal` | 在仓库根目录 `import libriichi` 会被 **Rust crate 目录 `libriichi/` 遮蔽**成空命名空间包（实测：`__path__=_NamespacePath([...])`、无任何属性，看起来像“没装”） |

### 10.2 跑 Rust CLI 时才必需

| 变量 | 适用范围 | 不设会怎样（实测） |
|---|---|---|
| `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`（= `/mnt/data8/miniconda3/envs/mortal/lib`） | `target/release/validate_logs`、`target/release/stat` | 两个二进制都链接了 `libpython3.12.so.1.0` → `error while loading shared libraries`，**退出码 127**；加上后退出码 0 |
| 同上 | **训练/评测的 python 进程不需要** | `import libriichi`、GPU 前向+AMP、牌谱解码在最小环境均正常（libpython 已在解释器进程内）→ 旧手册里的“必设”对训练是多余的，留着无害 |

### 10.3 重性能/选卡（建议每次显式设定）

| 变量 | 作用 | 推荐值 | 实测依据 |
|---|---|---|---|
| `CUDA_DEVICE_ORDER` | 消除 CUDA 序号与 nvidia-smi 不一致的歧义 | **`PCI_BUS_ID`** | §8 的映射表 |
| `CUDA_VISIBLE_DEVICES` | 选卡（config 里始终写 `cuda:0`） | GRP→`0/1/3/4`（P100）；主训→`2`（RTX 8000） | 训练日志的 `device: cuda:0 (…)` 行 |
| `RAYON_NUM_THREADS` | libriichi 特征编码的 rayon 全局池（**进程内只初始化一次，必须启动前设**） | **8**（管道吞吐最高） | nw=2 时：=8→3.77 it/s，=16→6.67 it/s（`env_probe.log`）；nw=8/=8 → 7.48 it/s |
| `OMP_NUM_THREADS` | torch intra-op 线程 | **1**（把核让给编码） | `torch.get_num_threads()`=1 生效 |
| `MKL_NUM_THREADS` | MKL BLAS 线程 | **1** | 同上 |
| （补充）`torch` 的 interop 线程不受 OMP 控制 | 实测 `get_num_interop_threads()`=28 | 如需收紧只能代码里 `torch.set_num_interop_threads()` | 一般不必动 |

### 10.4 按场景才需要

| 变量 | 谁读 | 默认 | 说明 |
|---|---|---|---|
| `TRAIN_PLAY_PROFILE` | `mortal/player.py:105`（TrainPlayer） | `default` | 仅**在线模式**用于选 `[train_play.<profile>]` 采样参数 |
| `MORTAL_IS_SUB_PROC` | `mortal/train.py:471` | 未设 | **由 `main()` 自设给子进程，手动设会绕过子进程包裹**（那层包裹是在线模式卡死 bug 的规避手段，不要动） |
| `MORTAL_REVIEW_MODE` | `mortal/mortal.py:27` | `0` | 推理入口的牌谱检讨模式（`1` 开启） |
| `AKOCHAN_DIR` | `one_vs_three.py:24` 写入；Rust `agent/akochan.rs:30` 读 | Rust 侧默认 `akochan` | 开 akochan 对手时必需（包含 `akochan.jar` 的目录） |
| `AKOCHAN_TACTICS` | `one_vs_three.py:25` 写入；`akochan.rs:35` 读 | Rust 侧默认 `tactics.json` | 同上；一般由 config 的 `[1v3.akochan]` 自动注入，无需手写 |

### 10.5 只属于脚本/文档（代码不读）

| 变量 | 用途 |
|---|---|
| `MORTAL_ROOT` | 本文档与 `run_*.sh` 里的路径变量（便利，代码从不读它） |
| `CONDA_PREFIX` | `conda activate mortal` 自动设；本手册用它拼 `LD_LIBRARY_PATH` |

### 10.6 可直接粘的标准启动块（第 2 轮主训练）

> ⚠️ **验证命令必须包含 `cd mortal/`**。直接在仓库根跑 `python -c "import ... libriichi; ...__version__"` 会报
> `AttributeError: module 'libriichi' has no attribute '__version__'` —— 因为根目录的 `libriichi/` 是 Rust crate 目录，会被当成**命名空间包**遮蔽 `mortal/libriichi.so`（`PYTHONPATH` 也救不了：`python -c` 的 `sys.path[0]` 就是 cwd）。

```bash
cd /mnt/data8/mortal/Mortal/mortal                              # ★ 必须在这个目录
export MORTAL_ROOT=/mnt/data8/mortal/Mortal
export MORTAL_CFG=$MORTAL_ROOT/mortal/config.round2.toml     # ★ 必需，绝对路径
export CUDA_DEVICE_ORDER=PCI_BUS_ID                          # ★ 消除序号歧义
export CUDA_VISIBLE_DEVICES=2                                # RTX 8000（PCI 序下就是 2）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=8
export LD_LIBRARY_PATH=${CONDA_PREFIX:-/mnt/data8/miniconda3/envs/mortal/lib}/lib:$LD_LIBRARY_PATH  # 训练可省，跑 CLI 时必需

python env_check.py --cfg config.round2.toml          # ★ 推荐：一条命令把所有项都校了
# 预期最后一行：结论：全部通过，可以启动训练/评测（退出码 0）

# 只要一行快速版也可以：
python -c "import torch,libriichi; print(torch.cuda.get_device_name(0), libriichi.__version__, libriichi.__profile__)"
# 预期：Quadro RTX 8000 0.1.0 release   ← 先确认卡名再开跑
```

`mortal/env_check.py` 逐项校验：cwd 与导入是否被遮蔽 → torch/CUDA/实际选中的卡 → `MORTAL_CFG`、`OMP/MKL/RAYON` 是否到位 → `obs_shape(4)`/`ACTION_SPACE` 常量 → config 能否解析、`dataset.globs` 命中文件数、四个产物目录 → Rust CLI 是否存在；任一项不过就打印修法并以退出码 1 退出。
额外可用品质检查：`CHECK_GPU_NAME='RTX 8000'`（或 `P100`）能反验选卡是否真落到了要的那张卡；已用 `tools/training/selftest_envcheck.sh` 测过 4 种场景（仓库根 / 正确目录 / 缺 MORTAL_CFG / 忘设 CUDA_DEVICE_ORDER），日志 `mortal/logs/envcheck_selftest.log`。

> 另：`python -u` 中的 `-u`（不缓冲日志）不是环境变量，但写日志文件时必须带，否则 `tail -f` 只能看到块缓冲后的片段。

## 11. 完成检查清单

- [ ] `conda activate mortal` 后 `python --version` = 3.12.x
- [ ] `torch.__version__` = `2.6.0+cu118`，`cuda.is_available()` = True，arch 含 sm_60/sm_75
- [ ] `import libriichi` 通过，obs_shape(4) = (1012, 34)
- [ ] `validate_logs` 对全部数据 0 错误
- [ ] GPU 冒烟通过；GRP/训练/评测的分卡规划确定（§8）
- [ ] `tensorboard` 可远程访问

**环境就绪后** → 按 [TRAINING_PLAN.md](TRAINING_PLAN.md) 继续：上传数据（§3）→ 写入 config.toml（§4）→ M1 GRP → M2 离线训练。

---

### 附：本机版本快照（配置完成后回填）

| 组件 | 版本（2026-10-09 实测回填） |
|---|---|
| Python | 3.12.11（conda 环境 `mortal`） |
| torch | 2.6.0+cu118（arch 含 sm_60/sm_75） |
| rustc / cargo | 1.99.0（无需回退 1.81.0，glibc 2.27 兼容） |
| libriichi | `0.1.0` / `__profile__=release`（只有从 `mortal/` 目录导入才能读到；`libriichi.so` 日期 2026-10-06 15:57） |
| GPU | 4×Tesla P100-16GB + 1×Quadro RTX 8000-48GB（PCI 序 2），驱动 580.173.02 |
| CPU / 内存 | 56 线程 / 251 GB |
| Mortal commit | 无（服务器目录不是 git 仓库；代码已打过补丁，清单见 MEMORY.md §16.4） |
