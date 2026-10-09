# Mortal 从零训练完整方案（Tesla P100 环境）

> ⚠️ **状态（2026-10-09）**：本文的数据基线（42,587 局）、路径、时间估算均**已被实跑推翻**。请以 [TRAINING_MANUAL.md §14 远程实跑记录与修正](./TRAINING_MANUAL.md#14-远程实跑记录与修正2026-10-0710-09) 与 [MEMORY.md §16](../../memory/MEMORY.md) 为准；本文仅作配方骨架与历史参考保留。

> **创建时间**：2026-10-02
> **训练环境**：torch-2.6.0+cu118 · Tesla P100-PCIE-16GB（服务器）
> **数据现状**：42,587 局 mjai 牌谱（2009 全年 + 2026-01/02 完整 + 2026-03 部分），约 190 MB
> **配套文档**：[MEMORY.md](../../memory/MEMORY.md)（项目全部背景）、[TASKLIST.md](../tenhou_data/TASKLIST.md)（数据管线）

---

## 目录

1. [环境评估结论](#1-环境评估结论)
2. [服务器环境搭建](#2-服务器环境搭建)
3. [数据上传](#3-数据上传)
4. [config.toml 全文（P100 版）](#4-configtoml-全文p100-版)
5. [bootstrap.py（随机 baseline）](#5-bootstrappy随机-baseline)
6. [执行顺序（M0–M5）](#6-执行顺序m0m5)
7. [监控指标与判停标准](#7-监控指标与判停标准)
8. [关键坑位](#8-关键坑位)
9. [各阶段时间估算](#9-各阶段时间估算)
10. [里程碑清单](#10-里程碑清单)

---

## 1. 环境评估结论

| 项 | 判断 |
|---|---|
| torch-2.6.0+**cu118** | 选择正确：cu118 wheel 含 sm_60（Pascal），CUDA 12 系 wheel 对 sm_60 支持不保证，**不要换 cu126**。驱动要求 ≥450.80.02，`nvidia-smi` 确认即可 |
| P100 无 Tensor Core / 无 TF32 | `enable_amp` 收益有限但可省显存，**先开 true**；若 TensorBoard 出现 q 值 NaN/异常再改 false（P100 FP32 ≈10.6 TFLOPS，关掉也没压力） |
| 16GB 显存 | batch 512（官方参考值）无压力，不需要梯度累积 |
| CPU 是真正瓶颈 | 特征编码（Rust/rayon）+ GRP float64 奖励推理在 CPU；建议 ≥8 核，`num_workers` 可升到 2–4 |
| P100 FP64 = 1/2 FP32 | GRP 是 float64 GRU，恰好适配 P100 的强 FP64，GRP 阶段飞快 |

## 2. 服务器环境搭建

假设 Linux + 已有 conda/python 环境（含 torch-2.6.0+cu118）。Windows 服务器见文末备注。

```bash
# ① 确认 torch 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出: 2.6.0+cu118 True Tesla P100-PCIE-16GB

# ② 补齐 Mortal 的三个纯 Python 依赖
pip install tqdm toml tensorboard

# ③ 安装 Rust（若服务器没有）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source ~/.cargo/env

# ④ 编译 libriichi（在仓库根目录，release + lto 编译较重）
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so

# ⑤ 验证
cd mortal && python -c "import libriichi; help(libriichi)"
```

**服务器目录布局**：

```
~/Mortal/                          # 代码仓库（git clone 官方仓库即可）
~/mortal_data/dataset/mjai/...     # 训练数据
~/Mortal/mortal/                   # 运行目录：config.toml + checkpoints/ + tb_log/
```

## 3. 数据上传

数据量很小（当前约 190 MB，2026 补全后也不到 1 GB），直接打包传输。

**本地（Windows，PowerShell）：**

```powershell
cd C:\Maj\Mortal\dataset
tar czf mjai_20261002.tar.gz mjai
scp mjai_20261002.tar.gz user@SERVER:~/mortal_data/
```

**服务器：**

```bash
mkdir -p ~/mortal_data/dataset && cd ~/mortal_data/dataset
tar xzf ~/mortal_data/mjai_20261002.tar.gz
find mjai -name '*.json.gz' | wc -l          # 应为 42,587
# 抽查首行应为 {"type":"start_game"...}
```

> glob 路径先 `find ~/mortal_data/dataset/mjai -name '*.json.gz' | head` 确认实际年月子目录结构后填写。

## 4. config.toml 全文（P100 版）

放在 `mortal/` 目录下，路径按实际用户名修改：

```toml
[control]
version = 4
online = false
state_file = 'checkpoints/mortal.pth'
best_state_file = 'checkpoints/best.pth'
tensorboard_dir = 'tb_log'
device = 'cuda:0'
enable_cudnn_benchmark = true
enable_amp = true            # P100 无 Tensor Core；若 q 值不稳改 false
enable_compile = false       # Pascal 上无收益，勿开
batch_size = 512             # 16GB 显存用标准值
opt_step_every = 1
save_every = 400
test_every = 20000           # 整除约束: test_every % save_every == 0
submit_every = 400

[train_play.default]
games = 800
log_dir = 'train_play'
boltzmann_epsilon = 0.005
boltzmann_temp = 0.05
top_p = 1.0
repeats = 1

[test_play]
games = 3000
log_dir = 'test_play'

[dataset]
globs = ['/home/USER/mortal_data/dataset/mjai/**/*.json.gz']
file_index = 'file_index.pth'    # ⚠ 数据更新后必须删除此缓存重建
file_batch_size = 15
reserve_ratio = 0.0
num_workers = 2              # Linux 下可试 2~4（worker_init_fn 会自动切分文件表）
player_names_files = []
num_epochs = 1
enable_augmentation = false  # 第二轮迭代时改 true
augmented_first = false

[env]
gamma = 1
pts = [6.0, 4.0, 2.0, 0.0]

[resnet]
conv_channels = 192
num_blocks = 40

[cql]
min_q_weight = 5

[aux]
next_rank_weight = 0.2

[freeze_bn]
mortal = false               # 在线阶段必须改 true

[optim]
eps = 1e-8
betas = [0.9, 0.999]
weight_decay = 0.1
max_grad_norm = 0

[optim.scheduler]
peak = 1e-4
final = 1e-4
warm_up_steps = 0
max_steps = 0

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
seed_key = -1
games_per_iter = 2000
iters = 500
log_dir = '1v3'

[1v3.challenger]
device = 'cuda:0'
name = 'mortal'
state_file = 'checkpoints/best.pth'
stochastic_latent = false
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true

[1v3.champion]
device = 'cuda:0'
name = 'baseline'
state_file = 'checkpoints/baseline.pth'
stochastic_latent = false
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true

[1v3.akochan]
enabled = false

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
train_globs = [
    '/home/USER/mortal_data/dataset/mjai/<2009目录>/**/*.json.gz',
    '/home/USER/mortal_data/dataset/mjai/houou4/202601/**/*.json.gz',
    '/home/USER/mortal_data/dataset/mjai/houou4/202602/**/*.json.gz',
]
val_globs = [
    '/home/USER/mortal_data/dataset/mjai/houou4/202603/**/*.json.gz',
]
file_index = 'grp_file_index.pth'
file_batch_size = 50

[grp.optim]
lr = 1e-5
```

## 5. bootstrap.py（随机 baseline）

保存为 `mortal/bootstrap.py`，解决首次训练时 `test_play` 没有 baseline 可用的冷启动问题：

```python
import torch
from datetime import datetime
from os import makedirs, path
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

## 6. 执行顺序（M0–M5）

在 `mortal/` 目录下运行，全程用 tmux/nohup 防断线：

```bash
# M0 数据闭环校验（强烈建议）
cargo build --release --bins && ./target/release/validate_logs /home/USER/mortal_data/dataset/mjai

# M1 GRP 预训练（奖励计算的前置，必须最先）
python bootstrap.py          # 先建随机 baseline
python train_grp.py          # tb 看 loss/val、acc/val，平台期即停（Ctrl+C 即存盘，可断点续训）

# M2 离线 CQL 第一轮
python train.py              # state_file 不存在自动从头训；存在则自动续训
# 盯 test_play/avg_rank：<2.5 即强于 baseline，持续下降说明在进步

# M3 评测
python one_vs_three.py       # challenger=best.pth vs champion=baseline.pth

# M4 轮次迭代（性能不再涨就停）
cp checkpoints/best.pth checkpoints/round1_best.pth    # 归档
cp checkpoints/best.pth checkpoints/baseline.pth       # 卫冕者换代
mv checkpoints/mortal.pth checkpoints/round1_state.pth # ⚠ 移走才会从头训
# 本轮可改 enable_augmentation = true
python train.py

# M5（可选）在线自博弈：online=true + freeze_bn.mortal=true 后
python server.py             # 终端1
python train.py              # 终端2
python client.py             # 终端3，可多开
```

远程看板：`tensorboard --logdir tb_log --bind_all`，本地 `ssh -L 6006:localhost:6006 user@SERVER` 后访问 `localhost:6006`。

## 7. 监控指标与判停标准

| 指标 | 健康表现 | 异常处置 |
|---|---|---|
| `loss/dqn_loss` | 稳步下降后趋平 | 不降 → 查数据 globs / grp.pth 路径 |
| `loss/cql_loss` | 存在且非负（仅离线） | 爆炸 → min_q_weight 降到 1–2.5 |
| `q_predicted vs q_target` 直方图 | 量级接近、同向移动 | q 飞天 → CQL 权重问题 |
| `test_play/avg_rank` | <2.5 并持续下降 | 长期=2.5 → 检查 baseline 是否随机权重 |
| `test_play/avg_pt` | 同步上升（[90,45,0,-135] 制） | — |
| GRP `acc/val` | 快速到平台 | — |

轮次停止条件：新旧 `best.pth` 在 1v3 上（≥4000 局再下结论，麻将方差大）avg_rank/avg_pt 无显著改善。

## 8. 关键坑位

1. **file_index 缓存**：`file_index.pth` / `grp_file_index.pth` 缓存文件列表，每次新增数据上传后必须删除重建；
2. **state_file 自动续训**：第二轮想从头训必须先移走 `mortal.pth`，否则接着旧状态训；
3. **AMP 兜底**：Pascal 无 Tensor Core，若 loss 曲线抖动异常，第一反应改 `enable_amp = false`；
4. **num_workers**：Windows 下 GRP 在 `__init__` 构造有坑，Linux 没有，可从 2 起试；
5. **在线训练**：`freeze_bn.mortal=true` 必改；源码有已知卡死 bug 靠子进程规避，勿"修复"；
6. **整除断言**：`test_every % save_every == 0`、`save_every % opt_step_every == 0`；
7. **合规**：数据只在训练机本地使用，不要入 git、不要对外提供下载（天凤 ToS）。

> Windows 服务器备注：libriichi 产物换成 `riichi.dll → libriichi.pyd`，路径全部换 Windows 风格，`num_workers` 保持 1。

## 9. 各阶段时间估算

### 9.1 估算依据

- 模型 ~10.8M 参数；单个决策步前向 ≈0.64 GFLOPs，batch 512 一步训练 ≈1 TFLOP（含反向）；
- P100 实效算力按 4–6 TFLOPS 估 → **理论 4–8 it/s（开 AMP）**，数据管道拖累时 3–5 it/s；
- 样本量：42,587 局 × ~660 样本/局 ≈ 2,810 万样本 → **batch 512 一个 epoch ≈ 5.5 万步**；
- 开 `enable_augmentation` 后样本/步数 ×2（≈11 万步/epoch）；
- 自博弈/评测对局：P100 推理估 3–8 局/秒（3000 局 ≈ 10–20 分钟）。

**⚠ 实测校准公式（训练启动 1 小时后务必执行一次）**：

```
从 TensorBoard 读实际步增速 it/s，则：
每 epoch 耗时(h) ≈ 55,000 ÷ (it/s × 3600)
```

下表全部按此口径修正。

### 9.2 分阶段估算（标"估"均为按上述假设的推算值）

| 阶段 | 内容 | 预计耗时 | 人工/挂机 |
|---|---|---|---|
| M0 环境 | rustup 安装 + libriichi 编译（lto 全量优化）+ 依赖 | 0.5–1 h | 人工为主 |
| M0 数据 | 190 MB 打包上传 + 解压 + validate_logs 全量校验 | 0.5–1 h | 人工 |
| M1 GRP | ~47 万前缀序列/epoch（≈920 步/epoch），小模型收敛快 | **0.5–2 h**（含观察收敛） | 挂机+偶尔看板 |
| M2 bootstrap | 随机 baseline 生成 | <5 min | 人工 |
| M2 离线一轮 | 5.5 万步 + 每轮 2–3 次 test_play（3000 局/次 ≈10–20 min/次） | **3–6 h/轮**（≥5 it/s 时）；数据管道瓶颈时 6–10 h/轮 | 挂机 |
| M3 1v3 评测 | 2000 局/次 ≈10–20 min；严谨对比需 4000–10000 局 | 0.5–2 h | 挂机 |
| M4 轮次迭代 | 每轮同"离线一轮"；通常 2–4 轮；第二轮起开增强（步数×2） | **1–4 天** | 挂机 |
| M5 在线（可选） | 流程验证 2–4 h；有效增强需 5–10 万自博弈局 | **1–5 天** | 挂机 |
| 并行：2026 续传 | 本地剩余 ~9.4 万局 @1.0s/局 + 分块冷却（计划任务托管） | ~30–36 h | 纯挂机（本机） |

### 9.3 汇总

| 目标 | 墙钟时间 | 其中人工 |
|---|---|---|
| **首个可用模型（M0→M3）** | **≈1–2 天** | 2–4 h |
| 迭代至性能平台（M4） | 再 +1–4 天 | 数小时（每轮检查归档） |
| 在线自博弈增强（M5，可选） | 再 +1–5 天 | 数小时 |
| 2026 数据补全（与本方案并行） | ≈1.5 天 | 0（计划任务自动） |

### 9.4 何时需要更多数据/算力

- 42,587 局只够"跑通闭环 + 出第一个能打的模型"；社区与作者级别的强度（凤凰卓上位）通常需要 50 万–250 万局量级；
- 2026 补全后约 13.7 万局，可支撑 2–3 轮有效迭代；
- 2009–2025 历史（~250 万局，18 个 tar 包已在 `D:\tenhou_hf`）释放到 NTFS 后是强度跃升的主要来源——届时每 epoch 步数线性放大，P100 单卡一轮将变成数百小时级，建议届时按年分批训练或考虑云 GPU。

## 10. 里程碑清单

| # | 里程碑 | 通过标准 |
|---|---|---|
| M0 | 环境就绪 | `import libriichi` 成功；validate_logs 0 错误 |
| M1 | GRP 完成 | `acc/val` 平台期，grp.pth 存在 |
| M2 | 离线第一轮启动 | dqn_loss 下降，avg_rank < 2.5 |
| M3 | 第一轮收敛 | avg_rank/avg_pt 趋平，best.pth 归档 |
| M4 | 迭代增益枯竭 | 相邻轮 1v3 无显著差异 |
| M5 | （可选）在线增强 | 自博弈轨迹持续回流，1v3 仍能涨 |
