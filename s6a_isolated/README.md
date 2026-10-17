# S6a 隔离工作区（2026-10-17）

S6a = 读牌×主模型结合的第一步：**逆转紧迫度特征注入**（obs version 4 → 5，+6 通道）。
本目录是**完全隔离**的实验版本：所有 S6a 专属代码、编译产物、配置与脚本都在这里，
共享树 `mortal/` + `libriichi/` 已还原为原版（v4），iql_beta / read_model 等并行实验不受影响。

## 目录

```
s6a_isolated/
├── code/                     # 复制自 mortal/*.py 的完整运行目录（cwd）
│   ├── libriichi.so          # ★ v5 编译产物（唯一使用它的地方）
│   ├── model.py / engine.py / env_check.py   # 含 v5 分支（共享树已还原为 v4 原版）
│   ├── train.py              # 取 git HEAD 版本（不含 iql_beta 的日志改动）
│   ├── config.s6a.toml / config.s6a_smoke.toml
│   ├── checkpoints/          # 本地输出 + 指向共享树的输入软链
│   │   ├── grp_h128.pth -> ../../../mortal/checkpoints/grp_h128.pth
│   │   ├── mortal_step_000200000.pth -> ...（baseline.test）
│   │   ├── reward_ablation -> ...（B/756k 手术源 + 判定对照）
│   │   └── s6a/ s6a_smoke/   # 真实输出
│   ├── indexes/              # 本地索引（s6a_r4_v2.pth，data_version=s6a-r4-v2，首次自动构建）
│   ├── runs/  logs/          # 本地 TB / 日志
├── tools/
│   ├── expand_ckpt_v5.py     # v4 -> v5 首层手术
│   └── test_urgency_features.py  # 六层回归（用 S6A_CODE 指向 code/）
├── patches/
│   └── v5_obs_features.patch # libriichi v5 源码 diff（177 行；共享树已还原，重编时应用）
├── run_s6a_smoke.sh          # 冒烟链（回归 + 手术 + 2k 步 + 行为门禁）
├── run_s6a.sh                # 正式链（回归 + 手术 + 867k + holdout + 双向 ladder + bootstrap）
└── README.md
```

## 运行

```bash
# 冒烟（GPU2，约 15 min）
tmux new -d -s s6a_smoke 'bash /mnt/data8/mortal/Mortal/s6a_isolated/run_s6a_smoke.sh'

# 正式轮（训练前自动等待 GPU2 空闲；最长 8h）
tmux new -d -s s6a 'bash /mnt/data8/mortal/Mortal/s6a_isolated/run_s6a.sh'
```

## 隔离边界

| 项 | 共享树（其他实验） | 本隔离区 |
|---|---|---|
| `libriichi.so` | v4（原版重建） | v5（+6 紧迫度通道） |
| `mortal/model.py` / `engine.py` / `env_check.py` | HEAD 原版 | 含 `case 5` 分支 |
| 训练代码 cwd | `mortal/` | `s6a_isolated/code/` |
| checkpoints / runs / logs | 各自目录 | `code/checkpoints|runs|logs` |
| 输入权重 | — | 经软链只读共享树的 grp/baseline/B/756k |
| 数据 | 共享只读 | 同（绝对路径 globs） |

## 复现 v5 编译（如需重建）

```bash
cd /mnt/data8/mortal/Mortal
git apply s6a_isolated/patches/v5_obs_features.patch   # 应用 v5 源码（libriichi/src 两文件）
cargo build -p libriichi --lib --release
cp target/release/libriichi.so s6a_isolated/code/libriichi.so
git checkout -- libriichi/src                          # 共享树还原
cargo build -p libriichi --lib --release               # 共享树重建 v4
cp target/release/libriichi.so mortal/libriichi.so
```
