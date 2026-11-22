<p align="center">
  <img src="https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/logo.png" width="550" />
</p>

# Mortal（本地实验改进版）

> **来源**：本仓库基于 [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) —— 一个免费开源的日本麻将 AI，由 Rust 规则引擎 `libriichi` 与 Python/PyTorch 训练侧构成，深度强化学习驱动，代码以 AGPL-3.0 发布。原项目文档见 [mortal.ekyu.moe](https://mortal.ekyu.moe)。
>
> 本仓库保留原项目代码骨架，在其之上自 2026-09 起开展了一系列训练方法实验、诊断工具与工程扩展。**核心结论一句话**：围绕「模仿学习到顶之后如何变得更强」做了约 30 条实验线，其中只有两个增益通过预注册统计判据被采纳（数据选择 **+2.86±0.59**、S6c 特征换代 **+0.422**），其余大量方向（IQL、自博弈、搜索/EI、守卫、EV 重排、数据扩窗等）以同样的判据**闭环关闭（null/负）**——每条线的结论、数据与坑位均有存档可查。

## 一、相比原版 Mortal 做了哪些工作

### 1. 主模型训练迭代（BC → 离线 RL → 特征注入）
- 完成 5 轮 BC（行为克隆）训练迭代：数据选择与名单收紧、奖励口径消融（`pts=[2,1,0,-3]` 对齐）、augmentation、NaN 三重防护与 fp32 口径治理。
- 引入离线强化学习：**IQL Stage A**（credit_mode 跨局信用分配）与 **Stage B**（真 IQL：next_obs bootstrap + expectile VHead + AWR），含价值尺度漂移的根因分析与 4 臂探针修复。
- 特征注入三代实验：S6a（紧迫度特征，隔离工作区）→ S6b（读牌危险度注入，FAIL 后代码回退）→ **S6c（自家速度特征，+0.422，77k sets 复判转正）**。现役最强模型为 **S6c/956k**（v4 obs + 2 行速度特征）。

### 2. 数据工程
- 全树 **263 万局**牌谱（2009-02 ~ 2026-10，天凤凤凰场 houou4 与 M-League），2026 牌谱 names 修复与验收工具、强度榜 v3 口径。
- holdout 规范（202609+，同 seed 同 batch 对比）、训练索引 manifest 化、数据扩窗/补量探针（D1/D2，均判负或无结论后关闭）。

### 3. 读牌模型线（`tools/read_model/`）
- 从零实现对手**听牌/待牌后验**预测模型 v1 → v2s/v2x（输出层因子化、逐牌列打分头、tenpai 监督），tenpai AUC **0.667 → 0.946**（A2b，预注册判据全满足；口径修正后 0.9685）。
- 配套：P0 口径修正、A3 纯 numpy 人类规则基线、Bayes 后验上界求解、零模型防守天花板审计、11 个诊断脚本与 **71 项门禁回归测试**（专治假 PASS）。
- 「注入主模型」路线经四臂对照判 FAIL，已冻结为独立资产与归档文档。

### 4. 推理期决策改造
- 防守守卫 **G3**：`Q' = Q − λ·risk`，含 λ 剂量曲线（λ=10 峰值 +0.778）与 risk_floor gating；后续更严复验判无结论，**+0.78 结论已正式撤回**。
- 推理期 EV 重排（SP 前瞻 + risk，判「结构冗余」不进 ladder）；状态依赖押し引き守卫 Guard-P2（离线无支配配置，不烧 ladder）。

### 5. 自博弈与在线训练线
- 离线档 A 自博弈（8000 半庄加权）两轮；在线自博弈 **M5**（修复原版 `test_play` 在 online 模式下**卡死多年**的问题）与 6 处「一开开关就坏」的工程问题；纯四席真自博弈 4M 及 top-5 探索、降方差 baseline、代际对手池等 7 个变体。
- 结论：强度**全部等于起点**，机制已查清（真实动作差异 < 单次回报噪声），整线按预注册判据关闭。

### 6. 搜索 / 专家迭代（EI）线
- 三档搜索设计（自博弈 MC / 1-ply 结构化 expectimax / 短 rollout）与成本、止损线规划。
- 短 rollout EI 落地：状态物化自检（1052/1052 分支可推进）、混合终值、多采样 M=16；SP-EV teacher 20k 对照一致率仅 0.537 → shadow rollout 判死，全部过程入档。

### 7. Oracle 全知诊断线
- 全知铳牌动作损失测量、押し引き（攻守判断）平衡设计、EV_push−EV_fold 攻击项、oracle guiding（invisible_obs 通道 + α 退火）。
- 用途：量化模型与人类在攻防判断上的真实差距（例如「模型在有人听牌处的铳牌选择已与人类持平」即出自此线）。

### 8. 评测与对照体系
- **ladder 标准臂**（16,000 sets，可分辨 δ≈0.6）+ block bootstrap CI 判据（下界 >0 才算赢）。
- **one_vs_three + akochan 外部锚点**：绝对强度锚定 2,388 局 **+0.00 ± 1.68** ⇒ ≈ 天凤凤凰场常客平均（距 p95 还差 5.4 pt）。期间定位并修复 akochan pipe 三处协议不匹配。
- **AI similarity 诊断**：与官方 Mortal v3 / 4.1c 同局对照（S6c 80.35% / v3-ref 80.98% / 官方 4.1c 78.42%），实证「一致率衡量拟人度、与强度反向」；对 4.1c 报告 7,791 个决策点榨取完整 Q 表，量化蒸馏上限（ρ 中位 0.846，剩余可补信息极少）。
- 行为画像口径统一（与 `libriichi` 的 `stat.rs` 对齐），证伪「模型太保守」的旧结论。

### 9. 风格模仿微调（`tools/style_ft/`）
- bc_style 白名单一致率微调流程（白名单提取/一致率评测/行为画像/选优），含职业选手风格配置（如 okada_sayaka、toi_takaharu）。

### 10. MortalSimulator 对局模拟器（`simulator/`）
- Windows 自包含运行包：3 家 Mortal AI + 1 名人类 + 读牌助手 + 实时张量检查器，双击启动，一键环境脚本；分发快照见 `dist/` 与根目录 zip。

### 11. 工程基建与文档
- **结构化项目记忆**（`memory/`）：`MEMORY.md` 六版全量梳理 + 各实验线 DESIGN 文档，新会话可从 §0「快速接手」直接续作。
- **技术原理报告**（`docs/TECHNICAL_REPORT.md`）：系统架构/数据管线/模型结构/奖励与 gamma 语义/训练目标三代演化/统计评估/工程可靠性，通俗版 + 严格版双写。
- 事故复盘与 **80+ 坑位台账**（如 resume 后 lr=0 静默冻结、TF32 破坏 fp32 口径、worker RSS 泄漏等），全部有修复与门禁。
- `libriichi` 本地改动：akochan agent pipe 应答时机修复、`AKO_TRACE` 埋点（env 门控）；`train.py` 扩展（dataset.extra_logs / extra_repeat、续训守卫、TF32 显式关闭）。

## 二、重要文件与目录

| 路径 | 说明 | 来源 |
|---|---|---|
| `libriichi/` | Rust 规则引擎：和牌/向听/牌效率/观测编码/对局竞技场（arena）/统计 | 原版，含本地修复与埋点 |
| `mortal/` | Python 训练与推理主代码。原版骨架（`train.py`、`model.py`、`client.py`、`server.py` 等）+ 本地大量扩展：`defense_guard.py`、`guard_risk.py`、`guard_state.py`（推理期守卫）、`ev_rerank.py`、`selfplay_gen.py`、`ladder.py`、`holdout_metrics.py`、`diag_ai_similarity.py`、`one_vs_three.py`、59 个实验配置 `config.*.toml`、各代模型权重 `checkpoints/`。**训练必须在 `mortal/` 目录下执行** | 原版骨架 + 本地扩展 |
| `tools/` | 实验工具集：`read_model/`（读牌模型全链 + 单一维护的 README）、`style_ft/`（风格模仿）、`training/`（编排）、`data_audit/`、`oracle_*`、`ei_*`（搜索线）、akochan 对接（`akochan_pipe_shim.py` 等）、`mjai_score.py`、`label_ceiling.py` 等 | 本地新增 |
| `simulator/` | MortalSimulator 对局模拟器（GUI/引擎/测试），见其 `README_FIRST.md` | 本地新增 |
| `s6a_isolated/` | S6a 隔离实验工作区，「完全隔离工作流」的范例与 v5 复现路径（patches + 首层手术工具） | 本地新增 |
| `memory/` | 项目上下文记忆：`MEMORY.md`（§0 快速接手 / 结论 / 坑位台账）+ 各线 `DESIGN_*.md` | 本地新增 |
| `docs/` | `TECHNICAL_REPORT.md` 技术原理报告（本地新增）+ 原版 mdBook 用户文档（`book.toml`、`src/`） | 原版 + 本地 |
| `log-viewer/` | mjai 牌谱查看器（静态页面） | 原版 |
| `exe-wrapper/` | Windows 可执行文件包装器（Cargo workspace 成员） | 原版 |
| `dist/` | 发布产物（`MortalMajsoulSuite` Windows 套件），以权重 sha256 标识版本 | 本地产物 |
| `dataset/` `logs/` `runs/` `backups/` | 牌谱数据、训练日志/产物、实验运行目录、备份（大体量，多数不入 git） | 本地产物 |
| `MortalSimulator_20261023.zip` | 模拟器分发快照（651 MB，附 `.sha256` 校验） | 本地产物 |

## 三、从哪里开始读

- 想快速接手项目现状 → [`memory/MEMORY.md`](memory/MEMORY.md) §0（TL;DR：当前坐标 / 七条结论 / 三条命令）
- 想理解技术原理 → [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)
- 想了解读牌线全部细节 → [`tools/read_model/docs/README.md`](tools/read_model/docs/README.md)
- 想跑对局模拟器 → [`simulator/README_FIRST.md`](simulator/README_FIRST.md)

## License

### Code
[![AGPL-3.0-or-later](https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/agpl.png)](https://github.com/Equim-chan/Mortal/blob/main/LICENSE)

Copyright (C) 2021-2022 Equim

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

本仓库相对上游的全部修改同样以 AGPL-3.0-or-later 发布。

### Logo and Other Assets
[![CC BY-SA 4.0](https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/by-sa.png)](https://creativecommons.org/licenses/by-sa/4.0/)

（Logo 版权与署名归原项目所有，来源见页面顶部链接。）
