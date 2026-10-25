# Mortal 项目上下文记忆文档

> **用途**：沉淀项目历史、关键结论、可复现配置与坑位，供新会话快速接手。
> **本版**：2026-10-18 结构化重构（按「快速接手 → 系统与数据 → 实验史 → 现状 → 工程规范 → 索引」重排，合并旧版 §12/§16/§17 重复叙述；旧版全文 `git show HEAD:memory/MEMORY.md`，重构前 2136 行）。**2026-10-19 增补 §3.15**（读牌模型复核与改进路线 RM1~RM11）：推翻 §3.12/§3.14 关于「tenpai_auc 与规则打平」与「读侧已近天花板」的两条判决，并同步更新 §0.1/§0.2/§4.1/§4.2/§5.2。**同日增补 §3.15.8**（局部手牌信息视角：架构与损失是否充分，新增 RM12~RM16 与坑位 49~55）：判定为「**集合级充分、顺序级不充分**」，时序流实测空转（配对反事实斜率 0.0144）；同时把全部诊断脚本从 `/tmp` 收进 **`tools/read_model/diagnostics/`**（含 `_common.py` 路径解析层、`README.md` 方法学纪律、`results/*.json` 原始产物）。**同日再把读牌模型的全部记录整体迁出本文**，改由 **`tools/read_model/docs/README.md`** 单一维护：§3.12 只留三句话摘要与链接，旧章节号（§3.12/§3.14/§3.15.x）映射见该 README **§16**；本文回归「主模型 + 数据 + 评测 + 工程规范」的项目级职责（1261 → 722 行）。**同日晚些时候 P0 零成本项（RM1~RM3+RM16）已全部落地并复跑验收**（实现与全部数字见该 README **§13.3**）：读法收口 `model.factorized_wait_probs`、val_score v2、CE 拆分/决策分层/R-P 双口径、`order_metrics.py` 常规诊断；回归 52 项 ALL PASS；并**实证否决**了复核清单第⑦条（chiitoi 0.109 确为 recall、向听桶 5 从不预测，坑 44 已修订）。**2026-10-20 更新**：① 读牌全量代码审计（22 py/8370 行 + 11 diagnostics/1684 行，`AUDIT_20261020.md`）修订两条地基——**A3 规则法并非顺序盲**（经 `early_discard_mask`/`timeline_features` 消费 `tl_tokens`），真顺序盲地板下 A2b 扣地板复现率由 41%/67%/−12% 上修为 **66.6%/81.9%/+31.0%**（§3.12-3、§0.2、§4.1 同步改口径；立直层仍是主要缺口）；resume 后 lr=0 静默冻结与门禁假 PASS 两个事故已修复并门禁化（坑 57/58）。② **本地 master 22 个提交已推送 `origin`（`eb4f660..69eaa6e`），§1.8「本地 ahead」状态解除**。③ A2c 200k 训练推进至 ~152k/200k，中途 best val_score **2.0552**（8192 eval 口径）已优于 A2b 的 2.0924。**2026-10-21 结项**：北京时间 00:46 完整跑完 200k 步并正常退出（末段 104k resume 起 15.11h，稳态 0.60–0.63 s/步，LR 余弦归零，全程无中断/发散/Traceback），best val_score **2.052388 @153k**（8192 口径：alloc_ce 0.6492 / wait_hit1 0.2033 / tenpai_auc 0.9705），RM4/RM5/RM7 三条判据点估计全过；**同日官方 20000 固定集三报告 + `order_metrics.py` 验收全通过（RM4 AUC 0.9698 / 标定 1.0364 / cond_ce 2.7532；RM5 hit@1 0.1993、立直 0.1724；RM7 val 2.0524 < 2.0924），A2c 取代 A2b 成为注入基线**（验收用 `best.pth`(153k) 而非 `last.pth`(200k，val 2.0613)；细节见 `tools/read_model/docs/README.md` **§13.4a**）；训练监控与 ETA 复盘见 `tools/read_model/docs/A2C_TRAINING_REPORT_20261021.md`（详见 §0.1/§3.12/§4.2）。**2026-10-21 读牌 P1b 两轮均 FAIL：RM12 v1 20k b=0.0131、RM14 next-tedashi aux 20k b=0.0038，均未启动 200k；P1b 判据已与历史 verdicts 解耦，下一步转显式 order 特征/输入消融（README §13.5~§13.7）。** **2026-10-22 注入 P0 三探针完成并判决 GO**（`tools/read_model/inject_p0.py`，产物 `diagnostics/results/p0_inject_{20k,100k}.json`）：① **P0-A 冻结主模型 backbone 的读牌探针**（5-fold by file，按文件分组避免同局泄漏）——`phi` 仅 alloc_ce **0.7588**/hit@1 **0.0775**/tenpai AUC **0.7993**，A2c 同批 **0.6517/0.2003/0.9701** ⇒ 主模型表示不含可泛化的读牌能力（5-fold std ≤0.005，非 split 噪声；探针 train hit@1 0.42 vs val 0.08 ⇒ 只在训练分布内可解码）；② **P0-B 决策分歧**（20000 样本）——主模型 top-1 点炮率 **1.32%** ≈ 人类 **1.40%**，A2c 最安全牌 **0.27%**（5× 空间）；`fixable`（主模型打危险牌且 A2c 有安全替代）**1.20%（239 个，30/30 文件，每场 ~8 个）**，集中在 turn≥8（2.28%）与有立直家（2.59%）；③ **P0-C deal 判据**——官方 20k 的 `deal_pure` **0.0108**（n_deal 1476，CI[0.0039,0.0202]）与 100k 的 **0.0147**（n_deal 7507，CI[0.0103,0.0198]）差 **36%** ⇒ 20k 的 deal 点估计不可作判据；④ **P0-D 零 `tl_tokens`**——alloc_ce **+0.0161（+2.5%）**、hit@1 **+0.0022**、AUC **−0.0251** ⇒ 读牌能力基本保留，P1 预计算可直接复用主模型 `GameplayLoader` 的 obs（无需 replay_file 对齐）。**2026-10-22 S6b 注入 P1 四臂 20k 探针判决 FAIL**：A0/A1/A2(安慰剂)/A3(限域) 同数据同种子（756k→776k），202609 20000 集 `deal_top1` **0.0120/0.0121/0.0118/0.0123**（A1≈A2，<1σ）、holdout top1 **0.8135/0.8143/0.8156/0.8136**；新通道 `|w|`mean A1 **0.0158** vs A2 **0.0062**（权重长起来但行为不变）⇒ **信息效应为零，S6b 关闭**（不再跑 100k/200k、不试逐家 3×34/action-bias 变体）；下一步转 RM10 轨迹监督或数据侧扩窗。详见 README §13.14.8。**2026-10-22 回退**：S6b 判 FAIL 后，`mortal/{model,dataloader,train,holdout_metrics,ladder}.py` 已整体恢复为注入前 `3a793a1`（commit `7bb5b7d`），删除 `config.s6b_a0~a3.toml`/`config.s6b_smoke.toml`；主模型不再依赖读牌侧任何特征（B/756k holdout top1=0.8223 / dqn_loss=0.3881 与注入前一致）。读牌侧自此**冻结为归档资产**，不再作为主线。
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
| 当前阶段 | IQL 路线已关闭；S6a 已判定；**2026-10-22 S6b 注入 P0 三探针 GO**（主模型 `phi` 缺读牌：tenpai AUC 0.7993 vs A2c 0.9701；决策 fixable 1.20%；100k `deal_pure` 0.0147）；**P1 四臂 20k 探针判决 FAIL（A1 真特征 ≈ A2 安慰剂，信息效应为零）⇒ S6b 关闭**；**主模型代码已回退到 `3a793a1`（commit `7bb5b7d`，无注入）**；**读牌模型已判决，注入基线由 A2b 换为 A2c（10-21 官方验收全通过），P0 口径修正已落地，P1（RM4+RM5+RM7）已完成并验收**，全部记录已从本文迁出**——实验史 / Bayes 上界 / 复核修正 / 改进路线 RM1~RM16 / 坑位 / 产物台账统一在 `tools/read_model/docs/README.md`，本文只留 §3.12 摘要。**读牌 P0 零成本项（RM1~RM3+RM16）已于 2026-10-19 落地并复跑验收**（README §13.3），注入特征口径可冻结。**A2c 结项（北京时间 10-21 00:46）**：200k 步完整跑完、正常退出、无中断（4 次运行：32.4k NW16 内存事故降 NW8 迁 RTX 8000 → 103k 命中 lr=0 坑 58 空转 1400 步 → 104k resume 起 15.11h 跑完，稳态 0.60–0.63 s/步），best val_score **2.052388 @153k**（8192 eval 口径：alloc_ce 0.6492 / wait_hit1 0.2033 / tenpai_auc 0.9705 / meanP(听) 0.1519 vs 实际 0.145 / 向听 acc 0.5590），终局 eval(200k) 2.0613 ⇒ **验收用 `best.pth`(153k)**；RM4（AUC≥0.9685）、RM5（hit@1≥0.16）、RM7（val_score<2.0924，−1.9%）三条判据点估计全部越过（8192 口径与 A2b 同集可比）。**同日 A2c 官方验收全通过**（`tools/read_model/docs/README.md` **§13.4a**）：202609/20000 固定集 seed 20261016，RM4（AUC **0.9698**／标定 **1.0364**／cond_ce **2.7532**）、RM5（hit@1 **0.1993**、立直 **0.1724**）、RM7（val_score **2.0524** < A2b 2.0924，−1.9%）均达标 ⇒ **A2c 取代 A2b 成为读牌注入基线，特征口径冻结**；对 A3 规则法立直 hit@1 由「不显著」转为显著 net 优，nlt 项由 rule 胜转为打平。顺序级仍 partially 未解：立直单张扣地板复现 31%→66%、非立直 82%→93%，但反事实斜率 b 仍 0.009（时序流空转）、立直窗口超地板仍显著为负 ⇒ P1b 已试两版：RM12 v1 20k b=0.0131、RM14 next-tedashi aux 20k b=0.0038，均 FAIL；不启动 200k，下一步转显式 order 特征/输入消融（README §13.5~§13.7）。同轮修 `order_metrics.py` 判据段 `NameError`（坑 61）并用 A2b 数值回归验证。 |
| 数据 | 全树 2,630,640 局（200902~202610，16 GB）；训练窗 813,145 局；holdout = 202609+ |
| 硬件 | Ubuntu 18.04 容器，56 线程，251 GB 内存，4×P100-16GB + 1×RTX 8000-48GB |

### 0.2 一句话结论

- **唯一统计显著的增益**：R2「top400 白名单 + GRP h128」= **+2.86 ± 0.59 pt/半庄**（t=4.83）。
- **已关闭的路线**：同分布数据 reshuffle（R3/R4/R5/消融 A/hp_ablation）、真实分数差奖励 C（−5.61）、IQL Stage A 跨局信用（−1.03）、IQL Stage B 真 IQL（v3 +0.646，CI 跨 0）、S6a 紧迫度特征（+0.182，CI 跨 0）、**S6c 自家速度特征（+0.268，bootstrap CI[−0.76,+1.27]；holdout top1 +0.40pp 但 dqn_loss +3.3%）、2019-2021 数据混入（δ=−1.40，bootstrap CI[−2.39,−0.44]）**。
- **当前主线（2026-10-22 起，读牌侧冻结）**：主模型侧三线——① **推理期防守守卫 G3**（放铳 14.05% vs 人类 12.55%、流局 10.9% vs 15.3%）：`Q'(t)=Q(t)−λ·risk(t)`（A2c 后验、连续、全对手；只改打牌 0..33），**已实施**（`mortal/defense_guard.py` + engine/ladder 钩子，commit `6fab9ce`）；20k 单步扫描 λ=20 令 `deal_top1` 0.0132→0.0081（−39%）、改选 100% 更安全；**2026-10-23 16:17（北京）剂量曲线全部完成（每臂双向 64000 半庄）：λ=5 +0.550（CI 跨 0）、λ=7.5 +0.712（CI[+0.07,+1.36]）、λ=10 +0.778（CI[+0.13,+1.42]，峰值）、λ=20 −0.496、λ=50/τ=0.03 −2.663 ⇒ GO，λ=10 最优，λ=7.5 备选**；② **ladder 扩容到 10000~20000 sets**（现有 5000 sets 的 CI ±0.6~1.0，看不见 <0.5 pt 的改进）；③ **S6c SP 盲区补全**（`SHANTEN_THRES=3`，shanten≥4 占 8.72% 决策点、SP 通道全零）。数据侧扩窗 2019+ 并重训 GRP 作并行长线。
- **读牌侧（全部详情见 `tools/read_model/docs/README.md`，本文 §3.12 有三句话摘要）**：2026-10-19 的只读复核**推翻两条旧判决**——「P(听牌) 过标定 6.3×、注入前必须重标定」实为读错输出（改 `1−sigmoid(wait_logits[34])` 即 AUC 0.9461→**0.9685**、标定比值 **0.998**，可直接当概率注入）；「读侧已近天花板」证据不足。当前判定是「**局部信息集合级充分；顺序级非立直层已捕捉约 8 成、立直层仍是主要缺口**」（10-20 审计修订「顺序级几乎为零」旧说法，见 §3.12-3）。改进路线 RM1~RM16，其中 RM1~RM3 + RM16 **零训练成本——已于 2026-10-19 全部落地并复跑验收**（README §13.3：全部 `wait_probs` 消费方统一为因子化读法；正确口径下 net tenpai AUC **0.9685 显著胜** rule 0.9468（CI 不跨 0），wait CE 优势全部来自条件分布 +0.2167 nats；`wait_ce_active` 官方值改为 3.4809=0.5848+2.8961；`order_metrics.py` 把 §12 三项顺序级指标沉淀为常规判据，A2b 基线 RM12~RM15 全 FAIL 存档；复核第⑦条 precision/recall「bug」经复跑实证否决）。**2026-10-21 A2c 官方验收全通过**：RM4（AUC 0.9698／标定 1.0364／cond_ce 2.7532）、RM5（hit@1 0.1993、立直 0.1724）、RM7（val 2.0524）均达标 ⇒ **A2c 取代 A2b 成为注入基线**；顺序级仍 partially 未解（立直单张 31%→66%、非立直 82%→93%，但反事实 b 仍 0.009）⇒ P1b 两版（RM12 v1、RM14 aux）20k 均 FAIL（b=0.0131/0.0038），不启动 200k；下一步显式 order 特征/输入消融（README §13.5~§13.7）。**2026-10-22 注入 P0（GO）**：主模型 `phi` 的读牌探针 AUC **0.7993**（A2c 0.9701）、hit@1 **0.0775**（0.2003）⇒ 主模型表示确实缺读牌；决策 `fixable` **1.20%**（239/20000，30/30 文件）；主模型点炮 **1.32%** ≈ 人类 **1.40%**、A2c 最安全 **0.27%**；20k 的 `deal_pure` 不可作判据（0.0108 vs 100k **0.0147**）；零 `tl_tokens` 只掉 2.5% ⇒ P1 预计算可直接复用主模型 obs。详见 README §13.14。**2026-10-22 S6b P1 判决 FAIL**：四臂 20k（A0/A1/A2/A3）202609 `deal_top1` 0.0120/0.0121/0.0118/0.0123、holdout top1 0.8135/0.8143/0.8156/0.8136，**A1≈A2（安慰剂）**；新通道 `|w|`mean A1 0.0158 vs A2 0.0062 ⇒ 权重长起来但行为不变，信息效应为零。S6b 关闭，转 RM10/数据侧。详见 README §13.14.8。**2026-10-22 回退**：`mortal/` 5 个文件恢复为注入前 `3a793a1`（commit `7bb5b7d`），删除 5 个 `config.s6b_*`；`mortal/` 下 grep 无 `extra_channels|danger_*` 残留；B/756k holdout top1=0.8223 / dqn_loss=0.3881 / agari 29/29 与注入前完全相同。读牌侧转**冻结资产**（A2c 归档，不再注入、不再改模型）；`tools/expand_ckpt_v6.py`/`eval_s6b_arms.py` 依赖已移除接口仅供查阅，`inject_p0.py`/`precompute_danger.py` 不受影响。
- **纪律**：每轮单变量；判定只认「全新种子块 + 双向同牌山 ≥5000 sets + block bootstrap CI」；CI 下界 >0 才算赢。**指标口径纪律（README §11.2）**：报 `wait_ce_active` 必须同时给 `−logP(t)` 与 `cond_ce`，否则跨模型比较无效。

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
| **读牌模型（暗牌/听牌/待牌预测）的一切** | **`tools/read_model/docs/README.md`**（本文仅 §3.12 摘要 + §13/§14 指针） |

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
- **当前状态**：`master` 与 `origin/master` **已同步**（10-20 一次性推送 22 个提交 `eb4f660..69eaa6e`，含读牌 P0/P1、两次事故修复与全量审计修正；此前本地一度 ahead 22）。推送前先 `git status -sb`；SSH 主机键缺失会导致 push 静默挂起（§5，用 `timeout` 包裹防挂）。
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

### 3.11 S6a / S6c：特征注入（2026-10-17~23）

**S6a（obs v5，紧迫度）**：v5=v4+6 行——与排序后 1/2/3/4 名的点差 `d1..d4` + `overtake_gap` + `lead_gap`；全部由 `state.scores` 确定性导出、归一化 [0,1]，不做剩余局数假设。首层 Conv1d 手术（新通道零初始化，前向与 v4 逐位等价）；**完全隔离**在 `s6a_isolated/`（独立 code/`libriichi.so`/index/ckpt/runs/logs，输入软链只读共享树；共享树已还原 v4）。回归 17/17、冒烟 ALL PASS；正式轮 867k / 7.8h / 4.5 batch/s；holdout 0.8160、lse +0.467、R²−0.995；δ=+0.182、解析 CI[−0.99,+1.35]、bootstrap CI[−0.82,+1.20] → **无结论**。归因：6 个特征全是 v4 已有分数通道的确定性函数 → **零新信息**，ResNet 本就能算；该 null 只否定“重排已知信息”，不否定 S6 方向。种子块 240000/250000 已消费。

**S6c 实施（2026-10-23，纯 Python 2 行，不动 Rust）**：下面那段原分析里「纯重算版不单烧轮」的前提被两条新证据修正——① v4 obs 的向听 one-hot（第 **862..868** 行）**在打牌后才更新**（`state/update.rs` 的 `update_shanten` 挂在 `dahai` 上），所以 3n+2（自摸后 14 张）状态里仍是摸牌前的值：实测 3n+1 状态 **507/507** 与 obs 逐位相等，3n+2 只有 74.5% 相等、且失配恒为 `obs = 真值 + 1`；② `state.shanten` 被 clamp 到 6，早期 5~8 向听的分辨率丢失。故「当前向听」不是纯冗余；「有效牌数」在 `SHANTEN_THRES=3` 盲区（8.7% 决策点）完全缺失。

- **特征**（`mortal/s6c_features.py`：libriichi 同款 tomohxx 向听表的 numpy 向量化，与 Rust 单测 **19/19** 逐位一致）：ch0 = `clamp(最优向听,0,8)/8`；ch1 = `max_候选(有效牌剩余枚数)/136`，剩余 = `4 − tiles_seen`（第 **835** 行，实战口径），候选 = 打牌后达到最优向听的每一张（3n+2）或当前手牌（3n+1），有效 = 摸进后向听前进 1；`best ≤ −1`（已和了）两通道置 0。34 列广播，obs 1012→1014。
- **口径回归**（`tools/test_s6c_features.py`，ALL PASS）：`obs[0:4]`=自家手牌整数、`seen>=own` 100%、3n+1 向听与 obs one-hot 全等、3n+2 滞后 ≤1、`attach` 不改前 1012 行且新行常量、缺表报错含修法。
- **成本**：**0.148 ms/样本**（2579 样本批，按文件批量；小批更贵），8 worker 下约占单核 2%。
- **管线**（复用 S6b 的 `extra_channels` 形态，配置键 `[control].extra_features='s6c'`）：`model.Brain(extra_channels)` + dataloader 按文件批量算并拼 + engine 推理时拼（防守守卫仍吃原始 1012 行）+ train/holdout/ladder/player/one_vs_three/mortal.py/client.py 全部由各自 ckpt 的 `[control]` 经 `s6c_features.resolve` 解析（旧 ckpt → 0/''）；手术 `tools/expand_ckpt_v6.py`（零初始化，**phi max|diff|=0**）。
- **透明性已验**（2026-10-23 12:22）：`config.s6c_base.toml`（关）与 `config.s6c.toml`（开）在同一 holdout（202609 / seed 20261009 / 24×512）上 `holdout_sample_digest=9c5362bf99d60661` 与 `top1=0.8127 / dqn_loss=0.3496 / agari 149/149 / lse +0.522` **逐位相同** ⇒ 手术 ckpt 与 B/756k 等价（注：0.8127 是这套抽样口径的基线，与 §0.2 记的 0.8223 不同种子/批数，比较必须在同一 digest 下做）。
- **编排**：`config.s6c{.toml,_probe,_smoke,_base}.toml` + `run_s6c.sh`（preflight/regression/surgery/transparent/smoke/probe/formal/ladder，tmux 承载）+ `run_s6c_chain.sh`（等 formal 结束 → 自动双向 ladder → 写 `logs/s6c/SUMMARY.md` 汇总）；20k 探针门禁 = holdout top1 与**同 digest 基线**比（容差 `TOP1_TOL=0.005`）+ `tools/s6c_channel_usage.py --ref` 证明首层两新通道的**权重变化量分位 ≥ 原始 p10**；正式 200k 后双向 5000 sets ladder（**λ=0** 隔离模型变量，**新种子块 280000/300000** 各 5×500 sets——260000/270000 已被守卫剂量曲线每向 8000 sets 消费）。
- **冒烟（2026-10-23 12:26~12:41，2k 步暖启）**：`obs shape: (1012,34) + extra 2 (s6c)`；**2.72~2.79 batch/s**（同日 s6b A0 无特征 2.59、A2 有特征 1.62~2.02 → 特征开销不构成回退，机器本身已比 10-17 的 4.5 慢一截）；test_play 200 半庄 avg_rank 2.47 / avg_pt 2.7、`behavior: agari 0.2277 houjuu 0.1404 fuuro 0.3095 riichi 0.1915`；`s6c_channel_usage --ref B756k_s6c_init`：初始 ΔL2=0 → 2k 步后 **5.01e-2 / 4.85e-2**，落在原始通道 ΔL2 分布的 **15.5% / 12.2% 分位（≥ p10）** ⇒ 特征真在学（S6a 式“没学没用”在 2k 步已被排除）；`ladder 1 set + guard λ=10` 跑通（验证 1014 obs + 守卫吃 1012 行）。
- **正式轮判定（2026-10-23 15:48 ~ 10-24 05:59，956k / 180k 步 / 13.1 h）**：全程 4.15~4.26 batch/s，**0 条非有限/异常**；每 50k 一次 test_play 行为画像 agari .2240/.2214/.2177/.2167、houjuu .1299/.1254/.1200/.1227、fuuro .3074/.2950/.2829/.3109、riichi .1933/.1957/.1950/.1855（放铳略降、其余在噪声内）。
  - **holdout（同 digest）**：756k 0.8127 / 776k 0.8115 / **956k 0.8167（+0.40pp ≈ 1.1σ）**；但 `dqn_loss` 0.3496→**0.3613**（+3.3%）、R² −1.2166→−1.2911、`var_q/var_t` 0.434→0.472（Q 尺度漂移）、lse +0.522→+0.525、agari 149/149。
  - **通道使用度**：ΔL2 **9.786e-1 / 4.106e-1** → 分位 **100% / 58.1%**（向听通道是全部 1014 个输入里学得最多的）⇒ 特征真被模型用上，不是 S6a 式的“没学没用”。
  - **ladder（λ=0，新种子块 280000/300000，双向各 5×500 sets = 20000 半庄）**：A +0.301（SE 0.845）、B −0.234（SE 0.843）→ **δ=+0.268**，解析 SE 0.597、t=0.45、CI[−0.90,+1.44]；**block bootstrap（2500 sets/向，ICC 0.186/0.175）SE 0.509、CI[−0.76,+1.27]**。
  - **判决：无结论（不采纳）**。两个独立量都为正但都不显著（holdout +1.1σ、ladder +0.45σ）；按纪律（CI 下界>0 才算赢）主模型保留 B/756k，956k ckpt 归档。若要分辨 +0.27 需 ~99k sets（工具估算），当前 5000 sets 的分辨率只有 ~±1。
- **状态（2026-10-24 08:40）**：S6c 全程跑完并判**无结论**；产晶 `checkpoints/s6c/{mortal(956k),best(900k),B756k_s6c_init,mortal_step_*}`、`logs/s6c/{SUMMARY.md,ladder_boot.log,ladder/ladder_summary.log}`。代码已入库但默认关闭（`extra_features=''`），想复核只需 `STAGE=ladder bash run_s6c.sh`（SEED_A/SEED_B 可换新块）。
- **20k 探针（2026-10-23 13:22~15:43，2.74~2.84 batch/s / 141 min）**：`obs shape (1012,34)+extra 2`，首次 test_play 在 760k（avg_rank 2.499 / avg_pt 0.225，behavior agari .2178 houjuu .1318 fuuro .3155 riichi .1753，与 B 基线 .217/.130 一致，无行为漂移）。**holdout（同 digest）：top1 0.8115 vs 基线 0.8127（−0.12pp ≈ 0.34σ）、dqn_loss 0.3479 vs 0.3496（更好）、R2 −1.2063 vs −1.2166、lse +0.505 vs +0.522、agari 149/149**；新通道 ΔL2 **62.4% / 16.8% 分位（均 ≥ p10）** ⇒ 特征确在学习。结论：**门禁 PASS（容差内不退），已启动 200k 正式轮**。
  - 修了两个探针工具 bug：① `grep -ao 'top1='` 会误中 `agari_top1=`（曾把 top1 读成 149，门禁假过），改为「空白+top1=」正则；② 严格「≥ 基线」在 12,288 样本上会被噪声卡死（SE≈0.35pp），改为 `TOP1_TOL=0.005`（~1.4σ）并打印 Δ。另：**脚本运行中不要改 `run_s6c.sh`**（bash 增量读取会报莫名 syntax error，本次已踩一次）。

**S6c 原分析（2026-10-18，S6b 已关闭、S6c 已按上方新证据单独实施）**：acceptance/期望巡目/牌型大多已被 v4 obs 显式编码——向听 6 槽 one-hot、自家待牌 34 维 one-hot、SP 每候选 `required_tiles` 2×34、逐巡 tenpai/win/EV 3×17、`doras_owned`/`tiles_seen`。SP 仅算 shanten≤3（`SHANTEN_THRES=3`），盲区 shanten≥4 占决策点 **8.72%**（3,622 点实测：0:14.8%/1:25.4%/2:31.6%/3:19.6%/≥4:8.72%）。结论：**纯重算版不单烧轮**；优先级 S6b（读牌后验，真新信息）> S6c（盲区补全）> S6d（GRP 扩展）；建议 S6b+S6c 合并成一次 **v6** 注入，先 30k 步探针（~2h，门禁 holdout/行为画像），下一块种子 260000/270000。

### 3.12 读牌模型（`tools/read_model/`）——完整记录已迁出

读牌模型的**全部实验记录、判决依据、复核修正、改进路线（RM1~RM16）、坑位与产物台账**，
自 2026-10-19 起统一维护在 **`tools/read_model/docs/README.md`**；诊断脚本与原始数字在
`tools/read_model/diagnostics/`。本文只保留下面三句话摘要与链接；历史文档中的旧章节号
（§3.12 / §3.14 / §3.15.x）对照见该 README 的 **§16 章节映射**。

**2026-10-19 P0 落地**：RM1（因子化读法收口 `model.factorized_wait_probs`，v1/v2/v2s/ONNX
统一语义）+ RM2（`val_score` v2 加 tenpai AUC 与标定项，版本护栏）+ RM3（CE 拆分/决策分层/
R-P 双口径）+ RM16（`order_metrics.py`）已实现，三份报告 + Bayes 对照按新口径重跑，
回归 52 项 ALL PASS；全部实现细节、验收数字与两条改判见 README **§13.3**。

1. **基线已定**：**A2c（v2s + RM4/RM5，best@153k，官方 20000 集验收全通过）已取代 A2b 成为注入基线**，特征口径冻结（数字见 README §13.4a）。与纯规则法 A3 对决：alloc/cond_ce/tenpai_auc/hit@1 全面显著胜过 A3，**立直 hit@1 由 A2b 时代的「不显著」转为显著 net 优（0.1724 vs 0.1121）**，nlt 项由 rule 胜（+0.039）转为打平。尚未注入 `mortal/`，强度只认路径 C 注入
   + ladder 在线验收，离线指标仅诊断。A2b（val 2.0924）归档保留作对照；A2c 结项报告见
   `tools/read_model/docs/A2C_TRAINING_REPORT_20261021.md`。
2. **两条旧判决已被推翻**（2026-10-19 只读复核，零训练成本）：① 「P(听牌) 过标定 6.3×、注入
   前必须重标定」实为**读错了输出**——BCE 标定的是 logit 而项目全程读 softmax，换
   `1−sigmoid(wait_logits[34])` 后标定比值 **0.998**、AUC **0.9461→0.9685**；② wait_ce 的
   「+0.758 nats 优势」**71% 是听牌因子伪影**，真实待牌条件分布优势仅 **+0.217**。⇒「读侧已近
   天花板」**证据不足**；实测空间在输出层参数化、H2 缺逐牌列特征（冻结骨干探针 hit@1 +35%
   相对）与训练规模（只用了全量数据的 **4.05%**）。
3. **「精确的局部 > 粗糙的整体」成立：集合级已充分，顺序级非立直层约 8 成、立直层仍缺口**（10-20 审计修订）：集合级局部推断（切了 x ⇒ x±1 不太可能
   有）A2b 已学 **75~79%**（A3 仅 13~19%）、CE 增益确实集中在邻接目标家手切的列；而切牌
   **顺序级**——立直层真值效应最大（+0.1895 张）。⚠️ 2026-10-20 审计修正了地板口径：A3 规则法
   并非顺序盲（消费 `tl_tokens`），旧「顺序盲地板」作废；换真顺序盲地板
   （`rule_predict(order_blind=True)`）后 A2b 扣地板复现率为全样本 **66.6%** / 非立直层
   **81.9%** / 立直层 **+31.0%**（旧值 41%/67%/−12%，立直层由「零」上修为「约 1/3」但仍是
   主要缺口）；区域级（同色 ±2 窗口）**显著为负**、配对反事实斜率 **b=0.0144** ⇒ 时序流空转。
   **2026-10-21 A2c 验收后重测（同一工具/同一地板）**：全样本 **85%**、非立直 **92.7%**、立直 **+66.2%**
   （立直层翻倍）；区域级 all **+0.0062（CI 跨 0）/ 非立直 +0.0155（CI>0）** 已转正，但**立直窗口仍显著为负
   （−0.0656，CI[−0.1134,−0.0142]）**、反事实斜率反降至 **b=0.0093** ⇒ 「逐牌列通路被激活、时序注意力仍空转」
   （tile 头之功，非 `tl_tokens`）→ P1b 两版实测：**RM12 v1**（dataset 侧车 tl_turn/tl_age/tl_consumed/tl_flags + tile-indexed 时序图）20k b=0.0131、r²=2.76%，RM13 86.9%、RM14 52.2%、RM15 −0.0857；**RM14 next-tedashi 辅助头** 20k b=0.0038、r²=0.28%，RM13 82.0%、RM14 51.2%、RM15 −0.0812。两版均 FAIL，整体 val 略优（2.1278/2.1236）但时序通路未被因果使用（obs 已带顺序，辅助头可绕过 e_col）；不启动 200k。P1b 判据已与历史 verdicts 解耦（`p1b_verdicts`：主 RM12/RM15，不退 RM13≥92.696/RM14≥66.2；RM14 +50 低于 A2c 66.2 无效）。逐格定位：A2c 立直窗口负 excess 集中 索7/万6/万3，详见 README §13.7。

4. **2026-10-22 注入 P0 三探针：P0 判决 GO → P1 四臂 20k FAIL**（`tools/read_model/inject_p0.py`；产物 `tools/read_model/diagnostics/results/p0_inject_{20k,100k}.json`；日志 `mortal/logs/read_model/p0_inject_{20k,100k}.log`）。**核心问题**：读牌输入与主模型是同一个 `encode_obs(version=4)`，注入不引入新信息，只补「主模型没学到的函数」——所以先测「主模型表示里有没有」，再决定是否烧 30k 步。**P0-A（主模型 `phi` 冻结探针，5-fold by file）**：linear alloc_ce **0.7761±0.0012** / hit@1 **0.0785±0.0089** / tenpai AUC **0.7273±0.0078**；mlp128 **0.7588±0.0014 / 0.0775±0.0182 / 0.7993±0.0049**；A2c 同批 **0.6517 / 0.2003 / 0.9701** ⇒ 主模型表示缺读牌（尤其「对手是否听牌」AUC 差 0.17）；探针 train hit@1 0.42（linear）远高于 val 0.08 ⇒ 只在训练分布内可解码、不泛化。**P0-B（决策分歧）**：主模型 top-1 与 A2c 最安全牌一致率 **28.5%**；deal 主模型 **1.32%** / A2c **0.27%** / 人类 **1.40%**；`fixable` **1.20%（239/20000，30/30 文件，每场 ~8 个）**，turn≥8 **2.28%**、有立直家 **2.59%**、turn<8 仅 0.49%。**P0-C（判据可信化）**：20k `deal_pure` **0.0108**（n_deal 1476，≈16 个点炮事件，CI[0.0039,0.0202]）vs 100k **0.0147**（n_deal 7507，CI[0.0103,0.0198]）**差 36%** ⇒ 此后 deal 类判据一律 ≥100k。**P0-D（零 `tl_tokens`）**：alloc_ce 0.6678（+2.5%）、hit@1 0.2025（+0.0022）、tenpai AUC 0.9451（−0.0251）、deal_pure 0.0129（+0.0020）⇒ 主模型 `GameplayLoader` 不提供 `tl_tokens` 也能预计算。**P1 设计已定**：注入形态 **2 行 obs**（row 1012 = 三家 union 危险度 `Σ_j P(听_j)P(待牌_j=t)`，row 1013 = `P(至少一家听牌)`），不是 102 维后验；工程路径 = `GameplayLoader(version=4)` → A2c(`tl=0`) 预计算（按文件存 fp16 npz）→ `mortal/dataloader.py::populate_buffer` 查表 + `np.concatenate`（**不动 Rust**）→ 首层手术 `Conv1d(192,1012,3)→(192,1014,3)` 新通道零初始化；实验臂 A0 基线 / A1 真特征 / **A2 安慰剂（同 batch 内 shuffle）** / A3 限域（仅 turn≥8）；判据 RM11 决策级 + holdout top1≥0.816 + ladder 260000/270000 双向 CI 下界>0。

5. **2026-10-22 S6b P1 四臂 20k 探针：判决 FAIL，注入关闭**（四臂 `config.s6b_a{0,1,2,3}.toml`，cwd `mortal/`，GPU2/1/3/4，同数据同种子 20261022，756k→776k；ckpt `mortal/checkpoints/inject_s6b/A{0,1,2,3}.pth`；判决脚本 `tools/read_model/eval_s6b_arms.py`，产物 `diagnostics/results/s6b_arms_20k.json`）。**四臂设计**：A0 基线（B/756k 副本，1012）/ A1 真特征（2 行 A2c 危险度+P(听)，1014）/ **A2 安慰剂**（同 cache，但在 16 文件批内跨文件全局置换）/ A3 限域（turn<8 置零）。**决策级（202609 20000 样本）**：`deal_top1` A0 **0.0120** / A1 **0.0121** / A2 **0.0118** / A3 **0.0123**（1σ≈0.077pp）；`agree_a2c` 0.2792/0.2806/0.2816/0.2827；`fixable` 0.0109/0.0109/0.0107/0.0111（A2c 最安全牌 0.0033）。**holdout（24×512）**：top1 **0.8135/0.8143/0.8156/0.8136**（1σ≈0.35pp）、dqn_loss 0.4006/0.3531/0.3547/0.3516（A1≈A2≈A3 均低于 A0，是「多 2 通道后的训练动力学」而非信息效应）、agari_top1 164/164 全臂。**新通道权重**：`|w|`mean A1 **0.0158** / A3 **0.0147** / A2 **0.0062** —— 真特征的梯度信号是安慰剂的 2.5×，**权重确实长起来了，但行为不变**。**判决**：A1 ≈ A2（安慰剂，全部指标 <1σ）⇒ **信息效应为零，S6b 关闭**；不再跑 100k/200k，不试逐家 3×34 / action-bias 变体（坑 64 已证「权重长起来 ≠ 行为改变」）。**归因（与 S6a/P0-A 合起来读）**：主模型 `phi` 确实缺读牌（探针 AUC 0.80 vs 0.97），但 **BC 目标不需要显式读牌**——人类舍牌行为已隐含其对危险的判断，2 行同源特征是冗余的（坑 72）。下一步转 **RM10 轨迹级监督**（唯一新监督）或数据侧扩窗。**回退（2026-10-22，commit `7bb5b7d`）**：`mortal/{model,dataloader,train,holdout_metrics,ladder}.py` 恢复为注入前 `3a793a1`（`git diff 3a793a1 -- <5 files>` 为空），删除 `config.s6b_a0~a3.toml`/`config.s6b_smoke.toml`；`mortal/` 下 grep 无 `extra_channels|danger_*` 残留。回归：B/756k holdout top1=0.8223 / dqn_loss=0.3881 / agari 29/29（与注入前完全相同）。`tools/expand_ckpt_v6.py`/`eval_s6b_arms.py` 依赖已移除接口，仅供查阅；`inject_p0.py`/`precompute_danger.py` 不受影响。A0~A3 ckpt 与 danger_cache 保留在磁盘（gitignore），A1~A3 需 checkout `27632a2` 才能重新加载。

### 3.12b 数据扩窗：2019-2021 混入（2026-10-24 起）

S6c 判无结论后，主线转数据侧（同分布 reshuffle 已到头，分布扩展是唯一没试过的新信息源）。

**年度审计**（`tools/data_audit/year_style_audit.py`，每年随机 4000 文件，单位：每席位每半庄）：

| 年 | 立直 | 副露 | 和了 | 自摸占比 | 流局/局 | 和了收入 | 局时长 | v2名单席位 |
|---|---|---|---|---|---|---|---|---|
| 2013 | 1.700 | 5.924 | 2.284 | .402 | .1622 | 6424 | 47.97 | 4.2% |
| 2016 | 1.746 | 5.590 | 2.263 | .399 | .1606 | 6548 | 47.96 | 6.0% |
| 2019 | 1.831 | 5.380 | 2.242 | .400 | .1623 | 6792 | 48.21 | 9.5% |
| 2020 | 1.880 | 5.293 | 2.241 | .405 | .1651 | 6831 | 48.29 | 13.2% |
| 2021 | 1.889 | 5.303 | 2.248 | .404 | .1644 | 6793 | 48.40 | 16.8% |
| 2022 | 1.891 | 5.277 | 2.230 | .410 | .1662 | 6808 | 48.35 | 23.7% |
| 2025 | 1.883 | 5.314 | 2.243 | .404 | .1564 | 6769 | 47.95 | 28.3% |

**结论**：① 牌风是**单调但温和**的漂移——2019-2021 与 2022 只差 ~3%（立直 +3%、副露 −2%），2013-2016 才差 ~10%/12%；② **玩家池换代**：用 2022+ 名单打旧年只覆盖 9.5~16.8% 席位（文件覆盖 33~44%），必须**按年代自建名单**（`top_players_400_2019_2021.txt`：扫 577,757 文件 / 4,762 人 / ≥300 局 574 人 / top400 阈值 avg_pt≥2.90，与 v2 只重叠 **18.8%**）——用年代名单后旧年文件覆盖升到 **57.4%/58.7%/66.6%**。

**数据池**（`tools/data_audit/build_mix_index.py`，`indexes/data_mix_2019_2021_20pct_v1.pth`）：B/756k 的 492,810 文件（2022-2025，top400_v2；**注意无 2026**——names 修复前 2026 全被名单滤掉）+ 旧年 577,757 候选→名单保留 354,564→按文件名 sha1 抽 20% = **70,588 文件** = **563,398（+14.3%）**；旧年占比 12.5%（2019 19,811 / 2020 25,819 / 2021 24,958）。

**探针 D1 设计**（`config.mix_v1.toml` + `run_mix.sh` + `run_mix_chain.sh`）：B/756k 暖启（不改 obs/奖励/超参），100k 步（756k→856k）；判据 holdout（同 digest 9c5362bf99d60661，基线 0.8127）+ ladder 5000 sets（λ=0，新种子块 **320000/340000**）。旧年局的 GRP 奖励沿用现有 `grp_h128.pth`（分数→顺位映射与年代关系小，作探针可接受；若混入有效再重训 GRP）。

**D1 判决（2026-10-24 17:31）：负迁移，关闭 2019-2021 混入**：

| 测量 | B/756k | D1/856k | 差 |
|---|---|---|---|
| holdout top1（同 digest） | 0.8127 | 0.8105 | −0.22pp（噪声内） |
| holdout dqn_loss | 0.3496 | 0.3721 | **+6.4%（变差）**，R² −1.217→−1.359 |
| ladder A（新当挑战者，10000 半庄） | — | **−1.224** ± 0.852 | — |
| ladder B（旧当挑战者，10000 半庄） | — | **+1.579** ± 0.834 | — |
| **δ（新−旧）** | — | **−1.402** | 解析 SE 0.596、t=−2.35、CI[−2.57,−0.23]；**bootstrap SE 0.497、CI[−2.39,−0.44]** ⇒ 变弱 |

- 行为：agari .2178→.2144、houjuu .1266→**.1192**、fuuro .2922→.2890、riichi .1823→.1852——**不是“学了旧年代爱鸣牌的风格”（副露反而降），更像是价值估计变差导致决策变差**。
- 可能的成因（未拆开）：① 年代牌风/强度差异（旧年副露高、立直低）；② 旧年名单阈值 avg_pt≥2.90 比 v2 宽松（强手密度低）；③ **GRP 奖励年代错配**（GRP 只见过 2022-2025，旧局的局初期望顺位偏）。三条都指向“旧年代不适合当增量数据”。
- **它不否定**：同年代的数据补齐（见下 D2）、更强过滤（旧年 top200）、以及数据质量加权。也提醒：D1 的 −1.40 比 S6c 的 +0.27 更接近“可判”，说明**数据侧变量的效应量比特征侧大一个量级**。
- 运行环境备注：ladder 期间其他用户的 4 个 3D 检测任务（tmux `m3fd_*`）占满 5 张卡，训练速率从 4.27 降到 2.64 batch/s；ladder 每 job 内部两引擎同卡受影响对称 → 不引入胜负偏。

**下一探针 D2（待启动）：补 2026H1 训练窗**——审计发现 B 的索引里**一个 2026 文件都没有**（names 修复前被名单滤掉），而修复树 `houou4_names/20260[1-8]` 有 **113,430** 文件、names 覆盖 100%、**v3 名单覆盖 69.3% 文件（席位 25.3%）** ⇒ 预计可加 ~78,600 文件（+16%），且**同年代无迁移问题**。单变量设计同 D1，仅换数据段；建议先做 100k 探针。

**状态（2026-10-25 01:40）**：D1 已跑完并判**负迁移**（δ=−1.40，bootstrap CI[−2.39,−0.44]）⇒ 2019-2021 混入关闭；**D2（补 2026H1）已在跑**：索引 `indexes/d2_2026_v1.pth` = 492,810(B) + **78,233**（houou4_names 113,430 → v3 名单保留 68.97%）= **571,043（+15.87%）**；`config.d2_2026.toml` + `run_d2.sh` + `run_d2_chain.sh`，因 RTX 8000 被其他用户占用，训练在 **GPU0（P100，2.65 batch/s）**，预计 ~10.5 h 到 856k（北京 ~10-25 20:05），之后自动 holdout + ladder（新种子块 **360000/380000**）。

### 3.13 项目准备期（2026-10-01~10-09，压缩）

- 10-01：完成 Rust+Python 架构分析；确认作者不公开权重、官方 Training 文档全为空占位；实测天凤 `list.cgi`/scc/`0/log`，年度 zip 404，滚动日归档可用。
- 10-02：发现 HF `tenhou-to-mjai` 2009–2025 全量；自建 2026 管线；实测 HF 2009/2026 包合并成功；D 盘 exFAT 簇 256 KiB 导致小文件膨胀 64×（13 万文件逻辑 521 MiB 实占 33 GiB，全量需 ~620 GiB）→ 用户拍板只保留 18 个年度 tar（16.86 GiB），不转换。
- 10-06：服务器环境落地（conda mortal/py3.12.11/torch 2.6.0+cu118/rust 1.99，`libriichi.so`+CLI 编译通过）。
- 10-07：HF 18 包 merge-hf + prune --apply → 2,630,640 局；GRP 30k 步（21 min）；R1 200k 步（22.9h）。
- 10-09：首次复盘——评测饱和、best.pth 陷阱、吞吐瓶颈、代码补丁清单；建本地 git 基线、玩家强度榜、holdout_metrics/ladder/bench_pipe；R2 启动。
- 采集预算存档（2026-10-01 精算）：2026 YTD 实测 129,475 局、raw+mjai≈1.53 GiB；历史 2009–2025 约 340–680 万局，仅粗估 40–80 GiB（未落地；最终走 HF 年度包）。

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
| 读牌模型（S6b 上游） | ✅ 资产冻结（2026-10-22 起不再注入） | 信息侧主线；全部记录在 `tools/read_model/docs/README.md`（§9 上界 / §10 实验 / §11 复核 / §12 局部信息 / §13 优先级 / **§13.4a A2c 验收**）。要点：**A2c（best@153k，官方 20000 集全项达标）已取代 A2b 成为注入基线、特征口径冻结**；「P(听) 需重标定」作废（sigmoid 口径已标定，可直接当概率注入）；「读侧近天花板」证据不足——**顺序级：立直单张复现 31%→66%、非立直 82%→93%，但反事实斜率仍 0.009（时序流空转）、立直窗口仍显著为负 → P1b 两版（RM12 v1 b=0.0131、RM14 aux b=0.0038）20k 均 FAIL，不启动 200k；下一步显式 order 特征/输入消融（§13.5~§13.7）**；P1b 不退线 RM13≥92.696%、RM14≥66.2%；RM14 +50 低于 A2c 66.2 无效（旧 +30% 仅存档）。**2026-10-22 起冻结**：S6b 四臂 FAIL 后主模型已回退（`7bb5b7d`），读牌侧不再作为主线，A2c 仅作归档资产 |
| S6b 注入（主模型 2 行危险度） | ❌ 关闭（代码已回退） | 四臂 20k：`deal_top1` 0.0120/0.0121/0.0118/0.0123、holdout top1 0.8135/0.8143/0.8156/0.8136，A1≈A2（<1σ）；新通道 `|w|`mean A1 0.0158 vs A2 0.0062（权重长但行为不变，坑 72）；回退 commit `7bb5b7d`（5 py 恢复 `3a793a1`）；完全记录见 README §13.14.8 |
| 扩窗 2019+ + 重训 GRP | ⏳ 数据侧主线 | 成本最高，IQL 关闭后接棒 |

### 4.2 下一步优先级

| 优先级 | 事项 | 要点 |
|---|---|---|
| **P0** ✅ | 读牌 RM1~RM3 + RM16（零训练成本）——**2026-10-19 已完成** | 实现与验收全录见 `tools/read_model/docs/README.md` **§13.3**：因子化读法收口模型层、val_score v2（H2 门禁验证 A2 型塌缩必被拒）、CE 拆分/决策分层/R-P 双口径、`order_metrics.py`（RM12~RM15 判据，A2b 基线全 FAIL 存档）。复跑改判：net tenpai AUC 显著胜 rule；⑦「precision/recall bug」实证否决（0.1087 确为 recall，chiitoi precision 0.413，桶 5 从不预测）|
| **P0** ❌ | S6b 注入（**P0 GO → P1 四臂 20k FAIL，代码已回退**） | **2026-10-22 P0 三探针判决 GO**：主模型 `phi` 读牌探针 AUC **0.7993** vs A2c **0.9701**、hit@1 **0.0775** vs **0.2003**（5-fold by file，std≤0.005）；决策分歧 `fixable` **1.20%**（239/20000，30/30 文件，turn≥8 2.28%/立直 2.59%）；主模型点炮 **1.32%** ≈ 人类 **1.40%**、A2c 最安全 **0.27%**。**注入形态定为 2 行 obs**（三家 union 危险度 + P(听)），不是 102 维后验；P1 = 预计算 A2c(`tl=0`) → dataloader 查表 + concat（不动 Rust）→ 首层手术 1012→1014（新通道零初始化）→ 30k 探针 A0/A1/A2(安慰剂)/A3(限域)；判据 RM11 决策级 + holdout top1≥0.816 + ladder 260000/270000。**deal 类判据必须用 100k 评估集**（20k 的 1476 个目标/≈16 事件：0.0108 vs 100k 0.0147，差 36%）。**P1 四臂 20k 判决 FAIL（2026-10-22）**：`deal_top1` A0/A1/A2/A3 **0.0120/0.0121/0.0118/0.0123**、holdout top1 **0.8135/0.8143/0.8156/0.8136**，**A1≈A2 安慰剂（全部 <1σ）**；新通道 `|w|`mean A1 **0.0158** vs A2 **0.0062** ⇒ 权重长但行为不变，信息效应为零，**S6b 关闭**（坑 72）；下一步转 RM10 轨迹监督或数据侧扩窗。P1b 两版 FAIL 不阻塞 |
| **P0** ✅ | 主模型：推理期防守守卫 G3（**GO：λ=10，+0.78 pt/半庄；λ=7.5 备选**） | `Q'=Q−λ·max(risk−τ,0)`（A2c `tl=0`，只改打牌 0..33）；代码 `mortal/defense_guard.py` + `engine`/`ladder` 钩子（`6fab9ce`）；**剂量曲线（每臂双向 64000 半庄）：λ=5 +0.550（CI[−0.10,+1.20]）、λ=7.5 +0.712（CI[+0.07,+1.36]）、λ=10 +0.778（CI[+0.13,+1.42]，峰值）、λ=20 −0.496（CI[−1.14,+0.15]）、λ=50/τ=0.03 −2.663（16000 半庄）**；λ=10：agari 0.217→0.207（−4.6%）、houjuu 0.130→0.111（−14.6%），两方向同号（+1.226/+0.330）；**默认已设为 10**（`mortal/mortal.py`，env 可覆盖，设 0 关闭），保守备选 7.5 |
| **P0** 🔜 | 主模型：ladder 扩容 10000~20000 sets | 现有 5000 sets 的 CI ±0.6~1.0，看不见 <0.5 pt 的改进；或保留 5000 但改配对同牌山 + 局分层 bootstrap |
| **P1** ❌ | 主模型：S6c 自家速度特征（**已跑完，判无结论 / 不采纳**） | 2 行（向听/8 + 实战有效牌/136，`mortal/s6c_features.py`）；手术 1012→1014 零初始化、透明性 holdout 逐位相同；通道使用度 ΔL2 **100% / 58.1% 分位**（特征确实被用）。但强度两个独立量都为正而不显著：holdout top1 **0.8167 vs 0.8127（+0.40pp ≈ 1.1σ，代价 dqn_loss +3.3%）**、ladder **δ=+0.268、解析 CI[−0.90,+1.44]、bootstrap CI[−0.76,+1.27]（t=0.45）** ⇒ 不采纳，主模型仍 B/756k；956k ckpt 归档备份。分辨率：要分辨 +0.27 需 ~99k sets |
| **P1** ❌ | 数据扩窗 2019-2021 混入（**D1 判负迁移，关闭**） | 年代审计 + 20% 混入（+70,588 文件 / +14.3%）100k 探针：holdout top1 0.8105（−0.22pp）、dqn_loss +6.4%；ladder **δ=−1.402（解析 CI[−2.57,−0.23]、bootstrap CI[−2.39,−0.44]，t=−2.35）** ⇒ 变弱。原因未拆（牌风差异 / 名单阈值宽松 / GRP 年代错配），但三条都指向“旧年代不适合当增量”。代码与配置保留供复现。
| **P1** 🔄 | 数据补齐 2026H1（**D2 在跑**） | B 的索引里 **0 个 2026 文件**（names 修复前被名单滤掉）；修复树 113,430 文件 / names 100% / v3 名单保留 **78,233（68.97%）** → 索引 **571,043（+15.87%）**；`config.d2_2026.toml` + `run_d2.sh`（GPU0 P100，2.65 batch/s，~10.5 h）；判据同 D1：holdout（同 digest）+ ladder 双向 5000 sets（新种子块 **360000/380000**） |
| **P2** 🔜 | 数据质量加权 / 放铳率辅助头 | 按局内 4 人强度或年代加权；aux 预测本局放铳（风险：辅助头可绕过，坑 63） |
| **P1** ✅ | 读牌 RM4~RM7 + RM12~RM15 — **RM4+RM5+RM7 已落地、A2c 验收全通过；RM12/RM14 两版探针 FAIL** | **A2c 取代 A2b 为注入基线（best@153k）**：RM4 AUC 0.9698（≥0.9685）/标定 1.0364（∈[0.95,1.05]）/cond_ce 2.7532（≤2.896）；RM5 hit@1 **0.1993**（≥0.16）、立直 **0.1724**（≥0.135）；RM7 val_score **2.0524** < 2.0924（−1.9%）。三报告 + `order_metrics.py` 已跑（数字与证据见 README **§13.4a** + `mortal/logs/read_model/a2c_accept_20261021/`）。同时修 `order_metrics` 判据段 `NameError`（坑 61）并完成 RM13/RM14 阈值重标定。**P1b 两版已试**：RM12 v1 20k b=0.0131、RM14 aux 20k b=0.0038，均 FAIL（§13.5~§13.6）；不启动 200k，下一步显式 order 特征/输入消融。P1b 判据见 §13.7（主 RM12/RM15；不退 RM13≥92.696/RM14≥66.2） |
| ~~P1~~ | 读牌顺序通路（**已关闭**） | P1b 五版 20k 全 FAIL（b 0.0038~0.0131），输入消融证明 alloc/wait 任务不需要牌河顺序；不再投入 |
| ~~P1~~ | 读牌路径 C 注入（**已关闭**） | 与 S6b 同一路线；四臂 20k A1≈A2 后判 FAIL，代码已回退（`7bb5b7d`） |
| **P2** | 外部锚点 akochan | 需 JRE + `AKOCHAN_DIR`/`AKOCHAN_TACTICS`；人类牌谱不能当 arena 对手（无 seed） |
| **P2** | 多卡并行对照 | 5 张卡各跑一组单变量配方，再用 ladder 选优；注意 CPU 争用会让双方慢 20–30% |
| **P2** | 工程债 | ~~git push~~ **已完成（10-20，22 个提交同步）**；异地备份、上游 `git diff` 校准、两套 holdout 同分布口径、agari-guard holdout、kan-select 回放对照、文档对账（`TRAINING_MANUAL.md` §4.x/健康判据） |
| **P3** | 在线自博弈（M5） | 仅在评测可信后开；`online=true` + `freeze_bn.mortal=true`；子进程规避卡死；P100×4 可当 client 集群 |
| **P3** | 防守行为专项 | 流局 10.9% vs 人类 15.3%、放铳 0.1405 vs 0.1255 → 弃和不足；试 cql 权重/奖励平滑/oracle 特征 |
| **P3** | 更强行为先验（若要继续压上界） | `tools/read_model/docs/README.md` §9 的参考策略弱于 A2b；要真正回答「还剩多少」需 ukeire/安全度策略或学习型 policy（如用 A2b 之外的人类舍牌模型），并保留 `hard` 作为下界 |

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
| 42~72 | 读牌 | **读牌专题坑位（P(听牌) 口径分裂、CE 必须先分解、分桶 support 命名、选型判据须覆盖曾退化指标、cosine 走平≠到顶、CLI 参数静默失效、设计动机与实现对账、必须减行为盲地板、合并跨格引入组成混杂、反事实要配逐样本期望、集合级掩盖顺序级、tile-agnostic 池化承载不了逐牌信息、三家共享同一份 `h`、切序直觉方向可能反、地板/参照系须先验证其对自变量不响应、同一符号只能有一种约定且明细必须落盘、判据段重构后必端到端跑通并做基线数值回归、冗余通路/辅助头可绕过目标通路、显式特征被权重使用≠反事实响应、输入消融先于通路工程、暖启探针不能用 --resume、枚举式 H2 三源不可行）已随读牌记录整体迁至 `tools/read_model/docs/README.md` §14，编号 42~67 原样保留便于回查；68~72 见下** |
| 68 | 评测 | **小样本决策指标不能当选型判据**（2026-10-22 P0 实证）：`deal_pure` 在官方 20000 集上 `n_deal=1476`（≈16 个点炮事件），点估计 0.0108；扩到 100k 后 0.0147（n_deal 7507），**差 36%**。处置：任何 deal 类结论必须 ≥100k 或按文件 block bootstrap 报 CI，且不得与 20k 的旧值直接横比 |
| 69 | 工程 | **同名 `model.py` 会静默串味**（2026-10-22）：`tools/read_model/model.py`（读牌 v1 骨干，含 `ReadBackbone`）与 `mortal/model.py`（主模型 `Brain`）同名；`_common.bootstrap_paths()` 把 `mortal/` 插到 `sys.path` 最前（脚本目录已存在时它不会再插 `tools/read_model`），裸 `import model` 会命中主模型版本，`model_v2.py` 报 `ReadBackbone` 缺失。处置：用 `spec_from_file_location` 显式占住名字（读牌=`model`、主模型=`mortal_model`），`train.py` 同理 |
| 70 | 工程 | **首层手术后必须重置 optimizer**（2026-10-22 S6b 冒烟实证）：`Conv1d(192,1012,3)→(192,1014,3)` 的模型已换形，但 ckpt 里的 AdamW 动量仍是 1012 维 ⇒ 第一步 `AdamW.step` 报 `size of tensor a (1012) must match b (1014)`。处置：config 设 `reset_optimizer='<新 token>'` 触发跳过 optimizer/scheduler 加载（新通道零初始化，旧动量本就不适用）；另注意新通道梯度 = 上游×输入，零填充样本对它的梯度为 0，预计算覆盖不全时信号会被稀释 |
| 71 | 工程 | **暖启训练的 `state_file` 绝不能指向基线/共享起手 ckpt**（2026-10-22 S6b 实证）：train.py 每个 `save_every` 会把训练中权重**原地写回** `control.state_file`。A0 臂直接指向 `reward_ablation/B_rank_pt_aligned/mortal.pth`（判定基线），2000 步后基线被覆盖（mtime 04:19、steps 756000→758000）。可精确恢复的唯一原因：`inject_s6b/B756k_v6_init.pth` 是手术前从基线复制的，首层前 1012 通道逐位相同（手术自检 phi diff=0），裁掉新增 2 通道即还原。处置：每个实验臂先复制一份独立起手 ckpt（A0.pth/A1.pth/...），`state_file` 指向副本；基线只作只读父代 |
| 72 | 建模 | **「权重长起来」+「行为不变」= 信息冗余，不是学习不足**（2026-10-22 S6b 四臂实证）：2 行读牌危险度注入后，A1 新通道 `|w|`mean **0.0158**（A2 安慰剂 0.0062，2.5×），但 `deal_top1` 0.0121 vs 0.0118、holdout top1 0.8143 vs 0.8156，全部 <1σ。合并 S6a（6 行紧迫度 null）与坑 64（显式 order 权重涨但反事实 b 不动）：**当注入特征与 obs 已有信息同源时，权重增长只反映「无害正则」，不构成行为改变的证据**。判据必须是 **A1 vs A2（安慰剂）的配对差**，而不是「权重是否非零」；同信息注入的天花板由「BC 目标是否需要它」决定，不由「表示里有没有它」决定（P0-A 的 0.80 vs 0.97 与 P1 的 null 同时成立） |

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
| `tools/read_model/` | **读牌模型全部工具**：v1/v2/v2s/v2x/A2b/**A2c(200k 结项 + 官方验收全通过)**/**A2d 20k 探针（RM12 v1 / RM14 aux，均 FAIL）**/A3 + 86 项回归 + ONNX 导出 + A3 规则法（`rule_baseline`/`a3_eval`/`compare_a3`/`calibrate_a3`）+ Bayes 上界（`bayes_bound`/`shanten_fast`）+ 验收链 `run_a2c_acceptance.sh`（四任务并行）+ 诊断脚本 `diagnostics/`。逐条清单与用法见 `tools/read_model/docs/README.md` §15 与 `diagnostics/README.md` |
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
| 读牌模型 | 全部资产在 `tools/read_model/`（对 `mortal/`、`libriichi/` **零改动**）；代码改动台账见该目录 `README.md` §15.2 |

### 6.3 数据与产物台账

| 路径 | 内容 | 体量 |
|---|---|---|
| `dataset/mjai/houou4/` | 2,630,640 局（200902~202610）；⚠ 2026 部分无 names，白名单模式 0 席位 | 16 GB |
| `dataset/mjai/houou4_names/` | 2026 names 补丁树（130,407 文件） | 607.6 MiB |
| `dataset/tenhou_hf/` | 18 个 HF 年度 tar | 17 GB |
| `dataset/raw/`、`dataset/index/` | 2026 原始 mjlog；清单/榜单/manifest | 1.2 GB / 197 MB |
| `mortal/checkpoints/` | R1–R5、reward_ablation A/B/C、hp_ablation×3、iql_a、iql_b_v2/v3、grp×3、baseline | ≈5 GB+ |
| `mortal/read_model_bayes/`、`mortal/read_model_v2s_a2b/`、`mortal/read_model_v2s_a2c/`、`mortal/logs/read_model/a2c_accept_20261021/`、`tools/read_model/diagnostics/results/` | 读牌全部产物（Bayes 上界报告 / A2b 对照 ckpt / **A2c 注入基线 ckpt：`best.pth`(153k) + `last.pth`(200k) + `train_log.jsonl` + 三份验收报告 `eval/strat/order_report.json`** / 验收全套日志与 A2b 回归 / 诊断 json） | 台账见 `tools/read_model/docs/README.md` §15.3 |
| `mortal/read_model_v2s_a2d_probe20k/`、`mortal/read_model_v2s_a2d_nt_probe20k/`、`tools/read_model/diagnostics/results/rm12_*_20261021.json` | P1b 两版 20k 探针（RM12 v1 / RM14 next-tedashi aux）ckpt + train_log + order_metrics；A2c 立直窗口逐格定位 + P1b 判决矩阵 | 见 README §13.5~§13.7 |
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
