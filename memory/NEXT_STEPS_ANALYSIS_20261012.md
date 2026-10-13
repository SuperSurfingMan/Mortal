# 下一步行动分析（2026-10-12）

> **用途**：沉淀 2026-10-12 对「项目代码现状 + MEMORY 文档」的完整分析结论，从**数据**与**模型**两个角度给出下一步行动建议。
> **信息来源**：本机代码通读（`mortal/`、`tools/`、`libriichi/src/`）+ `memory/MEMORY.md`（截止 §17.9）+ 既有产物实测（强度榜 CSV、checkpoint 扫描、日志、原始 mjlog 抽查）。
> **可靠性标注**：✅ 已读源码/已实测　🔶 来自 MEMORY 的历史记录　⬜ 尚未验证的推测
> **本次性质**：分析结论已实施；模型侧 P0 已完成代码接线与回归。**（后续更新）奖励三组消融 10-13 已判定（B 采纳、C 关）；R5（top200 收紧名单）已于 10-13 14:28 判定——无结论、方向偏负（详 §0/§1/§2.3/§4 与 MEMORY 会话 20）；下一轮转 IQL/Bellman。**

---

## 0. 一页结论

| 项 | 结论 |
|---|---|
| 训练闭环 | ✅ 已完整跑通（牌谱 → Rust 按座位产样本 → GRP 奖励 → Brain+DQN+AuxNet → holdout + 1v3 ladder） |
| 当前瓶颈 | 不是"能不能训"，而是 **① 实验信号是否可靠（holdout/ladder 口径）② 奖励是否与最终强度一致 ③ 数据选择是否还有质量红利** |
| 已证实有效 | 仅 **R2**：高手白名单 top400 + grp_h128 → **δ=+2.86±0.59 pt/半庄**（t=4.83）🔶 |
| 已证实无效 | **R3**（同池第 2 个 epoch）δ=−0.03±0.59；**R4**（席位视角增强）δ=+0.10±0.59；**R5**（top400→top200 收紧名单）δ=−0.61±0.59（方向偏负）→ “增加同分布样本量”与“换信息源收紧名单”均被否 🔶✅ |
| 下一步总纲 | **收紧名单已判定失败（10-13，见 MEMORY 会话 20）→ 离开数据 reshuffle**：转 **② IQL / advantage-weighted regression / 带 next-state + target network 的离线 Bellman**（跳出 MC 单样本回归天花板，唯一未试方向）；若仍无增益 → ③ 扩窗 2019+（需 GRP 重训 + 审计）。✅ 2026 names 已修、奖励消融已判（B 采纳、C 关）、gamma<1 已降级 |
| 现役最强 | `checkpoints/round4/mortal.pth`（645k，0 个非有限张量，fp32）✅ |

---

## 0bis. 实施状态（2026-10-12 当天更新）

**数据侧 P0 两条与模型侧 P0 三条均已实施并验收；奖励消融正在训练，尚无强度结论**：

| 条目 | 状态 | 关键证据 |
|---|---|---|
| §2.1 修复 2026 缺失 names | ✅ 完成 | 补丁树 `dataset/mjai/houou4_names/`（130,407 文件 / 607.6 MiB / 1.92 min）；`validate_logs` 全量 16 s 无告警；同一牌谱 `GameplayLoader` 由 **0 席位 → 4 席位 584 样本**；**全量复核**：带 names 100%、521,628 席位中白名单（v3/top400）命中 **130,415 席（25.00%）**、**89,317 文件（68.49%）**、榜单池命中 **99.36%**（旧树对照 0%） |
| §2.2 `file_index` manifest | ✅ 完成 | `train.py` `FILE_INDEX_VERSION=2`；manifest 比对项 `index_version/globs/player_names_files(sha256)/excludes_files/data_version`；新增可选 `dataset.data_version`；回归测试 E1~E6 全过（`tools/training/test_file_index_manifest.sh`） |
| §3.1 奖励形式对齐 | ✅ **已判定完毕（10-13）** | 三组同父代/同数据/同步数 ladder ≥5000 sets：**C 显著变弱**（δ=−5.61，bootstrap CI[−6.67,−4.55]，t=−9.38）；**B 方向正向未达显著**（δ=+0.18，CI[−0.85,+1.19]）；A 控制组无变化（−0.04）。C 的 holdout R2 首次转正（+0.01）但未转化为强度——"Q 拟合分数差"≠"会打牌"。**建议后续采用 B 口径（`pts=[2,1,0,-3]`）作为新基线**；详见 MEMORY §12 会话 18 |
| §3.2 `gamma` 语义 | ✅ 完成（结论已实测更正） | `tools/training/inspect_gamma_semantics.py`（200 文件 / 54,526 动作样本）验三条不变量全过：`apply_gamma == (label <= 37)`、`dones` 只在局边界、**折扣视界 = 当前局内**；杠(42，171 次) 不折扣；`gameplay.rs` 注释已改写为实测口径（仅注释、无行为变更）。**"gamma<1 = 跨局信用分配"作废**（MEMORY §14-31） |
| §3.5 判定执行参数固化 | ✅ 完成 | `ladder.py`：优先级 `CLI > [1v3.ladder] > 历史默认(=R2~R4 口径)`；双侧强制同参数；`[1v3.*]` 与生效值不一致时**打印告警**；输出 `exec_cfg` + 两侧 ckpt 的 `sha256/steps/resnet/mtime`，并写 `<log_dir>/run_meta.json`（`raw rankings`/`seed_start` 行不变，`agg_ladder.py` 兼容）。2-set 冒烟通过 |

**执行中发现的新事实（原分析未预见，已写入 MEMORY §14-30/31）**：

- `<UN>` 的 `n0~n3` 是 **URL 百分号编码**的玩家名，必须先 `unquote`；首轮直接写入时 `validate_logs` 全绿但白名单命中率 **0%** —— **结构校验无法发现语义错误**，所以新增了 `tools/tenhou_data/verify_names.py` 作为强制验收手段。
- **§3.2 的代码事实已在真实牌谱上实测确认**：每局最后一条动作 `steps_to_done=0`（不被折扣），下一局计数从 0 重启 → 折扣视界严格限于当前局内；参与折扣的只有打牌(0-36) 与立直(37)。
- 补丁树修复后重建强度榜（**v3**，窗口 = HF 2022-2025 + 修复后的 202601-202608，813,145 文件，失败 0）：玩家池 4,883→5,356、合格玩家 472→501、top400 门槛 +2.45→+2.57（**69 人进出**）、top200 门槛 +3.98。
- 新池覆盖度（`whitelist_coverage.py`）：**有 names 的文件 813,145（100%）**；top400 席位 **789,454**（原 735,249，+7.4%）/ ≈134M 样本 / **262k 步每 epoch**；top200 席位 **344,068** / ≈58M 样本 / **114k 步**。
- 由此对 §2.3 的影响：R5 的名单应改用 **v3 系列**（`top_players_{400,200}_v3.txt`），因为榜单已包含 2026；原 §16.22 用的 v2 席位/步数估算（111k 步）与新实测（114k）略有差异，以本表为准。

**R5 配置的接入写法**（`data_version` 会让旧索引自动失效，无需手工删索引）：

```toml
[dataset]
globs = [
    '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202[2-5]??/**/*.json.gz',      # HF 历史（自带 names）
    '/mnt/data8/mortal/Mortal/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz', # 修复后的 2026 训练窗
]
player_names_files = ['../dataset/index/top_players_200_v3.txt']
data_version = 'names-fix-v1'
file_index = 'indexes/main_r5_whitelist_top200_v3.pth'
```

**奖励消融（§3.1）的配置写法**（真实分数差口径已可用，见 §0bis）：

```toml
[env]
gamma = 1
# A 组 = 现状：等间距顺位点
pts = [6.0, 4.0, 2.0, 0.0]
# B 组：对齐天凤点数（同尺度、第 4 名重罚）
# pts = [2.0, 1.0, 0.0, -3.0]
# C 组：每局真实分数差（点）→ 量级 ~1e4，必须归一化，否则 CQL 权重与 lr 的相对尺度全变
# reward_mode = 'delta_points'
# reward_scale = 1e-3
```

- 三组必须从**同一父代**暖启（本轮实际使用 `checkpoints/round4/mortal.pth` 645k），同数据/同步数/同 holdout，判定用**全新种子块**的双向 ladder；本轮严格不改名单或增强。
- 本轮为奖励目标切换统一设置一次性 `control.reset_optimizer='reward-ablation-v1'`：只继承父代模型权重，跳过旧 optimizer/scheduler/scaler；因此 A 也是“同父权重 + 干净优化器”的控制组，A/B/C 之间奖励是唯一实质变量。
- 换口径后 `dqn_loss` 的绝对值**不可跨组比较**（目标尺度不同）——只比 ladder 与固定 holdout；holdout 日志必须核对相同 `holdout_sample_digest`，而 `holdout_target_digest` 应随奖励口径变化。强度结论仍只认 ladder。
- 每次判定后看 `<log_dir>/run_meta.json`：它记录 `exec_cfg` 与两侧 checkpoint 的 `sha256/steps`，是"同样条件"的唯一硬证据。
- 判定命令示例（`[1v3.ladder]` 可写 `enable_amp`/`enable_rule_based_agari_guard`；不写则用 R2~R4 口径并提示）：
  `python ladder.py <new>.pth <parent>.pth 100000 500 logs/ladder_x --tag reward-B`

---

## 1. 现有强度账（判定口径全部为 ladder 双向同牌山 ≥5000 sets）

| 轮 | 变量 | δ vs 父代 | 结论 |
|---|---|---|---|
| R1 | 2022-01~2026-08 全体 813,145 局 + grp h64/l2(30k) | —（基准） | 评测体系失效期产物 |
| R2 | 白名单 top400 + **grp_h128**（h128/l3/60k） | **+2.86 ± 0.59**（t=4.83）✅ | 唯一有效跳跃 |
| R3 | 与 R2 完全相同，仅 `num_epochs=2` | −0.03 ± 0.59 ❌ | 重复数据无增益 |
| R4 | 白名单 v2 + **augmentation**（+145k 步新样本） | +0.10 ± 0.59 ❌ | 视角旋转增强无增益 |
| 消融 A/B/C（10-13） | 奖励口径切换（top400，同父代 645k） | A −0.04 / **B +0.18** / **C −5.61** | B（[2,1,0,-3]）方向正向未显著→采纳为新基线；C 显著变弱→关 |
| **R5（10-13）** | **top400_v2 → top200_v3 收紧名单**（以 B 组/756k 为父代） | **−0.61 ± 0.59**（bootstrap CI[−1.61,+0.36]）❌ | **无结论、方向偏负 → 收紧名单无增益** |

**读法**：R2 同时改了「数据选择」和「奖励模型」两件事，因此 +2.86 无法拆开归因到其中某一项；后续所有"数据筛选类"实验都只能定性为"与 R2 同源"。**R5 把名单从 top400 单独收紧到 top200（席位砍到 ~45%、增强一遍仅 114k 步）仍无增益且方向偏负，证实“名单收紧”不是可复现的增益轴。**
**推论**（🔶 MEMORY §16.7 已给出，本次代码复核支持）：`gamma=1` 的单局 MC 奖励 + CQL 锚定行为策略的框架下，模型在提升"模仿人类精度"（holdout top1 单调上升）而非"赢的能力"（ladder 持平）。**“同分布层面挤数据”（R3/R4/消融 A）+“换信息源收紧名单”（R5）共 5 次否掉 → 数据 reshuffle 路线到头，下一轮必须改目标形式（IQL/Bellman）。**

---

## 2. 数据侧建议

### 2.1 【P0】修复 2026 牌谱缺失的 `names`（本轮新发现）

**问题**：自建 2026 mjai 的 `start_game` 只有 `rules`，没有 `names` 字段。

- 玩家强度榜扫描会把这些文件计入 `no_names` 单独计数，榜单只覆盖 HF 的 2009~2025：[`scan_player_ratings.py:41`](/mnt/data8/mortal/Mortal/tools/player_stats/scan_player_ratings.py:41)
- 训练索引构建时用 `start.get('names', [])` 防 KeyError，白名单模式下这类文件自然被排除：[`train.py:210`](/mnt/data8/mortal/Mortal/mortal/train.py:210)
- Rust 侧按座位过滤同样依赖 `names`：[`gameplay.rs:169`](/mnt/data8/mortal/Mortal/libriichi/src/dataset/gameplay.rs:169)

**实测**（✅ 本机抽查，未输出身份信息）：取一份 202608 的 mjai，其 `start_game` 键为 `['rules','type']`；对应 `dataset/raw/mjlog/` 下的原始 XML `<UN>` 节点**保留完整字段** `dan / n0 / n1 / n2 / n3 / rate / sx`。

→ 原始数据没有丢信息，是**转换管线丢了**。修回来即可让 13 万局 2026 数据重新进入白名单训练池。

**做法**：
1. 从 `dataset/raw/mjlog/*.xml.gz` 的 `<UN>` 读 `n0~n3`，在 `dump_mjai` 之后、落盘之前把 `names` 写回 `start_game`；
2. 输出到**带版本号的新目录**（如 `dataset/mjai/houou4_names/`），不覆盖现有 2.63M 局（避免 R1~R4 的可复现性被破坏）；
3. 用 `target/release/validate_logs` + `GameplayLoader` 抽样验证结构兼容；
4. 对比修复前后的：文件数 / `names` 覆盖率 / 白名单命中席位数 / 预计动作样本数；
5. 重跑 `build_whitelist.sh` 得到含 2026 的强度榜，与现有 v2 榜对比 top200/top400 的人员与门槛变化。

**相关入口**：[`tenhou_pipeline.py`](/mnt/data8/mortal/Mortal/tools/tenhou_data/tenhou_pipeline.py:357)（`_convert_one`）、[`build_whitelist.sh`](/mnt/data8/mortal/Mortal/tools/player_stats/build_whitelist.sh:16)。

**注意**：修好后 2026 数据的平均强度、规则微调版本都可能与 HF 历史数据不同，属于"换数据分布"，必须用同一 ladder 判定，不能直接假设为纯增益。

---

### 2.2 【P0】给 `file_index` 加 manifest，消灭"静默用旧索引"

**问题**：[`train.py:195`](/mnt/data8/mortal/Mortal/mortal/train.py:195) 只要索引文件存在就直接 `file_list = index['file_list']`，**完全不校验配置是否变化**。索引内只有 `{'file_list': ...}` 一个键。

→ 换白名单、扩 glob、修 `names`、改 `excludes` 后，若沿用旧索引名，会静默使用旧文件集合。MEMORY §14/§17.7 已把"名单变了必须换新名"列为坑位，但这是**靠人肉纪律**而非机制保证。

**做法**：把索引改写为

```
{
  'file_list': [...],
  'manifest': {
    'globs': [...], 'months': [...], 'n_files': ...,
    'player_names_sha256': ..., 'names_coverage': ...,
    'excludes_sha256': ..., 'data_version': 'names-fix-v1',
    'loader_version': ..., 'augmentation': false,
    'n_seats_est': ..., 'n_samples_est': ...,
  }
}
```

加载时逐项比对当前配置，不一致则 `logging.warning` 并重建。

---

### 2.3 【P1】R5（top200）必须带控制组，否则结论不可归因 —— ✅ **已判定（10-13）：收紧名单无增益、方向偏负**

> **结果（见 MEMORY 会话 20）**：实际以 **B 组/756k（top400_v2、[2,1,0,-3]）为父代与控制组**，唯一变量 top400→**top200_v3**（augmentation 仍为 true）；新种子 120000+/130000+ 双向各 5×500 sets → **δ=−0.61±0.59（bootstrap CI[−1.61,+0.36]）→ 无结论、方向偏负**。未出现 C 组那种“A 全负/B 全正”反对称形态，故为“无增益”而非“确证变弱”。**本条下方的双分支对照设计已被“单分支 vs B 组”取代。**

MEMORY §16.22 的 top200 方案本身成本合理，本机用 v2 榜单复核了覆盖度（✅ 实测）：

| top-N | 门槛 avg_pt | 最少局数 | 局数中位数 | 席位合计 | 占 top400 席位比 |
|---|---|---|---|---|---|
| 100 | 5.23 | 405 | 874 | 130,206 | 17.7% |
| **200** | **3.923** | 405 | 1068.5 | **334,907** | **45.6%** |
| 400 | 2.45 | 401 | 1260.5 | 735,249 | 100% |

**但现有方案把两个变量捆在一起**：`top400→top200` 与 `enable_augmentation: true→false` 同时发生（R4 的增强已判定无效，改回 false 是对的，但它仍是一个变更）。

**建议改为双分支对照**（都从 `round4/mortal.pth` 645k 暖启，同 `GRP`、同步数、同 holdout、全新 ladder 种子块 100000+/110000+）：

| 分支 | 名单 | 增强 | 预算 | 想回答的问题 |
|---|---|---|---|---|
| **A** | `top_players_200_v2` | false | 645k + 111k = 756k | 收紧名单是否有独立增益？ |
| **B** | `top_players_400_v2` | false | 645k + 111k = 756k | 关掉增强的影响（控制组） |

两分支都可跑（4×P100 空闲，CPU 争用会让双方慢 20~30%）。判定仍以 A/B 双向 ≥5000 sets + block bootstrap、95%CI 下界 > 0 为准。

**附带建议**：暖启时考虑重置 optimizer 动量与 scheduler（R4 是从 424k 带着 AdamW 二阶矩续训的，111k 步的新池 vs 旧动量是否匹配值得单独验证；若担心混入变量，至少在两分支间保持一致）。

---

### 2.4 【P1】重建两套固定 holdout，口径必须说清

当前矛盾：训练用「白名单 + 增强视角」，而 [`holdout_metrics.py`](/mnt/data8/mortal/Mortal/mortal/holdout_metrics.py:25) 用 `player_names=[]` + 不增强 = **全玩家原始视图**。它可以当泛化测试，但不能解释为"同分布验证"。

**建议固定两套**：

| 集合 | 名单 | 增强 | 月份 | 用途 |
|---|---|---|---|---|
| 同分布 | 与训练同一份 | 与训练一致 | 训练窗内留出 | 诊断"是否在学" |
| 时间外推 | 空（全玩家） | false | 202609 / 202610 | 诊断"是否泛化" |

- 所有 checkpoint 共用同一批文件 + 同一抽样 seed；
- bootstrap 单位取**牌谱**（或按牌的 set），不要把同一局内多个动作当独立样本；
- holdout 的 `q_target` 与 `gamma/pts` 依赖**运行时**配置而非 checkpoint 内配置（[`holdout_metrics.py:54`](/mnt/data8/mortal/Mortal/mortal/holdout_metrics.py:54)），跨配置比较旧 checkpoint 时必须固定目标参数，否则 Q target 已变、指标不可比。

---

### 2.5 【P2】扩窗到 2019+ 放在最后，且必须先审计

全树 263 万局看似充裕，但来源混杂（HF 18 包 + 2026 自建）、字段覆盖不一（`names` 只有 HF 侧有）、规则历史不同。直接扩窗 = 同时改变时间分布 + 数据质量 + 奖励模型训练分布，三个变量叠加无法归因。

**前置审计清单**：按年份统计 `names` 覆盖率、平均顺位点分布、动作类别分布、局数、文件大小；确认 GRP 是否需要重训（扩窗后 GRP 的输入分布也变了）。

---

## 3. 模型侧建议

### 3.1 【P0】奖励形式与评测目标不一致（本轮新发现，优先级高于 top200）——✅ 已实现（能力就绪，消融待跑；见 §0bis）

**矛盾**：

- 训练奖励：`env.pts = [6.0, 4.0, 2.0, 0.0]`（等间距顺位奖励），见 [`reward_calculator.py:34`](/mnt/data8/mortal/Mortal/mortal/reward_calculator.py:34) → `exp_pts[1:] - exp_pts[:-1]`
- 最终判定：天凤点数 `[90, 45, 0, -135]`，**第四名惩罚 (−135) 远超等间距假设**

→ 模型被训练成"第 2 名和第 3 名差不多"，而 ladder 判定下第 4 名是灾难。这与 MEMORY §16.8 观察到的"流局率 10.9% vs 人类 15.3%（弃和不足）、放铳偏高"方向一致。

**建议 ablation**（同数据、同初始 checkpoint、同步数、同 holdout、全新种子 ladder）：

| 组 | 奖励 | 说明 |
|---|---|---|
| 现状 | `[6,4,2,0]` | 等间距 |
| 对齐 | `[2,1,0,−3]` | 天凤点数归一化（同尺度，第 4 名惩罚重） |
| 真实分数 | `calc_delta_points` | 直接用每局真实分数差（[`reward_calculator.py:40`](/mnt/data8/mortal/Mortal/mortal/reward_calculator.py:40)，已有实现但未接入主数据流） |

⚠ 换奖励会改变目标幅度 → `cql.min_q_weight=5` 的相对权重随之变化，必须重新审视或一并扫；不要同时改名单/增强。

---

### 3.2 【P0】`gamma<1` ≠ 跨局信用分配（本轮新发现，别直接照 MEMORY 方案改）——✅ 已实测验证（结论见下，工具 `inspect_gamma_semantics.py`；MEMORY §14-31）

**代码事实**：

- `q_target = gamma ** steps_to_done * kyoku_reward`，其中 `steps_to_done` 用 `dones` 反向累加，而 `dones` 是**局边界**（`at_kyoku` 变化处置 true）：[`dataloader.py:110`](/mnt/data8/mortal/Mortal/mortal/dataloader.py:110)、[`gameplay.rs:272`](/mnt/data8/mortal/Mortal/libriichi/src/dataset/gameplay.rs:272)
- `kyoku_rewards[at_kyoku[i]]` 只取**当前局**的 GRP 排名点差

→ 现在把 `gamma` 从 1 改成 0.995，效果只是**把同一局内较早动作的目标值打折**，完全没有把后续局的收益传回前一局。MEMORY §13bis/§16.16 把 `gamma<1` 描述为"跨局信用分配"，**与代码实现不符**。

**另有注释/实现不一致**：[`gameplay.rs:428`](/mnt/data8/mortal/Mortal/libriichi/src/dataset/gameplay.rs:428) 注释写 "only discard and kan will discount"，代码是 `label <= 37` → 实际包含**出牌(0-36) + 立直(37)**，但**不含** action 42（杠）。`gamma=1` 时这个差异被完全掩盖。

**做法**：
1. 先用固定小牌谱（人工挑选含立直/吃/碰/杠/和了/流局/单候选杠/多候选杠）跑一遍 loader，打印每个动作的 `label / apply_gamma / steps_to_done / dones`，确认语义；
2. 若要真正的跨局 credit assignment，需要数据侧显式提供「下一局 obs + 未来 reward」并引入 target network（或至少一局一步的 Bellman），属于**训练框架改动**，应与奖励 ablation 分开做、单独判定。

---

### 3.3 【P1】模型规模不要扩大，先修 Q 校准 / 换目标

现状：DQN 回归单样本 MC 奖励，**无 next obs、无 target network、无 Bellman bootstrap**（[`train.py:265`](/mnt/data8/mortal/Mortal/mortal/train.py:265)）。holdout 显示 `R²` 为负（Q 回归还不如常数预测），但人类动作 top1 仍在升——"更像人"而非"更能赢"。

**建议顺序**：
1. 固定数据 → 奖励形式对照（3.1）；
2. 小扫 `cql.min_q_weight`（1 / 5 / 10），看 Q 幅度与 ladder 是否同步改善；
3. 验证 `aux.next_rank_weight=0.2` 的真实贡献（消融 0 vs 0.2）；
4. 仍无增益再考虑 IQL / advantage-weighted regression / 带 next state 的离线 Bellman；
5. **暂不建议**把 `192×40` 再扩大，也不建议只把步数推到 1M（R3 已证明重复数据无效）。

旁证（✅ 实测 holdout_r4.log）：424k→645k 的 top1 0.8052→0.8112，但 ladder δ≈0，说明 top1 与强度已解耦，不能当进度信号。

---

### 3.4 【P1】AMP 继续关闭，直到单独验证

R4 的 NaN 事故根因是 AMP(fp16) 偶发溢出 + BN running stats 前向即被污染（[`MEMORY.md` §16.20](/mnt/data8/mortal/Mortal/memory/MEMORY.md:1332)）。三重防护已就位，实测有效。

**建议**：R5 全程 `enable_amp=false`（fp32 ≈4.24 it/s vs AMP 6.7 it/s，用时间换可复现性与 checkpoint 纯净度）。若将来重开 AMP，须单独跑一轮稳定性验证（BN 行为 + finite guard + 退出码 + 快照回滚），不要与实验变量混在一起。

⚠ 另注意 R5 若同时跑两个分支，`test_play` 的 `log_dir` 会 `rmtree`（[`player.py:58`](/mnt/data8/mortal/Mortal/mortal/player.py:58)），**两个分支必须用不同 config / 不同 `tensorboard_dir` / 不同 `test_play.log_dir`**，否则互相删除。

---

### 3.5 【P0】最终模型选择只认全新种子的双向 ladder——✅ 执行参数已固化（`ladder.py` 现读 `[1v3.ladder]`/CLI、告警不一致、写 `run_meta.json`；见 §0bis）

| 信号 | 定位 | 可否作为强度判定 |
|---|---|---|
| `holdout_metrics` top1 / MSE / R² | 秒级诊断 | ❌ 只能看趋势（已与 ladder 解耦） |
| `test_play` avg_rank / avg_pt | 训练内监控 | ❌ 固定种子块已被多轮选模型污染（MEMORY §16.15：虚高 4.5 pt） |
| `best.pth` | 记录用 | ❌ 只在同一卫冕者下可比，不能当"最强模型" |
| **ladder 双向 + 新种子 + ≥5000 sets + block bootstrap** | 定成败 | ✅ 唯一标准 |

**执行口径问题的当前状态（10-12 更新）**：
- ✅ **已修**：`ladder.py` 原来硬编码 `enable_amp=False` / `enable_rule_based_agari_guard=True`、配置里的 `[1v3.*]` 对它无效。现改为 `CLI > [1v3.ladder] > 历史默认`，与 `[1v3.*]` 不一致会告警，双侧强制同参数，并把 `exec_cfg` 与两侧 checkpoint 的 `sha256/steps` 写进 stdout 与 `run_meta.json`。
- ⬜ **未做**：holdout 只算模型 argmax，不含 agari guard 与 quick-eval 路径差异（[`mortal.rs`](/mnt/data8/mortal/Mortal/libriichi/src/agent/mortal.rs:210)）→ 用 holdout 预筛 checkpoint 时仍要知道这层偏差（要做需要另写"带 guard 的 holdout"，属独立任务）。
- ◐ **可测未测**：训练 loader 一直用 Rust 默认 `always_include_kan_select=true`，部署 quick-eval 只在多 kan candidate 时才请求 kan-select 状态 → 训练侧现已可切换（`dataset.always_include_kan_select = false`，`dataloader`/`train`/`holdout_metrics` 三处已透传），但**"两种口径差多少"的回放对照尚未做**。

---

## 4. 建议执行顺序与验收标准

| 阶段 | 动作 | 成本量级 | 通过标准 |
|---|---|---|---|
| **S0** | 修 2026 `names` + 数据 manifest + 索引校验 | 1~2 天（CPU） | 新目录通过 `validate_logs`；白名单席位/样本数前后可比对；manifest 生效并能挡住旧索引 |
| **S1** | 建两套固定 holdout（同分布 + 时间外推） | 小时级 | 两套集合的样本量、seed、口径写进文档；跨 checkpoint 可比 |
| **S2a** | ~~R5-A：top200 + 无增强~~ → **✅ 已实跑（10-13）：以 B 组为父代的 top200_v3 单分支** | ≈7.8 h fp32 + 29 min 判定 | **δ=−0.61（bootstrap CI[−1.61,+0.36]）→ 无结论、方向偏负，收紧名单无增益** |
| ~~**S2b**~~ | ~~R5-B：top400 + 无增强（控制组）~~ → **控制组即 B 组本身（reward_ablation/B_rank_pt_aligned）** | — | B 组不单独跑，其 756k ckpt 直接用作 R5 的父代与对照 |
| **S3** | 奖励 ablation：A=`[6,4,2,0]` vs B=`[2,1,0,−3]` vs C=真实分数差 → **✅ 10-13 判定完毕**：C 显著变弱（−5.61），B 正向未显著（+0.18），A 无变化；产物见 `mortal/reward_ablation/*/ladder_summary.log` | 已花费 ≈12 h（3 GPU 并行）+ 45 min 判定 | B 口径建议采纳为新基线；C 路线关闭 |
| ~~**S4**~~ | ~~gamma 语义验证 → 若通过再试 `gamma=0.995`~~ → **✅ 语义已于 10-12 验证完毕**（`inspect_gamma_semantics.py`）：折扣只在当前局内、且不含后续局信息 → 单纯调 `gamma<1` 的收益预期很低，**降级为低优先级**（若仍要试，须先写清它想验证的假设，否则只是给同局较早动作加噪声） | 一轮 | ladder 判定 |
| **S5** | 仍无增益 → IQL/Bellman 类目标，或扩窗 2019+（含 GRP 重训与数据审计） | 大改动 | 单独设计、单独判定 |

**通用判定纪律**（沿用 MEMORY 已验证的方法论）：
- 一律用**全新种子块**（历史已用：10000-/30000-/60000-/70000-/80000-/90000-，下一轮建议 100000-/110000-）；
- 双向各 5×500 sets = 10,000 半庄/方向，SE(δ)≈±0.6；
- 按 set 做 block bootstrap（实测解析法偏保守，bootstrap SE≈0.51）；
- 判进标准：**95%CI 下界 > 0**；跨 0 即"无结论"，不得解读为"有效"；
- 每轮只改一个实质变量，其余配置锁死。

---

## 5. 关键代码证据索引

| 主题 | 位置 |
|---|---|
| 文件索引静默复用 | [`mortal/train.py:195`](/mnt/data8/mortal/Mortal/mortal/train.py:195) |
| `names` 缺失兼容 | [`mortal/train.py:210`](/mnt/data8/mortal/Mortal/mortal/train.py:210) |
| `excludes` 未接入主训练 | [`mortal/train.py:223`](/mnt/data8/mortal/Mortal/mortal/train.py:223) |
| 训练目标 `q_target` | [`mortal/train.py:265`](/mnt/data8/mortal/Mortal/mortal/train.py:265) |
| `steps_to_done` 构造 | [`mortal/dataloader.py:110`](/mnt/data8/mortal/Mortal/mortal/dataloader.py:110) |
| GRP 排名点奖励 | [`mortal/reward_calculator.py:34`](/mnt/data8/mortal/Mortal/mortal/reward_calculator.py:34) |
| 真实分数差奖励 | [`mortal/reward_calculator.py:40`](/mnt/data8/mortal/Mortal/mortal/reward_calculator.py:40)、[`mortal/dataloader.py`](/mnt/data8/mortal/Mortal/mortal/dataloader.py) 的 `reward_mode='delta_points'` |
| holdout 用运行时 GRP/gamma/pts | [`mortal/holdout_metrics.py:54`](/mnt/data8/mortal/Mortal/mortal/holdout_metrics.py:54) |
| holdout 全玩家/不增强 | [`mortal/holdout_metrics.py:35`](/mnt/data8/mortal/Mortal/mortal/holdout_metrics.py:35) |
| ladder 执行参数（**10-12 已改为可配置+记录**） | [`mortal/ladder.py`](/mnt/data8/mortal/Mortal/mortal/ladder.py)（`resolve_exec_cfg` / `run_meta.json`） |
| 奖励口径切换/干净暖启（**10-12 新增**） | [`mortal/dataloader.py`](/mnt/data8/mortal/Mortal/mortal/dataloader.py)（`reward_mode`/`reward_scale`）、[`mortal/train.py`](/mnt/data8/mortal/Mortal/mortal/train.py)（`env_reward_diff`/`reset_optimizer`） |
| 固定 holdout 与共享索引（**10-12 新增**） | [`mortal/holdout_metrics.py`](/mnt/data8/mortal/Mortal/mortal/holdout_metrics.py)（seed/digest）、[`tools/training/build_file_index.py`](/mnt/data8/mortal/Mortal/tools/training/build_file_index.py) |
| test_play 固定种子 + rmtree | [`mortal/player.py:58`](/mnt/data8/mortal/Mortal/mortal/player.py:58) |
| 按座位白名单过滤 | [`libriichi/src/dataset/gameplay.rs:169`](/mnt/data8/mortal/Mortal/libriichi/src/dataset/gameplay.rs:169) |
| `dones` 局边界 | [`libriichi/src/dataset/gameplay.rs:272`](/mnt/data8/mortal/Mortal/libriichi/src/dataset/gameplay.rs:272) |
| `apply_gamma` 注释/实现不一致（**10-12 注释已按实测修正**） | [`libriichi/src/dataset/gameplay.rs`](/mnt/data8/mortal/Mortal/libriichi/src/dataset/gameplay.rs)（`add_entry`）+ 验证工具 [`inspect_gamma_semantics.py`](/mnt/data8/mortal/Mortal/tools/training/inspect_gamma_semantics.py) |
| 席位视角增强实现 | [`libriichi/src/mjai/event.rs:187`](/mnt/data8/mortal/Mortal/libriichi/src/mjai/event.rs:187) |
| 增强的 tile 映射（数牌 ±9） | [`libriichi/src/tile.rs:154`](/mnt/data8/mortal/Mortal/libriichi/src/tile.rs:154) |
| mjlog→mjai 转换 | [`tools/tenhou_data/tenhou_pipeline.py:357`](/mnt/data8/mortal/Mortal/tools/tenhou_data/tenhou_pipeline.py:357) |
| 强度榜扫描（no_names 分支） | [`tools/player_stats/scan_player_ratings.py:41`](/mnt/data8/mortal/Mortal/tools/player_stats/scan_player_ratings.py:41) |
| 白名单构建脚本 | [`tools/player_stats/build_whitelist.sh:16`](/mnt/data8/mortal/Mortal/tools/player_stats/build_whitelist.sh:16) |
| R4 端到端链脚本 | [`mortal/run_round4.sh`](/mnt/data8/mortal/Mortal/mortal/run_round4.sh) |
| 奖励消融准备/启动/收尾 | [`tools/training/prepare_reward_ablation.py`](/mnt/data8/mortal/Mortal/tools/training/prepare_reward_ablation.py)、[`tools/training/run_reward_ablation.sh`](/mnt/data8/mortal/Mortal/tools/training/run_reward_ablation.sh)、[`tools/training/finish_reward_ablation.sh`](/mnt/data8/mortal/Mortal/tools/training/finish_reward_ablation.sh) |

---

## 6. 尚未验证的推测（需小规模对照实验才能确认）

1. **top200 收紧是否带来增益**——→ **10-13 已测：否，且方向偏负**。R5 以 B 组/756k 为父代的单变量对照（top400_v2→top200_v3），δ=−0.61±0.59（bootstrap CI[−1.61,+0.36]）；席位降到 top400 的 ~45%（344,068）、增强一遍仅 114k 步，“少而精”没兑现——**R2 的 +2.86 来自“全体→top400”这一大步，top400→top200 的边际效应为负**。收紧名单路线关闭。
2. **奖励对齐（第 4 名重罚）是否改善 ladder**——→ **10-13 已测**：B 口径 δ=+0.18±0.52，方向正向但未达显著（需 44 倍 sets 才到 2σ）；C（真实分数差）δ=−5.61 **显著变弱**。结论：顺位类奖励内部对齐值得采纳为基线（无实害、理论自洽），分数差路线关闭。
3. **2026 数据修好 `names` 后是否值得并入**——补齐的是最近数据，但可能与 HF 历史数据存在分布/规则差异。
4. **GRP 中段 hit@3≈37% 的奖励噪声**是否为当前强度天花板的主因——MEMORY 判断其受 7 维输入的信息量限制，但"换成真实分数差监督"这条替代路线的收益未测。→ **10-13 已间接否掉**：C 组直接用真实分数差做训练奖励，ladder δ=−5.61 显著变弱——分数差的逐局方差被对手行为主导，不适合做逐局奖励。
5. **训/部署的 kan-select 路径差异**是否造成实际强度损失——代码层面可定位，但需回放验证。
6. **optimizer 状态跨池暖启**（R5 换名单后继续用 R4 的 AdamW 二阶矩）是否有害——完全未测。

---

## 7. 本次分析的边界

- 全部结论基于 **2026-10-12 本机代码状态 + MEMORY.md + 既有日志/产物实测**；
- 本轮已修改训练/评测代码和工具，完成三组奖励消融并判定（结论见 §0bis 与 MEMORY §12 会话 18）；
- 未扫描或输出牌谱内容、玩家身份；
- 既有强度数字均引用 MEMORY 中已用 ladder 双向 ≥5000 sets 判定的结果；本轮新增运行状态以 `mortal/reward_ablation/experiment.json`、各组 `train.log` 和后续 `holdout_sample_digest/holdout_target_digest` 为准。
