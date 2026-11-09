# 读牌模型诊断脚本

`tools/read_model/docs/README.md` **§11**（旧 `MEMORY.md` §3.15，2026-10-19 已随读牌记录整体迁出）全部数字的**可复现来源**。本目录所有脚本一律**只读**：
不改 `mortal/`、不改 `libriichi/`、不写训练产物，只往 `results/` 落 json。

## 约定

- 路径全部由 `_common.py` 解析（仓库根自动推导），可用环境变量覆盖：

  | 变量 | 默认 |
  |---|---|
  | `RM_CKPT` | `mortal/read_model_v2s_a2b/best.pth` |
  | `RM_VAL_GLOB` | `dataset/mjai/houou4/202609/**/*.json.gz`（holdout） |
  | `RM_TRAIN_GLOB` | `dataset/mjai/houou4_names/202608/**/*.json.gz`（探针训练月，**必须与评估月不重叠**） |
  | `RM_OUT_DIR` | `./results` |
  | `RM_CACHE_DIR` | `/tmp/read_model_diag_cache`（大缓存放仓库外） |

- 评估集一律用 `train.build_fixed_eval_set(files, n, seed=20261016)`，与官方
  `evaluate.py` 同口径；不固定 seed 的抽样结果不可与主 README **§10/§11/§12** 的数字比较。
- 需要 GPU 的脚本请显式指定空闲卡，例如
  `CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python s315_head_probe.py`。

> ⚠️ **冒烟测试务必隔离 `RM_OUT_DIR`**：`results/*.json` 是主 README **§11 / §12**
> 表格数字的原始出处。用小样本参数试跑会**就地覆盖**它们（曾发生过：`probe.json` 被
> `s315_head_probe.py 1500 3000` 覆盖）。正确姿势：
> ```bash
> RM_OUT_DIR=/tmp/smoke_out RM_CACHE_DIR=/tmp/smoke_cache \
>   python s315_head_probe.py 1500 3000
> ```
> 若已污染，`results/` 里的正确值应能与 `MEMORY.md` 对应表格逐位对上；对不上就是被覆盖了。

## 脚本索引

### §11（2026-10-19 复核：推翻两条判决）

| 脚本 | 对应章节 | 产出 | 成本 |
|---|---|---|---|
| `s315_arch_audit.py` | §11 头部（架构/参数量/损失权重/课程/数据量逐位复核） | `results/diag1.json` | ~1 min，CPU |
| `s315_wait_calib.py [n]` | **§11.1** P(听牌) 双口径（`sigmoid(l34)` vs `softmax[34]`）标定与十桶可靠性曲线 | stdout | ~3 min（n=20000），GPU |
| `s315_strat_auc.py [n]` | **§11.1** 分层 AUC 重算（all / `called=1` / `riichi=0&called=0` / turn 分桶） | stdout | ~3 min（n=20000），GPU |
| `s315_wait_ce_decomp.py [n]` | **§11.2** `wait_ce_active = −logP(t) + cond_ce` 分解 | stdout | ~3 min（n=20000），GPU |
| `s315_head_probe.py [n_eval] [n_train]` | **§11.3** 冻结骨干探针 P0/P1/P2/P3（H2 逐牌列特征瓶颈） | `results/probe.json` | **最贵**，GPU；训练月 202608 |

### §12（局部手牌信息视角：架构与损失是否充分）

四个脚本按 Phase 1 → Phase 2 顺序跑；Phase 2 三个只读缓存、**不需要 GPU**。

| 脚本 | 内容 | 成本 |
|---|---|---|
| `s3158_local_extract.py [n]` | **Phase 1**：建 40000 样本评估集 → A2b / A3 / counts 三套后验 → tl_tokens 顺序对调的配对反事实 → 落 `CACHE` | ~2.5 min，GPU |
| `s3158_local_report.py` | **Phase 2a**：PART C（alloc CE 按列分解 + 邻接手切分组）、PART A（顺序模式 desc/asc/对照 + bootstrap + 立直分层 + 逐格 + 自然序格内分层）、PART A'（反事实）、PART D（tedashi vs tsumogiri）→ `results/local_order_report.json` | ~3 min，CPU |
| `s3158_local_strata.py` | **Phase 2b**：格内分层 ΔΔ（消除 m 混合混杂）× 立直分层；反事实改**逐样本期望回归**（斜率 b、r²）→ `results/local_order_report2.json` | ~4 min，CPU |
| `s3158_local_window.py` | **Phase 2c**：同色 ±2 窗口（5 张合计）版本，对应 「7s **附近**的索子张」这种区域性提问（仅 stdout） | ~3 min，CPU |
| `riichi_post_discard_probe.py` | **立直前后打牌信息的利用方式**（2026-11-03）：`zero_tg` / `drop_post` / `drop_pre` / `drop_all` 五变体消融 → `results/riichi_post_probe.json`；结论 CAPTURED（R=0.006）见 `results/RIICHI_POST_PROBE.md` | ~11 min，GPU |

> ⚠️ 地板口径提示（2026-10-20 审计）：Phase 2b/2c 原始跑批时用 **A3 规则法**当「顺序盲
> 地板」——该设定已作废（A3 消费 `tl_tokens`，对顺序有真实响应，见下方方法学 2）。
> `results/local_order_report2.json` 里的 A3 地板数字是**历史值**；现行地板是
> `rule_predict(order_blind=True)` 构造的真顺序盲变体，日常验收请用
> `order_metrics.py`（cache 自动区分两种地板，`meta.floor_kind` 标注来源）。

> ⚠️ `n_eval` 太小（<20000）时每格 (desc, asc) 不足 25 样本，格内 ΔΔ 会被全部丢弃；
> `s3158_local_strata.py` 会打印明确提示并跳过回归（不吐 nan）。**默认 40000 是复现
> §12 数字的必要条件**（改成别的 n 得到的值不可与文档比较）。

> 📌 **已常规化（RM16，2026-10-19；10-20 换地板口径）**：Phase 2b/2c 的三项指标（格内分层 ΔΔ
> 扣地板、±2 窗口版、配对反事实 b/r²）已沉淀为常规工具 **`tools/read_model/order_metrics.py`**
> ——一条命令产出 `<ckpt>/order_metrics.json` + RM12~RM15 四项 PASS/FAIL 判据，
> 自动复用/重建本目录 extract 的 cache（sidecar 记 ckpt 身份）。10-20 起默认判据用
> **真顺序盲地板**（`rule_predict(order_blind=True)`），A3 原样的顺序响应另报
> `a3_order_sensitivity`，`meta.floor_kind` 标注来源；对 A2b 复现的旧地板数字
> （复现率 all +41% / 非立直 +67% / 立直 −12%）见主 README §12 与 §13.3（历史存档）。
> 本目录的 s3158 脚本仍是**数字的原始出处**，保留不动；日常验收跑 `order_metrics.py` 即可。

### 工具

| 脚本 | 用途 |
|---|---|
| `md_table_check.py` | 校验 Markdown 表格列数一致性 + `R<n>`/`RM<n>` 编号是否串号（默认 `memory/MEMORY.md`，可传其它文档路径） |

## 关键方法学（复现或扩展前必读）

1. **counts 精确档是「行为盲地板」**，不是近似。`unseen` 池 = 三家暗牌 + 活牌山 +
   王牌区（−宝牌指示）；目标家的 `h_j = 13 − 3×副露` 张暗牌是该池的均匀随机子集，
   故 `P(count_i = k) = C(u_i,k)·C(N−u_i, h_j−k) / C(N, h_j)` 精确成立（超几何）。
   自检：`Σalloc == 13−3×melds` 应 100% 一致；`Σunseen − (三家暗牌 + 活牌山)`
   应 ≈ **+13**（王牌 14 − 宝牌指示 1）。

2. **一切「模型学到了 X」的断言都要减去地板。** ⚠️ 2026-10-20 审计修正：~~A3 规则法
   「天然顺序盲」~~ **不成立**——A3 经 `early_discard_mask`（前 3 次舍牌）与
   `timeline_features`（摸切/手切、末动作、巡目、舍牌数）消费 `tl_tokens`，对切牌顺序有
   真实响应（同一 token 置换下 21.5% 元素变化、`max|Δ|=1.34` 张），只能称「**规则法地板**」。
   真顺序盲地板 = `rule_predict(order_blind=True)`（token 规范化排序：破坏顺序、保留多重集），
   其格内 ΔΔ ≈ −0.008 ~ −0.041（≈0，证明格内设计真的消掉了组成混杂）。A2b 扣**真顺序盲
   地板**的复现率：全样本 **66.6%** / 非立直 **81.9%** / 立直 **+31.0%**——旧口径
   （扣 A3，41%/67%/−12%）系统性低估，已作废；立直层仍是主要缺口。

3. **合并跨格会引入组成混杂。** 「自然序 vs 非自然序」若直接池化，连只认集合的预测器都能
   "复现" 58%——因为两组的 `m` 组成不同。必须**格内（花色 × 中间张）算 ΔΔ 再加权**，这样每格两组
   的舍牌集合完全相同，预测器的格内 ΔΔ 只可能来自巡目/立直等组成差异（对真顺序盲地板而言）；
   A3 这类有顺序响应的参照系，其格内 ΔΔ 含真实顺序效应，不能当「上界」用（见上条）。

4. **反事实要配逐样本期望。** 用池化真值 ΔΔ 当期望会失真（池化值本身可能≈0，导致
   "实现比例"分母爆炸）。正确做法：逐样本 `expected = ∓ΔΔ_cell`，再回归
   `measured = a + b·expected`，看 `b`（1=完美，0=无响应）与 `r²`。

5. **区分「集合级」与「顺序级」。** 切了 x ⇒ x±1 不太可能有，是集合级（A2b 已学到
   75~79%）；先切 9s 还是先切 7s ⇒ 8s 概率不同，是顺序级（A2b 在立直层为零）。
   两者混在一个 CE 里会互相掩盖。

6. **`tl_tokens` 的巡目字段目前是常数**（`dataset.py` 把快照时目标家的 `at_turn`
   写进全部 26 个 token，见主 README §11.5-①，2026-10-20 复核仍未修），所以任何"时序流学到了时间信息"的推断都要
   先用反事实验证。实测斜率 `b = 0.0144`、`r² = 3.3%` ⇒ 时序流对顺序近乎无响应，
   A2b 表现出的顺序敏感性来自 **obs 空间流的按槽牌河通道**（`encode_kawa`，8 通道/槽）。

## results/

已落盘的 json 是主 README **§11 / §12** 表格数字的原始出处（§12 部分地板为 10-20 前
的 A3 口径，已被真顺序盲地板修订，现行值以 `order_metrics.py` 输出为准）：

| 文件 | 来源 |
|---|---|
| `diag1.json` | `s315_arch_audit.py` |
| `probe.json` | `s315_head_probe.py`（P0~P3 全表） |
| `local_order_report.json` | `s3158_local_report.py`（PART A/A'/C/D） |
| `local_order_report2.json` | `s3158_local_strata.py`（分层 + 反事实回归） |
| `local_order.json` | 早期 4000 样本冒烟版，仅留档，勿引用 |
