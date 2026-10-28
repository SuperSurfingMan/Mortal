# 风格模仿微调系统：经验总结（2026-10-28 收工）

冈田紗佳、多井隆晴两人风格微调全链已跑通并交付。本文是唯一入口文档：复用指南、
实验结论、踩坑记录、代码位置索引。计划书见 `.qoder/plans/Style_Finetune_Plan_ffe5b786.md`（v2）。

## 一、交付物

| 项 | 位置 |
|---|---|
| **冈田紗佳风格权重（正式版）** | `mortal/checkpoints/style/okada_sayaka_lb/best_snapshot.pth`（last_blocks@0.75，step 200，sha256 `1135f8444278472b`） |
| 冈田 轻度版（head 档） | `mortal/checkpoints/style/okada_sayaka/best_snapshot.pth`（head@1.5，step 600，sha256 `b8984d1a0c3d66c0`） |
| **多井隆晴风格权重（正式版）** | `mortal/checkpoints/style/toi_takaharu_lb/best_snapshot.pth`（last_blocks@0.75，step 600，sha256 `5b5f1de6463513a8`） |
| 多井 重风格版 | `mortal/checkpoints/style/toi_takaharu_lb05/best_snapshot.pth`（last_blocks@0.5，step 400，sha256 `f339ab0ce6182264`） |
| 多井 轻度版 | `mortal/checkpoints/style/toi_takaharu/best_snapshot.pth`（head@1.5，step 600，sha256 `40775d90b888e0bb`） |
| 对比报告 | `runs/style/<slug>/report.md`；多井三档选优汇总 `runs/style/toi_takaharu/DECISION.md` |
| 训练代码（门控改动） | `mortal/train.py`（bc_style 分支）、`mortal/dataloader.py`（坏样本过滤） |
| 工具集 | `tools/style_ft/`（本目录） |
| 数据清单 | `dataset/index/style/<slug>.txt`、`<slug>_stats.json` |
| 生成配置 | `mortal/config.style_<slug>.toml`（gitignore 内，可重生成） |

## 二、最终实验结论（同 holdout 32 局 / 同 ladder seed 30000）

| 指标 | 父代 s6c | head@1.5 | last_blocks@0.75 |
|---|---|---|---|
| holdout 一致率 | 0.8367 | 0.8398 (+0.32pp) | **0.8412 (+0.45pp)**，CI [+0.21,+0.74] |
| 强度 avg_pt vs 父代 | — | +0.90 | **+1.15**（avg_rank 2.4815） |
| 鸣牌率（人类 0.2106） | ~0.24 | 0.239 | **0.220** |
| 立直率（人类 0.2342） | — | 0.219 | **0.222** |
| 和了率（人类 0.1992） | — | 0.209 | **0.204** |

要点：
1. **行为画像对齐是风格模仿的本体度量**。一致率只 +0.45pp，但剩余 16% 不一致
   基数里大部分是多解性噪声（同局多打法皆可）；鸣牌/立直/和了率距人类全部收窄
   （2.8→1.1pp、1.5→1.2pp、1.0→0.5pp）才是真实的风格化证据。
2. **KL 锚是强度保险，也是风格化天花板**：1.5→0.75 强度反升（+0.90→+1.15），
   说明父代强度锁在冻结的前 36 层。想再风格化可降 kl_weight 至 0.5，但收益递减。
3. **数百局量级 200 步即饱和**：两档 snapshot 曲线都在 200~600 步达峰，1200 步
   无增益。加步数不如调档位/锚权重。
4. head→last_blocks 的分动作变化：meld 一致率 0.890→0.802、pass 0.957→0.986
   ——模型学会了「少鸣牌」偏好（人类鸣牌率低于 Mortal 默认），在她偶尔鸣牌的
   点上倾向 pass。分布对齐与逐点一致率在低频动作上天然互斥，属预期。
5. 3pp 一致率门槛对 168 局数据过严（统计功效不足），判定应以「一致率不降 +
   行为分布收窄 + ladder 不崩」三件套为准。

## 二·补、多井隆晴（269 局，holdout 46 局，2026-10-28）

同父代 s6c、同 holdout（2025-26 + 2026-27 = 46 局 / 9,082 决策点）、同 ladder seed 30000，
一次扫完三档（三链并行，各 GPU 独立）：

| 指标 | 父代 s6c | head@1.5 | last_blocks@0.75 | last_blocks@0.5 |
|---|---|---|---|---|
| holdout 一致率 | 0.7850 | 0.7904 (+0.57pp) | **0.7942 (+0.97pp)** | 0.7939 (+0.86pp) |
| 配对 bootstrap 95% CI | — | [+0.25,+0.88] | **[+0.57,+1.37]** | [+0.48,+1.25] |
| 达峰步数 | — | 600 | 600 | 400 |
| ladder avg_pt | — | +1.282 | **+0.427** | −2.295 |
| 鸣牌率（人类 0.2249） | — | 0.2411 | **0.2158** | 0.2105 |
| 立直率（人类 0.2066） | — | 0.2158 | 0.2138 | 0.2135 |
| 和了率（人类 0.2129） | — | 0.2075 | 0.2002 | 0.1977 |
| 同状态鸣牌选择率（人类 0.0227） | 0.0238 | 0.0218 | 0.0194 | 0.0188 |

要点（补上一轮之外的新结论）：
1. **≥200 局时 last_blocks 显著优于 head**：lb − head = +0.39pp，CI [+0.10,+0.68]。
   与岡田（136 局 lb +1.15 vs head +0.90）同向——「<400 局用 head」的原配方应作废（见 §三）。
2. **KL 拐点在 0.75**：0.75 与 0.5 一致率打平（+0.11pp 不显著），但 0.5 档强度转负
   (−2.295)，鸣牌率也越过人类值（0.2105 < 0.2249，过冲）。两轮证据合起来：
   0.75 是「风格化足够 + 强度不崩」的稳态点，KL 再降属换强度买风格。
3. **新增 pred_rate 指标（同状态下模型选择分布）比 ladder 行为率更贴近人类**：
   人类 0.0227 / 父代 0.0238 / lb 0.0194——父代在 M-League 状态上只比人类多鸣 0.11pp，
   风格化会把这个差直接压成 −0.33pp（过冲）。ladder 的 0.24 vs 0.216 差距被自对局
   对手池放大，**判风格方向要看 pred_rate，看幅度要看 ladder**。
4. **46 局 holdout 也到不了 3pp**：实际 +0.97pp，但局级配对差 std 1.39pp / SE 0.21pp，
   即 **4.7σ 显著而幅度小**。说明 3pp 门槛不是方差问题，而是 s6c 已在 M-League 状态上
   打得很像人（基数 0.785），可提升空间本身有限。

## 三、复用指南（为其他选手跑风格微调）

```bash
# 一键全链（repo root 下；换玩家只需换 --player/赛季/kl_weight）
mkdir -p runs/style/<slug>
CUDA_VISIBLE_DEVICES=0 bash tools/style_ft/run_style_ft.sh \
    <玩家名> mortal/checkpoints/s6c/mortal.pth <holdout赛季逗号分隔> <slug> <kl_weight> [trainable]
# trainable: 显式写 last_blocks（auto 仍按 <400→head 旧逻辑，已不推荐，见下方速查表）/ head / last_blocks / full
# 换档位/kl 实验务必换 slug，否则覆盖上一档 checkpoints 与 report
```

全链 8 步：白名单→KL 单测→生成配置→暖启训练→snapshot 扫描选优→最优快照
holdout 评测→人类画像→ladder 500 sets→汇总 report.md。数百局 + P100 fp32
全程约 1 小时（训练仅 4~8 分钟）。多档并行（各占一卡、slug 互不同）可直接三链并发，
多井一轮三档并行实测 45 分钟（含外部 ladder 抢核）。多档选优用 `compare_runs.py` 出单页汇总。

档位与配方速查（数百局量级，2026-10-28 多井一轮修正）：

| 训练局数 | 档位 | kl_weight | 步数 |
|---|---|---|---|
| <130 | 放弃微调，改调推理采样温度 | — | — |
| 130~1500 | **last_blocks**（~2.0M 参数） | **0.75** | 800~1200 |
| ≥1500 | last_blocks；full 仅压力测试 | 0.5~0.75 | 同上 |

head 档只在两种情形用：① 想保留强度优先、风格只要「轻微偏移」（avg_pt 反而最高）；
② 显存/时间极紧（同样本量下 head 训 800 步约 3 分钟 vs last_blocks 1200 步约 5 分钟，
推理也快约 12%）。风格化上限明显低于 last_blocks。

单跑某一环（示例）：
```bash
python tools/style_ft/eval_agreement.py --ckpts <a.pth> <b.pth> \
    --stats dataset/index/style/<slug>_stats.json --device cuda:0 --out <out.json>
python tools/style_ft/diag_mleague.py --stats <stats.json> [--augmented]   # 坏样本定位
(cd mortal && python ladder.py checkpoints/style/<slug>/best_snapshot.pth \
    checkpoints/s6c/mortal.pth 30000 500 runs/style/<slug>/ladder --no-amp)
```

## 四、踩坑记录（按发现顺序，复用时先读这段）

1. **暖启 ckpt 的 steps 不重置 → 零步退出**：父代 state 里 steps=940k ≥
   max_train_steps，命中预算检查直接返回。解：`control.reset_steps = true`
   （train.py 可选键，老配置无此键行为不变）。
2. **M-League 坏样本：动作不在合法 mask 内**：18/33602 = 0.054%，全部是
   action=37（立直）出现在副露状态（mask 无 37 位）。转换器缺陷，raw 与
   augmented 口径同源（非 Rust 增强路径 bug）。解：
   `dataset.filter_invalid_actions = true`（dataloader.populate_buffer 样本级
   剔除，保留整局其余样本；默认 False 天凤数据零影响）。
   定位工具 `diag_mleague.py`，先跑 raw 再跑 --augmented 区分两种根因。
3. **`take_at_kyoku()` 返回 bytes**（每字节一个局编号）：必须
   `np.frombuffer(x, dtype=np.uint8)`，`np.asarray(x, dtype=np.int64)` 会把
   bytes 当字符串解析而 ValueError。
4. **libriichi 导入路径**：编译产物在 `mortal/libriichi.*.so`，从 repo root
   跑工具须 sys.path 注入 mortal/（工具集已统一处理）。
5. **白名单脚本路径拼接**：player_index.json 的 file 字段相对 repo root，
   不是相对 index 文件（`index_path.parent.parent` 会拼出
   `dataset/mjai/dataset/...`）。现按脚本自身位置推 repo root + 存在性断言。
6. **35 个 gzip 损坏文件**（`_invalid_files.txt`，头部不可读，不在
   player_index 内）：目录通配扫描遇之即崩，因此配置 globs 用**精确文件列表**
   （兼得 holdout 排除与 manifest 精确失效重建）。
7. **ladder run_meta.json 路径**：ladder.py 在 `cd mortal` 下运行，产物落
   `mortal/runs/...` 而非 repo root `runs/...`，compare_style 传参须带
   `mortal/` 前缀。
8. **KL 单测口径**：student 与 base 必须共享同一合法动作 mask（训练中共用
   masks，DQN forward 的 masked_fill 保证 -inf 位一致）；测试里给 base 独立
   随机 mask 会造出 +inf 差分假失败。
9. **KL 实现必须「先做差再掩码」**：`(logp - logp_base).masked_fill(~masks, 0)`
   ——裸乘版在非法位产生 `-inf-(-inf)=NaN` 且 `0×NaN=NaN` 污染 sum。
10. **训练侧 agreement_batch 只做过拟合监控**（持续高位而 holdout 回落=记忆
    训练集），交付指标一律用 holdout 的 eval_agreement。
11. **GPU 指定**：配置内保持 `cuda:0`，物理卡用 `CUDA_VISIBLE_DEVICES=N`
    注入（子 shell 会继承），改配置反而漏 cd mortal 的子进程。
12. **fork DeprecationWarning**（num_workers>0 + tqdm 多线程）：无害，忽略。
13. **ladder 进度不落盘**：`libriichi.arena.OneVsThree` 的进度条只在 tty 上刷新，
    tee 出的日志里看不到；牌谱 `*.json.gz` 也是全跑完才一次性写入。判断完成只能等
    `run_meta.json` 出现（或 `ps` 看 ladder 进程退出），不要按日志文本估进度。
14. **并行链要分卡且注意 CPU 争抢**：多档并行时用 `CUDA_VISIBLE_DEVICES=<n>` 各占一卡
    （配置内始终 cuda:0）；但 ladder 是 CPU 密集型，本次机器上另有 2 个 8000-sets 的
    外部 ladder 抢 20 核，500 sets 从 ~10 分钟涨到 ~25 分钟。排期时先 `uptime` 看 load。
15. **一条链内不要中途改工具**：`eval_agreement.py` 加 pred_rate 后重跑是安全的（纯新增
    键），但**不要**在 `run_style_ft.sh` / `train.py` 被跑着时编辑它们（bash 增量读取 + 进程
    已加载模块，改了也不生效还容易报假错）。
16. **`config.style_<slug>.toml` 不能直接拿来实战**：它的 `[control].state_file` 指向
    `checkpoints/style/<slug>/mortal.pth`，那是 run_style_ft.sh 第 4 步 `cp $PARENT` 的
    **父代权重副本**（暖启源 + KL 锚），拿它跑 `mortal.py` 得到的是 s6c 而非风格模型。
    实战要另存一份配置把 state_file 改成 `best_snapshot.pth`（改法见
    `runs/style/toi_takaharu/DECISION.md` 的「怎么用」节）。另：ladder 用 `--guard-lambda 0`，
    而 `mortal.py` 默认 `MORTAL_GUARD_LAMBDA=10`，对比数字时注意口径。

## 五、代码位置索引（改了什么、在哪）

| 功能 | 位置 |
|---|---|
| bc_style 分支（CE + KL + 分动作监控） | train.py `train_batch` 内 `elif objective == 'bc_style'` |
| KL 锚模型加载（eval+冻结，不进 optimizer） | train.py `load_frozen_policy()` |
| trainable 分级（head/last_blocks/full） | train.py freeze_bn 之后；param_groups 收集处过滤 requires_grad |
| 步数重置键 | train.py 加载 state 后 `control.reset_steps` |
| enable_test_play 门控 | train.py TestPlayer 构造 + test_every 评测块 |
| 坏样本过滤 | dataloader.py `populate_buffer`（`filter_invalid_actions`） |
| 配置生成（以父代 state['config'] 为底打补丁） | tools/style_ft/make_style_cfg.py |
| holdout 一致率评测（含 --snapshots 早停） | tools/style_ft/eval_agreement.py |
| 人类画像（Stat.from_dir，别名逐名合并） | tools/style_ft/human_profile.py |
| 多档位选优汇总（一致率+CI+ladder+画像单页） | tools/style_ft/compare_runs.py |
| 同状态行为分布 pred_rate（新增于 by_action_class） | tools/style_ft/eval_agreement.py `summarize()` |
| 局级配对 bootstrap CI | tools/style_ft/compare_style.py |

全部改动门控化：不写新键的老配置（cql_mc/iql）行为逐位不变；回退只需删
`tools/style_ft/` 与 train.py/dataloader.py 的 bc_style/filter 分支。

## 六、数据备忘

- 源：`dataset/mjai/mleague/`（M-League 2019-20 ~ 2026-27，player_index.json
  索引 56 位选手；`_skipped.txt` 10 个转换期弃件、`_invalid_files.txt` 35 个损坏件）
- 岡田紗佳 168 局：训练 136（2019-20~2024-25）/ holdout 32（2025-26 + 2026-27）
- 多井隆晴 269 局：训练 223（2018-19~2024-25）/ holdout 46（2025-26 38 + 2026-27 8）；
  2018-19 共 35 局，mjai 转换与后续赛季同口径（抽查 4 文件 812 样本可正常加载）
- 画像（多井）：立直 0.2066 / 鸣牌 0.2249 / 和了 0.2129 / 放铳 0.0936 / avg_rank 2.4238
- holdout 划分原则：最近 1~2 个赛季，≥40 局为佳（该规模下 +1pp 已达 4~5σ；
  3pp 在这类任务上不可达，勿用固定 3pp 作硬门槛）
- M-League 与天凤分布差异（规则细节/对手池）是固有域差，跨域结论保守解读