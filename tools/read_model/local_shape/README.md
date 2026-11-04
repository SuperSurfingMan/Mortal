# Phase 2：局部形状读数（Local Shape）

> 目标：把「某一家在某张牌附近的搭子形状」从当前 A2c 的**逐张独立边际**，
> 推进到可测量、可改进的**局部联合形状**。P0 只做测量与基线，不训练。

## 最终结论（2026-11-03，覆盖全文）

**有条件成立，不重启 200k。**

1. **口径修正**：`active_any` 下最大类是 **EMPTY（空窗口）**，占 56.04%，
   使 micro top1 被「永远猜空」主导。cond 口径（筛非空 + EMPTY 置 0 重归一）下
   A2c 的归一化增益从 3.2% 升到 11.1%（§5c.1）。
2. **分层互有胜负**（§5d）：P2c 在 all/active_any **+1.20pp**、
   riichi/active_any **+5.52pp**、riichi/all_centers **+5.10pp**；
   A2c 在 all/active_both **+1.74pp**、clean_t8/active_both **+3.67pp**。
3. **语义标签 NO-GO**（§5c）：压缩损失 0.0539 nats > 共享红利 0.0500 nats。
4. **路由**：逐窗口不值得（全局 +0.07pp vs 2× 推理）⇒ 按场景选（§5d / `router.py`）。
5. **定位**：**P2c 在 15/18 个 cell 上 CE 最低**（3 个例外全是 active_both 系，
   那里评估集 oracle prior 更低）；P1/P2b 也全面优于 A2c；P2a 全面最差，
   立直显式特征路线证伪。
6. **上限探针（§5e / P0-a）**：cond 下 strict tier 信息只有 **+0.023 nats**
   （all/active_any）、+0.129（clean_t8/active_any）⇒ 任务信号确实稀薄；
   **但 tier 不是上界**：在 8 个“tier 优于 prior”的 cell 上，A2c 低 0.135~0.209 nats；
   且失配集中在 active_both/立直/called（mask 本身含信息，tier 未用）。
   ⇒ **P1/P2a/P2c 的 FAIL 是 top1 门禁的判定，不是能力不足**；在 CE 上它们
   已超过 A2c。“关闭”改为「**实现未达 A2c（top1），但上限未被证明**」。
   另：`bayes_ceiling.py` 的 Dirichlet α 语义错误已修（MC CE 曾偏高 0.22~0.35
   nats，top1 不受影响）。

下文 §1~§4 为 P0 口径与结果，§5 为前四轮（历史），§5b~§5e 为上限/口径探针，
**结论以本节与 §5b~§5e 为准**。


## 0. 为什么开这个阶段

既有诊断（README §12 / §13）已经证明：

- **集合级**局部信号很强：手切 x ⇒ x±1 持有下降，A2b 复现 75~79%；
- **联合形状**很弱：7s+9s 都手切 ⇒ 8s 几乎不在，A2b 只复现 49%；
- **顺序/区域级**更弱：单张大效应格只复现 14~22%，±2 窗口 all +0.006（CI 跨 0）、立直 −0.066；
- 继续加顺序特征/时序流已被五版 20k 全 FAIL 否定（b=0.0038~0.0131 vs 0.3）。

因此本阶段不再从「顺序」入手，而是从**输出参数化**入手：当前 H1 输出 34 个独立
`P(count=k)`，无法表达「8s 附近是 78 两面还是 79 嵌张还是 88 对子」。P0 先量清楚
当前独立边际到底能读到什么程度，再决定是否值得做局部形状头。

## 1. P0 口径

数据：A2c 的 40000 样本诊断缓存（202609 holdout，seed 20261016）：

```
/tmp/read_model_diag_cache_a2c/cache.npz
```

缓存含 `AP[n,3,34,5]`（A2c alloc_probs）、`RAP`（A3 规则法）、`EC/REC`
（期望计数）、`alloc`（真值计数）、`CPMF0`（counts-only P(count=0)）、
`riichi/called/turn/hc`。不需要 GPU。

### 1.1 三种局部读数

| 名称 | 定义 | 用途 |
|---|---|---|
| **成对共持** | 同花色相邻 `d=1`、隔张 `d=2`，两张计数都 >0 | 最细的联合信号；AUC/log-loss |
| **三张形状** | 中心 `m` 的 `(m−1, m, m+1)` 计数截断 0/1/2 ⇒ **27 类** | 局部“搭子形状”的主指标 |
| **五张窗口** | `m−2..m+2` 的 5 维计数 | MAE，衡量区域级整体读数 |

P0 的「形状」用**计数形状**，不是「两面/嵌张」分解：后者依赖手牌分解歧义，
留到 P1 用 `structure.py` 的软标签做。计数形状无歧义、可复现，适合做基线。

### 1.2 基线

- **prior_major**：同子集上「每个中心永远猜最常见形状」的准确率。
  这是**无信息先验**，且在评估集上拟合（oracle 先验）⇒ 对模型是**保守**基线。
- **A3 规则法**：`RAP` 的独立边际乘积，人类规则参照。
- **counts-only**：`CPMF0` 的独立乘积（只看可见张数，不看行为），成对指标用。
- **A2c-product**：A2c 的独立边际乘积。当前模型没有联合头，这就是它唯一能形成的
  「局部联合读数」；P0 要量的是它离先验/A3 有多远。

### 1.3 分层

`all / riichi / nonriichi / called / turn>=8 / clean_t8`，以及：

- `all_centers`：全部 21 个中心；
- `active_any`：目标家至少手切过 `m−1` 或 `m+1`（有局部动作信号）；
- `active_both`：两侧都手切过（信号最强，样本约 0.9%）。

## 2. 运行

```bash
# 只读缓存，CPU，约 1~2 分钟；必须用 conda mortal 的 python
bash tools/read_model/local_shape/run_p0.sh

# 或手动
python tools/read_model/local_shape/eval_local_shape.py \
  --cache /tmp/read_model_diag_cache_a2c/cache.npz \
  --out tools/read_model/local_shape/results
```

产物：

- `results/p0_local_shape.json`：全部数字；
- `results/P0_REPORT.md`：表格 + 结论；
- `mortal/logs/read_model/local_shape/p0_*.log`：运行日志。

如果缓存不存在，先跑（GPU，约 2.5 分钟）：

```bash
python tools/read_model/diagnostics/s3158_local_extract.py 40000
```

## 3. P0 结果（2026-11-02）

见 `results/P0_REPORT.md`。核心数字：

- 成对共持 d=1：A2c AUC **0.725** vs A3 0.679 vs counts-only 0.661（base 0.162）；
  d=2 为 0.748 / 0.707 / 0.679；
- 三张形状（active_any）：A2c top1 **57.5%** vs 先验 **56.0%**（Δ **+1.4pp**），
  top3 79.0% vs 先验 74.9%；A3 top1 54.7%；
- active_both：A2c top1 73.9% vs 先验 73.5%（Δ +0.4pp）；
- **立直 active_both：A2c 66.9% vs 先验 67.7%（Δ −0.8pp）；top3 91.7% vs 先验 97.7%**
  ——立直层连先验都没打过；副露 active_both Δ ≈ 0；
- 唯一像样的增量在 **clean_t8 active_any**：A2c 48.4% vs 先验 45.8%（+2.6pp）；
- 温度扫描 top1 不变、CE 在 T=1 最低 ⇒ **不是校准问题，是独立边际的表达问题**；
- 五张窗口 MAE：A2c 0.472 vs A3 0.495 张/格。

## 4. P0 判定与 P1 门禁

**GO → P1 冻结骨干局部形状头**：A2c 的 pairwise AUC 0.72~0.84 说明边际里有排序信息；
但 27 类 top1 只比先验高 0.4~1.4pp（全中心仅 +0.6pp），立直 active_both 甚至为负，
说明“联合读数”几乎没被表达出来，存在明确 headroom。

P1 门禁（预注册）：

1. 冻结 A2c 骨干，只训局部头；
2. 在 `active_any` 上比每中心先验高 **≥5pp top1**（或 **≥0.10 nats CE**）；
3. 在 `active_both` 上比先验高 **≥3pp**；
4. 且优于 A2c 独立乘积；
5. 若解冻微调，`alloc_ce`/`wait` 不得退线。

不满足 → 停止局部头，转立直显式特征 + 参考模型蒸馏；不要把预算烧在 200k 上。

## 4b. P1 结果（2026-11-02，FAIL）

实现：

- `features.py`：冻结 A2c，导出 `h/g/e_T/s_belief` + `alloc/unseen/turn/riichi/called/hc/tg`；
- `p1_local_head.py`：712→256→256→27 的 per-target 局部头（765,777 参数），
  输入 = h 的 m±2 五列 + e_T/s_belief/g + target one-hot + turn/riichi/called；
- `run_p1.sh`：GPU4 / tmux；train 20000（202608，按文件 80/20）、val 5000、eval 40000（202609）。

关键结果（all 层；完整表见 `results/p1/P1_REPORT.md`）：

| 子集 | head top1 | prior(train) | A2c product | head top3 | A2c top3 | head CE | A2c CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| active_any | 0.5603 | 0.5604 | **0.5746** | 0.8033 | 0.7902 | **1.4089** | 1.4323 |
| active_both | 0.7265 | 0.7350 | **0.7389** | 0.9125 | 0.9058 | 0.9300 | **0.9114** |
| riichi active_both | 0.6412 | 0.6547 | **0.6686** | 0.9159 | 0.9170 | 1.0261 | **1.0173** |
| clean_t8 active_any | 0.4647 | 0.4579 | **0.4839** | 0.7449 | 0.7396 | 1.6678 | **1.6641** |

- **active_any top1 = 先验（Δ −0.0pp），低于 A2c product 1.4pp**；
- **active_both −0.85pp、立直 active_both −1.35pp**；head 只在 CE/top3 上略好，
  但 top1 排序没有超过 A2c 独立乘积；
- 训练从 epoch 2 起过拟合（val top1 0.3240→0.2987），冻结 `h` 里没有足够的
  局部联合信号，头容量/正则化不是主因；
- **判定 FAIL**：按预注册门禁停止冻结局部头，不跑 200k。

## 4c. P2a 结果（2026-11-03，FAIL）

实现：`p2_riichi_head.py` + `run_p2.sh`；在 P1 的 712 维局部特征上追加 50 维立直特征
（宣言牌 one-hot 34、is_decl、同花色距离、立直后牌河计数、宣言后巡目、宣言牌摸切、
窗口内宣言牌/立直后牌河各 5 维）；头 762→256→256→27，dropout 0.1，冻结 A2c。

关键结果（all/riichi；完整表见 `results/p2/P2_REPORT.md`）：

| 子集 | P2 head top1 | P1 head | prior(train) | A2c product | P2 head CE | A2c CE |
|---|---:|---:|---:|---:|---:|---:|
| riichi active_both | 0.659 | 0.641 | 0.655 | **0.669** | 1.079 | **1.017** |
| riichi active_any | 0.473 | 0.444 | 0.465 | **0.478** | 1.657 | **1.770** |
| all active_any | 0.566 | 0.560 | 0.560 | **0.575** | 1.468 | **1.432** |

- 立直特征把 riichi active_both 从 P1 的 0.641 拉到 **0.659**（+1.8pp），
  但只比先验高 0.4pp，仍**低于 A2c product 0.669**；CE 也高于 A2c（1.079 vs 1.017）；
- 门禁：`riichi_both` CE 条件满足（1.519→1.079），但 **beat A2c = False**，总判 **FAIL**；
- 训练同样 epoch 1 后过拟合（val riichi any 0.5205→0.4709），显式立直特征不足以
  让冻结表示读出局部联合形状；
- **判定 FAIL**：不进入 200k，不把 P2a 当头继续调参。

## 4d. P2b 结果（2026-11-03，PASS via CE）

实现：`p2b_distill_data.py` + `p2b_distill_head.py` + `run_p2b.sh`。
在 train（202608）的立直子集上，用 `behave` 采样 1631 个目标 × 512 手，
生成局部 27 类软标签；训练时把硬标签与参考软标签按 w=0.5 混合，其余与 P2a 相同。

关键结果（202609；完整表见 `results/p2b/P2B_REPORT.md`）：

| 子集 | P2b head top1 | P2a head | prior | A2c product | P2b CE | A2c CE |
|---|---:|---:|---:|---:|---:|---:|
| riichi active_both | 0.667 | 0.659 | 0.655 | **0.669** | **1.006** | 1.017 |
| riichi all_centers | 0.363 | 0.356 | 0.347 | 0.347 | **2.047** | 2.187 |
| all active_any | 0.569 | 0.566 | 0.560 | **0.575** | **1.404** | 1.432 |
| clean_t8 active_any | 0.473 | 0.467 | 0.458 | **0.484** | **1.659** | 1.664 |

- **riichi active_both：top1 0.667（+1.2pp vs 先验、+0.8pp vs P2a），与 A2c 0.669 基本持平（−0.2pp）；
  CE 1.006 优于 A2c 1.017**；
- **门禁 PASS**：riichi_both CE 条件（1.519→1.006）满足，且 CE beat A2c；
  all active_any guard 通过（0.569 vs P1 0.560）；
- 但注意：**top1 仍略低于 A2c**，这次是**概率校准/CE 的胜利**，不是 top1 排序的胜利；
  top3 91.62% vs A2c 91.70%，也基本持平；
- 训练 best epoch 3，之后过拟合（val riichi any 0.523→0.501）。

## 4e. P2c 结果（2026-11-03，FAIL）

实现：`p2c_unfreeze.py` + `run_p2c.sh`。解冻最后 2 个 ResBlock + **H1（alloc 头）** + P2 局部头；
其余 A2c 部分冻结；BN 保持 eval。损失 = 局部 27 类硬标签 CE（active 加权）+ 1.0·alloc CE；
backbone lr 5e-6、H1 lr 1e-4、head lr 5e-4；202608 train / 202608 20% val / 202609 eval 40k；
计划 20k 步，实际 12k 早停（best step 6000）。

关键结果（202609；完整表见 `results/p2c/P2C_REPORT.md`）：

| 子集 | P2c top1 | A2c product | P2c top3 | A2c top3 | P2c CE | A2c CE |
|---|---:|---:|---:|---:|---:|---:|
| riichi all_centers | **0.3745** | 0.3471 | **0.6611** | 0.5796 | **1.9602** | 2.1872 |
| riichi active_any | **0.4815** | 0.4779 | **0.7662** | 0.6927 | **1.5477** | 1.7701 |
| **riichi active_both** | **0.6602** | **0.6686** | **0.9364** | 0.9170 | **0.9386** | 1.0173 |
| all active_any | 0.5714 | 0.5746 | **0.8128** | 0.7902 | **1.3536** | 1.4323 |

eval alloc CE：**0.6807**（A2c 0.6516，+0.0291）。

- **立直整体读数明显改善**：all_centers top1 +2.7pp、top3 +8.2pp、CE −0.227；
  active_any top1 +0.36pp、top3 +7.4pp、CE −0.222；
- **但关键子集 active_both top1 反降 0.84pp**（0.6602 vs 0.6686），虽然 CE 更好
  （0.9386 vs 1.0173）、top3 更高（0.9364 vs 0.9170）；
- **alloc CE 退化 +0.0291**（0.6807 vs 0.6516），超过 +0.005 容差；
- 门禁：strict top1 = False、CE = True、all_any 容差 = True、alloc = False ⇒ **FAIL**；
- 结论：解冻最后 2 个 Block + H1 能把**校准/top3/立直整体**做好，但**无法在不伤 alloc、
  且不牺牲 active_both top1 的前提下超过 A2c**。

## 5. 前四轮结果（P1/P2a/P2b/P2c，历史记录）

> ⚠️ 本节结论写于 cond 口径修正之前，其中的「关闭局部形状线」已被 §5d 改为
> 「有条件成立」。保留作历史记录。

- **P1 冻结局部头：FAIL**；**P2a 立直显式特征：FAIL**；**P2b 蒸馏：PASS via CE**；
  **P2c 受控解冻：FAIL**（strict top1 + alloc 未过）。
- **按预注册门禁关闭局部形状线**：不再跑 200k，不再调冻结/解冻头；
  P1/P2a/P2c 都没有把 `riichi active_both` top1 推过 A2c。
- 若目标只是守卫/风险概率，P2b/P2c 的 CE 改善（riichi active_both 1.017→1.006/0.939）
  可作为**校准**参考，但不能声称“局部形状读得更准”。
- **P3（仅在新任务/新标签下重启）**：5 类分解软标签或结构化解码；当前不启动。

## 5b. BAYES_CEILING：上限探针（2026-11-03）

`bayes_ceiling.py` 复用 `bayes_bound.py` 的三层 tier（counts / hard / behave）估计
`P(shape_j=k|o)`，判定 P1/P2a/P2c 的 FAIL 是否属于“后验本身接近均匀”。

**实现与口径**

- eval 划分与 P0 缓存逐样本校验 `alloc`（PASS）；mask/active 直接复用 `eval_local_shape`。
- `counts` 为**精确多元超几何**（全量 40000、无 MC 误差），与缓存 `CPMF0` 自检 **3e-08**；
  并额外给出“边际乘积 vs 联合（负相关）”分解。
- `hard`/`behave` 为 MC（6000 快照；M=4096；chains=256；iters=128），
  三档 M=256/1024/4096 同源截断做收敛检查（ΔCE ≤ **0.008 nats**）。
- MC 后验含 Dirichlet(α/K=0.5) 平滑，防 `-log 0` 爆炸。

**结果（top1 为主判据）**

| 子集 | prior | counts | hard | behave | A2c | P2c | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| all/active_any (n=39.3万) | 0.5604 | 0.5541 | 0.5553 | 0.5582 | 0.5746 | 0.5714 | SIGNAL_SHALLOW |
| all/active_both (n=2.2万) | 0.7350 | 0.7244 | 0.7432 | 0.7489 | 0.7389 | 0.7343 | HEADROOM（1.6σ） |
| clean_t8/active_both (n=1.05万) | 0.6853 | 0.6739 | 0.6914 | 0.7018 | 0.6897 | 0.6838 | HEADROOM（1.3σ） |
| riichi/active_both (n=2734) | 0.6770 | 0.6225 | 0.6714 | 0.6667 | 0.6686 | 0.6602 | SIGNAL_SHALLOW |

**关键结论**

1. 三层 tier 的信息集下，27 类形状 top1 增益 ≈ 0（甚至略负）；CE 全部高于标签边缘熵 ⇒
   “均匀子集”假设被手切行为打破（校准 gap **0.081**）⇒ **CE 不能作上限判据**，
   改用 top1（argmax 对单调失配稳健）。
2. **A2c 的 top1 高于所有 tier**（+1.6pp）⇒ tier 是信息子集（缺牌河顺序/宣言牌/
   逐巡事件），**不是紧上界**，“已达理论天花板”的说法不成立。
3. active_both 层 behave 比 prior 高 1.4~1.7pp、P2c 未拿到 ⇒ 名义 HEADROOM，
   但 n_tier 仅 3290/1536、**只有 1.3~1.6σ，不足支撑重启**。
4. `riichi/all_centers` 是唯一 **6.4σ** 显著异常：P2c 比 prior 高 2.8pp、比 tier 高 4.7pp，
   说明**立直相关特征有真实信息而 tier 未覆盖**；但该层 active_any/both 不显著。

**判定：P1/P2a/P2c 的 FAIL 主因是「任务信号稀薄」（A2c 全部超额仅 1.5pp），
不是训练/结构缺陷；同时不能声称已达理论天花板。局部形状线维持关闭。**
若未来重启，唯一有据的方向是立直子集的定向目标（all_centers 6.4σ），而非通用局部头。

## 5c. 语义标签路线验证（2026-11-03，NO-GO）

`semantic_label.py` 把 27 类计数形状映射到 **10 类局部结构角色**（EMPTY / MENJEN /
PAIR_C / PAIR_C_ONLY / PAIR_SIDE / RYANMEN / PENCHAN / KANCHAN / SINGLE_C /
SINGLE_SIDE）；`eval_semantic.py` 做零训练的四组诊断（90s）。

**设计前提的纠正**：`center` 本来就已知（21 中心分开建模），所以语义标签
**不增加信息**（同一中心内 `(1,1,0)` 固定是边张或两面）。唯一可能的收益是
**跨中心共享参数**（21×27 → 10 类 + 位置偏置）。

| 指标（all/active_any） | 27 类 | 10 类 |
|---|---:|---:|
| 跨中心 KL（可共享性） | 0.0716 | **0.0216**（降 70%） |
| 共享代价（全局先验 − 每中心先验） | 0.0716 nats | 0.0216 nats |
| A2c 归一化 top1 增益（cond） | **0.1108** | 0.0580 |
| A2c CE 降幅（cond） | **0.1041** | 0.0502 |

**判定**：压缩损失 **0.0539 nats** > 共享红利上限（KL 差）**0.0500 nats** ⇒ **NO-GO**，
不训练。差额仅 0.004 nats，是“差一点点但确实不够”。5 类 / 分解软标签同路线不再试。

### 5c.1 副产品：P0 以来的 micro top1 口径被 EMPTY 主导

`active_any` 下最大类是 **EMPTY（空窗口）**，占 **56.04%**。三种口径：

| 口径 | 27 类 prior | A2c top1 | 归一化增益 | CE 降幅 |
|---|---:|---:|---:|---:|
| micro（全样本） | 0.5604 | 0.5745 | 0.0322 | 0.1466 |
| nonempty（筛 y≠0，不重归一） | 0.2745 | 0.0873 | **−0.2580** | −0.1351 |
| **cond**（筛 y≠0 + EMPTY 置 0 重归一） | 0.2745 | **0.3549** | **0.1108** | **0.1041** |

读法：

1. **micro 的 +1.4pp 几乎全来自“更会判窗口有没有牌”**，不是“更会判形状”；
2. 只筛非空但不重归一时 A2c 只有 8.7%（低于先验 27.5%）⇒ 它的概率质量大量压在 EMPTY 上；
3. **公平口径（cond）下 A2c 真实增益是 +8.0pp top1 / 归一化 11.1%**，
   比此前认为的 +1.4pp 强得多。

另：`riichi/active_both` 的 cond 口径下 A2c top1 **0.6314 vs 先验 0.6875**、
CE **1.4052 vs 0.6454** ⇒ **A2c 在立直家的局部形状读数完全失效**；
10 类压缩后该 cell 的 CE 降幅仅 0.0063（几乎归零）。

**后续**：P1/P2a/P2b/P2c 的历史指标都应按 cond 口径重算，再决定“局部形状线关闭”
是否成立（当前关闭结论可能过严）。详见 `results/SEMANTIC_LABEL_REPORT.md`。

## 5d. cond 口径重算（2026-11-03）：P2c 分层互有胜负

`eval_cond_recheck.py` 把 4 个局部头 + A2c 放回 cond 口径（筛非空 + EMPTY 置 0
重归一），一次 replay + 一次前向，549s（含全部推理）。

**分层胜负（cond，相对 A2c；正 = 局部头更好）**

| 模型 | all/any top1 | all/both | riichi/both | clean_t8/both | all/any CE |
|---|---:|---:|---:|---:|---:|
| P1 | −0.0058 | −0.0319 | −0.0572 | −0.0479 | +0.0917 |
| P2a | −0.0301 | −0.0498 | −0.0551 | −0.0587 | −0.0514 |
| P2b | +0.0025 | −0.0331 | −0.0328 | −0.0467 | +0.1134 |
| **P2c** | **+0.0120** | −0.0174 | **+0.0169** | −0.0367 | **+0.1774** |

**主口径明细（all/active_any，n=172942）**

| 模型 | top1 | top3 | CE | 归一化增益 | CE 降幅 |
|---|---:|---:|---:|---:|---:|
| prior | 0.2745 | — | 2.2725 | 0 | 0 |
| A2c | 0.3549 | 0.6297 | 2.0359 | 0.1108 | 0.1041 |
| P1 | 0.3492 | 0.6750 | 1.9442 | 0.1029 | 0.1444 |
| P2a | 0.3248 | 0.6094 | 2.0874 | 0.0693 | 0.0814 |
| P2b | 0.3575 | 0.6784 | 1.9225 | 0.1143 | 0.1540 |
| **P2c** | **0.3670** | **0.6939** | **1.8585** | **0.1274** | **0.1822** |

**结论**

1. 原 P2c 的“strict top1 未过 A2c”**确实是 micro 口径造成的**：cond 下 +1.20pp
   top1、+6.4pp top3、−0.177 nats CE；立直 active_both 也 +1.69pp / −0.223 nats。
2. 但 **active_both 系反转**（all −1.74pp、clean_t8 −3.67pp）⇒ 真实情况是
   「**分层互有胜负**」，不是单向超越。
3. 叠加 alloc CE 退化 +0.029（P2c 解冻的真实代价）⇒ 原“关闭”应改为
   「**有条件成立**」：局部头可作**概率校准 / 立直子集**的组件，
   但不支撑通用局部形状头重启 200k。
4. P1/P2b 的价值全在 **CE/top3**（CE 降幅 0.144/0.154 vs A2c 0.104），top1 不占优；
   P2a 全面最差，确认立直特征注入无效。

**口径提醒**：`bayes_ceiling` 的 SIGNAL_SHALLOW 判定也是基于 micro top1。
已在 **§5e（P0-a）** 按 cond 重算完毕：strict tier 信息确实很小（all/active_any
+0.023 nats），但 tier **不是上界**（A2c 低 0.135~0.200 nats，16/18 cell 超出）。

## 5e. P0-a：BAYES_CEILING 三层 tier 按 cond 重算（2026-11-03）

`p0a_bayes_cond.py`：把 §5b 的三层 tier（counts/hard/behave）从 micro 改到
**cond** 口径重算（筛非空 + EMPTY 置 0 重归一），回答“P1/P2a/P2c 的 FAIL 是否
属于后验本身接近均匀的预期结果”。

**实现要点**

- MC **采样顺序与 `bayes_ceiling.py` 逐位相同**（同 seed、同循环），只改累加口径
  ⇒ cond 与 micro 是同一批后验样本的两种打分，可直接配对比较。
- `counts` 用**精确多元超几何**（全量，无 MC 误差）；`hard`/`behave` 为 MC
  （6000 快照，M=256/1024/4096，chains=256，iters=128）。
- 自检 **6/6**：counts 的 micro top1 与旧 JSON 逐位一致；cond prior 与
  `eval_cond_recheck.json` 的 **n 完全相同**（prior 值差 ≤0.3pp，因为本脚本按
  (中心, 目标家) 分 63 组、`cond_recheck` 按中心分 21 组）。
- 全量 MC 约 **91 分钟**（5451s）；`--reuse-mc` 可免重跑 MC 只重建报告（26s）。

**主表（cond CE，越小越好）**

| cell | n | prior | counts | hard | behave | A2c | P2c | strict 信息 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all/all_centers | 168.7万 | 2.662 | 2.624 | **2.609** | 2.808 | 2.473 | **2.409** | +0.038 | INFO_NEGLIGIBLE |
| all/active_any | 17.3万 | 2.272 | 2.249 | **2.209** | 2.496 | 2.036 | **1.858** | +0.023 | INFO_NEGLIGIBLE |
| all/active_both | 5865 | **1.674** | 1.834 | 1.819 | 2.231 | 1.593 | **1.499** | −0.160 | TIER_MISFIT |
| riichi/active_any | 22672 | **2.022** | 2.513 | 2.167 | 3.370 | 2.257 | **1.847** | −0.490 | TIER_MISFIT |
| riichi/active_both | 944 | **0.642** | 1.925 | 1.786 | 2.338 | 1.405 | 1.182 | −1.284 | TIER_MISFIT |
| clean_t8/active_any | 80145 | 2.339 | **2.211** | 2.213 | 2.474 | 2.040 | **1.942** | +0.129 | INFO_SMALL |

（粗体 = 该行最低；注意 `active_both` 系列和立直层的 **prior 才是最低**，tier 全部劣于 prior。）

**四条结论**

1. **tier 信息确实很小，但不是零**：strict（精确 counts，无行为假设）的 CE 增益
   在 `all/active_any` 只有 **+0.023 nats**，在 `clean_t8/active_any` 为 **+0.129**，
   `turn>=8/active_any` 为 **+0.076**。⇒ “27 类形状在给定可观测信息下接近不可预测”
   这个方向**成立**。
2. **但 tier 不是上界**：在 **8 个“tier 优于 prior”的 cell** 上，A2c 的 CE 比最强
   tier 低 **0.135~0.209 nats**（all/active_any 0.173、nonriichi/active_any 0.206、
   turn>=8/all_centers 0.209、clean_t8/all_centers 0.198）。其余 10 个 cell 的 prior
   已优于所有 tier（TIER_MISFIT），谈“tier 上界”无意义。
3. **失配集中在 active_both / 立直 / called**（strict 信息 −0.08~−1.28）：这几层的
   mask 本身含信息（“这两张曾被手切” ⇒ 窗口非空概率高），而 tier 的均匀子集假设
   完全不用 mask ⇒ 条件化后分布错位。这反而是“tier 是信息子集”的直接证据。
4. **P2c 在 15/18 个 cell 上 CE 最低**（3 个例外全是 active_both 系，那里
   评估集 oracle prior 更低）；P1/P2b 也全面优于 A2c。
   ⇒ **P1/P2a/P2c 的 FAIL 是 top1 门禁的判定，不是能力不足**；在 CE/校准口径上
   它们**已经超过 A2c**。

**副产品：`bayes_ceiling.py` 的 Dirichlet α 语义错误（已修）**

旧实现 `p = (cnt + α/27)/(wsum + α)` 中 `wsum = 1`，`α=0.5` 相当于 **33% 的均匀
混合**，把分布压平：与精确 counts 的 **L1 距离 0.43**，MC tier 的 CE 系统偏高
**0.22~0.35 nats**。**top1 不受影响**（加常数不改 argmax），所以 §5b 的 top1 结论
仍有效，但其 CE 结论（“CE 全部高于边缘熵 ⇒ 均匀子集被手切打破”）**归因不完整**——
主因是平滑口径。修正后按等效样本数计（`--alpha-mode count`），L1 **0.04**。

| cell | 精确 counts | MC（旧 α） | MC（新 α） |
|---|---:|---:|---:|
| all/active_any | 2.2486 | 2.4708 | 2.2514 |
| all/active_both | 1.8338 | 2.1857 | 1.8395 |
| clean_t8/active_any | 2.2105 | 2.4390 | 2.2151 |

**已知限制**：`behave` 的 M 收敛差（all/active_any Δ=0.157，riichi/active_both
Δ=0.494）⇒ M=4096 对 27 类长尾仍不足，其 CE 有正偏，“behave 比 hard 差”
有相当部分是采样偏差。**behave 仅作上界参考，不作判据**。

产物：`results/P0A_BAYES_COND_REPORT.md` / `p0a_bayes_cond.json`。

## 5f. 立直层温度校准（2026-11-04，负结论：收益 ≤0.025 nats，不落地）

P0-a 发现 `riichi/all_centers` 上 A2c 的 27 类 CE（**2.494**）比每中心先验
（**2.429**）还差 0.065，但 top1 更好 ⇒ 典型**过度自信**。本节回答：
这 0.065 里有多少是**校准误差**（温度缩放能回收），有多少是**形状错误**（救不了）。

`calib_riichi.py`：对后验做 `q ∝ p^(1/T)`，在同批样本上最小化 CE 拟合 T。
温度缩放是**单调变换** ⇒ top1/top3 完全不变，只回答“概率幅度对不对”。

**per-cell（A2c）**

| cell | n | prior CE | CE(T=1) | T* | CE(T*) | Δ |
|---|---:|---:|---:|---:|---:|---:|
| all/all_centers | 50k | 2.662 | 2.4680 | 1.055 | 2.4668 | −0.001 |
| all/active_any | 50k | 2.273 | 2.0413 | 1.126 | 2.0346 | −0.007 |
| **riichi/all_centers** | 50k | **2.430** | **2.4926** | **1.173** | **2.4825** | **−0.010** |
| **riichi/active_any** | 22672 | **2.024** | **2.2570** | **1.286** | **2.2292** | **−0.028** |
| riichi/active_both | 944 | 0.645 | 1.4052 | 0.945 | 1.4035 | −0.002 |
| clean_t8/all_centers | 50k | 2.750 | 2.4724 | 1.087 | 2.4698 | −0.003 |

**可落地分组（只看“对手是否立直”一个 bit）**

| 拟合 | T | 应用到 | CE(T=1) | CE(T) | Δ |
|---|---:|---|---:|---:|---:|
| riichi | 1.178 | all_centers | 2.4990 | 2.4881 | −0.011 |
| riichi | 1.178 | **active_any** | 2.2570 | 2.2323 | **−0.025** |
| riichi | 1.178 | **active_both** | 1.4052 | 1.4273 | **+0.022（变差）** |
| nonriichi | 1.054 | active_any | 1.9989 | 1.9957 | −0.003 |
| 交叉：nonriichi 用 T_riichi | 1.178 | all_centers | 2.4689 | 2.4729 | +0.004 |
| 交叉：riichi 用 T_nonriichi | 1.054 | all_centers | 2.4977 | 2.4918 | −0.006 |

**四条结论**

1. **A2c 确实过尖**：立直层 T=**1.17~1.29**（非立直 1.05~1.11），方向与“过度自信”一致。
2. **但校准只能回收 0.010~0.028 nats**：`riichi/all_centers` 只回收 **0.011/0.065 = 17%**，
   校准后 **2.483 仍差 prior 2.430**。⇒ **剩余 0.053 是分布形状错误**，温度缩放救不了。
3. **T 有层特异性**：把 T_riichi 用在非立直上变差 **+0.004**，反之只回收 1/4 效果。
   且 `active_both`（n=944）上用 T_riichi 会**变差 +0.022** ⇒ 一刀切不安全。
4. **P2c 已经完全校准（T_riichi=1.002、T_nonriichi=0.992，Δ≤0.0004）**
   ⇒ 它在 P0-a 里的 CE 优势是**真实的形状优势**（直接联合建模），不是校准假象。

**判定：不落地。** 收益上限 0.025 nats 且方向不稳（小样本 cell 变差）；守卫已关闭，
没有下游消费者。要真正修 A2c 的立直层 CE，必须**改联合建模**（P2c 的路），
而不是加温度。

产物：`results/RIICHI_CALIB_REPORT.md` / `riichi_calib.json`。

## 6. 已知边界

- 缓存是 40000 样本、202609 单月；bootstrap 为**样本级**（非文件块），CI 偏窄；
- `prior_major` 在评估集上拟合，是保守的 oracle 先验；P1 的 `prior_eval` 是**全局**
  27 类先验（CE 更大），与本目录的**每中心**先验口径不同，勿直接对比；
- counts/hard/behave 的 pool 含 ~9 张不可入手死牌（ura/备用指示牌）且未建模手切
  相关性 ⇒ 其 CE 是**保守（偏高）**估计，上限结论以 top1 判据为准；
- P0 的“形状”是计数形状，不是两面/嵌张分解；后者需要 P1 的软标签；
- 所有指标都是离线诊断，强度结论仍只认 ladder。
