# Phase 2：局部形状读数（Local Shape）

> 目标：把「某一家在某张牌附近的搭子形状」从当前 A2c 的**逐张独立边际**，
> 推进到可测量、可改进的**局部联合形状**。P0 只做测量与基线，不训练。

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

## 5. 后续（收口）

- **P1 冻结局部头：FAIL**；**P2a 立直显式特征：FAIL**；**P2b 蒸馏：PASS via CE**；
  **P2c 受控解冻：FAIL**（strict top1 + alloc 未过）。
- **按预注册门禁关闭局部形状线**：不再跑 200k，不再调冻结/解冻头；
  P1/P2a/P2c 都没有把 `riichi active_both` top1 推过 A2c。
- 若目标只是守卫/风险概率，P2b/P2c 的 CE 改善（riichi active_both 1.017→1.006/0.939）
  可作为**校准**参考，但不能声称“局部形状读得更准”。
- **P3（仅在新任务/新标签下重启）**：5 类分解软标签或结构化解码；当前不启动。

## 6. 已知边界

- 缓存是 40000 样本、202609 单月；bootstrap 为**样本级**（非文件块），CI 偏窄；
- `prior_major` 在评估集上拟合，是保守的 oracle 先验；
- P0 的“形状”是计数形状，不是两面/嵌张分解；后者需要 P1 的软标签；
- 所有指标都是离线诊断，强度结论仍只认 ladder。
