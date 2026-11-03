# 守卫 risk 口径修正与 λ 重扫（2026-11-02/03）

> 结论：`risk` 双乘 P(听) 已修正并有回归门禁；修正后在全新种子块上重扫
> λ=4.5/6/7.5，四臂（含 legacy λ=10）均为 **NO-CONCLUSION**。按预注册规则，
> 线上默认 `MORTAL_GUARD_LAMBDA=0`（关闭守卫），不保留 legacy 默认路径。
> 历史 +0.778 是旧口径下的结论，保留作历史，不再作为默认依据。

## 1. 问题

`factorized_wait_probs` 的输出 `wp[..., :34]` 已经是
`P(听)·P(待牌=t|听)`，行和恒等于 `1 − wp[...,34]`。旧实现又乘了一次
`P(听)`：

```python
p_tenpai = 1 - wp[..., 34]
risk = (p_tenpai[..., None] * wp[..., :34]).sum(1)
```

实际执行的是 `Σ_j P(听_j)^2·P(待牌_j=t|听_j)`。在 A2c / 202609 holdout
400 样本上（2026-11-02 实测）：

| 检查 | 结果 |
|---|---|
| `wp[:34].sum(-1)` vs `1 - wp[34]` | 相等（0.1495749489 vs 0.1495749503） |
| 文档公式 risk 均值 | 0.01358 |
| 旧代码 risk 均值 | 0.01030（约 0.76×） |
| 两版 top-1 牌一致率 | 90.5% |
| 逐样本 Pearson | 均值 0.995，最低 0.967 |

结论：牌序大体保留，但尺度和跨对手权重被改变；旧 λ=10 是在旧尺度上标定的。

## 2. 修复

- 新增 `mortal/guard_risk.py` 作为单一真源：
  - `risk_from_wait_probs(wp) = wp[..., :34].sum(-2).clamp(0,1)`；
  - `legacy_risk_from_wait_probs(wp)` 仅保留给回归/对照；
  - `apply_risk_penalty(q, risk, λ, τ, masks)` 只改动作 0..33。
- 改造调用方：`mortal/defense_guard.py`、`tools/read_model/precompute_danger.py`、
  `eval_defense_guard.py`、`eval_ev_rerank.py`、`eval_guard_state.py`、
  `eval_loss_pt.py`、`eval_s6b_arms.py`。`inject_p0.py` 原本就是正确公式，补注释并纳入交叉检查。
- `ladder.py` / `ladder_arm.sh` 增加 `--guard-risk-mode {fixed,legacy}`，只用于本次对照；
  生产入口 `mortal.py` 不暴露 legacy。
- 回归：`tools/test_guard_risk.py` 覆盖合成反例（旧公式必失败）、因子化恒等式、
  逐对手通道、硬掩码、Q 更新、真实 A2c 批次、调用方单一真源。
  `test_pipeline.py` **91/91 ALL PASS**；`test_guard_state`、`test_ev_rerank` ALL PASS。

合成反例锁死的数值：

```text
三家 P(听)=0.9/0.5/0.01，待牌 5/6/7
正确 risk: [5]=0.9, [6]=0.5, [7]=0.01
旧口径:    [5]=0.81, [6]=0.25, [7]=0.0001
关系:      legacy == p_ten * fixed
```

## 3. 20k 离线重扫（202609，seed 20261016）

`eval_defense_guard.py`，B/756k 冻结，A2c `tl=0`，20000 固定样本。
旧口径 legacy 与历史 `defense_guard_sweep_20k.json` 逐项一致（λ=10 change
7.98%、deal 0.0094、a2c 0.3098），证明复现无损。

| 口径 | λ | change | deal_top1 | risk_top1 | a2c_agree |
|---|---:|---:|---:|---:|---:|
| legacy | 10 | 7.98% | 0.0094 | 0.0103 | 0.3098 |
| fixed | 4.5 | 约 6.9%* | 约 0.0105* | — | — |
| fixed | 6 | 8.33% | 0.0097 | 0.0186 | 0.3118 |
| fixed | 7.5 | 9.33% | 0.0095 | 0.0180 | 0.3191 |
| fixed | 10 | 11.00% | 0.0089 | 0.0170 | 0.3312 |

\* 4.5 未单独跑 20k，按 4/5 插值；最终以 ladder 的 λ=4.5 为准。

修正后 risk 均值约为旧口径的 **1.55~1.66×**；`fixed λ=6` 与旧 `legacy λ=10`
工作点最接近（change 8.33% vs 7.98%、deal 0.0097 vs 0.0094、a2c 0.3118 vs 0.3098）。
所有 fixed 臂 `safer_ratio` 均为 **100%**，行为方向正确。

## 4. Ladder 受控实验

- challenger = champion = B/756k；守卫只挂 challenger；`--no-amp`；
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- 每臂 16 块 × 500 sets/向 = 8000 sets/向 = **64,000 半庄**；解析 SE≈0.33。
- 全新种子块 **1000000/1020000**，全部臂共用，可按方向配对。
- 空臂 N0（λ=0）2 块，δ=+0.000，链路 sanity 通过。

| 臂 | risk | λ | δ（pt/半庄） | bootstrap 95%CI | agari | houjuu |
|---|---|---:|---:|---|---:|---:|
| N0 | fixed | 0 | +0.000 | [0.00, 0.00] | — | — |
| L0 | legacy | 10 | **−0.184** | [−0.61, +0.27] | 0.2076 | 0.1126 |
| C0 | fixed | 6 | **−0.124** | [−0.56, +0.35] | 0.2092 | 0.1153 |
| C− | fixed | 4.5 | **+0.047** | [−0.37, +0.46] | 0.2105 | 0.1178 |
| C+ | fixed | 7.5 | **−0.103** | [−0.56, +0.39] | 0.2078 | 0.1129 |

四臂全部 **NO-CONCLUSION**（CI 跨 0），点估计在 −0.18~+0.05 之间。
行为护栏 `agari ≥0.200`、`houjuu ≤0.120` 全部通过；不是过度弃和导致的 null。

### 配对 Δδ（同种子、按方向配对）

| 对比 | Δδ | 95%CI | 判读 |
|---|---:|---|---|
| fixed λ=6 − legacy λ=10 | +0.059 | [−0.304, +0.416] | 边界非劣（差 0.004）/不确定 |
| fixed λ=4.5 − legacy λ=10 | +0.231 | [−0.159, +0.609] | 非劣 |
| fixed λ=7.5 − legacy λ=10 | +0.080 | [−0.278, +0.445] | 非劣 |

修正版相对旧口径没有变差，但也没有任何一臂显著为正。

## 5. 关键判读

1. **旧 +0.778 没有复现**：L0 在新种子块上是 −0.184，CI 上界 +0.27，已排除
   历史 +0.778。两轮使用同一 ckpt、同量级样本（64k 半庄），差异约 0.96 pt
   （差的标准误约 0.47）≈2σ。可能来源：种子/牌山方差、服务器环境或旧结论
   本身的乐观偏差；本实验不声称能拆分。
2. **旧 +0.778 本身不是标准 δ**：旧 `guard_pipeline` 两次都是“守卫当
   challenger”的同向运行（未 A/B 换边），`agg_guard_ladder` 报的是两次
   `avg_pt` 的均值（+1.226/+0.330 → +0.778）。若硬按 δ 粗算为 +0.448，
   SE≈0.33，CI 约 [−0.20,+1.10] 跨 0。因此旧“GO”不是标准双向臂结论；
   新 L0 的 −0.184 才是首次标准 A/B 测量。
3. **修正口径本身没有制造新收益**：fixed λ=4.5/6/7.5 与 legacy λ=10 同处
   null 区间，配对 Δδ 不为负，但也不为正。
4. **守卫机制仍在工作**：四臂 agari 约 0.208~0.211、houjuu 约 0.113~0.118，
   相对无守卫基线（历史 0.217/0.130）仍是“用少量和了换放铳”，只是没有转化为
   可测的 avg_pt。
5. **按预注册规则收口**：所有 fixed 臂不显著 ⇒ 默认 λ=0，不跑 fresh-seed
   确认（1040000/1060000 未消费），不保留 legacy 默认路径。

## 6. 最终状态

- `mortal/mortal.py`：`MORTAL_GUARD_LAMBDA` 默认 **0**（关闭）。
- `mortal/config.anchor_ako.toml`：`guard_lambda = 0`。
- `mortal/defense_guard.py`：默认 `risk_mode='fixed'`，公式已修正；legacy 仅
  通过 ladder CLI 复现历史，不作为生产路径。
- 历史 +0.778 文档保留，但加“旧口径”标注；新结论以本文为准。
- 种子台账：1000000/1020000 已消费；1040000/1060000 未消费（保留给后续实验）。

## 7. 产物

| 路径 | 内容 |
|---|---|
| `tools/read_model/diagnostics/results/guard_fix_20261102/{legacy,fixed}_20k.json` | 20k λ 扫描 |
| `.../paired_all_vs_l10.json` | 配对 Δδ |
| `mortal/runs/guard_fix/{legacy_l10,fixed_l60,fixed_l45,fixed_l75,null_l0}/` | ladder 牌谱与 run_meta |
| `mortal/logs/guard_fix/` | 全部运行日志 |
| `tools/test_guard_risk.py` | 口径回归门禁 |

## 8. 后续

守卫线（λ 剂量、EV 重排、Guard-P2、加性损失、本次口径修正）到此全部探完，
没有可复现的正收益。若未来重启，必须先有新机制（例如按打点分布的期望顺位
损失，或真正的新信息源），而不是继续在同一 risk frontier 上调 λ。
