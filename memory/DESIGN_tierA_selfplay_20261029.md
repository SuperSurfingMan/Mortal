# 档 A：自博弈 + MC 回报（2026-10-29 启动）

## 0. 为什么是它

搜索/EI 线（档 B/C）连续两次失败：teacher 分辨率不足（自洽性 0.52~0.54），
全知铳损线也关闭（模型弃牌率已与人类持平）。
**档 A 不依赖任何 teacher**：只换**数据分布**（用模型自己的对局）与**目标**（真实 MC 回报），
成本最低、可复用现有训练管线。

## 1. 已跑通：自博弈日志生成

`mortal/selfplay_gen.py`（4 席同一权重；`OneVsThree.py_vs_py` 的 challenger/champion 各一份实例）

```bash
cd mortal && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=config.s6c.toml \
  python -u selfplay_gen.py --sets 200 --seed 700000 --out ../logs/selfplay/s6c_700000
```

**冒烟实测**（2 sets = 8 半庄）：

```
[ selfplay ] state=checkpoints/s6c/mortal.pth steps=956000 sets=2 seed=700001
[ selfplay ] 完成：8 个日志 → logs/selfplay/smoke
耗时 ≈ 62s（含加载）⇒ 约 5~8 s/半庄
```

日志格式与人类日志一致，`GameplayLoader` 可直接读。

## 2. 规模与成本

| 目标局数（半庄） | 单卡耗时（按 6 s/半庄） | 4 卡并行 |
|---|---|---|
| 2,000 | ≈ 3.3 h | ≈ 50 min |
| 8,000 | ≈ 13 h | ≈ 3.3 h |
| 20,000 | ≈ 33 h | ≈ 8 h |

**建议起步 4,000~8,000 半庄**（≈ 与人类数据 1:60 的混合比例），足够看出方向。

## 3. 训练怎么改（最小改动）

现有 `train.py` 已经是「BC + MC 回报」的混合目标，因此档 A 的改动只有两处：

1. **数据来源**：把自博弈日志目录加入 `dataset` 列表（与人类日志按比例混合，
   `human:selfplay = 8:1 ~ 4:1`），避免分布漂移；
2. **损失权重**：对自博弈样本提高 MC 项的权重（例如 `1.0 → 1.5`），
   因为它们没有「人类示范」的意义，只有回报意义。

CLI 形式（示意）：

```bash
python train.py --config config.s6c.toml --init checkpoints/s6c/mortal.pth \
  --extra-logs ../logs/selfplay/s6c_700000 --mix 6 --mc-weight-selfplay 1.5 \
  --steps 200000 --save-every 20000
```

> 注：`--extra-logs` / `--mc-weight-selfplay` 需要新增（各约 20 行），
> 若不想改 `train.py`，可先把自博弈日志**软链**进人类日志目录，按 `--mix` 天然混采。

## 4. 判定

| 阶段 | 判据 |
|---|---|
| 冒烟 | 5k 步后 `dqn_loss` 不爆、holdout top1 不崩（−0.5pp 内） |
| 主判 | **16k sets 标准臂**（δ_min≈0.43），种子 **700000/720000**（与历史块不重叠） |
| 画像 | `agari` / `houjuu` / 平均打点 与 S6c 对比，确认没有「自博弈同质化」 |

## 5. 风险

| 风险 | 表现 | 对策 |
|---|---|---|
| 自我强化（同质化） | 自博弈胜率稳定但对外变弱 | 只混 10~20%；主判据用对外 ladder |
| 分布漂移 | holdout top1 下滑 | 混采 + 早停 |
| 回报噪声 | dqn_loss 高但无提升 | 加大混采比例，或加 GRP 平滑 |
