#!/usr/bin/env bash
# 读牌模型 A1 瓶颈归因消融（tmux 会话 read_v2x，会话 27c 计划阶段 A 第一跑）。
#
# 实验设计（预注册判据见 MEMORY.md「读牌模型」节 A1 条目）：
#   本跑 = v2 架构（--model v2，结构瓶颈 38 维，**不加宽**）+ v2s 全部新损失
#   （显式传参，与 v2s 轮逐项一致，防止默认值漂移）。与两个既有轮次构成
#   单因素对照（同数据/同步数/同种子/同预算/同评估集）：
#
#     单元            架构        损失        wait_hit1(15k, 20000评估集)
#     v2  (基线)      38 维      旧          0.1464
#     v2x (本跑)      38 维      新          W  ← 待测
#     v2s (15k)       46 维      新          0.1323
#
#   归因分解（正向效应；加性假设，2×2 缺第四格=46维+旧损失，无兴趣故不测）：
#     瓶颈效应 B = W − 0.1323（瓶颈加宽解释了多少代价）
#     损失效应 L = 0.1464 − W（新损失解释了多少代价）；B + L = 0.0141
#   预注册判据（80/20 可分辨阈值——wait_hit1 的 95% CI 半宽 ≈0.007，60/40
#      band 窄于噪声不可分辨）：
#     B ≥ 0.0113（W ≥ 0.1436，损失代价 ≤20%）→ 瓶颈加宽是主因 → 回退 38 维
#         保留全部损失，A1(v2x) 胜者直接晋级 A2 50k；
#     L ≥ 0.0113（W ≤ 0.1351，损失代价 ≥80%）→ 新损失是主因 → 转
#         --tenpai-tie-weight 0 消融（它直接塑形 H2 消费的向听信念）；
#     混合区（0.1351 < W < 0.1436）→ 比增益定去留：v2x 的 AUC/向听 ≥ v2s−0.005
#         则按简约原则用 38 维，否则保留 46 维（瓶颈在贡献增益而非仅代价）。
#   次级判读：v2x 结构头向听准确率（瓶颈不消费该头时的纯损失效果）vs v2s(0.526)；
#        wait 机制诊断（听牌目标 P(未听)/top1 概率/熵）复现 Sharpness 假设。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=$REPO/mortal/read_model_v2x
LOGDIR=$REPO/mortal/logs
mkdir -p "$OUT" "$LOGDIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0            # P100（GPU2 被主训练占用）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${STEPS:-15000}
NW=${NW:-8}

cd "$REPO/mortal"
echo "=== read_model v2x (A1 ablation) train start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW ==="
"$PY" -u "$REPO/tools/read_model/train.py" \
  --model v2 \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 \
  --stage-schedule 1:0.27,2:0.30,3:0.43 \
  --num-workers "$NW" --augment random \
  --eval-samples 8192 --eval-every 1000 --log-every 200 \
  --seed 20261016 \
  --w-shanten 0.15 \
  --struct-shanten-weight 0.5 \
  --consist-head-weight 0.3 \
  --consist-decomp-weight 0.2 \
  --tenpai-tie-weight 0.1 \
  2>&1 | tee -a "$LOGDIR/read_model_v2x_train.log"
echo "=== read_model v2x (A1 ablation) train done $(date -u +%FT%TZ) ==="
