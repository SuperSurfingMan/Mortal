#!/usr/bin/env bash
# 读牌模型 A2b 训练（tmux 会话 read_v2s_a2b）：修复 A2 发现的 P(未听) 塌缩。
#
# 背景（A2 结果，2026-10-18）：
#   v2s 的 l_wait 只在结构听牌目标（~14.5%）上激活，其余 85.5% not-tenpai 目标
#   对 H2 的 wait[34]（未听）输出无任何直接监督。50k 长训练中"听牌目标 P(未听)→0"
#   的梯度单向传递，使 P(未听) 在全体目标上集体塌缩（非听牌目标中位数 0.0030→0.0013），
#   tenpai 检测 AUC 0.7392→0.6671 显著退化（但 wait_hit5/点炮率显著改善）。
#
# A2b 的唯一改动：输出侧 tenpai 监督（--wait-tenpai-weight 0.2，全目标直接监督
#   wait[34] 拟合 1-tenpai，口径与 AUC 一致）。其余与 v2s 同数据/同种子/同预算/同课程。
#
# 预注册判据（vs A2 的 v2s_50k，同 20000 固定评估集 + bootstrap 200 CI）：
#   (a) 主判据：tenpai_auc ≥ 0.72（回收 0.072 退化的 ≥70%）；
#   (b) 护栏：deal_pure ≤ 0.0408（保住 A2 的点炮率改善，不退回 15k 水平）、
#       wait_hit5 ≥ 0.50；wait_hit1 Δ(vs v2) ≥ −0.015（不因新增监督恶化）；
#   (c) 若 (a) 成立且 (b) 保持 → A2b 定为注入基线；否则按失败归档并分析。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=${OUT:-$REPO/mortal/read_model_v2s_a2b}
LOGDIR=$REPO/mortal/logs
LOG=${LOG:-$LOGDIR/read_model/read_model_v2s_a2b_train.log}
RESUME=${RESUME:-}
PREFETCH=${PREFETCH:-2}
mkdir -p "$OUT" "$(dirname "$LOG")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0            # GPU0（GPU2 归 s6a 实验）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${STEPS:-50000}
NW=${NW:-8}

cd "$REPO/mortal"
echo "=== read_model v2s A2b train start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW ==="
"$PY" -u "$REPO/tools/read_model/train.py" \
  --model v2s \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 \
  --stage-schedule 1:0.27,2:0.30,3:0.43 \
  --num-workers "$NW" --prefetch-factor "$PREFETCH" --augment random \
  --eval-samples 8192 --eval-every 1000 --log-every 200 \
  --wait-tenpai-weight 0.2 \
  ${RESUME:+--resume "$RESUME"} \
  --seed 20261016 2>&1 | tee -a "$LOG"
echo "=== read_model v2s A2b train done $(date -u +%FT%TZ) ==="
