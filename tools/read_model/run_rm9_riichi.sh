#!/usr/bin/env bash
# RM9 立直层专项探针（2026-10-21）：A2c best.pth 暖启 + 立直/早期立直重加权。
#
# 依据（riichi_diag）：立直家 hit@1 随「立直后摸切张数」0→3+ 从 0.1305→0.2065，
# 早期立直（turn<=7）hit@1 仅 0.1016；缺口集中在立直宣言初期（post0 / 早期），
# 不是整层校准问题（温度扫描 T=1.2 仅 −0.0025 cond_ce）。
#
# 变量：唯一改动 = 样本权重（riichi_weight 3→6；riichi_early_weight 1→2），
# 其余与 A2c 锁死（v2s + tile 头 + pooled + 同数据/种子/batch）。
# 20k 暖启（--init：仅借权重，optimizer/scheduler 重置，避免 153k 绝对 LR 日程
# 让 lr 在起点就≈0）、lr 5e-5、stage 3 全量（A2c 已收敛，不需要课程）。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=${OUT:-$REPO/mortal/read_model_v2s_rm9_riichi}
LOGDIR=$REPO/mortal/logs
LOG=${LOG:-$LOGDIR/read_model/read_model_v2s_rm9_riichi.log}
INIT=${INIT:-$REPO/mortal/read_model_v2s_a2c/best.pth}
PREFETCH=${PREFETCH:-1}
mkdir -p "$OUT" "$LOGDIR/read_model"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

STEPS=${STEPS:-20000}
NW=${NW:-4}
LR=${LR:-5e-5}
RIICHI_WEIGHT=${RIICHI_WEIGHT:-6}
RIICHI_EARLY_WEIGHT=${RIICHI_EARLY_WEIGHT:-2}
MAX_FILES=${MAX_FILES:-0}
MAX_FILES_ARG=""
if [ "$MAX_FILES" -gt 0 ]; then MAX_FILES_ARG="--max-files $MAX_FILES"; fi

cd "$REPO/mortal"
echo "=== RM9 riichi probe start $(date -u +%FT%TZ) | steps=$STEPS lr=$LR rw=$RIICHI_WEIGHT ew=$RIICHI_EARLY_WEIGHT ==="
"$PY" -u "$REPO/tools/read_model/train.py" \
  --model v2s \
  --wait-head tile \
  --tl-mode pooled \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 --lr "$LR" \
  --stage-schedule "3:$STEPS" \
  --num-workers "$NW" --prefetch-factor "$PREFETCH" --augment random \
  --riichi-weight "$RIICHI_WEIGHT" --riichi-early-weight "$RIICHI_EARLY_WEIGHT" \
  --eval-samples 8192 --eval-every 1000 --log-every 100 \
  ${INIT:+--init "$INIT"} \
  $MAX_FILES_ARG \
  --seed 20261016 2>&1 | tee -a "$LOG"
echo "=== RM9 riichi probe done $(date -u +%FT%TZ) ==="
