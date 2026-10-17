#!/usr/bin/env bash
# 读牌模型 v1 首版训练入口（tmux 会话 read_v1）。
#
# 数据切分（方案 §2.4 按月 holdout）：
#   训练 = HF 2022-2025 原树 + names 补丁树 202601-202608（读牌模型不做玩家过滤，
#         names 无关紧要，两树并列只为覆盖训练窗）
#   验证 = 202609（与主模型 holdout 同月，口径一致）
# 注意：主训练（iql_b_v3）正占用 GPU2/RTX8000；读牌模型仅 1.13M 参数且数据管线是
# 瓶颈，用空闲的 P100（PCI 序号 0）即可，勿抢 GPU2。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=$REPO/mortal/read_model_v1
LOGDIR=$REPO/mortal/logs
mkdir -p "$OUT" "$LOGDIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0            # P100（GPU2 被主训练占用）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${STEPS:-15000}
NW=${NW:-8}

cd "$REPO/mortal"
echo "=== read_model v1 train start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW ==="
"$PY" -u "$REPO/tools/read_model/train.py" \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 \
  --stage-schedule 1:0.27,2:0.30,3:0.43 \
  --num-workers "$NW" --augment random \
  --eval-samples 8192 --eval-every 1000 --log-every 200 \
  --seed 20261016 2>&1 | tee -a "$LOGDIR/read_model_v1_train.log"
echo "=== read_model v1 train done $(date -u +%FT%TZ) ==="
