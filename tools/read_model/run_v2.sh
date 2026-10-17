#!/usr/bin/env bash
# 读牌模型 v2 首版训练（tmux 会话 read_v2）：v1 + 手切时序流 + 结构估计流。
# 与 v1（run_v1.sh）严格同预算同数据同种子，唯一差别是 --model v2。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=$REPO/mortal/read_model_v2
LOGDIR=$REPO/mortal/logs
mkdir -p "$OUT" "$LOGDIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0            # P100（GPU2 被主训练占用）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${STEPS:-15000}
NW=${NW:-8}

cd "$REPO/mortal"
echo "=== read_model v2 train start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW ==="
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
  --seed 20261016 2>&1 | tee -a "$LOGDIR/read_model_v2_train.log"
echo "=== read_model v2 train done $(date -u +%FT%TZ) ==="
