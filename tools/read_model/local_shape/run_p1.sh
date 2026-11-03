#!/usr/bin/env bash
# P1：冻结 A2c 骨干 + 局部形状头探针。
# 只使用 GPU4；长任务请在 tmux 里跑：
#   tmux new -d -s read_ls_p1 "bash tools/read_model/local_shape/run_p1.sh"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

# 当前唯一空闲卡 = GPU4；固定 PCI 顺序，避免多卡索引错位
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /mnt/data8/miniconda3/envs/mortal/bin/python ]]; then
    PYTHON=/mnt/data8/miniconda3/envs/mortal/bin/python
  else
    PYTHON=python
  fi
fi

CACHE_DIR="${CACHE_DIR:-$HERE/.cache}"
OUT="${OUT:-$HERE/results/p1}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/local_shape}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-1024}"
WORKERS="${WORKERS:-4}"
mkdir -p "$CACHE_DIR" "$OUT" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/p1_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "[run] cache_dir=$CACHE_DIR"
echo "[run] out=$OUT"
echo "[run] log=$LOG"

TRAIN_GLOB="$ROOT/dataset/mjai/houou4_names/202608/**/*.json.gz"
EVAL_GLOB="$ROOT/dataset/mjai/houou4/202609/**/*.json.gz"

if [[ ! -f "$CACHE_DIR/train.npz" || ! -f "$CACHE_DIR/val.npz" ]]; then
  echo "[features] train/val（202608，按文件 80/20）"
  "$PYTHON" "$HERE/features.py" --mode trainval \
    --glob "$TRAIN_GLOB" --train-n 20000 --val-n 5000 \
    --out "$CACHE_DIR" 2>&1 | tee -a "$LOG"
else
  echo "[features] reuse $CACHE_DIR/{train,val}.npz"
fi

if [[ ! -f "$CACHE_DIR/eval.npz" ]]; then
  echo "[features] eval（202609，与 P0 同源）"
  "$PYTHON" "$HERE/features.py" --mode eval \
    --glob "$EVAL_GLOB" --n 40000 --seed 20261016 --out "$CACHE_DIR" \
    --check-alloc /tmp/read_model_diag_cache_a2c/cache.npz \
    2>&1 | tee -a "$LOG"
else
  echo "[features] reuse $CACHE_DIR/eval.npz"
fi

echo "[train] frozen backbone + local shape head"
"$PYTHON" "$HERE/p1_local_head.py" \
  --cache-dir "$CACHE_DIR" --out "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" \
  2>&1 | tee -a "$LOG"
