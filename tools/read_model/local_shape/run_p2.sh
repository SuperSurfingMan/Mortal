#!/usr/bin/env bash
# P2：立直显式特征 + 局部形状头（冻结 A2c）。
# GPU4 / tmux：
#   tmux new -d -s read_ls_p2 "bash tools/read_model/local_shape/run_p2.sh"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /mnt/data8/miniconda3/envs/mortal/bin/python ]]; then
    PYTHON=/mnt/data8/miniconda3/envs/mortal/bin/python
  else
    PYTHON=python
  fi
fi

CACHE_DIR="${CACHE_DIR:-$HERE/.cache_p2}"
OUT="${OUT:-$HERE/results/p2}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/local_shape}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-1024}"
WORKERS="${WORKERS:-4}"
mkdir -p "$CACHE_DIR" "$OUT" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/p2_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "[run] cache_dir=$CACHE_DIR out=$OUT log=$LOG"

TRAIN_GLOB="$ROOT/dataset/mjai/houou4_names/202608/**/*.json.gz"
EVAL_GLOB="$ROOT/dataset/mjai/houou4/202609/**/*.json.gz"

if [[ ! -f "$CACHE_DIR/train.npz" || ! -f "$CACHE_DIR/val.npz" ]]; then
  echo "[features] train/val（202608，含立直宣言牌/立直后牌河）"
  "$PYTHON" "$HERE/features.py" --mode trainval \
    --glob "$TRAIN_GLOB" --train-n 20000 --val-n 5000 \
    --out "$CACHE_DIR" 2>&1 | tee -a "$LOG"
else
  echo "[features] reuse $CACHE_DIR/{train,val}.npz"
fi

if [[ ! -f "$CACHE_DIR/eval.npz" ]]; then
  echo "[features] eval（202609，seed=20261016，与 P0 同源）"
  "$PYTHON" "$HERE/features.py" --mode eval \
    --glob "$EVAL_GLOB" --n 40000 --seed 20261016 --out "$CACHE_DIR" \
    --check-alloc /tmp/read_model_diag_cache_a2c/cache.npz \
    2>&1 | tee -a "$LOG"
else
  echo "[features] reuse $CACHE_DIR/eval.npz"
fi

echo "[train] P2 riichi head（frozen A2c）"
"$PYTHON" "$HERE/p2_riichi_head.py" \
  --cache-dir "$CACHE_DIR" --out "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" \
  2>&1 | tee -a "$LOG"
