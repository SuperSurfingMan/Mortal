#!/usr/bin/env bash
# P2b：behave 参考蒸馏 + 局部形状头（冻结 A2c）。
# GPU4 / tmux：
#   tmux new -d -s read_ls_p2b "bash tools/read_model/local_shape/run_p2b.sh"
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
OUT="${OUT:-$HERE/results/p2b}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/local_shape}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-1024}"
WORKERS="${WORKERS:-4}"
DISTILL_W="${DISTILL_W:-0.5}"
mkdir -p "$CACHE_DIR" "$OUT" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/p2b_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "[run] cache_dir=$CACHE_DIR out=$OUT log=$LOG"

if [[ ! -f "$CACHE_DIR/train.npz" || ! -f "$CACHE_DIR/eval.npz" ]]; then
  echo "[error] 缺少 P2 feature cache；先跑 tools/read_model/local_shape/run_p2.sh" >&2
  exit 1
fi

TRAIN_GLOB="$ROOT/dataset/mjai/houou4_names/202608/**/*.json.gz"
if [[ ! -f "$CACHE_DIR/distill_train.npz" ]]; then
  echo "[ref] behave 软标签（train 立直子集）"
  "$PYTHON" "$HERE/p2b_distill_data.py" \
    --train-glob "$TRAIN_GLOB" \
    --train-cache "$CACHE_DIR/train.npz" \
    --out "$CACHE_DIR/distill_train.npz" \
    --max-snapshots 1500 --max-targets 3000 --n-ref 512 \
    2>&1 | tee -a "$LOG"
else
  echo "[ref] reuse $CACHE_DIR/distill_train.npz"
fi

echo "[train] P2b distill head（frozen A2c）"
"$PYTHON" "$HERE/p2b_distill_head.py" \
  --cache-dir "$CACHE_DIR" \
  --distill "$CACHE_DIR/distill_train.npz" \
  --out "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" \
  --distill-weight "$DISTILL_W" \
  2>&1 | tee -a "$LOG"
