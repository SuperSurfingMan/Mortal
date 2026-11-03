#!/usr/bin/env bash
# 训练逐牌存在性模型（GPU2）+ 202609 终评。
#   tmux new -d -s tp_train "bash tools/read_model/tile_presence/run_train.sh"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MALLOC_ARENA_MAX=2

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /mnt/data8/miniconda3/envs/mortal/bin/python ]]; then
    PYTHON=/mnt/data8/miniconda3/envs/mortal/bin/python
  else
    PYTHON=python
  fi
fi

CACHE="${CACHE:-$HERE/.cache}"
OUT="${OUT:-$HERE/results}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/tile_presence}"
mkdir -p "$OUT" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/train_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] cache=$CACHE out=$OUT"
echo "[run] log=$LOG"

"$PYTHON" "$HERE/train_presence.py" \
  --cache-dir "$CACHE" --out "$OUT" \
  --steps "${STEPS:-8000}" --batch "${BATCH:-256}" \
  --d "${D:-160}" --blocks "${BLOCKS:-2}" --heads "${HEADS:-8}" \
  --dropout "${DROPOUT:-0.2}" --lr "${LR:-2e-4}" --wd "${WD:-0.03}" \
  --w-sum "${W_SUM:-0.1}" --aug "${AUG:-1}" --aug-p "${AUG_P:-0.5}" \
  --eval-every "${EVAL_EVERY:-250}" --patience "${PATIENCE:-10}" \
  --workers "${WORKERS:-0}" \
  "$@" 2>&1 | tee "$LOG"
