#!/usr/bin/env bash
# 导出逐牌存在性 token 缓存：202608 train/val + 202609 eval（40k，对齐 A2c 诊断缓存）。
# 长任务请用 tmux：
#   tmux new -d -s tp_export "bash tools/read_model/tile_presence/run_export.sh"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /mnt/data8/miniconda3/envs/mortal/bin/python ]]; then
    PYTHON=/mnt/data8/miniconda3/envs/mortal/bin/python
  else
    PYTHON=python
  fi
fi

CACHE="${CACHE:-$HERE/.cache}"
TRAIN_GLOB="${TRAIN_GLOB:-dataset/mjai/houou4_names/202608/**/*.json.gz}"
EVAL_GLOB="${EVAL_GLOB:-dataset/mjai/houou4/202609/**/*.json.gz}"
DIAG_REF="${DIAG_REF:-/tmp/read_model_diag_cache_a2c/cache.npz}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/tile_presence}"
mkdir -p "$CACHE" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/export_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] cache=$CACHE"
echo "[run] log=$LOG"

{
  echo "=== trainval $(date) ==="
  "$PYTHON" "$HERE/export.py" --mode trainval --glob "$TRAIN_GLOB" \
    --train-n "${TRAIN_N:-40000}" --val-n "${VAL_N:-8000}" \
    --seed 20261103 --file-seed 20261103 --out "$CACHE"
  echo "=== eval $(date) ==="
  "$PYTHON" "$HERE/export.py" --mode eval --glob "$EVAL_GLOB" \
    --n "${EVAL_N:-40000}" --seed 20261016 --out "$CACHE" \
    --check-alloc "$DIAG_REF"
} 2>&1 | tee "$LOG"
