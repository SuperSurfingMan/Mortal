#!/usr/bin/env bash
# P0：局部形状读数基线（只读缓存，CPU，约 1~2 分钟）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /mnt/data8/miniconda3/envs/mortal/bin/python ]]; then
    PYTHON=/mnt/data8/miniconda3/envs/mortal/bin/python
  else
    PYTHON=python
  fi
fi

CACHE="${CACHE:-/tmp/read_model_diag_cache_a2c/cache.npz}"
OUT="${OUT:-$HERE/results}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/local_shape}"
mkdir -p "$OUT" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/p0_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] cache=$CACHE"
echo "[run] out=$OUT"
echo "[run] log=$LOG"

"$PYTHON" "$HERE/eval_local_shape.py" --cache "$CACHE" --out "$OUT" "$@" 2>&1 | tee "$LOG"
