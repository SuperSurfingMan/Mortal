#!/usr/bin/env bash
# P2c：受控解冻探针（最后 2 个 ResBlock + H1 + 局部头）。
# GPU4 / tmux：
#   tmux new -d -s read_ls_p2c "bash tools/read_model/local_shape/run_p2c.sh"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MALLOC_ARENA_MAX=2

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /mnt/data8/miniconda3/envs/mortal/bin/python ]]; then
    PYTHON=/mnt/data8/miniconda3/envs/mortal/bin/python
  else
    PYTHON=python
  fi
fi

OUT="${OUT:-$HERE/results/p2c}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/local_shape}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-512}"
WORKERS="${WORKERS:-8}"
mkdir -p "$OUT" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/p2c_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "[run] steps=$STEPS batch=$BATCH workers=$WORKERS out=$OUT log=$LOG"

"$PYTHON" "$HERE/p2c_unfreeze.py" \
  --steps "$STEPS" --batch "$BATCH" --workers "$WORKERS" \
  --w-alloc 1.0 --lr-bb 5e-6 --lr-h1 1e-4 --lr-head 5e-4 \
  --eval-every 2000 --patience 3 \
  --out "$OUT" 2>&1 | tee -a "$LOG"
