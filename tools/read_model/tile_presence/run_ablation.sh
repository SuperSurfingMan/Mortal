#!/usr/bin/env bash
# 根因诊断 2：训练期特征组消融（events / history / safety）。
# 与 run2（d160/aug/sum，40k）同配置，只关掉一组特征，比较 202609 eval AUC。
#   tmux new -d -s tp_abl "bash tools/read_model/tile_presence/run_ablation.sh"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MALLOC_ARENA_MAX=2

PYTHON="${PYTHON:-/mnt/data8/miniconda3/envs/mortal/bin/python}"
CACHE="${CACHE:-$HERE/.cache}"
LOG_DIR="${LOG_DIR:-$ROOT/mortal/logs/read_model/tile_presence}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

DROPS="${DROPS:-events history safety}"
for d in $DROPS; do
  OUT="$HERE/results/abl_no_$d"
  LOG="$LOG_DIR/abl_${d}_${STAMP}.log"
  echo "[abl] drop=$d out=$OUT log=$LOG"
  "$PYTHON" "$HERE/train_presence.py" \
    --cache-dir "$CACHE" --out "$OUT" --drop "$d" \
    --steps "${STEPS:-5000}" --batch "${BATCH:-256}" \
    --d 160 --blocks 2 --heads 8 --dropout 0.2 --lr 2e-4 --wd 0.03 \
    --w-sum 0.1 --aug 1 --aug-p 0.5 \
    --eval-every 250 --patience "${PATIENCE:-6}" --workers "${WORKERS:-0}" \
    2>&1 | tee "$LOG" | grep -E '^\[val|^\[best|^\[early|判定|all AUC Δ|^\[out\]'
done
