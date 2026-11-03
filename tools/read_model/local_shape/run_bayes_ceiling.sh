#!/usr/bin/env bash
# BAYES_CEILING：局部形状的贝叶斯后验 MC 探针（纯 CPU，不占 GPU）。
#
#   tmux new -d -s read_ls_bayes "bash tools/read_model/local_shape/run_bayes_ceiling.sh"
#
# 可调环境变量：
#   SNAPSHOTS=6000        MC 子集快照数（0=全量 40000）
#   M=256,1024,4096       收敛检查的三档采样量
#   METRIC=4096           M_max（三档从同一次采样截断）
#   CHAINS=256 ITERS=128  MH / 重要性采样的预算
#   TIERS=counts,hard,behave
#   CACHE=...             P0 诊断缓存
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
SNAPSHOTS="${SNAPSHOTS:-6000}"
M="${M:-256,1024,4096}"
METRIC="${METRIC:-4096}"
CHAINS="${CHAINS:-256}"
ITERS="${ITERS:-128}"
TIERS="${TIERS:-counts,hard,behave}"
mkdir -p "$OUT" "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/bayes_ceiling_${STAMP}.log"

echo "[run] python=$PYTHON"
echo "[run] cache=$CACHE"
echo "[run] snapshots=$SNAPSHOTS M=$M metric=$METRIC chains=$CHAINS iters=$ITERS"
echo "[run] tiers=$TIERS out=$OUT"
echo "[run] log=$LOG"

"$PYTHON" "$HERE/bayes_ceiling.py" \
  --cache "$CACHE" --out "$OUT" \
  --max-snapshots "$SNAPSHOTS" \
  --m-list "$M" --metric-samples "$METRIC" \
  --chains "$CHAINS" --iters "$ITERS" \
  --tiers "$TIERS" 2>&1 | tee -a "$LOG"
