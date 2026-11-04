#!/usr/bin/env bash
# P0-a：BAYES_CEILING 三层 tier 按 cond 口径重算（纯 CPU）。
#
#   tmux new -d -s read_ls_p0a "bash tools/read_model/local_shape/run_p0a.sh"
#
# 可调环境变量：
#   SNAPSHOTS=6000        MC 子集快照数（0=全量 40000）
#   M=256,1024,4096       收敛检查的三档采样量
#   METRIC=4096           M_max（三档从同一次采样截断）
#   CHAINS=256 ITERS=128  MH / 重要性采样的预算
#   TIERS=counts,hard,behave
#   CACHE=...             P0 诊断缓存
#   INDEX=...             样本定位缓存（与 bayes_ceiling 共用）
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
INDEX="${INDEX:-/tmp/bayes_ceiling_index.npz}"
OUT="${OUT:-$HERE/results}"
LOG_DIR="$ROOT/mortal/logs/read_model/local_shape"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/p0a_bayes_cond_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT"
"$PYTHON" tools/read_model/local_shape/p0a_bayes_cond.py \
  --cache "$CACHE" \
  --index-cache "$INDEX" \
  --max-snapshots "${SNAPSHOTS:-6000}" \
  --metric-samples "${METRIC:-4096}" \
  --m-list "${M:-256,1024,4096}" \
  --chains "${CHAINS:-256}" \
  --iters "${ITERS:-128}" \
  --tiers "${TIERS:-counts,hard,behave}" \
  --out "$OUT" 2>&1 | tee -a "$LOG"
