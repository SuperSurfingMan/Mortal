#!/usr/bin/env bash
# 读牌模型「修时钟 + 最小显式 order」20k 探针（独立复制路线）。
#
# 与主任务 tools/read_model/run_p1b.sh 的关系：
#   - 本脚本只调用 tools/read_model_clock/ 下的**副本**；
#   - dataset/model_v2/train/evaluate/order_metrics/diagnostics 均已复制并改路径；
#   - 主任务目录与正在跑的 read_p1b_order_probe 不被读写/覆盖。
#
# 本路线相对主任务 order 探针的唯一系统性差异：
#   1) 时钟：token 顺序用全局事件索引 tl_pos（严格递增、无并列），
#      衰减用 tl_age_ev（事件年龄），不再用 per-player at_turn/tl_turn；
#   2) 最小显式特征：e_last + pos[m−1]−pos[m+1] + valid（tile_proj 3d+3），
#      去掉主任务 3d+5 里的 last_turn/turn_mean/last_age；
#   3) 反事实交换同时交换 tl_pos/tl_age_ev（s3158_local_extract.py 副本）。
#
# 20k 早探针目的：只回答「修时钟 + 最小顺序特征能否让 RM12 b 离开 0.01」，
# 不直接烧 200k；判据仍用 order_metrics 的 p1b_verdicts。
#
# 用法：
#   GPU=3 NW=4 bash tools/read_model_clock/run_p1b_clock.sh
#   STEPS=2000 GPU=3 NW=2 TAG=clock_smoke bash ...   # 冒烟
#   SMOKE=1 ... 由调用方传 MAX_FILES=8
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
TOOLS=$REPO/tools/read_model_clock
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
STEPS=${STEPS:-20000}
TAG=${TAG:-a2d_clock_probe20k}
OUT=${OUT:-$REPO/mortal/read_model_v2s_${TAG}}
LOGDIR=$REPO/mortal/logs
LOG=${LOG:-$LOGDIR/read_model/read_model_v2s_${TAG}.log}
GPU=${GPU:-3}
NW=${NW:-4}
BATCH=${BATCH:-512}
PREFETCH=${PREFETCH:-1}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_SAMPLES=${EVAL_SAMPLES:-8192}
STAGE_SCHEDULE=${STAGE_SCHEDULE:-1:0.27,2:0.30,3:0.43}
MAX_FILES_ARG=""
if [ -n "${MAX_FILES:-}" ]; then MAX_FILES_ARG="--max-files ${MAX_FILES}"; fi
RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then RESUME_ARG="--resume ${RESUME}"; fi

mkdir -p "$OUT" "$LOGDIR/read_model"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

cd "$REPO/mortal"
echo "=== read_model clock-order probe start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW gpu=$GPU ==="
"$PY" -u "$TOOLS/train.py" \
  --model v2s \
  --wait-head tile \
  --tl-mode tile \
  --tile-order \
  --next-tedashi-weight 0 \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size "$BATCH" \
  --stage-schedule "$STAGE_SCHEDULE" \
  --num-workers "$NW" --prefetch-factor "$PREFETCH" --augment random \
  --eval-samples "$EVAL_SAMPLES" --eval-every "$EVAL_EVERY" --log-every 200 \
  ${MAX_FILES_ARG} ${RESUME_ARG} \
  --seed 20261016 2>&1 | tee -a "$LOG"
echo "=== read_model clock-order probe done $(date -u +%FT%TZ) ==="
