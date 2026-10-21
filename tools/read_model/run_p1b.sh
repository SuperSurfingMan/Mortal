#!/usr/bin/env bash
# 读牌模型 P1b（RM6 最小集 + RM12 tile-indexed 时序流 + RM14 next-tedashi 辅助）训练脚本。
#
# 与 A2c（run_a2c.sh）的唯一系统性差异：
#   --tl-mode tile：TimelineEncoder 从 mean-pool 的 [B,3,64] 改为逐牌列
#   [B,3,64,34]（token 在自身牌列 ±1/±2 散射、chi/pon consumed 也散射、
#   真实巡目 tl_turn/tl_age 进 token、recency=exp(-age) 加权）。
#   dataset 侧车字段：tl_turn / tl_age / tl_consumed / tl_flags（RM6 最小集）。
#   旧 43 维 token 末位仍写快照巡目常数 ⇒ A2b/A2c 旧 ckpt 逐位复现。
#   RM14：--next-tedashi-weight > 0 时建「预测目标家下一张手切牌」辅助头
#   （tile 时序图的 H1 逐列 logit + 无手切标量），直接给时序通路压力；
#   旧 tile ckpt 无该头（config 无 next_tedashi_weight）仍严格加载。
#   RM15/显式 order：TILE_ORDER=1 时给 tile 图加 last-turn/age/embedding、
#   邻域 mean-turn 与 turn[m−1]−turn[m+1] 局部先后差（config 记录 tile_order）。
#
# 用法：
#   20k 早探针（先判断 b/r² 是否离开 0.01 区间，别直接烧 200k）：
#     STEPS=20000 NTW=0 TILE_ORDER=1 TAG=a2d_order_probe20k \
#     OUT=$REPO/mortal/read_model_v2s_a2d_order_probe20k \
#     LOG=$REPO/mortal/logs/read_model/read_model_v2s_a2d_order_probe20k.log \
#     bash run_p1b.sh
#   复现 RM12 v1（无 next-tedashi 头、无显式 order）：NTW=0 TILE_ORDER=0
#   复现 RM14 aux：NTW=0.2 TILE_ORDER=0
#   正式 200k：
#     bash run_p1b.sh
#   resume（填剩余步数；LR 基准按 start+remaining 的绝对终点）：
#     RESUME=$REPO/mortal/read_model_v2s_a2d/last.pth STEPS=<剩余> \
#     OUT=$REPO/mortal/read_model_v2s_a2d bash run_p1b.sh
#
# 验收（README §12.8 / §13.4a-⑤）：
#   RM12  order_metrics 反事实 b >= 0.3 且 r² >= 0.2（A2c 0.0093 / 1.28%）
#   RM15  立直层 ±2 窗口超地板 CI95 下界 > 0（A2c [-0.113,-0.014]）
#   RM13  非立直单张扣地板复现 >= 92.7%（不退）
#   RM14  立直单张扣地板复现 >= 66.2%（不退）
#   RM14aux next-tedashi CE 必须学习（train_log 的 next_tedashi 分项下降）
#   TILE_ORDER=1 时另有：tile_proj 输入 3d+5，显式 turn_diff 通道进 H1/H2。
#   常规：alloc_ce / wait_hit1 / tenpai_auc / cond_ce 不退。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
STEPS=${STEPS:-200000}
NTW=${NTW:-0.2}
TILE_ORDER=${TILE_ORDER:-0}
TAG=${TAG:-a2d}
OUT=${OUT:-$REPO/mortal/read_model_v2s_${TAG}}
LOGDIR=$REPO/mortal/logs
LOG=${LOG:-$LOGDIR/read_model/read_model_v2s_${TAG}_train.log}
RESUME=${RESUME:-}
PREFETCH=${PREFETCH:-1}
NW=${NW:-8}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_SAMPLES=${EVAL_SAMPLES:-8192}
mkdir -p "$OUT" "$LOGDIR/read_model"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-2}       # RTX 8000-48GB；P100 请显式 GPU=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# DataLoader worker RSS 随训练时长累积（glibc arena），长轮次要限制 arena 数
export MALLOC_ARENA_MAX=2

STAGE_SCHEDULE=${STAGE_SCHEDULE:-1:0.27,2:0.30,3:0.43}
ORDER_ARG=""
if [ "$TILE_ORDER" = "1" ]; then ORDER_ARG="--tile-order"; fi

cd "$REPO/mortal"
echo "=== read_model P1b (RM6min+RM12) start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW ==="
"$PY" -u "$REPO/tools/read_model/train.py" \
  --model v2s \
  --wait-head tile \
  --tl-mode tile \
  --next-tedashi-weight "$NTW" \
  ${ORDER_ARG} \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 \
  --stage-schedule "$STAGE_SCHEDULE" \
  --num-workers "$NW" --prefetch-factor "$PREFETCH" --augment random \
  --eval-samples "$EVAL_SAMPLES" --eval-every "$EVAL_EVERY" --log-every 200 \
  ${RESUME:+--resume "$RESUME"} \
  --seed 20261016 2>&1 | tee -a "$LOG"
echo "=== read_model P1b done $(date -u +%FT%TZ) ==="
