#!/usr/bin/env bash
# 读牌模型 v2s 训练（tmux 会话 read_v2s）：v2 + 结构头专项。
#
# 与 v2（run_v2.sh）严格同预算同数据同种子，差别只在：
#   1) --model v2s：结构瓶颈注入向听分布（s_belief 38→46 维）
#   2) 新损失（train.py 默认开启，可用 CLI 消融）：
#      - 向听损失加权 ×3（aux 头 0.05→0.15）
#      - 结构头向听直接 CE（v2 完全缺失的一项，权重 0.5）
#      - 分解-向听一致性约束（头侧 0.3 + 分解侧 0.2，形制门控 normal）
#      - 向听-听牌一致性（shanten==0 ⟺ tenpai，权重 0.1）
#      - 向听/形制类均衡（--no-struct-balance 关闭）
# 背景：v2 结构头向听估计 25.7% < 28.4% 多数类基线，根因是该头零监督
#       （见 README §0ter 结论 3 与本脚本对应训练日志）。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=${OUT:-$REPO/mortal/read_model_v2s}          # 可用环境变量改出口（如 A2 50k 轮）
LOGDIR=$REPO/mortal/logs
LOG=${LOG:-$LOGDIR/read_model_v2s_train.log}
RESUME=${RESUME:-}                        # 续训 checkpoint 路径（含 optimizer/scheduler）
PREFETCH=${PREFETCH:-2}                   # DataLoader prefetch_factor（降内存压力时调小）
mkdir -p "$OUT" "$LOGDIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0            # P100（GPU2 被主训练占用）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${STEPS:-15000}
NW=${NW:-8}

cd "$REPO/mortal"
echo "=== read_model v2s train start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW ==="
"$PY" -u "$REPO/tools/read_model/train.py" \
  --model v2s \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 \
  --stage-schedule 1:0.27,2:0.30,3:0.43 \
  --num-workers "$NW" --prefetch-factor "$PREFETCH" --augment random \
  --eval-samples 8192 --eval-every 1000 --log-every 200 \
  ${RESUME:+--resume "$RESUME"} \
  --seed 20261016 2>&1 | tee -a "$LOG"
echo "=== read_model v2s train done $(date -u +%FT%TZ) ==="

# 用法备注：
#   15k 首版（默认）    : bash run_v2s.sh
#   A2 50k 正式轮       : OUT=$REPO/mortal/read_model_v2s_50k \
#                         LOG=$REPO/mortal/logs/read_model/read_model_v2s_50k_train.log \
#                         STEPS=50000 bash run_v2s.sh
#   A2 续训（崩溃后）   : OUT=$REPO/mortal/read_model_v2s_50k \
#                         LOG=$REPO/mortal/logs/read_model/read_model_v2s_50k_train.log \
#                         RESUME=$REPO/mortal/read_model_v2s_50k/last.pth \
#                         STEPS=43000 NW=4 PREFETCH=1 bash run_v2s.sh
#                         （end_step=start+total；续训填剩余步数凑满总量；
#                           NW=4/PREFETCH=1 缓解 DataLoader worker 内存压力）
