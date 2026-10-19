#!/usr/bin/env bash
# 读牌模型 A2c 训练（tmux 会话 read_a2c）：v2s + RM4/RM5 听牌头改造 + RM7 扩步数。
#
# 与 A2b（run_v2s.sh 50k 轮）的差别——README §11.7 P1：
#   1) --wait-head tile（RM5，§11.3）：H2 从 tile-agnostic 的 flat 头换成
#      逐牌列打分头（与 H1 同构）。冻结骨干探针证据：只加逐列 h_x 即
#      cond_ce −0.152、hit@1 +35% 相对、立直 hit@1 +34% 相对。
#      参数量 1,371,017 → 1,346,576（更小）。
#   2) RM4 输出层因子化（§11.1）：L_wait 从 35 类 softmax CE 改为 34 类
#      条件分布 CE（softmax34，与 evaluate 的 cond_ce 同口径）；tenpai
#      标量由 L_wait_tenpai BCE 独占监督，两个损失不再抢同一归一化
#      （35 类写法是 A2「P(未听) 塌缩」事故的结构性根源）。
#   3) RM7（§11.4）：步数 50k → 200k 并重设 cosine 终点（scheduler 自动
#      跟随 --steps）；DataLoader worker 8 → 16（obs 编码是吞吐瓶颈，
#      encode_obs 的 SP 表需改 libriichi，越界本次不做，README 已标注）。
#
# 同数据、同种子（20261016）、同 batch（512）、同阶段表——与 A2b 唯一的
# 系统性差异是上述三项 + 步数。验收判据（README §11.7）：
#   RM4：AUC ≥0.9685、meanP(听)/实际 ∈[0.95,1.05]、cond_ce ≤2.896、hit@1 不退
#   RM5：端到端 hit@1 ≥0.16、立直 hit@1 ≥0.135（A2b 0.1343/0.1116）
#   RM7：200k 的 val_score 显著低于 2.0924
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=${OUT:-$REPO/mortal/read_model_v2s_a2c}
LOGDIR=$REPO/mortal/logs
LOG=${LOG:-$LOGDIR/read_model/read_model_v2s_a2c_train.log}
RESUME=${RESUME:-}
PREFETCH=${PREFETCH:-2}
mkdir -p "$OUT" "$LOGDIR/read_model"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}       # P100；GPU=1 换卡（2026-10-19 重启时迁到 GPU 1）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${STEPS:-200000}
NW=${NW:-16}
STAGE_SCHEDULE=${STAGE_SCHEDULE:-1:0.27,2:0.30,3:0.43}

cd "$REPO/mortal"
echo "=== read_model A2c train start $(date -u +%FT%TZ) | steps=$STEPS workers=$NW ==="
"$PY" -u "$REPO/tools/read_model/train.py" \
  --model v2s \
  --wait-head tile \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 \
  --stage-schedule "$STAGE_SCHEDULE" \
  --num-workers "$NW" --prefetch-factor "$PREFETCH" --augment random \
  --eval-samples 8192 --eval-every 1000 --log-every 200 \
  ${RESUME:+--resume "$RESUME"} \
  --seed 20261016 2>&1 | tee -a "$LOG"
echo "=== read_model A2c train done $(date -u +%FT%TZ) ==="

# 用法备注：
#   正式轮（默认 200k）: bash run_a2c.sh
#   崩溃/重启续训      : OUT=$REPO/mortal/read_model_v2s_a2c \
#                        RESUME=$REPO/mortal/read_model_v2s_a2c/last.pth \
#                        STEPS=<剩余步数> NW=8 PREFETCH=1 bash run_a2c.sh
#   （end_step=start+total；续训填剩余步数凑满 200k 总量）
#   保持原课程边界续训（推荐）：按当前 step 手工配绝对阶段表，使三段之和
#   = 剩余步数。例：32k 处重启且原边界为 54k/114k/200k，则
#   STAGE_SCHEDULE=1:22000,2:60000,3:86000 STEPS=168000（=22000+60000+86000）。
#   背景：2026-10-19 首跑 NW=16 时 DataLoader worker RSS 涨到 ~15.7GB/个
#   （251GB 机器可用内存一度只剩 24GB，A2 事故同款前兆），已在 32k 处
#   重启降为 NW=8 PREFETCH=1 续训；worker 内存随 NW 线性增长，NW≤8 为安全线。
