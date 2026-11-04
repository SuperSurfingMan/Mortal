#!/usr/bin/env bash
# A2c 续训（第 2 段）：从 best@153k 温和重启，跑到绝对终点 653k。
#
# 动机（2026-11-04）：
#   A2c 的 val_score 最后 20 个点单调下降（2.198 → 2.0524），153k 时仍在降；
#   200k 的 2.0613 回升是 cosine LR 归零跑在次优点，不是平台。
#   原轮只覆盖 153k×512 = 7833 万样本 ≈ 全量 6.32 亿的 12.4%（一个 epoch 都没跑满）。
#   本脚本用 A2c best@153k 作初始化，重跑一条 600k 步的完整 cosine
#   （峰值 LR 1.2e-4、warmup 30k），数据覆盖 600k×512 = 3.07 亿 ≈ 全量 49%。
#
# 相对 run_a2c.sh 的差异（其余完全继承，保证可比）：
#   1) OUT 走新目录 read_model_v2s_a2c_cont —— 不覆盖 A2c 的 best.pth 资产。
#      --init 使 best_val 重置为 inf、ckpt 的 step 从 0 计数（日志里的 step
#      是「续训步」，与 A2c 的 153k 无关）；最终以两个目录的 best.pth 各自
#      跑官方 20000 集对比，而不是看 step 号。
#   2) 用 --init 而非 --resume：resume 时 train.py 会把 abs_total 设为
#      start_step+steps，而 scheduler 的 last_epoch 已恢复为 153k ⇒
#      lr_lambda(153000/653000) 虽然≈正常，但 400 步冒烟直接归零到 1.4e-9
#      （实测），参数静默冻结。--init 只借权重、重置 optimizer/scheduler，
#      得到一条完整 cosine（warmup 5% = 30k）——代价是丢掉 AdamW 动量，
#      在 30k warmup 内重建，占全程 5%。
#   3) seed 20261104（原 20261016）：让新一段的数据遍历顺序不同于前 12%，
#      避免重复采样同一批文件。
#   4) eval-every 2000：600k 步下 300 个评估点足够，减半写盘
#      （last.pth 132MB/次）与评估开销。
#   5) 阶段表 3:1.0：stage 只进日志/分组，不影响数据与增强。
#
# 冒烟（先验证 lr 在爬、loss 不炸，约 4 分钟）：
#   OUT=/tmp/a2c_cont_smoke LOG=/tmp/a2c_cont_smoke.log STEPS=4000 EVAL_EVERY=2000 \
#     bash tools/read_model/run_a2c_cont.sh
#
# 正式（约 60~70h @ P100，数据覆盖 600k×512 = 3.07 亿 ≈ 全量 49%）：
#   tmux new -d -s read_a2c_cont "bash tools/read_model/run_a2c_cont.sh"
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=${OUT:-$REPO/mortal/read_model_v2s_a2c_cont}
LOG=${LOG:-$REPO/mortal/logs/read_model/read_model_v2s_a2c_cont_train.log}
RESUME=${RESUME:-$REPO/mortal/read_model_v2s_a2c/best.pth}

STEPS=${STEPS:-600000}
LR=${LR:-1.2e-4}
NW=${NW:-8}
PREFETCH=${PREFETCH:-1}
EVAL_EVERY=${EVAL_EVERY:-2000}
SEED=${SEED:-20261104}
STAGE_SCHEDULE=${STAGE_SCHEDULE:-3:1.0}
# MODE=init   —— 只借权重，重置 optimizer/scheduler（LR 完全可控，但丢 AdamW 动量）
# MODE=resume —— 保留 optimizer/scheduler（保动量，需 --reset-base-lr + 大 steps）
MODE=${MODE:-init}

mkdir -p "$OUT" "$(dirname "$LOG")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# DataLoader worker 内存随训练时长累积（glibc arena 碎片）：A2c 首轮 NW=16 时
# 单 worker RSS 涨到 ~15.7GB，机器只剩 24GB。NW≤8 为安全线，且强制归还 arena。
export MALLOC_ARENA_MAX=2

cd "$REPO/mortal"
echo "=== A2c-cont start $(date -u +%FT%TZ) | mode=$MODE steps=$STEPS lr=$LR gpu=${GPU:-4} workers=$NW ==="
echo "    from=$RESUME"
echo "    out=$OUT"

if [[ "$MODE" == "resume" ]]; then
  LOAD_ARGS=(--resume "$RESUME" --reset-base-lr)
else
  LOAD_ARGS=(--init "$RESUME")
fi

"$PY" -u "$REPO/tools/read_model/train.py" \
  --model v2s \
  --wait-head tile \
  --train-globs \
    "$REPO/dataset/mjai/houou4/202[2-5]??/**/*.json.gz" \
    "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
  --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
  --out-dir "$OUT" \
  --steps "$STEPS" --batch-size 512 --lr "$LR" \
  --stage-schedule "$STAGE_SCHEDULE" \
  --num-workers "$NW" --prefetch-factor "$PREFETCH" --augment random \
  --eval-samples 8192 --eval-every "$EVAL_EVERY" --log-every 200 \
  "${LOAD_ARGS[@]}" \
  --seed "$SEED" 2>&1 | tee -a "$LOG"
echo "=== A2c-cont done $(date -u +%FT%TZ) ==="
