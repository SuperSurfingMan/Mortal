#!/usr/bin/env bash
# A2c（v2s + RM4/RM5 + 200k）官方验收：evaluate / strat_metrics / compare_a3 / order_metrics。
#
# 纪律（README §13.4）：固定 202609 holdout、20000 样本、seed 20261016；order_metrics 同源
# protocol（n=40000）。验收用 best.pth(153k) 而非 last.pth(200k)。四任务各占一张卡并行，
# order_metrics 用独立 cache 目录，避免覆盖 A2b 的 cache 证据。
#
# 用法：tmux new -d -s a2c_accept "bash tools/read_model/run_a2c_acceptance.sh"
set -u

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
CKPT="$REPO/mortal/read_model_v2s_a2c/best.pth"
OUTDIR="$REPO/mortal/read_model_v2s_a2c"
LOGDIR="$REPO/mortal/logs/read_model/a2c_accept_20261021"
STATUS="$LOGDIR/status"

mkdir -p "$LOGDIR"
: > "$STATUS"

export CUDA_DEVICE_ORDER=PCI_BUS_ID       # 0/1/3 = P100，2 = RTX 8000
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "$REPO"

note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }

note "START ckpt=$CKPT"
"$PY" - "$CKPT" <<'PY' 2>&1 | tee -a "$STATUS"
import sys, torch
s = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
c = s.get('config', {})
print(f"[ckpt] step={s.get('step')} model={c.get('model')} wait_head={c.get('wait_head')} "
      f"val_score_version={s.get('val_score_version')} best_val={s.get('best_val')}")
PY

# ---- 四任务并行（各一张卡）----
declare -A PIDS
CUDA_VISIBLE_DEVICES=0 "$PY" tools/read_model/evaluate.py \
    --state "$CKPT" --val-globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
    --samples 20000 --seed 20261016 --out "$OUTDIR/eval_report.json" \
    > "$LOGDIR/evaluate.log" 2>&1 &
PIDS[EVAL]=$!

CUDA_VISIBLE_DEVICES=1 "$PY" tools/read_model/strat_metrics.py \
    --state "$CKPT" --samples 20000 --out "$OUTDIR/strat_report.json" \
    > "$LOGDIR/strat_metrics.log" 2>&1 &
PIDS[STRAT]=$!

CUDA_VISIBLE_DEVICES=3 "$PY" tools/read_model/compare_a3.py \
    --net "$CKPT" --samples 20000 --n-boot 200 --seed 20261016 \
    > "$LOGDIR/compare_a3.log" 2>&1 &
PIDS[CMP]=$!

RM_CACHE_DIR=/tmp/read_model_diag_cache_a2c CUDA_VISIBLE_DEVICES=2 \
    "$PY" tools/read_model/order_metrics.py --state "$CKPT" --n 40000 --rebuild \
    > "$LOGDIR/order_metrics.log" 2>&1 &
PIDS[ORDER]=$!

for tag in EVAL STRAT CMP ORDER; do
    wait "${PIDS[$tag]}"
    note "${tag}_RC=$?"
done

# ---- 结果摘要（只抓判据相关行）----
{
    echo "===== evaluate ====="
    grep -E "alloc CE|alloc top1|wait CE|wait hit|听牌检测|听牌标定|结构性听牌率|立直家|all |riichi |called=1|clean_turn|向听 头|形制" \
        "$LOGDIR/evaluate.log" || true
    echo "===== compare_a3 ====="
    grep -E "alloc_ce|wait_hit1|riichi_hit1|wait_ce_active|wait_nlt_active|wait_cond_ce_active|tenpai_auc" \
        "$LOGDIR/compare_a3.log" || true
    echo "===== order_metrics ====="
    grep -E "single/|window/|斜率|RM1[2-5]:" "$LOGDIR/order_metrics.log" || true
} >> "$STATUS" 2>&1

note "DONE_ALL"
