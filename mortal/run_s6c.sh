#!/usr/bin/env bash
# S6c 端到端编排（2026-10-23）：预检 → 回归 → 首层手术 → 透明性校验 → 冒烟 → 20k 探针
#                                  → 200k 正式轮 → 双向 ladder
#
# 用法（tmux 承载，日志落 mortal/logs/s6c/）：
#   tmux new -d -s s6c "bash /mnt/data8/mortal/Mortal/mortal/run_s6c.sh all"
#   STAGE=probe bash run_s6c.sh            # 只跑某段：preflight|regression|surgery|transparent|
#                                          #          smoke|probe|formal|ladder|all
#   可选环境变量：CUDA_VISIBLE_DEVICES（默认 2=RTX 8000）、LADDER_BLOCKS（默认 5）、LADDER_SETS（默认 500）
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
CFG=config.s6c.toml
CFG_PROBE=config.s6c_probe.toml
CFG_SMOKE=config.s6c_smoke.toml
CFG_BASE=config.s6c_base.toml          # 与正式配置相同但 extra_features=''（holdout 对照口径）
BASE=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth
INIT_REF=checkpoints/s6c/B756k_s6c_init.pth   # 手术产物（零初始化，保留作通道使用度的 ref）
INIT=checkpoints/s6c/mortal.pth                # 训练实际续写的 state_file
LOGS=logs/s6c
STAGE=${STAGE:-all}
LADDER_BLOCKS=${LADDER_BLOCKS:-5}
LADDER_SETS=${LADDER_SETS:-500}

cd "$ROOT/mortal" || { echo "缺少 $ROOT/mortal"; exit 1; }
mkdir -p "$LOGS" checkpoints/s6c checkpoints/s6c_smoke

export MORTAL_ROOT=$ROOT
export MORTAL_CFG=$CFG                  # 默认口径；holdout/smoke 各自用行内覆盖
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # RTX 8000（fp32 实测最快）
export CHECK_GPU_NAME=${CHECK_GPU_NAME:-RTX 8000}

say () { echo "[$(date '+%F %T')] $*"; }
want () { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
holdout () {  # $1=cfg $2=ckpt $3=tag
    MORTAL_CFG="$1" "$PY" -u holdout_metrics.py "$2" --n 24 --workers 8 2>&1 \
        | tee "$LOGS/holdout_$3.log"
}
metric_line () { grep -a 'top1=' "$1" | tail -1 | sed 's/^[^ ]* *//'; }
digest_line () { grep -a 'holdout_sample_digest' "$1" | tail -1; }

preflight () {
    say '==== 0) 预检 ===='
    "$PY" env_check.py --cfg "$CFG" || { echo '预检未通过，终止'; exit 1; }
    "$PY" -c "import sys; sys.path.insert(0,'.'); import torch; print('torch', torch.__version__)"
}

regression () {
    say '==== 1) S6c 特征回归 ===='
    "$PY" -u ../tools/test_s6c_features.py 2>&1 | tee "$LOGS/regression.log"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo '特征回归未过，终止'; exit 1; }
}

surgery () {
    say '==== 2) 首层手术（B/756k → +2 通道，零初始化）===='
    if [ ! -f "$INIT_REF" ]; then
        "$PY" -u ../tools/expand_ckpt_v6.py "$BASE" "$INIT_REF" 2>&1 | tee "$LOGS/surgery.log"
        [ "${PIPESTATUS[0]}" = 0 ] || { echo '手术失败，终止'; exit 1; }
    else
        echo "已存在 $INIT_REF，跳过手术"
    fi
    # 训练用的 state_file 从手术产物复制（手术产物本身保留作零初始化 ref）
    [ -f "$INIT" ] || cp "$INIT_REF" "$INIT"
    echo "train state_file: $INIT（← $INIT_REF）"
}

transparent () {
    say '==== 3) 透明性校验（手术 ckpt 的 holdout 必须与 B/756k 逐位相同）===='
    # 两侧用**同一批量/种子/进程数**，只有 extra_features 开关不同
    holdout "$CFG_BASE" "$BASE" base
    holdout "$CFG" "$INIT" init
    local l_base l_init d_base d_init
    l_base=$(metric_line "$LOGS/holdout_base.log")
    l_init=$(metric_line "$LOGS/holdout_init.log")
    d_base=$(digest_line "$LOGS/holdout_base.log")
    d_init=$(digest_line "$LOGS/holdout_init.log")
    echo "  B/756k : $l_base"
    echo "  S6c初始化: $l_init"
    [ "$d_base" = "$d_init" ] || { echo "FAIL 抽样摘要不同（数据口径不一致）：$d_base / $d_init"; exit 1; }
    [ "$l_base" = "$l_init" ] || { echo 'FAIL 手术 ckpt 与父代 holdout 不一致 → 管线不透明'; exit 1; }
    say '透明性校验通过（top1/dqn_loss/agari 全部一致）'
}

smoke () {
    say '==== 4) 冒烟（2k 步）===='
    # 每次都从零初始化的手术产物重起（smoke 可重复跑；不会把 2k 步的垃圾权重带进探针）
    cp -f "$INIT_REF" checkpoints/s6c_smoke/mortal.pth
    local t0=$(date +%s)
    MORTAL_CFG="$CFG_SMOKE" "$PY" -u train.py > "$LOGS/smoke.log" 2>&1
    local dt=$(( $(date +%s) - t0 ))
    grep -aE 'obs shape|training budget reached|non-finite|Traceback|Error' "$LOGS/smoke.log" | tail -6
    grep -aq 'training budget reached' "$LOGS/smoke.log" || { echo '冒烟未跑到预算，终止'; exit 1; }
    say "冒烟完成，用时 ${dt}s"
    "$PY" -u ../tools/s6c_channel_usage.py checkpoints/s6c_smoke/mortal.pth \
        --ref "$INIT_REF" 2>&1 | tee "$LOGS/smoke_usage.log"
}

probe () {
    say '==== 5) 20k 探针 ===='
    [ -f "$INIT" ] || { echo "缺少 $INIT（先跑 STAGE=surgery）"; exit 1; }
    MORTAL_CFG="$CFG_PROBE" "$PY" -u train.py > "$LOGS/probe_train.log" 2>&1
    grep -aE 'obs shape|loaded|training budget reached|non-finite|Traceback|Error' \
        "$LOGS/probe_train.log" | tail -8
    grep -aq 'training budget reached' "$LOGS/probe_train.log" || { echo '探针未跑到预算，终止'; exit 1; }
    grep -a 'behavior:' "$LOGS/probe_train.log" | tail -2

    holdout "$CFG" "$INIT" probe
    "$PY" -u ../tools/s6c_channel_usage.py "$INIT" --ref "$INIT_REF" 2>&1 | tee "$LOGS/probe_usage.log"
    local usage_ok=${PIPESTATUS[0]}

    local top1
    top1=$(grep -ao 'top1=[0-9.]*' "$LOGS/holdout_probe.log" | tail -1 | cut -d= -f2)
    say "探针 holdout top1=$top1（同 digest 基线见 holdout_base.log）"
    [ "$usage_ok" = 0 ] || { echo '探针门禁未过：首层新通道权重变化量低于原始 p10（等同没注入）'; exit 1; }
    "$PY" - "$top1" "$LOGS/holdout_base.log" <<'PYEOF'
import re, sys

top1 = float(sys.argv[1])
txt = open(sys.argv[2], encoding='utf-8', errors='ignore').read()
m = re.findall(r'top1=([0-9.]+)', txt)
base = float(m[-1])
print(f'probe top1={top1:.4f}  baseline top1={base:.4f}')
if top1 < base:
    print('GATE FAIL: holdout top1 低于同 digest 基线（不退才算过）')
    sys.exit(1)
print('GATE OK: holdout top1 不退')
PYEOF
    [ $? = 0 ] || { echo '探针门禁未过：停在这里，不要进 200k'; exit 1; }
    say '探针门禁通过 → 可执行 STAGE=formal 继续 200k'
}

formal () {
    say '==== 6) 200k 正式轮 ===='
    MORTAL_CFG="$CFG" "$PY" -u train.py > "$LOGS/formal_train.log" 2>&1
    grep -aE 'obs shape|loaded|training budget reached|non-finite|Traceback|Error' \
        "$LOGS/formal_train.log" | tail -8
    grep -aq 'training budget reached' "$LOGS/formal_train.log" || { echo '正式轮未跑到预算，终止'; exit 1; }
    grep -a 'behavior:' "$LOGS/formal_train.log" | tail -2
    holdout "$CFG" "$INIT" formal
    "$PY" -u ../tools/s6c_channel_usage.py "$INIT" --ref "$INIT_REF" 2>&1 | tee "$LOGS/formal_usage.log"
    grep -a 'top1=' "$LOGS/holdout_formal.log" | tail -1
}

ladder () {
    say "==== 7) 双向 ladder（S6c vs B/756k，λ=0，每向 ${LADDER_BLOCKS}×${LADDER_SETS} sets）===="
    local OUT=$LOGS/ladder
    mkdir -p "$OUT"
    GPUS=(2 0 1 3)
    local i seed
    # 新种子块：260000/270000 已被守卫剂量曲线（每向 8000 sets）消费；80k/90k 已被更早轮次消费
    local SEED_A=${SEED_A:-280000}
    local SEED_B=${SEED_B:-300000}
    say '-- A 方向（S6c 当挑战者）--'
    for ((i = 0; i < LADDER_BLOCKS; i++)); do
        seed=$((SEED_A + i * 500))
        ladder_job "${GPUS[$((i % ${#GPUS[@]}))]}" "A_${seed}" "$INIT" "$BASE" "$seed" "$OUT"
        [ $(( (i + 1) % ${#GPUS[@]} )) = 0 ] && wait
    done
    wait
    say '-- B 方向（B/756k 当挑战者，独立种子块）--'
    for ((i = 0; i < LADDER_BLOCKS; i++)); do
        seed=$((SEED_B + i * 500))
        ladder_job "${GPUS[$((i % ${#GPUS[@]}))]}" "B_${seed}" "$BASE" "$INIT" "$seed" "$OUT"
        [ $(( (i + 1) % ${#GPUS[@]} )) = 0 ] && wait
    done
    wait
    "$PY" ../tools/training/agg_ladder.py --dir "$OUT" --new "$INIT" --old "$BASE" \
        2>&1 | tee "$LOGS/ladder_summary.log"
}

# $1=gpu $2=tag $3=challenger $4=champion $5=seed $6=out_dir
ladder_job () {
    ( CUDA_VISIBLE_DEVICES=$1 MORTAL_CFG=$CFG "$PY" -u ladder.py "$3" "$4" "$5" "$LADDER_SETS" \
          "$6/$2" --guard-lambda 0 > "$6/$2.log" 2>&1
      echo "[$2] $(grep -a 'avg_pt' "$6/$2.log" | tail -1)" ) &
}

want preflight   && preflight
want regression  && regression
want surgery     && surgery
want transparent && transparent
want smoke       && smoke
want probe       && probe
want formal      && formal
want ladder      && ladder
say "STAGE=$STAGE 完成"
