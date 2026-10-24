#!/usr/bin/env bash
# 数据扩窗探针 D1（2026-10-24）：B/756k 暖启 + 现窗口混入 2019-2021 的 20%
#
# 数据：`indexes/data_mix_2019_2021_20pct_v1.pth`
#   = B 的 492,810 文件（2022-2025，top400_v2）+ 70,588 文件（2019-2021，
#     年代名单过滤后按文件名 sha1 抽 20%；+14.3%）
# 单变量：obs/奖励/超参全部同 B，只改数据池。
#
# 用法（tmux 承载，日志落 mortal/logs/mix_v1/）：
#   STAGE=preflight bash run_mix.sh
#   STAGE=smoke     bash run_mix.sh    # 2k 步冒烟（验证数据池/吞吐/无 NaN）
#   STAGE=formal    bash run_mix.sh    # 100k 步正式轮（756k → 856k）
#   STAGE=ladder    bash run_mix.sh    # 双向 5×500 sets（λ=0，新种子块 320000/340000）
# 环境变量：CUDA_VISIBLE_DEVICES（默认 2）、SEED_A/SEED_B、LADDER_BLOCKS、LADDER_SETS、TOPS_TOL
# 注意：脚本运行期间不要改本文件（bash 增量读取会报 syntax error）。
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
CFG=config.mix_v1.toml
CFG_SMOKE=config.mix_v1_smoke.toml
BASE=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # 756k 父代
STATE=checkpoints/mix_v1/mortal.pth
LOGS=logs/mix_v1
STAGE=${STAGE:-formal}
LADDER_BLOCKS=${LADDER_BLOCKS:-5}
LADDER_SETS=${LADDER_SETS:-500}

cd "$ROOT/mortal" || exit 1
mkdir -p "$LOGS" checkpoints/mix_v1 checkpoints/mix_v1_smoke

export MORTAL_ROOT=$ROOT
export MORTAL_CFG=$CFG
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export CHECK_GPU_NAME=${CHECK_GPU_NAME:-RTX 8000}

say () { echo "[$(date '+%F %T')] $*"; }
want () { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
holdout () { MORTAL_CFG="$1" "$PY" -u holdout_metrics.py "$2" --n 24 --workers 8 2>&1 | tee "$LOGS/holdout_$3.log"; }

preflight () {
    say '==== 0) 预检 + 索引一致性 ===='
    "$PY" env_check.py --cfg "$CFG" || { echo '预检未通过'; exit 1; }
    "$PY" - <<'PYEOF'
import torch
st = torch.load('indexes/data_mix_2019_2021_20pct_v1.pth', weights_only=True)
m = st['manifest']
print(f"index n_files={m['n_files']:,} data_version={m['data_version']}")
print(f"mix: {m['mix']}")
assert m['n_files'] == len(st['file_list'])
assert m['mix']['old_added'] > 0
PYEOF
}

smoke () {
    say '==== 1) 冒烟（2k 步）===='
    cp -f "$BASE" checkpoints/mix_v1_smoke/mortal.pth
    local t0=$(date +%s)
    MORTAL_CFG="$CFG_SMOKE" "$PY" -u train.py > "$LOGS/smoke.log" 2>&1
    grep -aE 'obs shape|file index|training budget reached|non-finite|Traceback|Error' "$LOGS/smoke.log" | tail -6
    grep -aq 'training budget reached' "$LOGS/smoke.log" || { echo '冒烟未跑到预算'; exit 1; }
    say "冒烟完成，用时 $(( $(date +%s) - t0 ))s"
    grep -a 'behavior:' "$LOGS/smoke.log" | tail -1
}

formal () {
    say '==== 2) 100k 正式轮 ===='
    [ -f "$STATE" ] || cp "$BASE" "$STATE"
    MORTAL_CFG="$CFG" "$PY" -u train.py > "$LOGS/formal_train.log" 2>&1
    grep -aE 'obs shape|file index|total steps|training budget reached|non-finite|Traceback|Error' \
        "$LOGS/formal_train.log" | tail -8
    grep -aq 'training budget reached' "$LOGS/formal_train.log" || { echo '正式轮未跑到预算'; exit 1; }
    grep -a 'behavior:' "$LOGS/formal_train.log" | tail -2
    holdout "$CFG" "$STATE" formal
    grep -a 'top1=' "$LOGS/holdout_formal.log" | tail -1
}

# $1=gpu $2=tag $3=challenger $4=champion $5=seed $6=out_dir
ladder_job () {
    ( CUDA_VISIBLE_DEVICES=$1 MORTAL_CFG=$CFG "$PY" -u ladder.py "$3" "$4" "$5" "$LADDER_SETS" \
          "$6/$2" --guard-lambda 0 > "$6/$2.log" 2>&1
      echo "[$2] $(grep -a 'avg_pt' "$6/$2.log" | tail -1)" ) &
}

ladder () {
    say "==== 3) 双向 ladder（混入 vs B/756k，λ=0，每向 ${LADDER_BLOCKS}×${LADDER_SETS} sets）===="
    local OUT=$LOGS/ladder
    mkdir -p "$OUT"
    GPUS=(2 0 1 3)
    local SEED_A=${SEED_A:-320000}   # 260/270 守卫消费；280/300 S6c 消费
    local SEED_B=${SEED_B:-340000}
    local i seed
    for dir in A B; do
        say "-- $dir 方向 --"
        for ((i = 0; i < LADDER_BLOCKS; i++)); do
            if [ "$dir" = A ]; then
                seed=$((SEED_A + i * 500))
                ladder_job "${GPUS[$((i % ${#GPUS[@]}))]}" "A_${seed}" "$STATE" "$BASE" "$seed" "$OUT"
            else
                seed=$((SEED_B + i * 500))
                ladder_job "${GPUS[$((i % ${#GPUS[@]}))]}" "B_${seed}" "$BASE" "$STATE" "$seed" "$OUT"
            fi
            [ $(( (i + 1) % ${#GPUS[@]} )) = 0 ] && wait
        done
        wait
    done
    "$PY" ../tools/training/agg_ladder.py --dir "$OUT" --new "$STATE" --old "$BASE" \
        --logs-root "$OUT" --challenger-name challenger 2>&1 | tee "$LOGS/ladder_summary.log"
}

want preflight && preflight
want smoke    && smoke
want formal   && formal
want ladder   && ladder
say "STAGE=$STAGE 完成"
