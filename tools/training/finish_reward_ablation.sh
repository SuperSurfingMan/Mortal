#!/usr/bin/env bash
# 奖励消融收尾链：等待三组训练到 756k → 固定 holdout → 双向 ladder → block bootstrap。
set -euo pipefail

ROOT=/mnt/data8/mortal/Mortal
MORTAL=$ROOT/mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
PARENT=$MORTAL/checkpoints/round4/mortal.pth
TAGS=(A_rank_pt B_rank_pt_aligned C_delta_points)
GPUS=(0 1 3 4 2)
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p reward_ablation

wait_for_training() {
    while true; do
        done_count=0
        for tag in "${TAGS[@]}"; do
            state="checkpoints/reward_ablation/$tag/mortal.pth"
            log="reward_ablation/$tag/train.log"
            steps=$($PY - "$state" <<'PY'
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location='cpu', weights_only=True).get('steps', -1))
except Exception:
    print(-1)
PY
)
            if [ "$steps" -ge 756000 ] && grep -aq 'training budget reached' "$log"; then
                done_count=$((done_count + 1))
            elif grep -aqE 'non-finite state|Traceback|exit=[1-9]' "$log"; then
                echo "[错误] $tag 训练异常终止（steps=$steps），停止收尾链。请检查 $log"
                exit 1
            else
                echo "[$(date '+%F %T')] $tag steps=$steps waiting"
            fi
        done
        if [ "$done_count" -eq "${#TAGS[@]}" ]; then
            echo "[$(date '+%F %T')] 三组训练均已完成 756k"
            break
        fi
        sleep 60
    done
}

run_holdout() {
    local tag=$1
    local cfg="$PWD/reward_ablation/$tag/config.toml"
    local ckpt="$PWD/checkpoints/reward_ablation/$tag/mortal.pth"
    local out="$PWD/reward_ablation/$tag/holdout_202609.log"
    MORTAL_CFG="$cfg" CUDA_VISIBLE_DEVICES=4 "$PY" -u holdout_metrics.py \
        --month 202609 --n 24 --seed 20261009 --workers 1 "$ckpt" > "$out" 2>&1
    grep -aE 'holdout_sample_digest|holdout_target_digest|steps=|env:' "$out" || true
}

run_ladder_job() {
    local gpu=$1 tag=$2 direction=$3 seed=$4 challenger=$5 champion=$6
    local cfg="$PWD/reward_ablation/$tag/config.toml"
    local root="$PWD/reward_ablation/$tag/ladder"
    local out="$root/${direction}_${seed}"
    mkdir -p "$root"
    (
        export MORTAL_CFG="$cfg"
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$PY" -u ladder.py "$challenger" "$champion" "$seed" 500 "$out" \
            --no-amp --agari-guard --tag "${tag}-${direction}-${seed}" > "$out.log" 2>&1
        grep -aE 'avg_pt|run_meta|exec_cfg' "$out.log" | tail -3 || true
    ) &
    PIDS+=("$!")
}

run_ladder_variant() {
    local tag=$1
    local candidate="$PWD/checkpoints/reward_ablation/$tag/mortal.pth"
    local root="$PWD/reward_ablation/$tag/ladder"
    PIDS=()
    i=0
    for seed in 100000 100500 101000 101500 102000; do
        run_ladder_job "${GPUS[$i]}" "$tag" A "$seed" "$candidate" "$PARENT"
        i=$((i + 1))
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done

    PIDS=()
    i=0
    for seed in 110000 110500 111000 111500 112000; do
        run_ladder_job "${GPUS[$i]}" "$tag" B "$seed" "$PARENT" "$candidate"
        i=$((i + 1))
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done

    "$PY" ../tools/training/agg_ladder.py \
        --dir "$root" --new "$candidate" --old "$PARENT" \
        --logs-root "$root" --challenger-name challenger \
        --bootstrap 5000 --bseed 20261012 --workers 28 \
        > "$PWD/reward_ablation/$tag/ladder_summary.log" 2>&1
    cat "$PWD/reward_ablation/$tag/ladder_summary.log"
}

wait_for_training

echo '===== 固定 holdout（同 month/seed，核对 sample_digest） ====='
for tag in "${TAGS[@]}"; do
    run_holdout "$tag"
done

echo '===== 双向 ladder + block bootstrap ====='
for tag in "${TAGS[@]}"; do
    echo "===== $tag ====="
    run_ladder_variant "$tag"
done

echo "[$(date '+%F %T')] 奖励消融收尾完成"
