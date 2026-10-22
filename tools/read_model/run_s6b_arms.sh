#!/usr/bin/env bash
# S6b 探针编排：等训练集预计算完成 → 跑 202609 holdout 预计算 → 启动 A1/A2/A3。
# A0 已单独在跑（GPU0）。用法：
#   tmux new -d -s s6b_arms "bash tools/read_model/run_s6b_arms.sh > mortal/logs/read_model/s6b_arms_outer.log 2>&1"
set -u

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
GPUS=(0 1 3 4)
cd "$REPO"

echo "[arms] $(date -u +%FT%TZ) 等待训练集预计算..."
while ps -eo args | grep -q "[p]recompute_danger.py --globs"; do
    sleep 60
done
echo "[arms] $(date -u +%FT%TZ) 训练集预计算结束；npz=$(find mortal/danger_cache -name '*.npz' | wc -l)"

echo "[arms] 202609 holdout 预计算（8 分片）..."
pids=()
for i in $(seq 0 7); do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES="$gpu" MALLOC_ARENA_MAX=2 RAYON_NUM_THREADS=2 "$PY" -u \
        tools/read_model/precompute_danger.py \
        --globs "$REPO/dataset/mjai/houou4/202609/**/*.json.gz" \
        --num-shards 8 --shard "$i" --out-dir mortal/danger_cache --batch-files 25 \
        > "mortal/logs/read_model/precompute_holdout_shard${i}.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[arms] $(date -u +%FT%TZ) holdout 完成；npz=$(find mortal/danger_cache -name '*.npz' | wc -l)"

# 启动 A1/A2/A3（各自 tmux 会话；A0 已在跑）
declare -A ARM_GPU=([1]=1 [2]=3 [3]=4)
for arm in 1 2 3; do
    gpu=${ARM_GPU[$arm]}
    tmux kill-session -t "s6b_a${arm}" 2>/dev/null
    tmux new -d -s "s6b_a${arm}" \
        "cd $REPO/mortal && CUDA_VISIBLE_DEVICES=$gpu MALLOC_ARENA_MAX=2 RAYON_NUM_THREADS=8 MORTAL_CFG=config.s6b_a${arm}.toml $PY -u train.py > logs/s6b_a${arm}.log 2>&1"
    echo "[arms] A${arm} → GPU ${gpu} (tmux s6b_a${arm})"
done
echo "[arms] $(date -u +%FT%TZ) A1/A2/A3 已启动"
