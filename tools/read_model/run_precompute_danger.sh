#!/usr/bin/env bash
# S6b 正式预计算：2026 01-08 names 树（113k 文件），8 分片并行。
# 每个分片独立进程、独立日志；GPU 在 0/1/3/4 间轮转（每张 2 个进程）。
#
# 用法：
#   tmux new -d -s pre_danger "bash tools/read_model/run_precompute_danger.sh > mortal/logs/read_model/precompute_danger_outer.log 2>&1"
# 环境变量：OUT（默认 mortal/danger_cache）、LOGP（默认 mortal/logs/read_model/precompute_danger）
set -u

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
OUT=${OUT:-$REPO/mortal/danger_cache}
LOGP=${LOGP:-$REPO/mortal/logs/read_model/precompute_danger}
NSHARD=${NSHARD:-8}
GPUS=(0 1 3 4)

mkdir -p "$OUT" "$(dirname "$LOGP")"

echo "[start] $(date -u +%FT%TZ)  shards=$NSHARD  out=$OUT"
pids=()
for i in $(seq 0 $((NSHARD - 1))); do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    # RAYON_NUM_THREADS=2：GameplayLoader 默认 rayon 会开满 56 线程；8 进程 × 56
    # 会严重过度订阅（实测 8 进程总吞吐仅 9.5 文件/s，单进程却有 4.0）。
    CUDA_VISIBLE_DEVICES="$gpu" MALLOC_ARENA_MAX=2 RAYON_NUM_THREADS=2 "$PY" -u \
        "$REPO/tools/read_model/precompute_danger.py" \
        --globs "$REPO/dataset/mjai/houou4_names/20260[1-8]/**/*.json.gz" \
        --num-shards "$NSHARD" --shard "$i" \
        --out-dir "$OUT" --batch-files 25 \
        > "${LOGP}_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "  shard $i → GPU $gpu  pid=$!"
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=$((fail + 1))
done
n=$(find "$OUT" -name '*.npz' | wc -l)
echo "[done] $(date -u +%FT%TZ)  npz=$n  failed_shards=$fail"
