# 第 3 轮判定：new=round3/mortal.pth(650k) vs old=round2/mortal.pth(424k)
# 与上一轮同方法：双向各 5 块 × 500 sets = 5000 sets = 20,000 半庄，5 卡两波并行
# 用法: tmux new -d -s ladder_r3 'bash /mnt/data8/mortal/Mortal/mortal/run_ladder_r3.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python

setup_env () {
    export MORTAL_ROOT=$ROOT
    export MORTAL_CFG=$ROOT/mortal/config.round3.toml
    export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export RAYON_NUM_THREADS=8
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    cd $ROOT/mortal
}
setup_env

NEW=checkpoints/round3/mortal.pth                  # 650,000 步（epoch 2 结束）
OLD=checkpoints/round2/mortal.pth                  # 424,847 步（epoch 1 结束）
SETS=500
OUT=logs/ladder_r3
mkdir -p $OUT

job () {   # $1=gpu $2=tag $3=challenger $4=champion $5=seed_start
    local gpu=$1 tag=$2 chal=$3 cham=$4 seed=$5
    (
        setup_env
        export CUDA_VISIBLE_DEVICES=$gpu
        local t0=$(date +%s)
        $PY -u ladder.py "$chal" "$cham" "$seed" "$SETS" "$OUT/$tag" > "$OUT/$tag.log" 2>&1
        echo "[$(date '+%F %T')] $tag gpu=$gpu 用时 $(( ($(date +%s)-t0)/60 )) 分  $(grep -a 'avg_pt' "$OUT/$tag.log" | tail -1)"
    ) &
}

echo "==== 起跑 $(date) | $NEW(650k) vs $OLD(424k) ===="
GPUS=(2 0 1 3 4)

echo "---- 波 1：A 方向（650k 当挑战者）$(date '+%H:%M:%S') ----"
i=0
for seed in 60000 60500 61000 61500 62000; do
    job ${GPUS[$i]} "A_${seed}" "$NEW" "$OLD" "$seed"; i=$((i+1))
done
wait
echo "---- 波 1 完成 $(date '+%H:%M:%S') ----"

echo "---- 波 2：B 方向（424k 当挑战者，独立种子块）$(date '+%H:%M:%S') ----"
i=0
for seed in 70000 70500 71000 71500 72000; do
    job ${GPUS[$i]} "B_${seed}" "$OLD" "$NEW" "$seed"; i=$((i+1))
done
wait
echo "---- 波 2 完成 $(date '+%H:%M:%S') ----"

echo "==== 汇总判读 ===="
$PY ../tools/training/agg_ladder.py --dir $OUT --new $NEW --old $OLD
echo "ALL DONE $(date)"
