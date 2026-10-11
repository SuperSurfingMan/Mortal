# 第 2 轮判定实验（加大功效版）：new(round2 424k) vs old(round1 200k)
# 设计：双向各 5 个种子块 × 500 sets = 共 5000 sets = 20,000 半庄
#       δ = (avg_pt_A − avg_pt_B) / 2，SE(δ) ≈ 0.62 pt（把每次对战当独立样本的保守估计）
# 5 张卡并行，两波；每波 5 个任务，每任务独占一个日志目录（避免互相 rmtree）
# 用法: tmux new -d -s ladder_r2 'bash /mnt/data8/mortal/Mortal/mortal/run_r2_ladder.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python

# ==== 每个任务启动前都完整重设一遍环境变量（一个都不能省）====
setup_env () {
    export MORTAL_ROOT=$ROOT
    export MORTAL_CFG=$ROOT/mortal/config.round2.toml          # 必需：配置文件绝对路径
    export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export RAYON_NUM_THREADS=8
    export CUDA_DEVICE_ORDER=PCI_BUS_ID                        # 必需：序号对齐 nvidia-smi
    cd $ROOT/mortal
}
setup_env

NEW=checkpoints/round2/mortal.pth              # 424,000 步（第 2 轮终态：白名单 + grp_h128）
OLD=checkpoints/mortal_step_000200000.pth      # 200,000 步（第 1 轮终态）
SETS=500
OUT=logs/ladder_r2
mkdir -p $OUT

job () {   # $1=gpu $2=tag $3=challenger $4=champion $5=seed_start
    local gpu=$1 tag=$2 chal=$3 cham=$4 seed=$5
    (
        setup_env                                       # 子 shell 内再设一遍，避免继承污染
        export CUDA_VISIBLE_DEVICES=$gpu
        local t0=$(date +%s)
        $PY -u ladder.py "$chal" "$cham" "$seed" "$SETS" "$OUT/$tag" > "$OUT/$tag.log" 2>&1
        echo "[$(date '+%F %T')] $tag gpu=$gpu 用时 $(( ($(date +%s)-t0)/60 )) 分  $(grep -a 'avg_pt' "$OUT/$tag.log" | tail -1)"
    ) &
}

echo "==== 起跑 $(date) | 新=$NEW vs 旧=$OLD | 每任务 $SETS sets × 10 任务 ===="
GPUS=(2 0 1 3 4)

echo "---- 波 1：A 方向（新当挑战者，旧当卫冕者）$(date '+%H:%M:%S') ----"
i=0
for seed in 44000 44500 45000 45500 46000; do
    job ${GPUS[$i]} "A_${seed}" "$NEW" "$OLD" "$seed"; i=$((i+1))
done
wait
echo "---- 波 1 完成 $(date '+%H:%M:%S') ----"

echo "---- 波 2：B 方向（旧当挑战者，新当卫冕者，独立种子块消除牌山偏置）$(date '+%H:%M:%S') ----"
i=0
for seed in 54000 54500 55000 55500 56000; do
    job ${GPUS[$i]} "B_${seed}" "$OLD" "$NEW" "$seed"; i=$((i+1))
done
wait
echo "---- 波 2 完成 $(date '+%H:%M:%S') ----"

echo "==== 汇总判读 ===="
$PY ../tools/training/agg_ladder.py --dir $OUT --new $NEW --old $OLD
echo "ALL DONE $(date)"
