# 第 4 轮端到端：预检 → 训练（增强样本一遍）→ 双向判定 → 汇总，单 tmux 会话串行
# 用法: tmux new -d -s r4_chain 'bash /mnt/data8/mortal/Mortal/mortal/run_round4.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export MORTAL_CFG=$ROOT/mortal/config.round4.toml
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2                    # RTX 8000（PCI 序）
export CHECK_GPU_NAME='RTX 8000'                 # env_check 用来反验选卡

cd $ROOT/mortal
mkdir -p checkpoints/round4 runs logs

echo "==== 0) 预检 $(date) ===="
$PY env_check.py --cfg config.round4.toml || { echo '预检未通过，终止'; exit 1; }

echo "==== 1) 暖启（round2/424k → round4）$(date) ===="
if [ ! -f checkpoints/round4/mortal.pth ]; then
    cp checkpoints/round2/mortal.pth checkpoints/round4/mortal.pth
    echo "已拷贝 checkpoints/round4/mortal.pth"
else
    echo "round4/mortal.pth 已存在，直接续训"
fi

echo "==== 2) 训练（max_train_steps=645000，到点自停）$(date) ===="
$PY -u train.py > logs/main_r4_train.log 2>&1
echo "训练进程结束 $(date)"
tr '\r' '\n' < logs/main_r4_train.log | grep -aE 'training budget reached|Traceback|Error|device:|file list size|resetting best_perf' | tail -6

NEW=checkpoints/round4/mortal.pth
OLD=checkpoints/round2/mortal.pth
OUT=logs/ladder_r4
SETS=500
GPUS=(2 0 1 3 4)
mkdir -p $OUT

job () {   # $1=gpu $2=tag $3=challenger $4=champion $5=seed_start
    (
        export CUDA_VISIBLE_DEVICES=$1
        local t0=$(date +%s)
        $PY -u ladder.py "$3" "$4" "$5" "$SETS" "$OUT/$2" > "$OUT/$2.log" 2>&1
        echo "[$(date '+%F %T')] $2 gpu=$1 用时 $(( ($(date +%s)-t0)/60 )) 分  $(grep -a 'avg_pt' "$OUT/$2.log" | tail -1)"
    ) &
}

echo "==== 3) 判定：round4 vs round2（双向各 5×500 sets，新种子块）$(date) ===="
echo "---- 波 1：A 方向（round4 当挑战者）----"
i=0
for seed in 80000 80500 81000 81500 82000; do
    job ${GPUS[$i]} "A_${seed}" "$NEW" "$OLD" "$seed"; i=$((i+1))
done
wait
echo "---- 波 2：B 方向（round2 当挑战者，独立种子块）----"
i=0
for seed in 90000 90500 91000 91500 92000; do
    job ${GPUS[$i]} "B_${seed}" "$OLD" "$NEW" "$seed"; i=$((i+1))
done
wait

echo "==== 4) 汇总判读 $(date) ===="
$PY ../tools/training/agg_ladder.py --dir $OUT --new "$NEW" --old "$OLD"
echo "ALL DONE $(date)"
