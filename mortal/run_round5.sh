# 第 5 轮端到端：预检 → 暖启（B 组 756k）→ 训练（top200_v3 新管线）→ 双向 ladder vs B 组 → 汇总
# 唯一实质变量：数据选择 top400_v2 → top200_v3（含 names 修复后的 2026 训练窗）
# 用法: tmux new -d -s r5_chain 'bash /mnt/data8/mortal/Mortal/mortal/run_round5.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export MORTAL_CFG=$ROOT/mortal/config.round5.toml
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2                    # RTX 8000（PCI 序）——实测 fp32 4.28 it/s vs P100 2.48
export CHECK_GPU_NAME='RTX 8000'                 # env_check 反验选卡

cd $ROOT/mortal
mkdir -p checkpoints/round5 runs logs

echo "==== 0) 预检 $(date) ===="
$PY env_check.py --cfg config.round5.toml || { echo '预检未通过，终止'; exit 1; }

echo "==== 1) 暖启（B 组 756k → round5）$(date) ===="
# B 组 = 同奖励口径（pts=[2,1,0,-3]）下的 top400 控制组；R5 只换数据选择
if [ ! -f checkpoints/round5/mortal.pth ]; then
    cp checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth checkpoints/round5/mortal.pth
    echo "已暖启: checkpoints/round5/mortal.pth ← B_rank_pt_aligned/mortal.pth (steps=756000)"
else
    echo "round5/mortal.pth 已存在，直接续训"
fi

echo "==== 2) 训练（max_train_steps=870000，到点自停）$(date) ===="
echo "---- $(date '+%F %T') 训练进程启动 ----" >> logs/main_r5_train.log
$PY -u train.py >> logs/main_r5_train.log 2>&1
train_code=$?
echo "训练进程结束 $(date)，退出码 $train_code"
tr '\r' '\n' < logs/main_r5_train.log | grep -aE 'training budget reached|non-finite|Traceback|Error|device:|file list size|loaded 200 players|reward:' | tail -8
if [ "$train_code" -ne 0 ]; then
    echo "!! 训练非正常结束（退出码 $train_code）→ 跳过判定。"
    echo "   若出现 'non-finite state'：从 checkpoints/round5/mortal_step_*.pth 取最新干净快照"
    echo "   cp 成 round5/mortal.pth 后重跑本脚本即可。"
    exit "$train_code"
fi

# ===== 判定：R5(870k, top200_v3) vs B 组(756k, top400_v2)——同奖励口径，只差数据选择 =====
NEW=checkpoints/round5/mortal.pth
OLD=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth
OUT=logs/ladder_r5
SETS=500
GPUS=(2 0 1 3 4)
mkdir -p $OUT

job () {   # $1=gpu $2=tag $3=challenger $4=champion $5=seed_start
    local gpu=$1 tag=$2 chal=$3 cham=$4 seed=$5
    (
        export MORTAL_CFG=$ROOT/mortal/config.round5.toml
        export CUDA_VISIBLE_DEVICES=$gpu
        local t0=$(date +%s)
        $PY -u ladder.py "$chal" "$cham" "$seed" "$SETS" "$OUT/$2" \
            --no-amp --agari-guard --tag "r5-$2" > "$OUT/$2.log" 2>&1
        echo "[$(date '+%F %T')] $2 gpu=$gpu 用时 $(( ($(date +%s)-t0)/60 )) 分  $(grep -a 'avg_pt' "$OUT/$2.log" | tail -1)"
    ) &
}

echo "==== 3) 判定：round5(top200_v3) vs B组(top400_v2)，双向各 5×500 sets，新种子块 120000+/130000+ $(date) ===="
echo "---- 波 1：A 方向（R5 当挑战者）----"
i=0
for seed in 120000 120500 121000 121500 122000; do
    job ${GPUS[$i]} "A_${seed}" "$NEW" "$OLD" "$seed"; i=$((i+1))
done
wait
echo "---- 波 1 完成 $(date '+%H:%M:%S') ----"

echo "---- 波 2：B 方向（B 组当挑战者，独立种子块）----"
i=0
for seed in 130000 130500 131000 131500 132000; do
    job ${GPUS[$i]} "B_${seed}" "$OLD" "$NEW" "$seed"; i=$((i+1))
done
wait
echo "---- 波 2 完成 $(date '+%H:%M:%S') ----"

echo "==== 4) 汇总判读 $(date) ===="
$PY ../tools/training/agg_ladder.py --dir $OUT --new "$NEW" --old "$OLD" \
    --logs-root $OUT --challenger-name challenger \
    --bootstrap 5000 --bseed 20261013 --workers 28
echo "ALL DONE $(date)"
