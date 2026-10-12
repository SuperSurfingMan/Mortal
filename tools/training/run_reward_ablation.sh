#!/usr/bin/env bash
# 奖励口径消融：A 现状 / B 对齐顺位 / C 真实分数差。
# 三组共享 R4 v2 数据池、父模型、seed 和新增预算；仅 reward_mode/pts/scale 不同。
set -euo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"

if [ ! -f reward_ablation/experiment.json ]; then
    echo '缺少实验配置；先运行 prepare_reward_ablation.py'
    exit 1
fi

INDEX=indexes/reward_ablation_r4_v2.pth
if [ ! -f "$INDEX" ]; then
    echo "构建共享 file_index: $INDEX"
    MORTAL_CFG="$PWD/reward_ablation/A_rank_pt/config.toml" "$PY" \
        "$ROOT/tools/training/build_file_index.py"
fi

for tag in A_rank_pt B_rank_pt_aligned C_delta_points; do
    cfg="$PWD/reward_ablation/$tag/config.toml"
    state="$PWD/checkpoints/reward_ablation/$tag/mortal.pth"
    if [ ! -f "$cfg" ] || [ ! -f "$state" ]; then
        echo "缺少 $tag 的配置或父 checkpoint: $cfg / $state"
        exit 1
    fi
done

run_one() {
    local gpu=$1 tag=$2
    local cfg="$PWD/reward_ablation/$tag/config.toml"
    local log="$PWD/reward_ablation/$tag/train.log"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export MORTAL_CFG="$cfg"
        echo "[$(date '+%F %T')] start tag=$tag gpu=$gpu cfg=$cfg" > "$log"
        set +e
        "$PY" -u train.py >> "$log" 2>&1
        code=$?
        set -e
        echo "[$(date '+%F %T')] exit=$code tag=$tag" >> "$log"
        exit "$code"
    )
}

# 先只启动训练；三组日志彼此隔离。建议调用者在新 shell/tmux 中执行本脚本。
echo "启动奖励消融训练：A GPU=0，B GPU=1，C GPU=3；父代 round4/645k，预算 756k。"
run_one 0 A_rank_pt & pids[0]=$!
run_one 1 B_rank_pt_aligned & pids[1]=$!
run_one 3 C_delta_points & pids[2]=$!
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
