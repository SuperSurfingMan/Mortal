# GRP 容量实验：hidden 64→128、layers 2→3，从零训 60k 步，与 grp_v2 同口径比较
# 用法: tmux new -d -s grp_cap 'bash /mnt/data8/mortal/Mortal/mortal/run_grp_capacity.sh'
set -euo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python

export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1            # P100（float64；PCI 序 1/3/4 与 nvidia-smi 一致）

cd "$ROOT/mortal"
mkdir -p checkpoints runs logs

# 从 config.round2.toml 派生：只改 [grp] 的容量、落盘路径、看板目录与预算
sed -e "s|^state_file = 'checkpoints/grp_v2.pth'|state_file = 'checkpoints/grp_h128.pth'|" \
    -e "s|^tensorboard_dir = 'runs/grp_v2'|tensorboard_dir = 'runs/grp_h128'|" \
    -e "s|^hidden_size = 64|hidden_size = 128|" \
    -e "s|^num_layers = 2|num_layers = 3|" \
    -e "s|^max_train_steps = 150000|max_train_steps = 60000|" \
    config.round2.toml > config.grp_h128.toml
grep -n -E "hidden_size|num_layers|^state_file|tensorboard_dir = 'runs/grp_h128'|max_train_steps" config.grp_h128.toml | head -8

echo "开始 GRP 容量实验（从零，60k 步）；$(date)"
MORTAL_CFG=$ROOT/mortal/config.grp_h128.toml $PY -u train_grp.py > logs/grp_h128_train.log 2>&1
echo "结束；$(date)"
grep -a 'total steps' logs/grp_h128_train.log | tail -3

echo '=== 与 grp_v2 / grp 同口径对比 ==='
for ck in "checkpoints/grp.pth 64 2" "checkpoints/grp_v2.pth 64 2" "checkpoints/grp_h128.pth 128 3"; do
    set -- $ck
    $PY ../tools/training/grp_quality_probe.py --state "$1" --hidden "$2" --layers "$3" --n 150
done
