# 启动第 3 轮训练（白名单配方 + 第 2 个 epoch），全程 tmux + 完整环境变量
set -euo pipefail
ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python

# ==== 全体环境变量：每次跑之前重新设一遍（缺一即可能跑偏）====
export MORTAL_ROOT=$ROOT
export MORTAL_CFG=$ROOT/mortal/config.round3.toml               # 必需
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID                             # 必需：序号对齐 nvidia-smi
export CUDA_VISIBLE_DEVICES=2                                   # RTX 8000

cd $ROOT/mortal

# 暖启：round2 终态 → round3 起点（不动任何旧文件）
mkdir -p checkpoints/round3
if [ ! -f checkpoints/round3/mortal.pth ]; then
    cp checkpoints/round2/mortal.pth checkpoints/round3/mortal.pth
    echo "已暖启: checkpoints/round3/mortal.pth ← round2/mortal.pth (steps=424000)"
fi

$PY -c "import torch,libriichi; print('前置检查:', torch.cuda.get_device_name(0), libriichi.__version__, libriichi.__profile__)"
echo "MORTAL_CFG = $MORTAL_CFG"
tmux new -d -s train_r3 "cd $ROOT/mortal && export MORTAL_ROOT=$ROOT MORTAL_CFG=$MORTAL_CFG LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=8 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 && $PY -u train.py > logs/main_r3_train.log 2>&1"
sleep 40
tmux ls
echo '--- 启动日志 ---'
tr '\r' '\n' < logs/main_r3_train.log | grep -a -E "device:|loaded|file list size|total steps|loaded .* players" | head -8
