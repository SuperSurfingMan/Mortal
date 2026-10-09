# 逐个实测环境变量的必要性：最小环境 vs 完整环境，以及缺某个变量时的真实报错
# 用法: tmux new -d -s env_probe 'bash /mnt/data8/mortal/Mortal/tools/training/probe_env_vars.sh 2>&1 | tee /mnt/data8/mortal/Mortal/mortal/logs/env_probe.log'
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
ENVM=/mnt/data8/miniconda3/envs/mortal
CONDA_LIB=$ENVM/lib

echo '########## T1 最小环境（只给 PATH/HOME）：import torch + libriichi ##########'
env -i PATH=/usr/bin:/bin HOME=/root "$PY" -c '
import sys
try:
    import torch
except Exception as e:
    print("import torch FAIL:", type(e).__name__, str(e)[:220]); sys.exit(1)
print("torch", torch.__version__)
try:
    import libriichi
    print("import libriichi OK")
except Exception as e:
    print("import libriichi FAIL:", type(e).__name__, str(e)[:300])
' 2>&1 | tail -8

echo '########## T2 最小环境 + LD_LIBRARY_PATH=conda/lib ##########'
env -i PATH=/usr/bin:/bin HOME=/root LD_LIBRARY_PATH=$CONDA_LIB "$PY" -c '
import torch, libriichi
print("both OK; cuda avail =", torch.cuda.is_available())
' 2>&1 | tail -5

echo '########## T3 最小环境 + LD_LIBRARY_PATH，再看 CUDA 实际能否用 ##########'
env -i PATH=/usr/bin:/bin HOME=/root LD_LIBRARY_PATH=$CONDA_LIB CUDA_VISIBLE_DEVICES=2 "$PY" -c '
import torch, libriichi
x = torch.randn(4, 1012, 34, device="cuda:0")
from model_stub import *
' 2>/dev/null
env -i PATH=/usr/bin:/bin HOME=/root LD_LIBRARY_PATH=$CONDA_LIB CUDA_VISIBLE_DEVICES=2 "$PY" -c '
import torch, libriichi
print("device:", torch.cuda.get_device_name(0), "| arch:", torch.cuda.get_arch_list()[-3:])
' 2>&1 | tail -3

echo '########## T4 不设 MORTAL_CFG（cwd=mortal，存在 config.toml 吗？） ##########'
cd /mnt/data8/mortal/Mortal/mortal
ls config.toml 2>&1 | head -2
env -i PATH=/usr/bin:/bin HOME=/root LD_LIBRARY_PATH=$CONDA_LIB "$PY" -c '
import os
print("MORTAL_CFG =", repr(os.environ.get("MORTAL_CFG")))
try:
    from config import config
    print("load OK:", config["control"]["state_file"])
except Exception as e:
    print("load FAIL:", type(e).__name__, str(e)[:160])
' 2>&1 | tail -3

echo '########## T5 不设 MORTAL_CFG 但 cwd 里有 config.offline.toml.bak 之类 → 确认默认文件名 ##########'
env -i PATH=/usr/bin:/bin HOME=/root "$PY" -c '
import re
src = open("/mnt/data8/mortal/Mortal/mortal/config.py", encoding="utf-8").read()
print([l for l in src.splitlines() if "environ" in l])
' 2>&1 | tail -2

echo '########## T6 RAYON_NUM_THREADS 是否真的改变线程数（看 nlwp） ##########'
cd /mnt/data8/mortal/Mortal/mortal
for ray in 2 8 16; do
    MORTAL_CFG=/mnt/data8/mortal/Mortal/mortal/config.round2.toml RAYON_NUM_THREADS=$ray OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 \
    LD_LIBRARY_PATH=$CONDA_LIB "$PY" bench_pipe.py 2 $ray > /tmp/rb_$ray.txt 2>&1 &
    pid=$!
    sleep 12
    n=$(ps -o nlwp= -p $pid 2>/dev/null | tr -d ' ')
    # 统计该进程树的线程总数（含 worker 子进程）
    tree=$(pgrep -P $pid 2>/dev/null | tr '\n' ' ')
    tot=$n
    for c in $tree; do tot=$((tot + $(ps -o nlwp= -p $c 2>/dev/null | tr -d ' ' || echo 0))); done
    echo "RAYON=$ray: 主进程线程=$n 进程树线程合计≈$tot  $(tail -1 /tmp/rb_$ray.txt 2>/dev/null | head -c 90)"
    wait $pid 2>/dev/null
done

echo '########## T7 torch 侧 OMP/MKL 设置的实际生效值 ##########'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_LIB "$PY" -c '
import torch
print("torch.get_num_threads() =", torch.get_num_threads(), "| interop =", torch.get_num_interop_threads())
import os
print("env 里存在的相关变量:")
for k in ["MORTAL_CFG","LD_LIBRARY_PATH","OMP_NUM_THREADS","MKL_NUM_THREADS","RAYON_NUM_THREADS","CUDA_VISIBLE_DEVICES","TRAIN_PLAY_PROFILE","MORTAL_IS_SUB_PROC","MORTAL_REVIEW_MODE","AKOCHAN_DIR","AKOCHAN_TACTICS"]:
    print("  ", k, "=", os.environ.get(k, "<未设置>"))
' 2>&1 | tail -18

echo '########## T8 exe-wrapper 需要的 env（Rust 侧读 PATH） ##########'
grep -n -B2 -A6 'let path_env' /mnt/data8/mortal/Mortal/exe-wrapper/src/main.rs | head -20
echo 'ALL DONE'
