# 实测 ① CUDA_VISIBLE_DEVICES 每个取值实际落到哪张卡 ② LD_LIBRARY_PATH 到底必需吗
# 用法: tmux new -d -s env_map 'bash /mnt/data8/mortal/Mortal/tools/training/probe_gpu_and_lib.sh 2>&1 | tee /mnt/data8/mortal/Mortal/mortal/logs/env_map.log'
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
CONDA_LIB=/mnt/data8/miniconda3/envs/mortal/lib
cd /mnt/data8/mortal/Mortal/mortal

echo '########## A. CUDA_VISIBLE_DEVICES=0..4 各自实际是哪张卡 ##########'
for i in 0 1 2 3 4; do
    name=$(CUDA_VISIBLE_DEVICES=$i LD_LIBRARY_PATH=$CONDA_LIB "$PY" -c \
        'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)
    echo "  CUDA_VISIBLE_DEVICES=$i  →  $name"
done

echo '########## B. 同一实验但显式 CUDA_DEVICE_ORDER=PCI_BUS_ID ##########'
for i in 0 1 2 3 4; do
    name=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$i LD_LIBRARY_PATH=$CONDA_LIB "$PY" -c \
        'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)
    echo "  PCI_BUS_ID + CUDA_VISIBLE_DEVICES=$i  →  $name"
done

echo '########## C. 不给 LD_LIBRARY_PATH：GPU 前向 + AMP 缩放 ##########'
env -i PATH=/usr/bin:/bin HOME=/root CUDA_VISIBLE_DEVICES=0 "$PY" -c '
import torch
x = torch.randn(512, 1012, 34, device="cuda:0")
w = torch.randn(192, 1012, 3, device="cuda:0")
y = torch.nn.functional.conv1d(x, w, padding=1)
torch.cuda.synchronize()
with torch.autocast("cuda", enabled=True):
    z = torch.nn.functional.relu(y)
torch.cuda.synchronize()
print("无 LD_LIBRARY_PATH 时 GPU 前向 OK:", tuple(z.shape), torch.cuda.get_device_name(0))
' 2>&1 | tail -4

echo '########## D. 不给 LD_LIBRARY_PATH：libriichi 真实解码+特征编码（含 rayon） ##########'
env -i PATH=/usr/bin:/bin HOME=/root MORTAL_CFG=$PWD/config.round2.toml \
    RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -c '
import glob
from libriichi.dataset import GameplayLoader
fl = sorted(glob.glob("/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202503/*.json.gz"))[:4]
ld = GameplayLoader(version=4, oracle=False, player_names=[], excludes=None, augmented=False)
data = ld.load_gz_log_files(fl)
n = sum(sum(len(g.take_obs()) for g in file) for file in data)
print("无 LD_LIBRARY_PATH 时 libriichi 解码 OK, 样本数 =", n)
' 2>&1 | tail -4

echo '########## E. 不给 LD_LIBRARY_PATH：Rust CLI validate_logs / stat ##########'
cd /mnt/data8/mortal/Mortal
ls -l target/release/validate_logs target/release/stat 2>&1 | head -3
env -i PATH=/usr/bin:/bin HOME=/root ./target/release/validate_logs dataset/mjai/houou4/202610 2>&1 | tail -3
echo '--- 同上，给了 LD_LIBRARY_PATH ---'
env -i PATH=/usr/bin:/bin HOME=/root LD_LIBRARY_PATH=$CONDA_LIB ./target/release/validate_logs dataset/mjai/houou4/202610 2>&1 | tail -3

echo '########## F. libriichi.so 与二进制的动态库依赖中非系统部分 ##########'
ldd mortal/libriichi.so 2>/dev/null | grep -v -E 'linux-vdso|ld-linux|libc\.so|libm\.so|libpthread|libdl|librt' | head -12
echo '--- validate_logs ---'
ldd target/release/validate_logs 2>/dev/null | grep -v -E 'linux-vdso|ld-linux|libc\.so|libm\.so|libpthread|libdl|librt' | head -12
echo 'ALL DONE'
