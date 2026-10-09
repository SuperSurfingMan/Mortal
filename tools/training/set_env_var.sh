echo "退出码=$?"

echo
echo '##### B) cd mortal + 完整环境变量（预期全 OK）#####'
cd "$ROOT/mortal"
export MORTAL_CFG=$PWD/config.round2.toml
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=8
export CHECK_GPU_NAME='RTX 8000'
$PY env_check.py --cfg config.round2.toml
echo "退出码=$?"

echo
echo '##### C) 不设 MORTAL_CFG（预期抓出 MORTAL_CFG 缺失）#####'
unset MORTAL_CFG
env -u MORTAL_CFG $PY env_check.py | tail -8
echo "退出码=${PIPESTATUS[0]}"

echo
echo '##### D) 故意用算力序选卡（不设 CUDA_DEVICE_ORDER，CUDA_VISIBLE_DEVICES=2 实际是 P100）#####'
export MORTAL_CFG=$PWD/config.round2.toml
env -u CUDA_DEVICE_ORDER $PY -c 'import torch; print("cuda:0 =", torch.cuda.get_device_name(0))'
env -u CUDA_DEVICE_ORDER CHECK_GPU_NAME='RTX 8000' $PY env_check.py | grep -A1 '选卡'
echo 'ALL DONE'
