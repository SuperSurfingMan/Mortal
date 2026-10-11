# 验证 best_perf 自动重置逻辑（四路）：
#   A) 旧 state 无 best_perf_baseline → 应重置，并在存盘时写入卫冕者标识
#   B) 同一卫冕者再启动          → 不应重置
#   C) 换成另一个卫冕者再启动      → 应重置（reason: 卫冕者已更换）
#   D) 手动令牌 reset_best_perf     → 是一次性：D1 生效并存盘（写入令牌），D2 同值不再重置
set -u
ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python

export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0   # 用空闲 P100，避开正在训练 RTX 8000
cd $ROOT/mortal

T=checkpoints/_bestperf_test
rm -rf $T runs/_bestperf_test
mkdir -p $T
cp checkpoints/round3/mortal.pth $T/mortal.pth     # steps=650000，且无 best_perf_baseline 字段

mk_cfg () {   # $1=输出路径  $2=champion 文件
    sed -e "s|^state_file = 'checkpoints/round3/mortal.pth'.*|state_file = '$T/mortal.pth'|" \
        -e "s|^best_state_file = 'checkpoints/round3/best.pth'|best_state_file = '$T/best.pth'|" \
        -e "s|^tensorboard_dir = 'runs/main_r3'|tensorboard_dir = 'runs/_bestperf_test'|" \
        -e "s|^max_train_steps = 650000.*|max_train_steps = 651000        # 只再训 1000 步（存一次盘）就自停，用于验证|" \
        -e "s|^num_workers = 8|num_workers = 2|" \
        -e "s|^file_batch_size = 16.*|file_batch_size = 8|" \
        -e "s|^state_file = 'checkpoints/mortal_step_000200000.pth'.*|state_file = '$2'|" \
        config.round3.toml > "$1"
}
mk_cfg config._bpA.toml 'checkpoints/mortal_step_000200000.pth'
mk_cfg config._bpB.toml 'checkpoints/mortal_step_000200000.pth'
mk_cfg config._bpC.toml 'checkpoints/round2/mortal.pth'
grep -n "reset_best_perf\|^state_file = 'checkpoints/round3/best" config._bpA.toml | head -2

echo '########## A) 旧 state（无标识）→ 期望 resetting best_perf + 训 1000 步后存盘 ##########'
MORTAL_CFG=$PWD/config._bpA.toml $PY -u train.py > logs/bp_A.log 2>&1
tr '\r' '\n' < logs/bp_A.log | grep -aE "loaded:|resetting best_perf|total steps: 651,000|budget" | head -5
echo '--- 存盘后的 state 关键字段 ---'
$PY -c "
import torch
s=torch.load('$T/mortal.pth', map_location='cpu', weights_only=True)
print('steps =', s['steps'])
print('best_perf =', s['best_perf'])
print('best_perf_baseline =', repr(s.get('best_perf_baseline')))
assert s.get('best_perf_baseline') == 'checkpoints/mortal_step_000200000.pth', '存盘里应写入卫冕者标识'
print('[OK] 存盘已带 best_perf_baseline')
"

echo '########## B) 同卫冕者重启 → 期望【无】resetting 行 ##########'
MORTAL_CFG=$PWD/config._bpB.toml $PY -u train.py > logs/bp_B.log 2>&1
echo "  B 里 resetting 出现次数（应为 0）: $(grep -ac 'resetting best_perf' logs/bp_B.log)"
tr '\r' '\n' < logs/bp_B.log | grep -aE "loaded:|budget already reached|resetting" | head -4

echo '########## C) 换卫冕者重启 → 期望 resetting（reason: 卫冕者已更换）##########'
MORTAL_CFG=$PWD/config._bpC.toml $PY -u train.py > logs/bp_C.log 2>&1
tr '\r' '\n' < logs/bp_C.log | grep -aE "resetting best_perf|budget already reached" | head -3

echo '########## D) 手动令牌（一次性）：D1 应重置并存盘写入令牌，D2 同值不应再重置 ##########'
sed -e 's|^save_every = 1000|save_every = 1000\nreset_best_perf = true|' \
    -e 's|^max_train_steps = 651000.*|max_train_steps = 652000        # D1 再训 1000 步以触发一次存盘（把令牌写进 state）|' \
    config._bpB.toml > config._bpD.toml
MORTAL_CFG=$PWD/config._bpD.toml $PY -u train.py > logs/bp_D1.log 2>&1
echo "  D1 resetting 次数（应为 1）: $(grep -ac 'resetting best_perf' logs/bp_D1.log)"
tr '\r' '\n' < logs/bp_D1.log | grep -aE "resetting best_perf|total steps: 652,000|budget" | head -4
MORTAL_CFG=$PWD/config._bpD.toml $PY -u train.py > logs/bp_D2.log 2>&1
echo "  D2 resetting 次数（应为 0，令牌已一次性消费）: $(grep -ac 'resetting best_perf' logs/bp_D2.log)"
$PY -c "
import torch
s = torch.load('$T/mortal.pth', map_location='cpu', weights_only=True)
print('  存盘里的 token =', repr(s.get('best_perf_reset_token')), '| steps =', s['steps'])
"

echo '########## 清理 ##########'
rm -rf $T runs/_bestperf_test config._bpA.toml config._bpB.toml config._bpC.toml config._bpD.toml logs/bp_A.log logs/bp_B.log logs/bp_C.log logs/bp_D1.log logs/bp_D2.log
git -C $ROOT status --short | head -8
echo DONE
