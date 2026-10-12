#!/usr/bin/env bash
# file_index manifest 机制回归测试（2026-10-12 加，防"静默复用过期文件表"）
#
#   U)  纯逻辑单测（tool 层，不训练）
#   E1) 无索引 → 构建并写入 manifest（含 globs/名单 sha256/names 覆盖率）
#   E2) 同配置再跑 → manifest 校验通过、复用，且索引文件字节不变（不重建）
#   E3) 只改 data_version → 判定不一致 → 重建，manifest 里换成新值
#   E4) 索引退化成旧格式（仅 file_list）→ 判"旧格式" → 重建
#   E5) 索引文件损坏 → 判"不可读" → 重建（不崩训练）
#
# 设计要点：离线模式一个 epoch 就退出，且 steps >= max_train_steps 时会在**索引逻辑之前**
# 提前 return（train.py 的 "training budget already reached"）→ 每轮把预算 +8 步，
# 保证每轮都真正走到索引代码；训练用 CPU + 极小 batch（8 步/轮，约 1 分钟）。
set -euo pipefail
ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
T=checkpoints/_idx_test
IDX=indexes/_idx_test.pth
CFG=config._idx_test.toml

export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=8
cd $ROOT/mortal
mkdir -p $T indexes logs
rm -rf $T/* runs/_idx_test indexes/_idx_test.pth indexes/_idx_test_grp.pth logs/idx_*.log

mk_cfg () {   # $1=输出路径  $2=max_train_steps  $3=data_version
cat > "$1" <<EOF
# 自动生成：test_file_index_manifest.sh（勿手工编辑）
[control]
version = 4
online = false
state_file = '$T/mortal.pth'
best_state_file = '$T/best.pth'
tensorboard_dir = 'runs/_idx_test'
device = 'cpu'
enable_cudnn_benchmark = false
enable_amp = false
enable_compile = false
batch_size = 8
opt_step_every = 1
save_every = 8
test_every = 50000
submit_every = 8
max_train_steps = $2
snapshot_every = 50000

[train_play.default]
games = 8
log_dir = 'runs/_idx_test/train_play'
boltzmann_epsilon = 0.005
boltzmann_temp = 0.05
top_p = 1.0
repeats = 1

[test_play]
games = 8
log_dir = 'runs/_idx_test/test_play'
enable_amp = false
enable_rule_based_agari_guard = true

[dataset]
globs = ['$ROOT/dataset/mjai/houou4_names/202610/**/*.json.gz']
player_names_files = ['../dataset/index/top_players_400_v2.txt']
file_index = '$IDX'
data_version = '$3'
file_batch_size = 2
reserve_ratio = 0.0
num_workers = 2
num_epochs = 1
enable_augmentation = false
augmented_first = false

[env]
gamma = 1
pts = [6.0, 4.0, 2.0, 0.0]

[resnet]
conv_channels = 192
num_blocks = 40

[cql]
min_q_weight = 5

[aux]
next_rank_weight = 0.2

[freeze_bn]
mortal = false

[optim]
eps = 1e-8
betas = [0.9, 0.999]
weight_decay = 0.1
max_grad_norm = 5.0

[optim.scheduler]
peak = 1e-4
final = 1e-4
warm_up_steps = 2
max_steps = 8

[baseline.train]
device = 'cpu'
enable_compile = false
state_file = 'checkpoints/mortal_step_000200000.pth'

[baseline.test]
device = 'cpu'
enable_compile = false
state_file = 'checkpoints/mortal_step_000200000.pth'

[online]
history_window = 50
enable_compile = false

[online.remote]
host = '127.0.0.1'
port = 5000

[online.server]
buffer_dir = '/tmp/buffer'
drain_dir = '/tmp/drain'
sample_reuse_rate = 0
sample_reuse_threshold = 0
capacity = 1600
force_sequential = false

[1v3]
seed_key = -1
games_per_iter = 8
iters = 1
log_dir = 'runs/_idx_test/1v3'
[1v3.challenger]
device = 'cpu'
name = 'mortal'
state_file = 'checkpoints/mortal_step_000200000.pth'
stochastic_latent = false
enable_compile = false
enable_amp = false
enable_rule_based_agari_guard = true
[1v3.champion]
device = 'cpu'
name = 'baseline'
state_file = 'checkpoints/mortal_step_000200000.pth'
stochastic_latent = false
enable_compile = false
enable_amp = false
enable_rule_based_agari_guard = true
[1v3.akochan]
enabled = false
dir = '/tmp/akochan'
tactics = '/tmp/tactics.json'

[grp]
state_file = 'checkpoints/grp_h128.pth'
[grp.network]
hidden_size = 128
num_layers = 3
[grp.control]
device = 'cpu'
enable_cudnn_benchmark = false
tensorboard_dir = 'runs/_idx_test/grp'
batch_size = 8
save_every = 2000
val_steps = 10
max_train_steps = 0
[grp.dataset]
train_globs = ['$ROOT/dataset/mjai/houou4_names/202610/**/*.json.gz']
val_globs = ['$ROOT/dataset/mjai/houou4_names/202610/**/*.json.gz']
file_index = 'indexes/_idx_test_grp.pth'
file_batch_size = 2
[grp.optim]
lr = 1e-4
EOF
}

run_train () {   # $1=cfg $2=log名
    MORTAL_CFG=$PWD/$1 $PY -u train.py > "logs/$2" 2>&1
    local code=$?
    tr '\r' '\n' < "logs/$2" | grep -aE "file index|building file index|file list size|budget reached" | sed 's/^/    /'
    echo "    退出码=$code"
    return $code
}

echo '########## U) 纯逻辑单测 ##########'
$PY $ROOT/tools/training/test_file_index_manifest.py

echo
echo '########## E1) 无索引 → 构建 + 写 manifest ##########'
mk_cfg $CFG 8 idx-test-v1
run_train $CFG idx_E1.log
$PY - <<'PY'
import torch
s = torch.load('indexes/_idx_test.pth', map_location='cpu', weights_only=True)
m = s['manifest']
print('    manifest 键:', sorted(m))
print('    index_version =', m['index_version'], '| data_version =', repr(m['data_version']))
print('    n_files =', m['n_files'], '| names_coverage =', m['names_coverage'])
print('    player_names_count =', m['player_names_count'], '| built_at =', m['built_at'])
assert m['index_version'] == 2 and m['data_version'] == 'idx-test-v1'
assert m['names_coverage']['without_names'] == 0, '202610 应已全部带 names'
assert m['player_names_count'] == 400
print('    [OK] E1 manifest 内容正确')
PY

echo
echo '########## E2) 同配置再跑 → 应复用（索引字节不变）##########'
sha_before=$(sha256sum $IDX | cut -d' ' -f1)
mk_cfg $CFG 16 idx-test-v1
run_train $CFG idx_E2.log
sha_after=$(sha256sum $IDX | cut -d' ' -f1)
echo "    索引 sha256: ${sha_before:0:16} → ${sha_after:0:16}"
if [ "$sha_before" = "$sha_after" ]; then
    echo '    [OK] 未重建（字节不变）'
else
    echo '    !! 索引被无谓重写'
    exit 1
fi
n=$(grep -ac '校验通过，复用' logs/idx_E2.log)
echo "    '校验通过，复用' 出现 $n 次（应为 1）"
[ "$n" -eq 1 ]

echo
echo '########## E3) 只改 data_version → 应重建 ##########'
mk_cfg $CFG 24 idx-test-v2
run_train $CFG idx_E3.log
$PY -c "
import torch
m = torch.load('indexes/_idx_test.pth', map_location='cpu', weights_only=True)['manifest']
print('    重建后 data_version =', repr(m['data_version']))
assert m['data_version'] == 'idx-test-v2', '应写入新的 data_version'
print('    [OK] E3 已按新 data_version 重建')
"

echo
echo '########## E4) 索引退化成旧格式 → 应判旧格式并重建 ##########'
$PY -c "
import torch
s = torch.load('indexes/_idx_test.pth', map_location='cpu', weights_only=True)
torch.save({'file_list': s['file_list']}, 'indexes/_idx_test.pth')   # 旧格式：只有 file_list
print('    已把索引退化为旧格式（file_list 保留，去掉 manifest）')
"
mk_cfg $CFG 32 idx-test-v2
run_train $CFG idx_E4.log
$PY -c "
import torch
s = torch.load('indexes/_idx_test.pth', map_location='cpu', weights_only=True)
assert 'manifest' in s, '旧格式应被重建为带 manifest 的新索引'
print('    [OK] E4 旧格式索引已被重建（manifest 回来了，n_files=%d）' % s['manifest']['n_files'])
"

echo
echo '########## E5) 索引损坏 → 应判不可读并重建（不崩）##########'
printf 'this is not a torch file\x00\x01\x02' > $IDX
mk_cfg $CFG 40 idx-test-v2
run_train $CFG idx_E5.log
$PY -c "
import torch
s = torch.load('indexes/_idx_test.pth', map_location='cpu', weights_only=True)
assert 'manifest' in s
print('    [OK] E5 损坏索引已被重建（n_files=%d）' % s['manifest']['n_files'])
"

echo
echo '########## E6) 续训时改奖励口径 → 应打印 env 差异告警（2026-10-12 加的守卫）##########'
mk_cfg $CFG 48 idx-test-v2
sed -i "s|^pts = \[6.0, 4.0, 2.0, 0.0\]|pts = [2.0, 1.0, 0.0, -3.0]|" $CFG
grep -n '^pts' $CFG | sed 's/^/    /'
run_train $CFG idx_E6.log
n=$(grep -ac 'env 奖励口径与 checkpoint 记录不同' logs/idx_E6.log || true)
echo "    'env 奖励口径…不同' 告警出现 $n 次（应为 1）"
if [ "$n" -ge 1 ]; then
    echo '    [OK] E6 目标变化被显式告警'
else
    echo '    !! 未告警'
    exit 1
fi

echo
echo '########## 清理 ##########'
rm -rf $T runs/_idx_test indexes/_idx_test.pth indexes/_idx_test_grp.pth $CFG logs/idx_E*.log
git -C $ROOT status --short | head -20
echo DONE
