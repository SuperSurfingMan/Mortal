#!/usr/bin/env bash
# 数据质量加权 100k 探针（P0-b，2026-11-02）
#
# 变量：S6c/956k 暖启 + [dataset.strength_weight]（局内最弱一席 avg_pt 分档重复
#       [1,1,2,3]，mean≈1.35；只扩 file_list，不改损失/索引）。
# 产物：checkpoints/strength_probe/（mortal.pth、20k 快照、best.pth）
#       logs/strength_probe_train.log / logs/strength_probe_holdout.log
#
# 用法（tmux 承载）：
#   tmux new -d -s sp100k "cd /mnt/data8/mortal/Mortal/mortal && STAGE=all bash run_strength_probe.sh > logs/strength_probe_run.log 2>&1"
#   STAGE=train|holdout|all（默认 all）
#
# 门禁（holdout，202609，digest 9c5362bf）：
#   top1 ≥ 0.8117（基线 0.8167 − 0.5pp）；dqn_loss ≤ 0.365（基线 0.3613 +1%）
#   两项同时满足 → 上 76k sets 臂（种子 1000000/1020000）；否则关闭该轴。
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
CFG=config.strength_probe.toml
LOGS=logs
STAGE=${STAGE:-all}

cd "$ROOT/mortal" || { echo "缺少 $ROOT/mortal"; exit 1; }
mkdir -p checkpoints/strength_probe "$LOGS"

if [ ! -f checkpoints/strength_probe/mortal.pth ]; then
    echo "复制 S6c/956k → checkpoints/strength_probe/mortal.pth"
    cp checkpoints/s6c/mortal.pth checkpoints/strength_probe/mortal.pth
fi

export MORTAL_ROOT=$ROOT
export MORTAL_CFG=$CFG
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CHECK_GPU_NAME=${CHECK_GPU_NAME:-RTX 8000}

say () { echo "[$(date '+%F %T')] $*"; }
want () { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

train () {
    say "==== 预检 ===="
    "$PY" env_check.py --cfg "$CFG" || { echo '预检未通过，终止'; exit 1; }
    say "==== 100k 训练（S6c/956k 暖启）===="
    local t0=$(date +%s)
    "$PY" -u train.py > "$LOGS/strength_probe_train.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    grep -aE 'strength_weight|file list size|total steps: 1,056,000|budget reached|non-finite|Traceback' \
        "$LOGS/strength_probe_train.log" | tail -8
    if [ $rc -ne 0 ] || ! grep -aq 'training budget reached' "$LOGS/strength_probe_train.log"; then
        echo "训练未正常到达预算（rc=$rc），终止"; exit 1
    fi
    say "训练完成，用时 ${dt}s"
}

holdout () {
    say "==== holdout（202609，与 S6c 同 digest）===="
    MORTAL_CFG="$CFG" "$PY" -u holdout_metrics.py \
        checkpoints/strength_probe/mortal.pth --n 24 --workers 8 2>&1 \
        | tee "$LOGS/strength_probe_holdout.log"
    say "基线（S6c/956k）：top1=0.8167 dqn_loss=0.3613；判据 top1≥0.8117 且 dqn_loss≤0.365"
}

want train && train
want holdout && holdout
say "完成（STAGE=$STAGE）"
