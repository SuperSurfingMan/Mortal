#!/usr/bin/env bash
# anchor3：绝对锚点（Mortal×1 vs akochan×3）
#   phase 1：S6c + 守卫 λ=10，1600 局（8 进程 × 200 局）
#   phase 2：S6c 无守卫，800 局（8 进程 × 100 局）——同协议对照
#
# 用法（tmux 承载）：
#   tmux new -d -s anchor3 "bash /mnt/data8/mortal/Mortal/mortal/run_anchor3.sh"
#
# 种子块：1040000（guard）/ 1060000（noguard），符合 MEMORY §2.4。
set -uo pipefail

cd /mnt/data8/mortal/Mortal/mortal || exit 1
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
GPUS=${GPUS:-0,1,3,4}

say () { echo "[$(date '+%F %T')] $*"; }

wait_tag () {  # $1 = tag
    while tmux ls 2>/dev/null | grep -q "^anchor_$1_p"; do sleep 60; done
}

say '==== phase 1: S6c + guard λ=10, 1600 局 ===='
BASE_CFG=config.anchor3_guard.toml bash run_anchor.sh \
    --procs 8 --games-per-proc 200 --seed-key 1040000 --gpus "$GPUS" --tag anchor3
wait_tag anchor3
"$PY" agg_anchor.py --dir logs/anchor3 | tee logs/anchor3/summary.txt
say '==== phase 1 完成 ===='

say '==== phase 2: S6c 无守卫, 800 局（对照）===='
BASE_CFG=config.anchor3_noguard.toml bash run_anchor.sh \
    --procs 8 --games-per-proc 100 --seed-key 1060000 --gpus "$GPUS" --tag anchor3_noguard
wait_tag anchor3_noguard
"$PY" agg_anchor.py --dir logs/anchor3_noguard | tee logs/anchor3_noguard/summary.txt
say '==== phase 2 完成；anchor3 全部结束 ===='
