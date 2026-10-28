#!/usr/bin/env bash
# S6c 守卫复验第二波：再跑 4 个副本（r5~r8），凑满 8 副本。
#
# 协议与剂量曲线完全一致：challenger=champion=`checkpoints/s6c/mortal.pth`，
# `--guard-lambda 10` **只作用于挑战者席**；座位在 4 个 split 间轮转，单副本内即自平衡。
# 判据：8 副本 avg_pt 的均值（理论 SE = 82/√(8000×4×8) ≈ 0.16；经验跨副本 SD 更小）。
#
# 用法（tmux，GPU2）：bash run_guard_s6c_wave2.sh
set -u -o pipefail
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID MORTAL_CFG=config.s6c.toml CUDA_VISIBLE_DEVICES=2
OUT=logs/guard_s6c
mkdir -p "$OUT"

for pair in "520000 540000" "560000 580000"; do
    for seed in $pair; do
        echo "[$(date '+%F %T')] start seed=$seed"
        ( "$PY" -u ladder.py checkpoints/s6c/mortal.pth checkpoints/s6c/mortal.pth \
              "$seed" 8000 "$OUT/l10_$seed" --guard-lambda 10 --tag "guard_s6c_w2_$seed" \
              > "$OUT/l10_$seed.log" 2>&1
          echo "RC=$?" >> "$OUT/l10_$seed.log" ) &
    done
    wait
    echo "[$(date '+%F %T')] wave done: $pair"
done
echo "[$(date '+%F %T')] ALL DONE"
