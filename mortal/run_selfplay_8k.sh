#!/bin/bash
# 档 A 数据生成：8000 半庄，4 卡并行（GPU 0/2/3/4，避开会过热降频的 GPU1）
# 每卡 500 sets（=2000 半庄），种子块互不重叠：[700000,700500) [700500,701000) [701000,701500) [701500,702000)
set -u
cd "$(dirname "$0")"
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MORTAL_CFG=config.s6c.toml

launch () {
  local gpu=$1 seed=$2
  local out="../logs/selfplay/s6c_${seed}"
  tmux kill-session -t "sp${gpu}" 2>/dev/null
  tmux new -d -s "sp${gpu}" \
    "cd $(pwd) && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu} MORTAL_CFG=config.s6c.toml \
     ${PY} -u selfplay_gen.py --sets 500 --seed ${seed} --out ${out} > logs/spgen_${seed}.log 2>&1; \
     echo RC=\$? >> logs/spgen_${seed}.log"
  echo "gpu=${gpu} seed=${seed} out=${out} log=logs/spgen_${seed}.log"
}

launch 0 700000
launch 2 700500
launch 3 701000
launch 4 701500

sleep 5
tmux ls | grep -E '^sp[0-9]:' || true
