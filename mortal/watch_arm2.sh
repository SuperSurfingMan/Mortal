#!/bin/bash
# 判定臂看门狗：每 10 分钟写一行状态到 logs/arm_tierA2_monitor.log
cd "$(dirname "$0")"
OUT=../logs/arm_tierA2
MON=logs/arm_tierA2_monitor.log
while true; do
  ts=$(date -u +%H:%M)
  n=$(ls $OUT/*.log 2>/dev/null | wc -l)
  gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "g%s:%s/%s ", $1, $2, $3}' | sed 's/ MiB//g')
  los=$(grep -a "avg_pt" logs/arm_tierA2.log 2>/dev/null | tail -3 | tr -d '\n' | cut -c1-160)
  procs=$(pgrep -cf "ladder.py" || echo 0)
  echo "[$ts] blocks=$n procs=$procs | $gpu | $los" >> $MON
  if [ "$procs" -lt 5 ] && [ "$n" -lt 152 ]; then
    echo "[$ts] ⚠️ worker 存活 $procs < 5（异常）" >> $MON
  fi
  sleep 600
done
