#!/usr/bin/env bash
# 20k 修时钟探针的常驻监控：每 60s 写一行进度；训练结束后可选自动跑 order_metrics。
# 只读训练日志/ckpt，不修改训练进程；监控日志与训练日志分离。
set -u

REPO=/mnt/data8/mortal/Mortal
TOOLS=$REPO/tools/read_model_clock
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
LOG=${LOG:-$REPO/mortal/logs/read_model/read_model_v2s_a2d_clock_probe20k.log}
MON=${MON:-$REPO/mortal/logs/read_model/read_model_v2s_a2d_clock_probe20k.monitor.log}
CKPT=${CKPT:-$REPO/mortal/read_model_v2s_a2d_clock_probe20k/best.pth}
TARGET=${TARGET:-20000}
INTERVAL=${INTERVAL:-60}
AUTO_EVAL=${AUTO_EVAL:-1}
N_EVAL=${N_EVAL:-40000}
EVAL_LOG=${EVAL_LOG:-$REPO/mortal/logs/read_model/read_model_v2s_a2d_clock_probe20k.order_metrics.log}
EVAL_OUT=${EVAL_OUT:-$REPO/mortal/read_model_v2s_a2d_clock_probe20k/order_metrics.json}
GPU=${GPU:-3}

mkdir -p "$(dirname "$MON")"
echo "[monitor] start $(date -u +%FT%TZ) target=$TARGET interval=${INTERVAL}s auto_eval=$AUTO_EVAL" >> "$MON"

while true; do
    ts=$(date -u +%FT%TZ)
    if pgrep -f '[t]ools/read_model_clock/train.py' >/dev/null 2>&1; then
        last=$(grep -E '^step [0-9]+ ' "$LOG" 2>/dev/null | tail -1)
        step=$(printf '%s' "$last" | sed -E 's/^step ([0-9]+).*/\1/')
        elapsed=$(printf '%s' "$last" | sed -E 's/.*\[([0-9.]+)s\]$/\1/')
        if [ -z "$step" ] || [ -z "$elapsed" ]; then
            echo "$ts RUN waiting_for_first_log" >> "$MON"
        else
            eta=$(awk -v n="$step" -v t="$elapsed" -v T="$TARGET" \
                'BEGIN{ if(n>0 && t>0) printf "%.1f", (T-n)*t/n/60; else printf "?" }')
            gpu=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used \
                  --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
            echo "$ts RUN step=$step/$TARGET elapsed=${elapsed}s eta=${eta}min gpu=${gpu:-?}" >> "$MON"
        fi
        sleep "$INTERVAL"
        continue
    fi

    # 训练进程已退出：区分正常完成与异常退出
    if grep -q '\[done\]' "$LOG" 2>/dev/null; then
        echo "$ts DONE $(grep '\[done\]' "$LOG" | tail -1)" >> "$MON"
        if [ "$AUTO_EVAL" = "1" ]; then
            echo "$ts EVAL start $(date -u +%FT%TZ)" >> "$MON"
            RM_CACHE_DIR=${RM_CACHE_DIR:-/tmp/read_model_diag_cache_clock} \
            RM_OUT_DIR=${RM_OUT_DIR:-$REPO/mortal/read_model_v2s_a2d_clock_probe20k/diag} \
            CUDA_VISIBLE_DEVICES="$GPU" \
            "$PY" "$TOOLS/order_metrics.py" --state "$CKPT" --n "$N_EVAL" \
                --out "$EVAL_OUT" > "$EVAL_LOG" 2>&1
            rc=$?
            echo "$ts EVAL done rc=$rc out=$EVAL_OUT log=$EVAL_LOG" >> "$MON"
            if [ "$rc" = "0" ]; then
                grep -E '^  (RM12|RM13|RM14|RM15) ' "$EVAL_LOG" >> "$MON" 2>/dev/null || true
            fi
        fi
        break
    fi

    echo "$ts EXIT_WITHOUT_DONE (check log: $LOG)" >> "$MON"
    break
done
echo "[monitor] stop $(date -u +%FT%TZ)" >> "$MON"
