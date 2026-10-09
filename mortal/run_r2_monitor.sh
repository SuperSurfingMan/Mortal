# 第 2 轮自动验收：每个新 snapshot 出现就跑一次未见月离线指标；训练结束后自动跑定胜负的 ladder
# 刻意避开 RTX 8000（训练在用）：评测全部放 P100 idx1，且 nice + 低 worker 数，别抢训练的 CPU
# 用法: tmux new -d -s r2_watch 'bash /mnt/data8/mortal/Mortal/mortal/run_r2_monitor.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
cd "$ROOT/mortal"

export MORTAL_CFG=$ROOT/mortal/config.round2.toml
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib
export CUDA_VISIBLE_DEVICES=1          # 空闲 P100，不碰训练的卡
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4   # 压低，避免拖累训练管道

REF=checkpoints/mortal_step_000200000.pth       # 第 1 轮终态 = 对照组兼卫冕者
MON=logs/r2_monitor.log
DONE=logs/r2_evaldone
mkdir -p "$DONE"

echo "==== 监控启动 $(date) | 对照组=$REF ====" >> "$MON"

# 注意必须用锚定模式：无锚定时连 bash 包装进程（cmdline 里含同样字符串）也会匹上，永远退不出循环
while [ "$(pgrep -c -f '^python -u train\.py$' || true)" -gt 0 ]; do
    for s in checkpoints/round2/mortal_step_*.pth; do
        [ -e "$s" ] || continue
        tag=$(basename "$s" .pth)
        [ -e "$DONE/$tag" ] && continue
        echo "---- [$tag] holdout $(date) ----" >> "$MON"
        nice -n 10 "$PY" holdout_metrics.py --workers 2 --n 12 \
            "$s" "$REF" >> "$MON" 2>&1
        touch "$DONE/$tag"
    done
    sleep 240
done

echo "==== 训练进程已退出 $(date) ====" >> "$MON"
tail -6 logs/main_r2_train.log | tr '\r' '\n' | grep -aE "total steps|avg rank|avg pt|budget|record" >> "$MON"

LAST=$(ls -1 checkpoints/round2/mortal_step_*.pth 2>/dev/null | tail -1)
[ -n "${LAST:-}" ] || { echo '没有 round2 snapshot，跳过 ladder' >> "$MON"; exit 0; }

echo "---- 定胜负：ladder 新 vs 第1轮200k（500 sets = 2000 局，四座位同牌山） $(date) ----" >> "$MON"
nice -n 10 "$PY" ladder.py "$LAST" "$REF" 40000 500 2>&1 \
    | grep -av "%|" >> "$MON"
echo "---- 反向对照：ladder 第1轮200k 当挑战者 vs 新（不同种子块，消除牌山偏置） $(date) ----" >> "$MON"
nice -n 10 "$PY" ladder.py "$REF" "$LAST" 41000 500 2>&1 | grep -av "%|" >> "$MON"

echo "==== 监控结束 $(date) | 判读规则：单侧 avg_pt > 0 且超出 ±3.6 pt 噪声（500 sets 时 SE≈±2）才算本轮有效 ====" >> "$MON"
