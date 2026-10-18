# β 扫描探针收尾（2026-10-18）：d 臂迁移到 GPU0 后，链已在 12:57 收尾（仅含 a/b/c）。
# 本脚本等待 d 训练到 786k → 跑 holdout → 提取 d 指标 → 重建含四臂的完整汇总。
# 同时修正 extract_metrics 中的一个提取 bug：awr_w 的 p99 之前会先匹配到同行的 q_target p99。
# 日志仍全部在 mortal/logs/iql_beta/ 下，不产生散落文件。
# 用法: nohup bash run_iql_beta_d_finish.sh > logs/iql_beta/d_finish.out 2>&1 &
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
LOGDIR=$MORTAL/logs/iql_beta
CHAIN=$LOGDIR/chain.log
PT_THR=2.0
T1_TOL=0.005

export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

say "==== d 收尾开始：等待 d 训练到 786k ===="
while ! grep -aq 'training budget reached' "$LOGDIR/d/train.log" 2>/dev/null; do
    sleep 60
    if ! pgrep -f '^/mnt/data8/miniconda3/envs/mortal/bin/python -u train\.py$' >/dev/null 2>&1; then
        # 进程已退出但无 budget 标记 → 可能异常，再等一轮确认后退出
        sleep 30
        grep -aq 'training budget reached' "$LOGDIR/d/train.log" && break
        say "!! d 训练进程已退出但未见 budget 标记，终止收尾（检查 logs/iql_beta/d/train.log）"
        exit 1
    fi
done
say "d 训练完成（budget reached）"

# holdout（GPU0，d 的训练卡）
CUDA_VISIBLE_DEVICES=0 MORTAL_CFG=config.iql_beta_d.toml $PY -u holdout_metrics.py \
    --month 202609 --n 24 --seed 20261009 --workers 1 checkpoints/iql_beta_d/mortal.pth \
    > "$LOGDIR/d/holdout_202609.log" 2>&1
say "d holdout: $(grep -a 'steps=' "$LOGDIR/d/holdout_202609.log" | tail -1)"

# ---- 从 train.log 提取单臂指标（修正 awr_w 提取：只取 'awr_w ess=.. p99=.. max=..' 段）----
extract () {   # $1=arm  $2=beta
    local x=$1 beta=$2 log=$LOGDIR/$1/train.log
    local rk pt t1 lq maxp50 awr ess p99 mx beh
    rk=$(grep -a 'avg rank:' "$log" | tail -1 | sed 's/.*avg rank: //' | awk '{print $1}')
    pt=$(grep -a 'avg pt:' "$log" | tail -1 | sed 's/.*avg pt: //' | awk '{print $1}')
    maxp50=$(grep -a 'iql scale:' "$log" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 \
        | awk 'BEGIN{m=-1e9}{v=$1+0; if(v>m)m=v}END{printf "%.3f", m}')
    awr=$(grep -a 'iql scale:' "$log" | grep -aoE 'awr_w ess=[0-9.]+ p99=[-+0-9.]+ max=[-+0-9.]+' | tail -1)
    ess=$(printf '%s' "$awr" | grep -aoE 'ess=[0-9.]+' | cut -d= -f2)
    p99=$(printf '%s' "$awr" | grep -aoE 'p99=[-+0-9.]+' | cut -d= -f2)
    mx=$(printf '%s' "$awr" | grep -aoE 'max=[-+0-9.]+' | cut -d= -f2)
    beh=$(grep -a 'behavior:' "$log" | tail -1 | sed 's/.*behavior:/behavior:/')
    t1=$(grep -a 'steps=' "$LOGDIR/$1/holdout_202609.log" 2>/dev/null | tail -1 | grep -aoE 'top1=[0-9.]+' | head -1 | cut -d= -f2)
    lq=$(grep -a 'steps=' "$LOGDIR/$1/holdout_202609.log" 2>/dev/null | tail -1 | grep -aoE 'logsumexp_q=[-+]?[0-9.]+' | head -1 | cut -d= -f2)
    # 健康门禁
    local fail=0
    awk -v m="${maxp50:--1e9}" 'BEGIN{exit !(m<=8)}' || fail=1
    [ -n "${rk:-}" ] && awk -v x="$rk" 'BEGIN{exit !(x<=2.60)}' || fail=1
    [ -n "${t1:-}" ] && awk -v x="$t1" 'BEGIN{exit !(x>=0.78)}' || fail=1
    [ -n "${lq:-}" ] && awk -v x="$lq" 'BEGIN{if(x<0)x=-x; exit !(x<=10)}' || fail=1
    cat > "$LOGDIR/$x/metrics.env" <<EOF
ARM=$x
BETA=$beta
MAX_P50=${maxp50:--1e9}
AWRE=${ess:-?}
AWRP=${p99:-?}
AWRM=${mx:-?}
AVG_RANK=${rk:-?}
AVG_PT=${pt:-?}
TOP1=${t1:-?}
LSE=${lq:-?}
BEHAVIOR="$beh"
HEALTH_FAIL=$fail
EOF
    say "[$x] metrics: beta=$beta max_p50=${maxp50:-?} avg_rank=${rk:-?} avg_pt=${pt:-?} top1=${t1:-?} lse=${lq:-?} awr_w=${ess:-?}/${p99:-?}/${mx:-?} health=$([ "$fail" -eq 0 ] && echo PASS || echo FAIL)"
}

for pair in "a 3.0" "b 1.0" "c 0.5" "d 10.0"; do
    set -- $pair
    extract "$1" "$2"
done

# ---- 重建完整汇总 ----
out=$LOGDIR/summary.log
{
echo "==== β 扫描探针完整汇总（$(date '+%F %T')，d 已迁移 GPU0 续训；awr_w p99 提取已修正）===="
printf '%-4s %-6s %-9s %-9s %-9s %-8s %-8s %-22s %s\n' \
    arm beta max_p50 avg_rank avg_pt top1 lse 'awr_w(ess/p99/max)' health
for x in a b c d; do
    [ -f "$LOGDIR/$x/metrics.env" ] || continue
    ( unset ARM BETA MAX_P50 AWRE AWRP AWRM AVG_RANK AVG_PT TOP1 LSE BEHAVIOR HEALTH_FAIL
      . "$LOGDIR/$x/metrics.env"
      printf '%-4s %-6s %-9s %-9s %-9s %-8s %-8s %-22s %s\n' \
          "$ARM" "$BETA" "$MAX_P50" "$AVG_RANK" "$AVG_PT" "$TOP1" "$LSE" \
          "${AWRE}/${AWRP}/${AWRM}" "$([ "$HEALTH_FAIL" -eq 0 ] && echo PASS || echo FAIL)" )
done
echo
echo "---- 行为分解（末次 test_play，500 sets）----"
for x in a b c d; do
    [ -f "$LOGDIR/$x/metrics.env" ] || continue
    ( . "$LOGDIR/$x/metrics.env"; echo "  arm=$x beta=$BETA  $BEHAVIOR" )
done
echo
echo "---- 判读（对照=臂a β=3.0；阈值 avg_pt±$PT_THR / top1±$T1_TOL；探针级低功效）----"
( . "$LOGDIR/a/metrics.env"; echo "  对照: avg_pt=$AVG_PT top1=$TOP1" )
for x in a b c d; do
    [ -f "$LOGDIR/$x/metrics.env" ] || continue
    ( . "$LOGDIR/$x/metrics.env"
      cpt=$(grep -a '^AVG_PT=' "$LOGDIR/a/metrics.env" | cut -d= -f2)
      ct1=$(grep -a '^TOP1=' "$LOGDIR/a/metrics.env" | cut -d= -f2)
      if [ "$HEALTH_FAIL" -ne 0 ]; then
          echo "  arm=$x beta=$BETA: FAIL(health)"
      elif awk -v a="$AVG_PT" -v c="$cpt" -v t="$PT_THR" 'BEGIN{exit !(a>=c+t)}' \
        && awk -v a="$TOP1" -v c="$ct1" -v d="$T1_TOL" 'BEGIN{exit !(a>=c-d)}'; then
          echo "  arm=$x beta=$BETA: PROMISING → 建议正式轮（种子块 240000/250000）"
      elif awk -v a="$AVG_PT" -v c="$cpt" -v t="$PT_THR" 'BEGIN{exit !(a<=c-t)}' \
        || awk -v a="$TOP1" -v c="$ct1" -v d="$T1_TOL" 'BEGIN{exit !(a<c-d)}'; then
          echo "  arm=$x beta=$BETA: DEGRADED"
      else
          echo "  arm=$x beta=$BETA: NO-SEP（探针无分离）"
      fi )
done
echo
echo "---- 旋钮生效检查：ess 应随 β 下降、p99 应随 β 上升（见上表 awr_w 列）----"
} | tee -a "$CHAIN" > "$out"
say "完整汇总已写入 $out"
say "ALL DONE（d 收尾）"
