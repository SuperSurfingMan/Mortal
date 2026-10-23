#!/usr/bin/env bash
# S6c 全链收尾（无人值守）：等正式轮跑完 → 校验退出码 → 双向 ladder → 汇总摘要
#
# 前置：formal 阶段已在跑（tmux `s6c_formal`；`run_s6c.sh formal` 结束会把
#       `FORMAL=<code>` 写进 logs/s6c/formal_stage.log）。
# 用法：
#   tmux new -d -s s6c_chain "bash /mnt/data8/mortal/Mortal/mortal/run_s6c_chain.sh > logs/s6c/chain.log 2>&1"
#
# ladder 口径（README §13.16）：S6c(956k) vs B/756k，λ=0 隔离模型变量，
# 双向 5 块 × 500 sets，新种子块 280000 / 300000，block bootstrap CI 下界 >0 才算赢。
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
cd "$ROOT/mortal" || exit 1
LOG=logs/s6c
say () { echo "[$(date '+%F %T')] $*"; }

say '等待 formal 阶段结束（每 60 s 检查一次）...'
while ! grep -qa 'FORMAL=' "$LOG/formal_stage.log" 2>/dev/null; do
    sleep 60
done
code=$(grep -ao 'FORMAL=[0-9]*' "$LOG/formal_stage.log" | tail -1 | cut -d= -f2)
say "formal 退出码 = $code"
if [ "$code" != '0' ]; then
    say 'formal 未正常结束 → 不跑 ladder（排查：tail -80 logs/s6c/formal_train.log）'
    exit 1
fi

say '==== 双向 ladder（λ=0，280000/300000，各 5×500 sets）===='
STAGE=ladder bash run_s6c.sh > "$LOG/ladder_stage.log" 2>&1
say "ladder 退出码 = $?"
grep -a 'avg_pt' "$LOG/ladder_stage.log" | tail -4

say '==== 汇总摘要 ===='
{
    echo "# S6c 结果摘要（自动生成 $(date '+%F %T')）"
    echo
    echo "## holdout（同 digest 9c5362bf99d60661；B/756k → 20k 探针 → 956k 正式轮）"
    grep -a 'steps=' "$LOG/holdout_base.log" "$LOG/holdout_probe.log" "$LOG/holdout_formal.log" 2>/dev/null
    echo
    echo "## 首层新通道使用度（ΔL2 分位 ≥ 原始 p10 才算被学习）"
    grep -aE 'ΔL2|OK\]|WARN\]|绝对 L2' "$LOG/formal_usage.log" 2>/dev/null
    echo
    echo "## 训练行为画像（每次 test_play 的 behavior 行）"
    grep -a 'behavior:' "$LOG/formal_train.log" 2>/dev/null
    echo
    echo "## ladder 汇总（δ = (A − B)/2；CI 下界 >0 才算赢）"
    tail -45 "$LOG/ladder_summary.log" 2>/dev/null
} > "$LOG/SUMMARY.md"
say "摘要已写 $LOG/SUMMARY.md"
say 'ALL DONE'
