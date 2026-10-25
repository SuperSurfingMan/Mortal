#!/usr/bin/env bash
# 数据补齐探针 D2 收尾链：等 100k 正式轮跑完 → 双向 ladder → 写 SUMMARY.md
#
# 前置：formal 已在跑（tmux `d2_formal`；run_d2.sh formal 结束会写 `FORMAL=<code>`）。
# 用法：tmux new -d -s d2_chain "bash /mnt/data8/mortal/Mortal/mortal/run_d2_chain.sh > logs/d2_2026/chain.log 2>&1"
#
# ladder 口径：D2(856k) vs B/756k，λ=0，双向 5×500 sets，新种子块 360000/380000。
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
cd "$ROOT/mortal" || exit 1
LOG=logs/d2_2026
say () { echo "[$(date '+%F %T')] $*"; }

say '等待 formal 阶段结束（每 60 s 检查一次）...'
while ! grep -qa 'FORMAL=' "$LOG/formal_stage.log" 2>/dev/null; do
    sleep 60
done
code=$(grep -ao 'FORMAL=[0-9]*' "$LOG/formal_stage.log" | tail -1 | cut -d= -f2)
say "formal 退出码 = $code"
if [ "$code" != '0' ]; then
    say "formal 未正常结束 → 不跑 ladder（tail -80 logs/d2_2026/formal_train.log）"
    exit 1
fi

say '==== 双向 ladder（λ=0，360000/380000，各 5×500 sets）===='
STAGE=ladder bash run_d2.sh > "$LOG/ladder_stage.log" 2>&1
say "ladder 退出码 = $?"
grep -a 'avg_pt\|强度差\|判读' "$LOG/ladder_stage.log" | tail -5

say '==== 汇总摘要 ===='
{
    echo "# 数据补齐探针 D2 摘要（自动生成 $(date '+%F %T')）"
    echo
    echo "## 数据池"
    "$PY" - <<'PYEOF'
import torch
st = torch.load('indexes/d2_2026_v1.pth', weights_only=True)
m = st['manifest']
print(f"n_files={m['n_files']:,}  data_version={m['data_version']}")
print(f"mix={m['mix']}")
PYEOF
    echo
    echo "## holdout（同 digest 9c5362bf99d60661；基线 = B/756k 0.8127）"
    grep -a 'steps=' logs/s6c/holdout_base.log logs/d2_2026/holdout_formal.log 2>/dev/null
    echo
    echo "## 训练行为画像"
    grep -a 'behavior:' logs/d2_2026/formal_train.log 2>/dev/null
    echo
    echo "## ladder 汇总（δ = (A − B)/2；CI 下界 >0 才算赢）"
    tail -45 logs/d2_2026/ladder_summary.log 2>/dev/null
} > "$LOG/SUMMARY.md"
say "摘要已写 $LOG/SUMMARY.md"
say 'ALL DONE'
