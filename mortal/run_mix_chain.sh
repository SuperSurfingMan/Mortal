#!/usr/bin/env bash
# 数据扩窗探针 D1 收尾链：等 100k 正式轮跑完 → 双向 ladder → 写 SUMMARY.md
#
# 前置：formal 已在跑（tmux `mix_formal`；run_mix.sh formal 结束会写 `FORMAL=<code>`）。
# 用法：tmux new -d -s mix_chain "bash /mnt/data8/mortal/Mortal/mortal/run_mix_chain.sh > logs/mix_v1/chain.log 2>&1"
#
# ladder 口径：混入(856k) vs B/756k，λ=0，双向 5×500 sets，新种子块 320000/340000。
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
cd "$ROOT/mortal" || exit 1
LOG=logs/mix_v1
say () { echo "[$(date '+%F %T')] $*"; }

say '等待 formal 阶段结束（每 60 s 检查一次）...'
while ! grep -qa 'FORMAL=' "$LOG/formal_stage.log" 2>/dev/null; do
    sleep 60
done
code=$(grep -ao 'FORMAL=[0-9]*' "$LOG/formal_stage.log" | tail -1 | cut -d= -f2)
say "formal 退出码 = $code"
if [ "$code" != '0' ]; then
    say "formal 未正常结束 → 不跑 ladder（tail -80 logs/mix_v1/formal_train.log）"
    exit 1
fi

say '==== 双向 ladder（λ=0，320000/340000，各 5×500 sets）===='
STAGE=ladder bash run_mix.sh > "$LOG/ladder_stage.log" 2>&1
say "ladder 退出码 = $?"
grep -a 'avg_pt\|强度差\|判读' "$LOG/ladder_stage.log" | tail -5

say '==== 汇总摘要 ===='
{
    echo "# 数据扩窗探针 D1 摘要（自动生成 $(date '+%F %T')）"
    echo
    echo "## 数据池"
    "$PY" - <<'PYEOF'
import torch
st = torch.load('indexes/data_mix_2019_2021_20pct_v1.pth', weights_only=True)
m = st['manifest']
print(f"n_files={m['n_files']:,}  data_version={m['data_version']}")
print(f"mix={m['mix']}")
PYEOF
    echo
    echo "## holdout（同 digest 9c5362bf99d60661；基线 = B/756k 0.8127）"
    grep -a 'steps=' logs/s6c/holdout_base.log logs/mix_v1/holdout_formal.log 2>/dev/null
    echo
    echo "## 训练行为画像"
    grep -a 'behavior:' logs/mix_v1/formal_train.log 2>/dev/null
    echo
    echo "## ladder 汇总（δ = (A − B)/2；CI 下界 >0 才算赢）"
    tail -45 logs/mix_v1/ladder_summary.log 2>/dev/null
} > "$LOG/SUMMARY.md"
say "摘要已写 $LOG/SUMMARY.md"
say 'ALL DONE'
