# S6a 冒烟链（隔离版，2026-10-17）：读牌×主模型结合第一步——紧迫度特征注入。
#   本脚本只使用 s6a_isolated/code/ 下的代码与 libriichi.so（v5），
#   不触碰 mortal/ 共享树（iql_beta / read_model 等实验继续用原版）。
# 验收：① 回归 ALL PASS ② 日志 version=5 / obs (1018,34) ③ 无 non-finite
#       ④ 到 758k 预算 ⑤ 快照在 ⑥ test_play avg_rank ≤ 2.60（父代 B 同阶 2.466）
#       ⑦ holdout top1 ≥ 0.78 且 |logsumexp_q| ≤ 10 ⑧ ckpt 标记 version=5/首层 1018
# 用法: tmux new -d -s s6a_smoke 'bash /mnt/data8/mortal/Mortal/s6a_isolated/run_s6a_smoke.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
ISO=$ROOT/s6a_isolated
CODE=$ISO/code
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # 经软链指向共享树 B/756k
CK=checkpoints/s6a_smoke/mortal.pth
CFG=config.s6a_smoke.toml
LOG=s6a_smoke/train.log

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export S6A_CODE=$CODE

cd "$CODE"
mkdir -p s6a_smoke checkpoints/s6a_smoke runs/s6a_smoke/test_play logs

say () { echo "[$(date '+%F %T')] $*" | tee -a s6a_smoke/smoke.log; }

say "==== 0) 回归测试（隔离 code 目录，六层）===="
$PY "$ISO/tools/test_urgency_features.py" > s6a_smoke/test_urgency_features.log 2>&1 \
    || { say "!! 回归测试未通过，终止（见 s6a_smoke/test_urgency_features.log）"; exit 1; }
say "回归测试 ALL PASS（$(grep -c '\[OK\]' s6a_smoke/test_urgency_features.log) 项）"

say "==== 1) 预检 ===="
CUDA_VISIBLE_DEVICES=1 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > s6a_smoke/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 s6a_smoke/env_check.log）"; exit 1; }
say "env_check OK"

say "==== 2) 首层手术（B/756k v4 -> v5）===="
if [ ! -f "$CK" ]; then
    [ -f "$PARENT" ] || { say "!! 缺少手术源 $PARENT"; exit 1; }
    mkdir -p "$(dirname "$CK")"
    $PY "$ISO/tools/expand_ckpt_v5.py" "$PARENT" "$CK" > s6a_smoke/surgery.log 2>&1 \
        || { say "!! 手术失败（见 s6a_smoke/surgery.log）"; exit 1; }
    say "手术完成：$PARENT -> $CK"
else
    say "复用已有 $CK"
fi

say "==== 3) 训练（756k -> 758k，GPU2 RTX 8000，fp32）===="
(
    export CUDA_VISIBLE_DEVICES=1
    export MORTAL_CFG=$CFG
    echo "---- $(date '+%F %T') s6a smoke train start ----" >> "$LOG"
    $PY -u train.py >> "$LOG" 2>&1
    echo "---- $(date '+%F %T') s6a smoke train exit=$? ----" >> "$LOG"
)
rc=$?
say "训练阶段结束 rc=$rc"

say "==== 4) 冒烟验收 ===="
fail=0

grep -aq 'version: 5' "$LOG" && say "  [OK] 日志 version: 5" \
    || { say "  [FAIL] 日志缺 version: 5"; fail=1; }
grep -aq '(1018, 34)' "$LOG" && say "  [OK] obs shape (1018, 34)" \
    || { say "  [FAIL] obs shape 非 (1018,34)"; fail=1; }

if grep -aqE 'non-finite|Traceback' "$LOG"; then
    say "  [FAIL] 出现 non-finite/Traceback"; grep -aE 'non-finite|Traceback' "$LOG" | head -3 | sed 's/^/      /'
    fail=1
else say "  [OK] 无 non-finite / 无异常栈"; fi

grep -aq 'training budget reached' "$LOG" && say "  [OK] 到达 758k 预算" \
    || { say "  [FAIL] 未到 758k 预算"; fail=1; }
[ -f "checkpoints/s6a_smoke/mortal_step_000758000.pth" ] \
    && say "  [OK] 冒烟快照已生成" \
    || say "  [WARN] 冒烟快照缺失（检查 snapshot_every=2000 是否触发）"

grep -aq 'avg rank' "$LOG" && say "  [OK] test_play 输出 avg rank/avg pt" \
    || { say "  [FAIL] test_play 无输出"; fail=1; }

step=$(grep -aE 'total steps:' "$LOG" | tail -1)
say "  末次步数: $step"
say "  吞吐: $(grep -aoE '[0-9.]+batch/s' "$LOG" | tail -1)（与 B 同阶预期）"

rk=$(grep -a 'avg rank:' "$LOG" | tail -1 | grep -aoE '[0-9]+\.[0-9]+')
pt=$(grep -a 'avg pt:' "$LOG" | tail -1 | grep -aoE '\-?[0-9]+\.[0-9]+')
if [ -n "${rk:-}" ] && awk -v x="$rk" 'BEGIN{exit !(x<=2.60)}'; then
    say "  [OK] test_play avg_rank=$rk avg_pt=${pt:-?} 未劣化（门禁 ≤2.60）"
else
    say "  [FAIL] test_play avg_rank=$rk avg_pt=${pt:-?} 已劣化（门禁 ≤2.60）"; fail=1
fi

CUDA_VISIBLE_DEVICES=1 MORTAL_CFG=$CFG $PY -u holdout_metrics.py \
    --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
    > s6a_smoke/holdout_202609.log 2>&1
ho=$(grep -a 'steps=' s6a_smoke/holdout_202609.log | tail -1)
say "  holdout: $ho"
t1=$(printf '%s' "$ho" | grep -aoE 'top1=[0-9.]+' | head -1 | cut -d= -f2)
lq=$(printf '%s' "$ho" | grep -aoE 'logsumexp_q=[-+]?[0-9.]+' | head -1 | cut -d= -f2)
if [ -n "${t1:-}" ] && awk -v x="$t1" 'BEGIN{exit !(x>=0.78)}'; then
    say "  [OK] holdout top1=$t1 保持（门禁 ≥0.78）"
else
    say "  [FAIL] holdout top1=${t1:-?} 明显下降（门禁 ≥0.78）"; fail=1
fi
if [ -n "${lq:-}" ] && awk -v x="$lq" 'BEGIN{if(x<0)x=-x; exit !(x<=10)}'; then
    say "  [OK] logsumexp_q=$lq 幅度合理（门禁 |·|≤10）"
else
    say "  [FAIL] logsumexp_q=${lq:-?} 尺度异常（门禁 |·|≤10）"; fail=1
fi

ver=$($PY -c "
import torch
st = torch.load('$CK', map_location='cpu', weights_only=False)
print(st['config']['control']['version'], tuple(st['mortal']['encoder.net.0.weight'].shape)[1])
" 2>/dev/null)
if [ "$ver" = "5 1018" ]; then
    say "  [OK] ckpt 标记 version=5 / 首层 1018"
else
    say "  [FAIL] ckpt 标记异常：$ver（期望 '5 1018'）"; fail=1
fi

if [ "$fail" -ne 0 ]; then say "SMOKE FAIL（rc_train=$rc）—— 先查实现，不要启动正式轮"; exit 1; fi
say "SMOKE ALL PASS（rc_train=$rc）—— 正式轮可用 bash s6a_isolated/run_s6a.sh 启动"
