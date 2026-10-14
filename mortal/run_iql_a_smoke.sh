# IQL/Bellman Stage A 冒烟链（2026-10-14，对应计划 S-A1）：
#   暖启 B/756k → credit_mode='game' 训练 2k 步 → 守卫/非有限/预算/test_play 四项检查。
# 验收：① 恰好 1 条 credit_mode 差异告警 ② 无 non-finite ③ 'training budget reached'
#       ④ test_play 正常跑完且 avg rank/avg pt 有输出。产物全部隔离在 *_smoke 命名空间。
# 用法: tmux new -d -s iql_a_smoke 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_a_smoke.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k 暖启源
CK=checkpoints/iql_a_smoke/mortal.pth
CFG=config.iql_a_smoke.toml
LOG=iql_a_smoke/train.log

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p iql_a_smoke checkpoints/iql_a_smoke runs/iql_a_smoke/test_play

say () { echo "[$(date '+%F %T')] $*" | tee -a iql_a_smoke/smoke.log; }

say "==== 0) 预检 ===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > iql_a_smoke/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 iql_a_smoke/env_check.log）"; exit 1; }
say "env_check OK"

if [ ! -f "$CK" ]; then
    [ -f "$PARENT" ] || { say "!! 缺少暖启源 $PARENT"; exit 1; }
    mkdir -p "$(dirname "$CK")"
    cp "$PARENT" "$CK"
    say "warm start: $PARENT → $CK"
fi

say "==== 1) 训练（756k → 758k，GPU2 RTX 8000，fp32）===="
(
    export CUDA_VISIBLE_DEVICES=2
    export MORTAL_CFG=$CFG
    echo "---- $(date '+%F %T') iql_a_smoke train start ----" >> "$LOG"
    $PY -u train.py >> "$LOG" 2>&1
    echo "---- $(date '+%F %T') iql_a_smoke train exit=$? ----" >> "$LOG"
)
rc=$?

say "==== 2) 冒烟验收 ===="
fail=0

# ① 续训目标守卫：credit_mode 差异告警应恰好 1 条（B ckpt 无 credit_mode 键 → 按默认 'kyoku' 处理）
n_guard=$(grep -ac "credit_mode: 'kyoku' -> 'game'" "$LOG" || true)
if [ "$n_guard" -eq 1 ]; then say "  [OK] credit_mode 守卫告警恰好 1 条"
else say "  [FAIL] credit_mode 守卫告警条数=$n_guard（期望 1）"; fail=1; fi

# ② 目标口径日志
grep -aq 'credit_mode=game' "$LOG" && say "  [OK] env 日志含 credit_mode=game" \
    || { say "  [FAIL] env 日志缺 credit_mode=game"; fail=1; }
grep -aq 'credit=game' "$LOG" && say "  [OK] reward_desc 含 credit=game 后缀和口径" \
    || { say "  [FAIL] reward_desc 缺 credit=game"; fail=1; }

# ③ 无 non-finite / 异常栈
if grep -aqE 'non-finite|Traceback' "$LOG"; then
    say "  [FAIL] 出现 non-finite/Traceback"; grep -aE 'non-finite|Traceback' "$LOG" | head -3 | sed 's/^/      /'
    fail=1
else say "  [OK] 无 non-finite / 无异常栈"; fi

# ④ 预算达成 + 存盘
grep -aq 'training budget reached' "$LOG" && say "  [OK] 到达 758k 预算" \
    || { say "  [FAIL] 未到 758k 预算"; fail=1; }
[ -f "checkpoints/iql_a_smoke/mortal_step_000758000.pth" ] \
    && say "  [OK] 冒烟快照已生成" \
    || say "  [WARN] 冒烟快照缺失（检查 snapshot_every=2000 是否触发）"

# ⑤ test_play 正常跑完
grep -aq 'avg rank' "$LOG" && say "  [OK] test_play 输出 avg rank/avg pt" \
    || { say "  [FAIL] test_play 无输出"; fail=1; }

step=$(grep -aE 'total steps:' "$LOG" | tail -1)
say "  末次步数: $step"

if [ "$fail" -ne 0 ]; then say "SMOKE FAIL（rc_train=$rc）"; exit 1; fi
say "SMOKE ALL PASS（rc_train=$rc）—— 正式轮可用 bash run_iql_a.sh 启动"
