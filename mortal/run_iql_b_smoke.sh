# IQL/Bellman Stage B 冒烟链（2026-10-15，对应计划 S-B1b）：
#   暖启 B/756k → objective='iql' 训练 2k 步 → 守卫/非有限/预算/test_play/新键往返 五项检查。
# 验收：① "缺 v_net/EMA" 告警恰 1 条（从 cql_mc 父代暖启属预期）② 无 non-finite
#       ③ 'training budget reached' ④ test_play 正常 ⑤ 冒烟快照已生成。
# 用法: tmux new -d -s iql_b_smoke 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_b_smoke.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k 暖启源
CK=checkpoints/iql_b_smoke/mortal.pth
CFG=config.iql_b_smoke.toml
LOG=iql_b_smoke/train.log

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p iql_b_smoke checkpoints/iql_b_smoke runs/iql_b_smoke/test_play logs

say () { echo "[$(date '+%F %T')] $*" | tee -a iql_b_smoke/smoke.log; }

say "==== 0) 预检 ===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > iql_b_smoke/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 iql_b_smoke/env_check.log）"; exit 1; }
say "env_check OK"

if [ ! -f "$CK" ]; then
    [ -f "$PARENT" ] || { say "!! 缺少暖启源 $PARENT"; exit 1; }
    mkdir -p "$(dirname "$CK")"
    cp "$PARENT" "$CK"
    say "warm start: $PARENT → $CK"
fi

say "==== 1) 训练（756k → 758k，GPU2 RTX 8000，fp32，3× encoder 前向预计 ~2 it/s）===="
(
    export CUDA_VISIBLE_DEVICES=2
    export MORTAL_CFG=$CFG
    echo "---- $(date '+%F %T') iql_b_smoke train start ----" >> "$LOG"
    $PY -u train.py >> "$LOG" 2>&1
    echo "---- $(date '+%F %T') iql_b_smoke train exit=$? ----" >> "$LOG"
)
rc=$?

say "==== 2) 冒烟验收 ===="
fail=0

# ① 目标口径日志
grep -aq 'objective=iql' "$LOG" && say "  [OK] env 日志含 objective=iql" \
    || { say "  [FAIL] env 日志缺 objective=iql"; fail=1; }
grep -aq 'objective=iql(next_obs bootstrap' "$LOG" && say "  [OK] reward_desc 含 iql 口径" \
    || { say "  [FAIL] reward_desc 缺 iql 口径"; fail=1; }

# ② 暖启新键告警恰 1 条（B ckpt 缺 v_net/EMA 键）
n_wnet=$(grep -ac '缺 v_net/EMA target 键' "$LOG" || true)
if [ "$n_wnet" -eq 1 ]; then say "  [OK] v_net/EMA 暖启告警恰好 1 条"
else say "  [FAIL] v_net/EMA 暖启告警条数=$n_wnet（期望 1）"; fail=1; fi

# ③ 无 non-finite / 异常栈
if grep -aqE 'non-finite|Traceback' "$LOG"; then
    say "  [FAIL] 出现 non-finite/Traceback"; grep -aE 'non-finite|Traceback' "$LOG" | head -3 | sed 's/^/      /'
    fail=1
else say "  [OK] 无 non-finite / 无异常栈"; fi

# ④ 预算达成 + 存盘
grep -aq 'training budget reached' "$LOG" && say "  [OK] 到达 758k 预算" \
    || { say "  [FAIL] 未到 758k 预算"; fail=1; }
[ -f "checkpoints/iql_b_smoke/mortal_step_000758000.pth" ] \
    && say "  [OK] 冒烟快照已生成" \
    || say "  [WARN] 冒烟快照缺失（检查 snapshot_every=2000 是否触发）"

# ⑤ test_play 正常跑完
grep -aq 'avg rank' "$LOG" && say "  [OK] test_play 输出 avg rank/avg pt" \
    || { say "  [FAIL] test_play 无输出"; fail=1; }

step=$(grep -aE 'total steps:' "$LOG" | tail -1)
say "  末次步数: $step"
say "  吞吐: $(grep -aoE '[0-9.]+batch/s' "$LOG" | tail -1)（3× encoder 前向基线，供正式轮估时）"

if [ "$fail" -ne 0 ]; then say "SMOKE FAIL（rc_train=$rc）"; exit 1; fi
say "SMOKE ALL PASS（rc_train=$rc）—— 正式轮可用 bash run_iql_b.sh 启动"
