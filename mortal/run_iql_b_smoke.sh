# IQL/Bellman Stage B 冒烟链（2026-10-15，对应计划 S-B1b）：
#   暖启 B/756k → objective='iql' 训练 2k 步 → 结构检查 + **行为门禁**。
# 验收：① "缺 v_net/EMA" 告警恰 1 条 ② 无 non-finite ③ 'training budget reached' ④ test_play 跑完
#       ⑤ 快照已生成 ⑥ **avg_rank 未劣化（≤ 2.60；父代 B 同阶 2.466，v1 受损版 2.789）**
#       ⑦ **holdout top1 ≥ 0.78 且 |logsumexp_q| ≤ 10（v1 受损版 0.6917 / -176.1）**
# ★ v2：v1 产物（checkpoints/iql_b_smoke/）为 r_t 双重计入缺陷版，含受损权重，不可续用（详 MEMORY §16.26）。
# 用法: tmux new -d -s iql_b_smoke_v3 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_b_smoke.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k 暖启源
CK=checkpoints/iql_b_smoke_v3/mortal.pth
CFG=config.iql_b_smoke.toml
LOG=iql_b_smoke_v3/train.log

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p iql_b_smoke_v3 checkpoints/iql_b_smoke_v3 runs/iql_b_smoke_v3/test_play logs

say () { echo "[$(date '+%F %T')] $*" | tee -a iql_b_smoke_v3/smoke.log; }

say "==== 0) 预检 ===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > iql_b_smoke_v3/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 iql_b_smoke_v3/env_check.log）"; exit 1; }
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
[ -f "checkpoints/iql_b_smoke_v3/mortal_step_000758000.pth" ] \
    && say "  [OK] 冒烟快照已生成" \
    || say "  [WARN] 冒烟快照缺失（检查 snapshot_every=2000 是否触发）"

# ⑤ test_play 正常跑完
grep -aq 'avg rank' "$LOG" && say "  [OK] test_play 输出 avg rank/avg pt" \
    || { say "  [FAIL] test_play 无输出"; fail=1; }

step=$(grep -aE 'total steps:' "$LOG" | tail -1)
say "  末次步数: $step"
say "  吞吐: $(grep -aoE '[0-9.]+batch/s' "$LOG" | tail -1)（3× encoder 前向基线，供正式轮估时）"

# ⑥ 行为门禁Ⅰ：2k 步内 test_play avg_rank 不得劣化（父代 B 同阶 2.466；v1 受损版 2.789/avg_pt −19.3）
rk=$(grep -a 'avg rank:' "$LOG" | tail -1 | grep -aoE '[0-9]+\.[0-9]+')
pt=$(grep -a 'avg pt:' "$LOG" | tail -1 | grep -aoE '\-?[0-9]+\.[0-9]+')
if [ -n "${rk:-}" ] && awk -v x="$rk" 'BEGIN{exit !(x<=2.60)}'; then
    say "  [OK] test_play avg_rank=$rk avg_pt=${pt:-?} 未劣化（门禁 ≤2.60）"
else
    say "  [FAIL] test_play avg_rank=$rk avg_pt=${pt:-?} 已劣化（门禁 ≤2.60）→ 目标实现极可能有错，不得开正式轮"; fail=1
fi

# ⑦ 行为门禁Ⅱ：holdout top1 与 Q 幅度（v1 受损版 top1=0.6917、logsumexp_q=-176.1）
CUDA_VISIBLE_DEVICES=4 MORTAL_CFG=$CFG $PY -u holdout_metrics.py \
    --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
    > iql_b_smoke_v3/holdout_202609.log 2>&1
ho=$(grep -a 'steps=' iql_b_smoke_v3/holdout_202609.log | tail -1)
say "  holdout: $ho"
t1=$(printf '%s' "$ho" | grep -aoE 'top1=[0-9.]+' | head -1 | cut -d= -f2)
lq=$(printf '%s' "$ho" | grep -aoE 'logsumexp_q=[-+]?[0-9.]+' | head -1 | cut -d= -f2)
if [ -n "${t1:-}" ] && awk -v x="$t1" 'BEGIN{exit !(x>=0.78)}'; then
    say "  [OK] holdout top1=$t1 保持（门禁 ≥0.78）"
else
    say "  [FAIL] holdout top1=${t1:-?} 明显下降（门禁 >=0.78）-> 策略已被打坏"; fail=1
fi
if [ -n "${lq:-}" ] && awk -v x="$lq" 'BEGIN{if(x<0)x=-x; exit !(x<=10)}'; then
    say "  [OK] logsumexp_q=$lq 幅度合理（门禁 |·|≤10）"
else
    say "  [FAIL] logsumexp_q=${lq:-?} 尺度异常（门禁 |·|≤10）→ 目标量级可能被重复计入"; fail=1
fi

# ⑧ 价值尺度门禁（10-16 加，v2 漂移教训）：冒烟窗内 q_target p50 ≤ 10。
#    v2 的漂移时间尺度 ~3 万步，2k 冒烟结构性无感——此门禁是兜底，充分证据靠探针/加长冒烟。
sc_max=$(grep -a 'iql scale:' "$LOG" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 \
    | awk 'BEGIN{m=-1e9}{v=$1+0; if(v>m)m=v}END{printf "%.3f", m}')
if [ -n "$sc_max" ] && awk -v m="$sc_max" 'BEGIN{exit !(m<=10)}'; then
    say "  [OK] 价值尺度 max p50=$sc_max 未漂移（门禁 ≤10）"
else
    say "  [FAIL] 价值尺度 max p50=${sc_max:-?} 已漂移（门禁 ≤10）→ expectile/bootstrap 可能自激"; fail=1
fi

if [ "$fail" -ne 0 ]; then say "SMOKE FAIL（rc_train=$rc）—— 先查实现，不要启动正式轮"; exit 1; fi
say "SMOKE ALL PASS（rc_train=$rc）—— 正式轮可用 bash run_iql_b.sh 启动"
