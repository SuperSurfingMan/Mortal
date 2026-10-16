# IQL 价值尺度漂移 4 臂探针（2026-10-16，MEMORY 会话 25）：
#   iql_b_v2 正式轮 q_target 中位数 105k 步 +0.6→+61 纯指数发散（倍增 ~32k 步，超理论界 [−5,+5]
#   ~12 倍）；参数级诊断：膨胀 confined 在价值通道（dqn v_head/VHead），a_head 统计不变。
#   机制：τ=0.7 expectile 不对称放大 × γ≈1 × 边界稀疏奖励 → bootstrap 反馈增益 >1。
# 4 臂全部 B/756k 暖启、同数据(indexes/iql_b_r4_v2)/30k 步/同 batch，只改一个旋钮：
#   A 对照=v2 原配置（预期 30k 步 p50 ~+15，复现漂移）
#   B bootstrap 改用 V 的 EMA target（iql_v_target_decay=0.99）
#   C γ=0.99（直接压反馈环增益）
#   D τ=0.5（expectile 对称化；机制检查——若稳定则证实不对称放大是根源）
# 判据：q_target p50 全程 ≤ 8（train.py 每 1000 步打 'iql scale:' 行）。
# 用法: tmux new -d -s iql_probe 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_probe.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k 暖启源
ARMS=(a b c d)
GPUS=(0 1 3 4)                                                    # GPU2 仍被 iql_b_v2 训练/判定占用

export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p logs
for x in "${ARMS[@]}"; do mkdir -p "iql_probe_$x" "checkpoints/iql_probe_$x"; done
CHAIN=logs/iql_probe_chain.log
say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

run_arm () {   # $1=arm 字母  $2=gpu
    local x=$1 gpu=$2
    local CK=checkpoints/iql_probe_$x/mortal.pth
    local CFG=config.iql_probe_$x.toml
    local log=iql_probe_$x/train.log
    say "[$x] env_check (GPU$gpu)"
    CUDA_VISIBLE_DEVICES=$gpu MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
        > "iql_probe_$x/env_check.log" 2>&1 \
        || { say "[$x] !! env_check 未通过，终止该臂"; return 1; }
    if [ ! -f "$CK" ]; then
        cp "$PARENT" "$CK"
        say "[$x] warm start: B/756k → $CK"
    fi
    say "[$x] train 756k→786k (GPU$gpu, fp32)"
    (
        export CUDA_VISIBLE_DEVICES=$gpu MORTAL_CFG=$CFG
        echo "---- $(date '+%F %T') iql_probe_$x train start ----" >> "$log"
        $PY -u train.py >> "$log" 2>&1
        echo "---- $(date '+%F %T') iql_probe_$x train exit=$? ----" >> "$log"
    )
    # 汇总：q_target p50 轨迹与最大值
    local traj maxp50 last
    traj=$(grep -a 'iql scale:' "$log" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 | tr '\n' ' ')
    maxp50=$(grep -a 'iql scale:' "$log" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 \
        | awk 'BEGIN{m=-1e9}{v=$1+0; if(v>m)m=v}END{printf "%.3f", m}')
    last=$(grep -a 'iql scale:' "$log" | tail -1)
    say "[$x] scale 轨迹: $traj"
    say "[$x] max p50=$maxp50 | $last"
    if grep -aqE 'non-finite|Traceback' "$log"; then say "[$x] !! 出现 non-finite/异常"; return 1; fi
    if awk -v m="$maxp50" 'BEGIN{exit !(m<=8)}'; then
        say "[$x] [PASS] 尺度门禁：max p50=$maxp50 ≤ 8（30k 步内未漂移）"
    else
        say "[$x] [FAIL] 尺度门禁：max p50=$maxp50 > 8（30k 步内漂移，对照臂预期 ~+15）"
    fi
    say "[$x] test_play: $(grep -a 'avg rank' "$log" | tail -1)"
}

say "==== 4 臂探针启动（A 对照 / B v-ema / C γ=0.99 / D τ=0.5；GPU ${GPUS[*]}）===="
wp=()
for i in "${!ARMS[@]}"; do
    run_arm "${ARMS[$i]}" "${GPUS[$i]}" &
    wp+=($!)
done
for p in "${wp[@]}"; do wait "$p"; done
say "==== 判读：PASS 臂 = 30k 步内价值尺度稳定（≤8）；FAIL 臂 = 漂移未止 ===="
say "ALL DONE"
