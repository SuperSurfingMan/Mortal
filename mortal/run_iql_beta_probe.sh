# IQL AWR 温度 β 扫描探针（2026-10-17）：
#   背景：v3（τ=0.5）δ=+0.646 / CI[−0.35,+1.63] 无结论 → IQL 路线关闭（MEMORY 会话 28）；
#   归因第 3 条：AWR 权重≈uniform（top1 0.8167，剩余分歧 ~18% 多 indifferent）。τ=0.5 下
#   V=E[Q̄]（行为均值），w=exp((Q−V)/β) 是价值→策略的唯一改进机制，β=3.0 从未扫过。
#   优势尺度实测 ~O(1)（a_head std 1.6~1.7）：β=3→1.4×/单位优势（软）、β=1→2.7×、β=0.5→7.4×（锐）、
#   β=10→1.11×（≈纯 BC）。
# 设计：4 臂全部 B/756k 暖启、同数据(indexes/iql_b_r4_v2)/同步数(30k)/同 batch/同 seed(20261016)，
#   唯一变量 = iql_beta：a=3.0(对照) / b=1.0 / c=0.5 / d=10.0(≈BC 极限)。
#   假设：H1 锐化有帮助（优势有真信号）；H2 锐化有害（1:10 SNR 优势被运气污染）；H3 无效（权重本就均匀）。
# 流程：① 2k 预检冒烟（臂a 配置 + train.py 新日志行 + v3 同款门禁）→ 通过才启动 4 臂；
#       ② 4 臂并行 30k（GPU1/2/3/4）→ 每臂健康五门 + 指标提取 → 汇总判读。
# 预注册判据：
#   健康五门（每臂必过）：无 non-finite / q_target p50≤8 / avg_rank≤2.60 / top1≥0.78 / |lse|≤10
#   旋钮生效：awr_w ess 随 β 下降、p99 随 β 上升（ess=1/mean(w^2)，w 已均值归一）
#   探针级方向（500 局 test_play 噪声 ~1.5σ，仅筛查；强度只认正式轮 ladder）：
#     avg_pt ≥ 对照+2.0 且 top1 ≥ 对照−0.005 → PROMISING（建议正式轮，种子块 240000/250000）
#     avg_pt ≤ 对照−2.0 或 top1 < 对照−0.005 或健康 FAIL → DEGRADED
#     其余 → NO-SEP（探针无分离）
# 日志纪律：一切日志进 mortal/logs/iql_beta/（chain + 每臂 train/env_check/holdout/metrics +
#   汇总），不在 mortal/ 顶层或臂工作目录散落文件；ckpt 在 checkpoints/iql_beta_*/，TB 与对局
#   日志在 runs/iql_beta_*/（既有约定，均已被 .gitignore 覆盖）。
# GPU 分配（2026-10-17 13:20 实测校正）：**GPU1 热降频**（79°C 时 SM 556/1328MHz，SW Thermal，
#   MEMORY 会话 21 同一张卡的老问题）→ 长时任务一律避开 GPU1。冒烟与臂 a/b/c 用 GPU2(RTX
#   8000)/3/4；臂 d 等读牌模型 v2s@50k 让出 GPU0（轮询等待，超时则回退 GPU2 空闲或 GPU1 并告警）。
# 用法: tmux new -d -s iql_beta 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_beta_probe.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k 暖启源
LOGDIR=$MORTAL/logs/iql_beta
ARMS=(a b c d)
BETAS=(3.0 1.0 0.5 10.0)                                          # 与 ARMS 对齐
# GPU1 热降频（556/1328MHz@79°C）避开；臂 d 等 GPU0（读牌模型让出，见 wait_gpu_free）
GPUS=(2 3 4 0)
SMOKE_GPU=2
GPU0_WAIT_MIN=150                                                 # 臂 d 等 GPU0 的最长分钟数
PT_THR=2.0                                                        # 探针级 avg_pt 分离阈值
T1_TOL=0.005                                                      # top1 容许退让

export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p "$LOGDIR"
CHAIN=$LOGDIR/chain.log
say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

# 从 train.log / holdout 日志提取指标 → $LOGDIR/<arm>/metrics.env（$2=beta 由调用方传入）
extract_metrics () {   # $1=arm  $2=beta
    local x=$1 beta=$2
    local ALOG=$LOGDIR/$1 log=$LOGDIR/$1/train.log
    local sc traj maxp50 last beh rk pt t1 lq ho
    sc=$(grep -a 'iql scale:' "$log" 2>/dev/null || true)
    traj=$(printf '%s\n' "$sc" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 | tr '\n' ' ')
    maxp50=$(printf '%s\n' "$sc" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 \
        | awk 'BEGIN{m=-1e9}{v=$1+0; if(v>m)m=v}END{printf "%.3f", m}')
    last=$(printf '%s\n' "$sc" | tail -1)
    beh=$(grep -a 'behavior:' "$log" 2>/dev/null | tail -1 | sed 's/.*behavior:/behavior:/')
    rk=$(grep -a 'avg rank:' "$log" 2>/dev/null | tail -1 | sed 's/.*avg rank: //' | awk '{print $1}')
    pt=$(grep -a 'avg pt:' "$log" 2>/dev/null | tail -1 | sed 's/.*avg pt: //' | awk '{print $1}')
    ho=$(grep -a 'steps=' "$ALOG/holdout_202609.log" 2>/dev/null | tail -1)
    t1=$(printf '%s' "$ho" | grep -aoE 'top1=[0-9.]+' | head -1 | cut -d= -f2)
    lq=$(printf '%s' "$ho" | grep -aoE 'logsumexp_q=[-+]?[0-9.]+' | head -1 | cut -d= -f2)
    cat > "$ALOG/metrics.env" <<EOF
ARM=$x
BETA=$beta
TRAJ="$traj"
MAX_P50=${maxp50:--1e9}
AWRE=$(printf '%s' "$last" | grep -aoE 'ess=[0-9.]+' | head -1 | cut -d= -f2)
AWRP=$(printf '%s' "$last" | grep -aoE 'p99=[-+0-9.]+' | head -1 | cut -d= -f2)
AWRM=$(printf '%s' "$last" | grep -aoE 'max=[-+0-9.]+' | head -1 | cut -d= -f2)
AVG_RANK=${rk:-?}
AVG_PT=${pt:-?}
TOP1=${t1:-?}
LSE=${lq:-?}
BEHAVIOR="$beh"
EOF
    say "[$x] metrics: max_p50=${maxp50:-?} avg_rank=${rk:-?} avg_pt=${pt:-?} top1=${t1:-?} lse=${lq:-?} awr_w=$(printf '%s' "$last" | grep -aoE 'ess=[0-9.]+ p99=[-+0-9.]+ max=[-+0-9.]+')"
    say "[$x] scale 轨迹: $traj"
}

# 轮询等待某张 GPU 显存占用低于阈值（默认 500MiB）；超时返回 1
wait_gpu_free () {   # $1=gpu_idx  $2=max_minutes
    local gpu=$1 maxm=$2 i=0
    while [ "$i" -lt "$maxm" ]; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
        if [ -n "${used:-}" ] && [ "$used" -lt 500 ]; then return 0; fi
        sleep 60; i=$((i+1))
    done
    return 1
}

# 一次训练尝试是否"跑完且健康"（2026-10-18 修正）：原判据只认 'training budget reached'，
# 但该行发生在 test_play **之后**——test_play 被外部信号打断时（10-17 实测到一次 SIGINT
# 静默 exit=0）会被误判为失败并白白重训。改认两种预算消息，且必须无异常栈。
# 续训场景（checkpoint 已达 max_train_steps）train.py 打的是 'training budget already reached'。
train_ok () {   # $1=log
    grep -aqE 'training budget (reached|already reached)' "$1" \
        && ! grep -aqE 'Traceback|Segmentation fault|exited unexpectedly' "$1"
}

# 训练 + 崩溃自动续训（2026-10-17）：偶发 DataLoader worker 段错误（fork 多线程进程的
# 已知风险，历史 111k 步正式轮 0 次，10-17 并发环境下遇到 2 次）会让 4h 探针白跑。
# train.py 每 1000 步存盘且能从 state_file 的 steps 续训，故最多重试 5 次；每次重试把
# 旧日志轮转为 <log>.pre-retry-N（保留取证，且避免旧 Traceback 污染门禁）。
train_with_retry () {   # $1=cfg $2=gpu $3=log $4=tag
    local cfg=$1 gpu=$2 log=$3 tag=$4 attempt=1
    while [ "$attempt" -le 5 ]; do
        (
            export CUDA_VISIBLE_DEVICES=$gpu MORTAL_CFG=$cfg
            echo "---- $(date '+%F %T') $tag train attempt=$attempt start ----" >> "$log"
            $PY -u train.py >> "$log" 2>&1
            rc=$?
            echo "---- $(date '+%F %T') $tag train attempt=$attempt exit=$rc ----" >> "$log"
            exit "$rc"
        )
        local rc=$?
        if train_ok "$log"; then
            return 0
        fi
        say "[$tag] !! train 第 $attempt 次未完成（rc=$rc，异常见 $log），10s 后从最近 checkpoint 续训"
        [ "$attempt" -lt 5 ] && mv "$log" "$log.pre-retry-$attempt"
        attempt=$((attempt+1))
        sleep 10
    done
    say "[$tag] !! train 连续 5 次失败，放弃该臂"; return 1
}

run_smoke () {
    local CK=checkpoints/iql_beta_smoke/mortal.pth
    local CFG=config.iql_beta_smoke.toml
    local SLOG=$LOGDIR/smoke
    local log=$SLOG/train.log
    mkdir -p "$SLOG" checkpoints/iql_beta_smoke runs/iql_beta_smoke/test_play runs/iql_beta_smoke/ladder
    say "==== 0) 预检冒烟（臂a 配置 β=3.0，2k 步，GPU$SMOKE_GPU）===="
    CUDA_VISIBLE_DEVICES=$SMOKE_GPU MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
        > "$SLOG/env_check.log" 2>&1 \
        || { say "!! env_check 未通过，终止（见 $SLOG/env_check.log）"; exit 1; }
    say "env_check OK"
    # 冒烟必须"从父代全新开始"（v_net/EMA 告警门禁依赖首次暖启）→ 崩溃即清盘重跑而非续训。
    # 成功判据（2026-10-18 修正）：预算消息 + **test_play 结果已产出**（冒烟本身要验证 test_play
    # 与 behavior 行；10-17 曾出现跑完 758k 但在 test_play 中被外部 SIGINT 打断 → 误判为通过的风险）。
    local attempt=1
    while :; do
        rm -rf checkpoints/iql_beta_smoke
        rm -f "$log"
        mkdir -p checkpoints/iql_beta_smoke
        cp "$PARENT" "$CK"
        say "warm start: B/756k → $CK (attempt=$attempt)"
        (
            export CUDA_VISIBLE_DEVICES=$SMOKE_GPU MORTAL_CFG=$CFG
            echo "---- $(date '+%F %T') iql_beta_smoke train attempt=$attempt start ----" >> "$log"
            $PY -u train.py >> "$log" 2>&1
            echo "---- $(date '+%F %T') iql_beta_smoke train attempt=$attempt exit=$? ----" >> "$log"
        )
        if train_ok "$log" && grep -aq 'avg rank:' "$log"; then
            break
        fi
        mv "$log" "$SLOG/train.attempt-$attempt.log"
        say "!! 冒烟第 $attempt 次未跑完（异常见 $SLOG/train.attempt-$attempt.log）"
        attempt=$((attempt+1))
        if [ "$attempt" -gt 5 ]; then say "冒烟连续 5 次失败，终止"; exit 1; fi
        sleep 15
    done
    local fail=0
    grep -aq 'objective=iql' "$log" && say "  [OK] env 日志含 objective=iql" \
        || { say "  [FAIL] env 日志缺 objective=iql"; fail=1; }
    local n_wnet; n_wnet=$(grep -ac '缺 v_net/EMA target 键' "$log" || true)
    if [ "$n_wnet" -eq 1 ]; then say "  [OK] v_net/EMA 暖启告警恰好 1 条"
    else say "  [FAIL] v_net/EMA 暖启告警条数=$n_wnet（期望 1）"; fail=1; fi
    if grep -aqE 'non-finite|Traceback' "$log"; then
        say "  [FAIL] 出现 non-finite/Traceback"; fail=1
    else say "  [OK] 无 non-finite / 无异常栈"; fi
    if train_ok "$log"; then say "  [OK] 到达 758k 预算"
    else say "  [FAIL] 未到 758k 预算"; fail=1; fi
    grep -aq 'avg rank' "$log" && say "  [OK] test_play 输出 avg rank/avg pt" \
        || { say "  [FAIL] test_play 无输出"; fail=1; }
    # train.py 新增日志行必须出现（awr_w 集中度 + behavior 分解）
    grep -aq 'awr_w ess=' "$log" && say "  [OK] 新日志行 awr_w ess= 已产出" \
        || { say "  [FAIL] awr_w 日志行缺失（train.py 改动未生效？）"; fail=1; }
    grep -aq 'behavior:' "$log" && say "  [OK] 新日志行 behavior: 已产出" \
        || { say "  [FAIL] behavior 日志行缺失（train.py 改动未生效？）"; fail=1; }
    local rk pt
    rk=$(grep -a 'avg rank:' "$log" | tail -1 | sed 's/.*avg rank: //' | awk '{print $1}')
    pt=$(grep -a 'avg pt:' "$log" | tail -1 | sed 's/.*avg pt: //' | awk '{print $1}')
    if [ -n "${rk:-}" ] && awk -v x="$rk" 'BEGIN{exit !(x<=2.60)}'; then
        say "  [OK] test_play avg_rank=$rk avg_pt=${pt:-?} 未劣化（门禁 ≤2.60）"
    else
        say "  [FAIL] test_play avg_rank=$rk avg_pt=${pt:-?} 已劣化（门禁 ≤2.60）"; fail=1
    fi
    CUDA_VISIBLE_DEVICES=$SMOKE_GPU $PY -u holdout_metrics.py \
        --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
        > "$SLOG/holdout_202609.log" 2>&1
    local ho t1 lq
    ho=$(grep -a 'steps=' "$SLOG/holdout_202609.log" | tail -1)
    say "  holdout: $ho"
    t1=$(printf '%s' "$ho" | grep -aoE 'top1=[0-9.]+' | head -1 | cut -d= -f2)
    lq=$(printf '%s' "$ho" | grep -aoE 'logsumexp_q=[-+]?[0-9.]+' | head -1 | cut -d= -f2)
    if [ -n "${t1:-}" ] && awk -v x="$t1" 'BEGIN{exit !(x>=0.78)}'; then
        say "  [OK] holdout top1=$t1 保持（门禁 ≥0.78）"
    else say "  [FAIL] holdout top1=${t1:-?}（门禁 ≥0.78）"; fail=1; fi
    if [ -n "${lq:-}" ] && awk -v x="$lq" 'BEGIN{if(x<0)x=-x; exit !(x<=10)}'; then
        say "  [OK] logsumexp_q=$lq 幅度合理（门禁 |·|≤10）"
    else say "  [FAIL] logsumexp_q=${lq:-?}（门禁 |·|≤10）"; fail=1; fi
    local sc_max
    sc_max=$(grep -a 'iql scale:' "$log" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 \
        | awk 'BEGIN{m=-1e9}{v=$1+0; if(v>m)m=v}END{printf "%.3f", m}')
    if [ -n "$sc_max" ] && awk -v m="$sc_max" 'BEGIN{exit !(m<=10)}'; then
        say "  [OK] 价值尺度 max p50=$sc_max 未漂移（门禁 ≤10）"
    else say "  [FAIL] 价值尺度 max p50=${sc_max:-?} 已漂移（门禁 ≤10）"; fail=1; fi
    say "  awr_w 末行: $(grep -a 'awr_w ess=' "$log" | tail -1)"
    say "  behavior 末行: $(grep -a 'behavior:' "$log" | tail -1)"
    say "  吞吐: $(grep -aoE '[0-9.]+batch/s' "$log" | tail -1)（供正式轮估时）"
    if [ "$fail" -ne 0 ]; then
        say "SMOKE FAIL——先查 train.py 改动/配置，不启动 4 臂探针"; exit 1
    fi
    say "SMOKE ALL PASS——启动 4 臂 30k 探针"
}

run_arm () {   # $1=arm 字母  $2=gpu  $3=beta
    local x=$1 gpu=$2 beta=$3
    local CK=checkpoints/iql_beta_$x/mortal.pth
    local CFG=config.iql_beta_$x.toml
    local ALOG=$LOGDIR/$x
    local log=$ALOG/train.log
    mkdir -p "$ALOG" "checkpoints/iql_beta_$x" "runs/iql_beta_$x/test_play" "runs/iql_beta_$x/ladder"
    say "[$x] env_check (GPU$gpu, β=$beta)"
    CUDA_VISIBLE_DEVICES=$gpu MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
        > "$ALOG/env_check.log" 2>&1 \
        || { say "[$x] !! env_check 未通过，终止该臂"; return 1; }
    if [ ! -f "$CK" ]; then
        cp "$PARENT" "$CK"
        say "[$x] warm start: B/756k → $CK"
    fi
    say "[$x] train 756k→786k (GPU$gpu, fp32, β=$beta)"
    train_with_retry "$CFG" "$gpu" "$log" "$x" \
        || { say "[$x] !! 训练连续失败，跳过该臂评估"; return 1; }
    CUDA_VISIBLE_DEVICES=$gpu $PY -u holdout_metrics.py \
        --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
        > "$ALOG/holdout_202609.log" 2>&1
    extract_metrics "$x" "$beta"
    local maxp50 rk pt t1 lq fail=0
    maxp50=$(grep -a 'iql scale:' "$log" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 \
        | awk 'BEGIN{m=-1e9}{v=$1+0; if(v>m)m=v}END{printf "%.3f", m}')
    rk=$(grep -a 'avg rank:' "$log" | tail -1 | sed 's/.*avg rank: //' | awk '{print $1}')
    pt=$(grep -a 'avg pt:' "$log" | tail -1 | sed 's/.*avg pt: //' | awk '{print $1}')
    t1=$(grep -a 'steps=' "$ALOG/holdout_202609.log" | tail -1 | grep -aoE 'top1=[0-9.]+' | head -1 | cut -d= -f2)
    lq=$(grep -a 'steps=' "$ALOG/holdout_202609.log" | tail -1 | grep -aoE 'logsumexp_q=[-+]?[0-9.]+' | head -1 | cut -d= -f2)
    awk -v m="$maxp50" 'BEGIN{exit !(m<=8)}' \
        && say "[$x] [PASS] 尺度门禁 max p50=$maxp50 ≤ 8" \
        || { say "[$x] [FAIL] 尺度门禁 max p50=$maxp50 > 8"; fail=1; }
    if [ -n "${rk:-}" ] && awk -v x="$rk" 'BEGIN{exit !(x<=2.60)}'; then
        say "[$x] [PASS] avg_rank=$rk ≤ 2.60"
    else say "[$x] [FAIL] avg_rank=${rk:-?} > 2.60"; fail=1; fi
    if [ -n "${t1:-}" ] && awk -v x="$t1" 'BEGIN{exit !(x>=0.78)}'; then
        say "[$x] [PASS] holdout top1=$t1 ≥ 0.78"
    else say "[$x] [FAIL] holdout top1=${t1:-?} < 0.78"; fail=1; fi
    if [ -n "${lq:-}" ] && awk -v x="$lq" 'BEGIN{if(x<0)x=-x; exit !(x<=10)}'; then
        say "[$x] [PASS] |logsumexp_q|=$lq ≤ 10"
    else say "[$x] [FAIL] |logsumexp_q|=${lq:-?} > 10"; fail=1; fi
    echo "HEALTH_FAIL=$fail" >> "$ALOG/metrics.env"
    say "[$x] 健康门禁: $([ "$fail" -eq 0 ] && echo ALL-PASS || echo FAIL)（avg_pt=$pt）"
}

summarize () {
    local out=$LOGDIR/summary.log
    {
    echo "==== β 扫描探针汇总（$(date '+%F %T')）===="
    printf '%-4s %-6s %-9s %-9s %-9s %-8s %-8s %-24s %s\n' \
        arm beta max_p50 avg_rank avg_pt top1 lse 'awr_w(ess/p99/max)' health
    for x in "${ARMS[@]}"; do
        [ -f "$LOGDIR/$x/metrics.env" ] || continue
        ( unset ARM BETA TRAJ MAX_P50 AWRE AWRP AWRM AVG_RANK AVG_PT TOP1 LSE BEHAVIOR HEALTH_FAIL
          . "$LOGDIR/$x/metrics.env"
          printf '%-4s %-6s %-9s %-9s %-9s %-8s %-8s %-24s %s\n' \
              "$ARM" "$BETA" "$MAX_P50" "$AVG_RANK" "$AVG_PT" "$TOP1" "$LSE" \
              "${AWRE:-?}/${AWRP:-?}/${AWRM:-?}" \
              "$([ "${HEALTH_FAIL:-1}" -eq 0 ] && echo PASS || echo FAIL)" )
    done
    echo
    echo "---- 行为分解（末次 test_play，500 局）----"
    for x in "${ARMS[@]}"; do
        [ -f "$LOGDIR/$x/metrics.env" ] || continue
        ( . "$LOGDIR/$x/metrics.env"; echo "  arm=$x beta=$BETA  $BEHAVIOR" )
    done
    echo
    echo "---- 判读（对照=臂a β=3.0；阈值 avg_pt±$PT_THR / top1±$T1_TOL；探针级低功效）----"
    local cpt ct1
    cpt=$(grep -a 'avg pt:' "$LOGDIR/a/train.log" 2>/dev/null | tail -1 | sed 's/.*avg pt: //' | awk '{print $1}')
    ct1=$(grep -a 'steps=' "$LOGDIR/a/holdout_202609.log" 2>/dev/null | tail -1 | grep -aoE 'top1=[0-9.]+' | head -1 | cut -d= -f2)
    echo "  对照: avg_pt=$cpt top1=$ct1"
    for x in "${ARMS[@]}"; do
        [ -f "$LOGDIR/$x/metrics.env" ] || continue
        ( . "$LOGDIR/$x/metrics.env"
          if [ "${HEALTH_FAIL:-1}" -ne 0 ]; then
              echo "  arm=$x beta=$BETA: FAIL(health)"
          elif awk -v a="$AVG_PT" -v c="$cpt" -v t="$PT_THR" 'BEGIN{exit !(a>=c+t)}' \
            && awk -v a="$TOP1" -v c="$ct1" -v d="$T1_TOL" 'BEGIN{exit !(a>=c-d)}'; then
              echo "  arm=$x beta=$BETA: PROMISING → 建议正式轮（种子块 240000/250000，ladder 双向 ≥5000 sets）"
          elif awk -v a="$AVG_PT" -v c="$cpt" -v t="$PT_THR" 'BEGIN{exit !(a<=c-t)}' \
            || awk -v a="$TOP1" -v c="$ct1" -v d="$T1_TOL" 'BEGIN{exit !(a<c-d)}'; then
              echo "  arm=$x beta=$BETA: DEGRADED"
          else
              echo "  arm=$x beta=$BETA: NO-SEP（探针无分离）"
          fi )
    done
    echo
    echo "---- H3 专项（d vs a：若两臂指标同阶且 awr_w ess 同阶 → 权重本就≈均匀）----"
    echo "---- 旋钮生效检查：ess 应随 β 下降、p99 应随 β 上升（见上表 awr_w 列）----"
    } | tee -a "$CHAIN" > "$out"
    say "汇总已写入 $out"
}

say "==== β 扫描探针启动（a=3.0 对照 / b=1.0 / c=0.5 / d=10.0；臂 a/b/c→GPU2/3/4，臂 d 等 GPU0）===="
say "预注册：健康五门 + 旋钮生效(awr_w 单调) + avg_pt≥对照+$PT_THR 且 top1≥对照−$T1_TOL → PROMISING"
run_smoke
wp=()
for i in 0 1 2; do
    run_arm "${ARMS[$i]}" "${GPUS[$i]}" "${BETAS[$i]}" &
    wp+=($!)
done
# 臂 d（β=10）：等读牌模型 v2s@50k 让出 GPU0；超时回退"首个空闲 GPU"（仍避开 GPU1，除非万不得已）
if wait_gpu_free 0 "$GPU0_WAIT_MIN"; then
    say "[d] GPU0 已让出（读牌模型结束或腾出显存），臂 d 上 GPU0"
    run_arm d 0 10.0 &
    wp+=($!)
else
    say "[d] !! 等待 GPU0 超时（${GPU0_WAIT_MIN} 分钟），回退扫描空闲 GPU（避开 GPU1）"
    fb=""
    for g in 2 3 4 0; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        if [ -n "${used:-}" ] && [ "$used" -lt 500 ]; then fb=$g; break; fi
    done
    [ -z "$fb" ] && { say "[d] !! 无空闲 GPU，臂 d 改上 GPU1（热降频卡，预计 ~0.85 batch/s，仅兜底）"; fb=1; }
    say "[d] 臂 d 最终上 GPU$fb"
    run_arm d "$fb" 10.0 &
    wp+=($!)
fi
for p in "${wp[@]}"; do wait "$p"; done
summarize
say "ALL DONE"
