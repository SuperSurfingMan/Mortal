# S6a 正式轮端到端链（隔离版，2026-10-17，会话 29，对应 NEXT_STEPS §4bis）：
#   紧迫度特征注入：B/756k 首层手术（v4 -> v5，+6 通道零初始化）→ 训练 +111k（→867k）
#   → holdout → 双向 ladder vs B/756k → block bootstrap。
#   **代码隔离**：只使用 s6a_isolated/code/（含 v5 libriichi.so 与新版 model/engine/env_check）；
#   共享树 mortal/ 已还原为原版，iql_beta / read_model 等实验不受影响。
#   **GPU 隔离**：训练前等待 GPU2（RTX 8000）空闲（iql_beta 探针可能在用），超时 8h 放弃。
# 用法: tmux new -d -s s6a 'bash /mnt/data8/mortal/Mortal/s6a_isolated/run_s6a.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
ISO=$ROOT/s6a_isolated
CODE=$ISO/code
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # 经软链指向共享树 B/756k
CK=checkpoints/s6a/mortal.pth
CFG=config.s6a.toml
CHAIN=logs/s6a_chain.log
GPUS=(2 0 1 3 4)                                                 # 训练=GPU2(RTX8000)；ladder 波内并行
GPU2_FREE_MB=2000                                                # 训练前等待阈值
GPU2_WAIT_MIN=480                                                # 最长等待 8h

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export S6A_CODE=$CODE

cd "$CODE"
mkdir -p s6a checkpoints/s6a logs

say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

say "==== 0) 预检 + 回归测试（隔离 code 目录）===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > s6a/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 s6a/env_check.log）"; exit 1; }
say "env_check OK"
$PY "$ISO/tools/test_urgency_features.py" > s6a/test_urgency_features.log 2>&1 \
    || { say "!! 回归测试未通过，终止（见 s6a/test_urgency_features.log）"; exit 1; }
say "回归测试 ALL PASS（$(grep -c '\[OK\]' s6a/test_urgency_features.log) 项）"

say "==== 1) 首层手术（B/756k v4 -> v5，新通道零初始化）===="
[ -f "$PARENT" ] || { say "!! 缺少手术源 $PARENT"; exit 1; }
if [ ! -f "$CK" ]; then
    $PY "$ISO/tools/expand_ckpt_v5.py" "$PARENT" "$CK" > s6a/surgery.log 2>&1 \
        || { say "!! 手术失败（见 s6a/surgery.log）"; exit 1; }
    say "手术完成：$PARENT -> $CK"
else
    say "复用已有 $CK（续训）"
fi

say "==== 2) 等待 GPU2 空闲（避免与 iql_beta/read_model 等并行实验冲突）===="
waited=0
while :; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null | tr -d ' ')
    if [ -n "${mem:-}" ] && [ "$mem" -lt "$GPU2_FREE_MB" ]; then
        say "GPU2 空闲（mem=${mem}MiB），继续"
        break
    fi
    if [ "$waited" -ge "$GPU2_WAIT_MIN" ]; then
        say "!! 等待 GPU2 超时（${GPU2_WAIT_MIN}min，当前 mem=${mem:-?}MiB）→ 终止，可改 GPUS 配置手动启动"
        exit 1
    fi
    [ $((waited % 10)) -eq 0 ] && say "  GPU2 仍占用（mem=${mem:-?}MiB），已等 ${waited}min，最长 ${GPU2_WAIT_MIN}min"
    sleep 60; waited=$((waited + 1))
done

say "==== 3) 训练（756k -> 867k，GPU2 RTX 8000，fp32）===="
(
    export CUDA_VISIBLE_DEVICES=2
    export MORTAL_CFG=$CFG
    log=s6a/train.log
    echo "---- $(date '+%F %T') s6a train start ----" >> "$log"
    $PY -u train.py >> "$log" 2>&1
    echo "---- $(date '+%F %T') s6a train exit=$? ----" >> "$log"
)
rc=$?
say "训练阶段结束 rc=$rc"
log=s6a/train.log
if [ "$rc" -ne 0 ]; then say "!! 训练退出码非 0（rc=$rc）→ 终止判定链"; exit 1; fi
if grep -aqE 'non-finite|Traceback' "$log"; then say "!! non-finite/异常 → 终止判定链"; exit 1; fi
if ! grep -aq 'training budget reached' "$log"; then say "!! 未到 867k 预算 → 终止"; exit 1; fi
say "  OK  $(grep -aE 'total steps:' "$log" | tail -1)"

say "==== 4) 固定 holdout（202609，同 month/seed，仅诊断，不入强度结论）===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY -u holdout_metrics.py \
    --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
    > s6a/holdout_202609.log 2>&1
say "  holdout: $(grep -aE 'top1|agari' s6a/holdout_202609.log | tail -3 | tr '\n' ' ')"

say "==== 5) 双向 ladder vs B/756k（新种子块 A=240000- / B=250000-，各 5×500 sets）===="
ladder_wave () {   # $1=tag $2=dir(A/B) $3=challenger $4=champion $5..=seeds
    local tag=$1 dir=$2 chal=$3 cham=$4; shift 4
    local root="s6a/ladder"; mkdir -p "$root"; local i=0; local wp=()
    for seed in "$@"; do
        (
            export MORTAL_CFG=$CFG CUDA_VISIBLE_DEVICES=${GPUS[$i]}
            out="$root/${dir}_${seed}"
            $PY -u ladder.py "$chal" "$cham" "$seed" 500 "$out" \
                --no-amp --agari-guard --tag "s6a-$dir-$seed" > "$out.log" 2>&1
            echo "  [$tag $dir $seed] $(grep -a 'avg_pt' "$out.log" | tail -1)" >> "$CHAIN"
        ) &
        wp+=($!); i=$((i+1))
    done
    for p in "${wp[@]}"; do wait "$p"; done
}

say "---- 波A（新当挑战者, seed 240000..242000）----"
ladder_wave s6a A "$CK" "$PARENT" 240000 240500 241000 241500 242000
say "---- 波B（父代当挑战者, seed 250000..252000）----"
ladder_wave s6a B "$PARENT" "$CK" 250000 250500 251000 251500 252000

say "==== 6) 汇总 + block bootstrap ===="
$PY "$ROOT/tools/training/agg_ladder.py" --dir "s6a/ladder" \
    --new "$CK" --old "$PARENT" --logs-root "s6a/ladder" \
    --challenger-name challenger --bootstrap 5000 --bseed 20261017 --workers 28 \
    > s6a/ladder_summary.log 2>&1
say "$(grep -aE '强度差|bootstrap δ|判读|A（|B（' s6a/ladder_summary.log | tr '\n' ' ')"

say "==== 7) 判读守则（预注册）===="
say "  - 判进（CI 下界 > 0）→ 紧迫度特征有效，S6 路线成立，推进 S6b（读牌后验注入）。"
say "  - 无结论 → 紧迫度不是缺口或形式需调（改用 GRP 顺位概率差 / 加权安全边际），先小消融再决定。"
say "  - 显著变弱 → 特征形式有害（查归一化/冗余），回滚 version=4 并记录。"
say "ALL DONE"
