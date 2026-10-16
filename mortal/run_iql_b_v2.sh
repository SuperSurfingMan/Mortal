# IQL/Bellman Stage B 正式轮 v2 端到端链（2026-10-15，对应计划 S-B2 重跑）：
#   暖启 B/756k → objective='iql' 训练 +111k（→867k）→ holdout 诊断 → 双向 ladder vs B/756k
#   → block bootstrap 判定。唯一实质变量 = control.objective；数据/名单/增强/奖励 pts/GRP 全锁死 = B。
# 与 v1 的差异：① 全新命名空间 iql_b_v2（v1 产物含受损权重，封存不续用）② 新种子块
#   A=200000-/B=210000-（180000/190000 已被 v1 消费）③ reset 令牌 iql-b-v2。
# 前置条件（均已在启动前人工确认）：两处实现缺陷已修（r_t 仅局边界；L_V 只对数据动作）、
#   test_iql_data.py 八层 ALL PASS、冒烟 v3 10/10 通过（avg_rank 2.527 / top1 0.8134 / lse +1.343）。
# 判定：双向各 5×500 sets，bootstrap 95%CI 下界 > 0 判进；跨 0 = 无结论。
# 用法: tmux new -d -s iql_b_v2 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_b_v2.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k（父代+判定对照）
CK=checkpoints/iql_b_v2/mortal.pth
CFG=config.iql_b_v2.toml
CHAIN=logs/iql_b_v2_chain.log
GPUS=(2 0 1 3 4)                                                 # 训练=GPU2(RTX8000)；ladder 波内并行

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p iql_b_v2 checkpoints/iql_b_v2 logs

say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

say "==== 0) 预检 ===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > iql_b_v2/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 iql_b_v2/env_check.log）"; exit 1; }
say "env_check OK"
[ -f "$PARENT" ] || { say "!! 缺少父代 ckpt $PARENT"; exit 1; }
if [ ! -f "$CK" ]; then
    mkdir -p "$(dirname "$CK")"
    cp "$PARENT" "$CK"
    say "warm start: $PARENT → $CK"
else
    say "复用已有 $CK（续训）"
fi

say "==== 1) 训练（756k → 867k，GPU2 RTX 8000，fp32，3× encoder 前向，按冒烟吞吐 2.93 batch/s 估算）===="
(
    export CUDA_VISIBLE_DEVICES=2
    export MORTAL_CFG=$CFG
    log=iql_b_v2/train.log
    echo "---- $(date '+%F %T') iql_b_v2 train start ----" >> "$log"
    $PY -u train.py >> "$log" 2>&1
    echo "---- $(date '+%F %T') iql_b_v2 train exit=$? ----" >> "$log"
)
rc=$?
say "训练阶段结束 rc=$rc"
log=iql_b_v2/train.log
if grep -aqE 'non-finite|Traceback' "$log"; then say "!! non-finite/异常 → 终止判定链"; exit 1; fi
if ! grep -aq 'training budget reached' "$log"; then say "!! 未到 867k 预算 → 终止"; exit 1; fi
say "  OK  $(grep -aE 'total steps:' "$log" | tail -1)"

say "==== 2) 固定 holdout（202609，同 month/seed，仅诊断，不入强度结论）===="
CUDA_VISIBLE_DEVICES=4 MORTAL_CFG=$CFG $PY -u holdout_metrics.py \
    --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
    > iql_b_v2/holdout_202609.log 2>&1
say "  holdout: $(grep -aE 'top1|agari' iql_b_v2/holdout_202609.log | tail -3 | tr '\n' ' ')"
say "  注：objective='iql' → 无 MC 口径 q_target，R²/dqn_loss 不适用；只看 top1/agari 趋势。"

say "==== 3) 双向 ladder vs B/756k（新种子块 A=200000-/B=210000-，各 5×500 sets）===="
ladder_wave () {   # $1=tag $2=dir(A/B) $3=challenger $4=champion $5..=seeds
    local tag=$1 dir=$2 chal=$3 cham=$4; shift 4
    local root="iql_b_v2/ladder"; mkdir -p "$root"; local i=0; local wp=()
    for seed in "$@"; do
        (
            export MORTAL_CFG=$CFG CUDA_VISIBLE_DEVICES=${GPUS[$i]}
            out="$root/${dir}_${seed}"
            $PY -u ladder.py "$chal" "$cham" "$seed" 500 "$out" \
                --no-amp --agari-guard --tag "iql-b-v2-$dir-$seed" > "$out.log" 2>&1
            echo "  [$tag $dir $seed] $(grep -a 'avg_pt' "$out.log" | tail -1)" >> "$CHAIN"
        ) &
        wp+=($!); i=$((i+1))
    done
    for p in "${wp[@]}"; do wait "$p"; done
}

say "---- 波A（新当挑战者, seed 200000..202000）----"
ladder_wave iql_b_v2 A "$CK" "$PARENT" 200000 200500 201000 201500 202000
say "---- 波B（父代当挑战者, seed 210000..212000）----"
ladder_wave iql_b_v2 B "$PARENT" "$CK" 210000 210500 211000 211500 212000

say "==== 4) 汇总 + block bootstrap ===="
$PY ../tools/training/agg_ladder.py --dir "iql_b_v2/ladder" \
    --new "$CK" --old "$PARENT" --logs-root "iql_b_v2/ladder" \
    --challenger-name challenger --bootstrap 5000 --bseed 20261015 --workers 28 \
    > "iql_b_v2/ladder_summary.log" 2>&1
say "$(grep -aE '强度差|bootstrap δ|判读|A（|B（' iql_b_v2/ladder_summary.log | tr '\n' ' ')"

say "==== 5) 对照参考：B/756k δ vs round4/645k = +0.18；Stage A(867k) δ = −1.03（160000/170000）；v1 无效轮 δ = −31.04（实现缺陷，不作判决）===="
say "==== 6) 收尾守则："
say "  - 判进（CI 下界>0）→ Stage B 采纳为新基线与现役最强候选；补 cql on/off 归因消融。"
say "  - 无结论/变弱 → IQL 路线关闭，转扩窗 2019+ + 重训 GRP + 按年份审计（另立设计）。"
say "ALL DONE"
