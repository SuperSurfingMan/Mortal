# IQL/Bellman Stage B 正式轮 v3 端到端链（2026-10-16，对应计划 S-B2 第二轮）：
#   暖启 B/756k → objective='iql'（+反漂移修复）训练 +111k（→867k）→ holdout → 双向 ladder vs B/756k
#   → block bootstrap。唯一实质变量 = v2 的 objective 基础上叠加探针选出的最小反漂移修复；
#   数据/名单/增强/奖励 pts/GRP 全锁死 = B。
# 与 v2 的差异：① 反漂移修复（config.iql_b_v3.toml，由 prepare_iql_b_v3.py 生成）② 新种子块
#   A=220000-/B=230000-（200000/210000 已被 v2 消费）③ reset 令牌 iql-b-v3 ④ seed 20261016。
# 前置条件：4 臂探针（run_iql_probe.sh）已判定哪项修复 30k 步内价值尺度稳定（p50 ≤ 8）。
# 用法: tmux new -d -s iql_b_v3 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_b_v3.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k（父代+判定对照）
CK=checkpoints/iql_b_v3/mortal.pth
CFG=config.iql_b_v3.toml
CHAIN=logs/iql_b_v3_chain.log
GPUS=(2 0 1 3 4)                                                 # 训练=GPU2(RTX8000)；ladder 波内并行

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p iql_b_v3 checkpoints/iql_b_v3 logs

say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

say "==== 0) 预检 ===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > iql_b_v3/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 iql_b_v3/env_check.log）"; exit 1; }
say "env_check OK"
[ -f "$PARENT" ] || { say "!! 缺少父代 ckpt $PARENT"; exit 1; }
if [ ! -f "$CK" ]; then
    mkdir -p "$(dirname "$CK")"
    cp "$PARENT" "$CK"
    say "warm start: $PARENT → $CK"
else
    say "复用已有 $CK（续训）"
fi

say "==== 1) 训练（756k → 867k，GPU2 RTX 8000，fp32；反漂移修复见 config.iql_b_v3.toml 头注）===="
(
    export CUDA_VISIBLE_DEVICES=2
    export MORTAL_CFG=$CFG
    log=iql_b_v3/train.log
    echo "---- $(date '+%F %T') iql_b_v3 train start ----" >> "$log"
    $PY -u train.py >> "$log" 2>&1
    echo "---- $(date '+%F %T') iql_b_v3 train exit=$? ----" >> "$log"
)
rc=$?
say "训练阶段结束 rc=$rc"
log=iql_b_v3/train.log
if grep -aqE 'non-finite|Traceback' "$log"; then say "!! non-finite/异常 → 终止判定链"; exit 1; fi
if ! grep -aq 'training budget reached' "$log"; then say "!! 未到 867k 预算 → 终止"; exit 1; fi
say "  OK  $(grep -aE 'total steps:' "$log" | tail -1)"
sc_max=$(grep -a 'iql scale:' "$log" | grep -aoE 'q_target p50=[-+0-9.]+' | cut -d= -f2 \
    | awk 'BEGIN{m=-1e9}{v=$1+0; if(v>m)m=v}END{printf "%.3f", m}')
say "  价值尺度: max q_target p50=$sc_max（v2 同点为 +60.8；修复有效应 ≤ ~5）"

say "==== 2) 固定 holdout（202609，同 month/seed，仅诊断，不入强度结论）===="
CUDA_VISIBLE_DEVICES=4 MORTAL_CFG=$CFG $PY -u holdout_metrics.py \
    --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
    > iql_b_v3/holdout_202609.log 2>&1
say "  holdout: $(grep -aE 'top1|agari' iql_b_v3/holdout_202609.log | tail -3 | tr '\n' ' ')"

say "==== 3) 双向 ladder vs B/756k（新种子块 A=220000-/B=230000-，各 5×500 sets）===="
ladder_wave () {   # $1=tag $2=dir(A/B) $3=challenger $4=champion $5..=seeds
    local tag=$1 dir=$2 chal=$3 cham=$4; shift 4
    local root="iql_b_v3/ladder"; mkdir -p "$root"; local i=0; local wp=()
    for seed in "$@"; do
        (
            export MORTAL_CFG=$CFG CUDA_VISIBLE_DEVICES=${GPUS[$i]}
            out="$root/${dir}_${seed}"
            $PY -u ladder.py "$chal" "$cham" "$seed" 500 "$out" \
                --no-amp --agari-guard --tag "iql-b-v3-$dir-$seed" > "$out.log" 2>&1
            echo "  [$tag $dir $seed] $(grep -a 'avg_pt' "$out.log" | tail -1)" >> "$CHAIN"
        ) &
        wp+=($!); i=$((i+1))
    done
    for p in "${wp[@]}"; do wait "$p"; done
}

say "---- 波A（新当挑战者, seed 220000..222000）----"
ladder_wave iql_b_v3 A "$CK" "$PARENT" 220000 220500 221000 221500 222000
say "---- 波B（父代当挑战者, seed 230000..232000）----"
ladder_wave iql_b_v3 B "$PARENT" "$CK" 230000 230500 231000 231500 232000

say "==== 4) 汇总 + block bootstrap ===="
$PY ../tools/training/agg_ladder.py --dir "iql_b_v3/ladder" \
    --new "$CK" --old "$PARENT" --logs-root "iql_b_v3/ladder" \
    --challenger-name challenger --bootstrap 5000 --bseed 20261016 --workers 28 \
    > "iql_b_v3/ladder_summary.log" 2>&1
say "$(grep -aE '强度差|bootstrap δ|判读|A（|B（' iql_b_v3/ladder_summary.log | tr '\n' ' ')"

say "==== 5) 对照参考：v2（无修复）holdout top1 0.8069/lse +45；Stage A δ=−1.03；B vs round4/645k δ=+0.18===="
say "==== 6) 收尾守则："
say "  - 判进（CI 下界>0）→ 稳定版 IQL 采纳为新基线与现役最强候选。"
say "  - 无结论/变弱 → IQL 路线关闭（价值稳定条件下的干净否证），转扩窗 2019+ + 重训 GRP + 按年份审计。"
say "ALL DONE"
