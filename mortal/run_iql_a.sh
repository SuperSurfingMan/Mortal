# IQL/Bellman Stage A 端到端链（2026-10-14，对应计划 S-A2）：
#   暖启 B/756k → credit_mode='game' 训练 +111k（→867k）→ holdout 诊断 → 双向 ladder vs B/756k
#   → block bootstrap 判定。唯一实质变量 = env.credit_mode；数据/名单/增强/奖励 pts/GRP 全锁死 = B。
# 判定：新种子块 160000/170000（140000/150000 已被 hp_ablation 用掉），双向各 5×500 sets，
#   bootstrap 95%CI 下界 > 0 判进；跨 0 = 无结论。
# 用法: tmux new -d -s iql_a 'bash /mnt/data8/mortal/Mortal/mortal/run_iql_a.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
PARENT=checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth   # B/756k（父代+判定对照）
CK=checkpoints/iql_a/mortal.pth
CFG=config.iql_a.toml
CHAIN=logs/iql_a_chain.log
GPUS=(2 0 1 3 4)                                                 # 训练=GPU2(RTX8000)；ladder 波内并行

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p iql_a checkpoints/iql_a logs

say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

say "==== 0) 预检 ===="
CUDA_VISIBLE_DEVICES=2 MORTAL_CFG=$CFG $PY env_check.py --cfg "$CFG" \
    > iql_a/env_check.log 2>&1 \
    || { say "!! env_check 未通过，终止（见 iql_a/env_check.log）"; exit 1; }
say "env_check OK"
[ -f "$PARENT" ] || { say "!! 缺少父代 ckpt $PARENT"; exit 1; }
if [ ! -f "$CK" ]; then
    mkdir -p "$(dirname "$CK")"
    cp "$PARENT" "$CK"
    say "warm start: $PARENT → $CK"
else
    say "复用已有 $CK（续训）"
fi

say "==== 1) 训练（756k → 867k，GPU2 RTX 8000，fp32 ≈4.28 it/s，预计 ≈7.3h）===="
(
    export CUDA_VISIBLE_DEVICES=2
    export MORTAL_CFG=$CFG
    log=iql_a/train.log
    echo "---- $(date '+%F %T') iql_a train start ----" >> "$log"
    $PY -u train.py >> "$log" 2>&1
    echo "---- $(date '+%F %T') iql_a train exit=$? ----" >> "$log"
)
rc=$?
say "训练阶段结束 rc=$rc"
log=iql_a/train.log
if grep -aqE 'non-finite|Traceback' "$log"; then say "!! non-finite/异常 → 终止判定链"; exit 1; fi
if ! grep -aq 'training budget reached' "$log"; then say "!! 未到 867k 预算 → 终止"; exit 1; fi
say "  OK  $(grep -aE 'total steps:' "$log" | tail -1)"

say "==== 2) 固定 holdout（202609，同 month/seed，仅诊断，不入强度结论）===="
CUDA_VISIBLE_DEVICES=4 MORTAL_CFG=$CFG $PY -u holdout_metrics.py \
    --month 202609 --n 24 --seed 20261009 --workers 1 "$CK" \
    > iql_a/holdout_202609.log 2>&1
say "  holdout: $(grep -aE 'top1|R2|acc' iql_a/holdout_202609.log | tail -3 | tr '\n' ' ')"
say "  注：credit_mode='game' → holdout_target_digest 与 B 口径不同，属预期；只比 top1 趋势。"

say "==== 3) 双向 ladder vs B/756k（新种子块 A=160000-/B=170000-，各 5×500 sets）===="
ladder_wave () {   # $1=tag $2=dir(A/B) $3=challenger $4=champion $5..=seeds
    local tag=$1 dir=$2 chal=$3 cham=$4; shift 4
    local root="iql_a/ladder"; mkdir -p "$root"; local i=0; local wp=()
    for seed in "$@"; do
        (
            export MORTAL_CFG=$CFG CUDA_VISIBLE_DEVICES=${GPUS[$i]}
            out="$root/${dir}_${seed}"
            $PY -u ladder.py "$chal" "$cham" "$seed" 500 "$out" \
                --no-amp --agari-guard --tag "iql-a-$dir-$seed" > "$out.log" 2>&1
            echo "  [$tag $dir $seed] $(grep -a 'avg_pt' "$out.log" | tail -1)" >> "$CHAIN"
        ) &
        wp+=($!); i=$((i+1))
    done
    for p in "${wp[@]}"; do wait "$p"; done
}

say "---- 波A（新当挑战者, seed 160000..162000）----"
ladder_wave iql_a A "$CK" "$PARENT" 160000 160500 161000 161500 162000
say "---- 波B（父代当挑战者, seed 170000..172000）----"
ladder_wave iql_a B "$PARENT" "$CK" 170000 170500 171000 171500 172000

say "==== 4) 汇总 + block bootstrap ===="
$PY ../tools/training/agg_ladder.py --dir "iql_a/ladder" \
    --new "$CK" --old "$PARENT" --logs-root "iql_a/ladder" \
    --challenger-name challenger --bootstrap 5000 --bseed 20261014 --workers 28 \
    > "iql_a/ladder_summary.log" 2>&1
say "$(grep -aE '强度差|bootstrap δ|判读|A（|B（' iql_a/ladder_summary.log | tr '\n' ' ')"

say "==== 5) 对照参考：B 组本身 δ vs round4/645k = +0.18（种子块 100000/110000，见 reward_ablation）===="
say "==== 6) 收尾守则："
say "  - 判进（CI 下界>0）→ Stage A 采纳为新基线；Stage B（真 IQL）以较强者为父代。"
say "  - 跨 0 → 无结论，不采纳；直接转 Stage B（真 IQL：next_obs + V 网络 + expectile + AWR）。"
say "ALL DONE"
