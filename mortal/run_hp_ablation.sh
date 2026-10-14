# 便宜前置实验（超参消融）端到端链：并行训练 3 分支 → holdout 诊断 → 双向 ladder → block bootstrap。
# 唯一变量：cql.min_q_weight 1/10、aux.next_rank_weight 0；控制组 = B 组(cql5/aux0.2, 756k)。
# 全部严格复刻 B：父代 round4/645k、top400_v2、[2,1,0,-3]、augmentation=true、+111k→756k、reset_optimizer。
# 用法: tmux new -d -s hp_chain 'bash /mnt/data8/mortal/Mortal/mortal/run_hp_ablation.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal

# ==== 每次跑前完整重设全部环境变量 ====
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
mkdir -p logs hp_ablation

TAGS=(cql_min1 cql_min10 aux_next0)
declare -A TRAIN_GPU=([cql_min1]=0 [cql_min10]=1 [aux_next0]=2)   # 0/1=P100, 2=RTX8000(快)
GPUS=(2 0 1 3 4)                                                  # ladder 波内并行用卡
PARENT=checkpoints/round4/mortal.pth
CHAIN=logs/hp_ablation_chain.log
cfg_of () { echo "$PWD/hp_ablation/$1/config.toml"; }
ck_of  () { echo "checkpoints/hp_ablation/$1/mortal.pth"; }

say () { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

say "==== 0) 预检 + 校验三配置可解析 ===="
for t in "${TAGS[@]}"; do
    CUDA_VISIBLE_DEVICES=${TRAIN_GPU[$t]} MORTAL_CFG="$(cfg_of $t)" \
        $PY env_check.py --cfg "hp_ablation/$t/config.toml" >>"$CHAIN" 2>&1 \
        || { say "!! $t env_check 未通过，终止"; exit 1; }
    [ -f "$(ck_of $t)" ] || { say "!! 缺少暖启 ckpt: $(ck_of $t)"; exit 1; }
done

say "==== 1) 并行训练 3 分支（各 +111k → 756k，父代 round4/645k）===="
pids=()
for t in "${TAGS[@]}"; do
    (
        export CUDA_VISIBLE_DEVICES=${TRAIN_GPU[$t]}
        export MORTAL_CFG="$(cfg_of $t)"
        log="hp_ablation/$t/train.log"
        echo "---- $(date '+%F %T') $t train start gpu=${TRAIN_GPU[$t]} ----" >> "$log"
        $PY -u train.py >> "$log" 2>&1
        echo "---- $(date '+%F %T') $t train exit=$? ----" >> "$log"
    ) &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
say "训练阶段结束 rc=$rc"
for t in "${TAGS[@]}"; do
    log="hp_ablation/$t/train.log"
    step=$(grep -aE 'total steps:' "$log" | tail -1)
    if grep -aqE 'non-finite|Traceback' "$log"; then say "!! $t 出现 non-finite/异常 → 终止判定链"; exit 1; fi
    if ! grep -aq 'training budget reached' "$log"; then say "!! $t 未到 756k 预算 → 终止"; exit 1; fi
    say "  $t OK  $step"
done

say "==== 2) 固定 holdout（202609，同 month/seed，仅诊断，不入强度结论）===="
for t in "${TAGS[@]}"; do
    CUDA_VISIBLE_DEVICES=4 MORTAL_CFG="$(cfg_of $t)" $PY -u holdout_metrics.py \
        --month 202609 --n 24 --seed 20261009 --workers 1 "$(ck_of $t)" \
        > "hp_ablation/$t/holdout_202609.log" 2>&1
    say "  $t holdout: $(grep -aE 'top1|R2|acc' hp_ablation/$t/holdout_202609.log | tail -3 | tr '\n' ' ')"
done

say "==== 3) 双向 ladder vs round4/645k（新种子块 A=140000-/B=150000-，各 5×500 sets）===="
ladder_wave () {   # $1=tag $2=dir(A/B) $3=challenger $4=champion $5..=seeds
    local tag=$1 dir=$2 chal=$3 cham=$4; shift 4
    local root="hp_ablation/$tag/ladder"; mkdir -p "$root"; local i=0; local wp=()
    for seed in "$@"; do
        (
            export MORTAL_CFG="$(cfg_of $tag)" CUDA_VISIBLE_DEVICES=${GPUS[$i]}
            out="$root/${dir}_${seed}"
            $PY -u ladder.py "$chal" "$cham" "$seed" 500 "$out" \
                --no-amp --agari-guard --tag "$tag-$dir-$seed" > "$out.log" 2>&1
            echo "  [$tag $dir $seed] $(grep -a 'avg_pt' "$out.log" | tail -1)"
        ) &
        wp+=($!); i=$((i+1))
    done
    for p in "${wp[@]}"; do wait "$p"; done
}

for t in "${TAGS[@]}"; do
    NEW=$(ck_of $t); OLD=$PARENT
    say "---- $t 波A（新当挑战者, seed 140000..142000）----"
    ladder_wave "$t" A "$NEW" "$OLD" 140000 140500 141000 141500 142000
    say "---- $t 波B（父代当挑战者, seed 150000..152000）----"
    ladder_wave "$t" B "$OLD" "$NEW" 150000 150500 151000 151500 152000
    say "---- $t 汇总 + block bootstrap ----"
    $PY ../tools/training/agg_ladder.py --dir "hp_ablation/$t/ladder" \
        --new "$NEW" --old "$OLD" --logs-root "hp_ablation/$t/ladder" \
        --challenger-name challenger --bootstrap 5000 --bseed 20261013 --workers 28 \
        > "hp_ablation/$t/ladder_summary.log" 2>&1
    say "$(grep -aE '强度差|bootstrap δ|判读|A（|B（' "hp_ablation/$t/ladder_summary.log" | tr '\n' ' ')"
done

say "==== 4) 对照参考：B 组本身(控制) δ vs round4/645k = +0.18（种子块 100000/110000，见 reward_ablation）===="
say "ALL DONE"
