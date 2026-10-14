# hp_ablation 续训+判定链（因 GPU1 热降频改挪凉卡）：
#   1) 续训未完成分支 cql_min1(GPU4 凉 P100)/cql_min10(GPU2 RTX 凉) 到 756k（原地续，不重跑 prepare）
#   2) 三条分支（含已完成的 aux_next0）跑 holdout + 双向 ladder(新种子 140000/150000) + block bootstrap
# 配置不变：沿用各分支 hp_ablation/<tag>/config.toml（state_file 即现存 mortal.pth → train.py 自动续训）；
# reset_optimizer/reset_best_perf 令牌已在 ckpt 记录，续训不会二次触发。
# 用法: tmux new -d -s hp_resume 'bash /mnt/data8/mortal/Mortal/mortal/resume_and_judge_hp.sh'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
MORTAL=$ROOT/mortal
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$MORTAL"
TAGS=(aux_next0 cql_min1 cql_min10)          # 判定用全部三条
RESUME=(cql_min1 cql_min10)                  # 仅这两条需续训（aux 已 756k）
declare -A RGPU=([cql_min1]=4 [cql_min10]=2) # 凉卡：4=P100(32°C) 2=RTX(30°C)
GPUS=(2 0 1 3 4)
PARENT=checkpoints/round4/mortal.pth
CHAIN=logs/hp_resume_chain.log
cfg_of(){ echo "$PWD/hp_ablation/$1/config.toml"; }
ck_of(){ echo "checkpoints/hp_ablation/$1/mortal.pth"; }
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

say "==== 0) 校验现存 ckpt 完整性与当前 steps（应 >645k 且 <756k，方可原地续训）===="
$PY - <<'PY'
import torch
for t in ['aux_next0','cql_min1','cql_min10']:
    c=torch.load(f'checkpoints/hp_ablation/{t}/mortal.pth',map_location='cpu',weights_only=True)
    print(f'  {t}: steps={c.get("steps")}')
PY

say "==== 1) 续训未完成分支 → 756k（凉卡，原地续训）===="
pids=()
for t in "${RESUME[@]}"; do
    (
        export CUDA_VISIBLE_DEVICES=${RGPU[$t]} MORTAL_CFG="$(cfg_of $t)"
        log="hp_ablation/$t/train.log"
        echo "---- $(date '+%F %T') $t RESUME start gpu=${RGPU[$t]} (凉卡续训) ----" >> "$log"
        $PY -u train.py >> "$log" 2>&1
        echo "---- $(date '+%F %T') $t RESUME exit=$? ----" >> "$log"
    ) &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
say "续训阶段结束 rc=$rc"
for t in "${RESUME[@]}"; do
    log="hp_ablation/$t/train.log"
    grep -aqE 'non-finite|Traceback' "$log" && { say "!! $t non-finite/异常 → 终止"; exit 1; }
    grep -aq 'training budget reached' "$log" || { say "!! $t 未到 756k → 终止"; exit 1; }
    say "  $t OK  $(grep -aE 'total steps:' "$log" | tail -1)"
done

say "==== 2) 固定 holdout（202609，同 month/seed，仅诊断）===="
for t in "${TAGS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 MORTAL_CFG="$(cfg_of $t)" $PY -u holdout_metrics.py \
        --month 202609 --n 24 --seed 20261009 --workers 1 "$(ck_of $t)" \
        > "hp_ablation/$t/holdout_202609.log" 2>&1
    say "  $t holdout: $(grep -aE 'top1|R2|acc|digest' hp_ablation/$t/holdout_202609.log | tail -3 | tr '\n' ' ')"
done

say "==== 3) 双向 ladder vs round4/645k（新种子 A=140000-/B=150000-，各 5×500 sets）===="
ladder_wave(){ local tag=$1 dir=$2 chal=$3 cham=$4; shift 4
    local root="hp_ablation/$tag/ladder"; mkdir -p "$root"; local i=0; local wp=()
    for seed in "$@"; do
        ( export MORTAL_CFG="$(cfg_of $tag)" CUDA_VISIBLE_DEVICES=${GPUS[$i]}
          out="$root/${dir}_${seed}"
          $PY -u ladder.py "$chal" "$cham" "$seed" 500 "$out" \
              --no-amp --agari-guard --tag "$tag-$dir-$seed" > "$out.log" 2>&1
          echo "  [$tag $dir $seed] $(grep -a 'avg_pt' "$out.log" | tail -1)" ) &
        wp+=($!); i=$((i+1))
    done
    for p in "${wp[@]}"; do wait "$p"; done; }

for t in "${TAGS[@]}"; do
    NEW=$(ck_of $t); OLD=$PARENT
    say "---- $t 波A（新当挑战者）----"
    ladder_wave "$t" A "$NEW" "$OLD" 140000 140500 141000 141500 142000
    say "---- $t 波B（父代当挑战者）----"
    ladder_wave "$t" B "$OLD" "$NEW" 150000 150500 151000 151500 152000
    say "---- $t 汇总 + bootstrap ----"
    $PY ../tools/training/agg_ladder.py --dir "hp_ablation/$t/ladder" \
        --new "$NEW" --old "$OLD" --logs-root "hp_ablation/$t/ladder" \
        --challenger-name challenger --bootstrap 5000 --bseed 20261013 --workers 28 \
        > "hp_ablation/$t/ladder_summary.log" 2>&1
    say "$(grep -aE '强度差|bootstrap δ|判读|A（|B（' "hp_ablation/$t/ladder_summary.log" | tr '\n' ' ')"
done

say "==== 4) 对照：B 组(控制) δ vs round4/645k = +0.18（种子块 100000/110000）===="
say "ALL DONE"
