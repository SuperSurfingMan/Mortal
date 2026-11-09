#!/usr/bin/env bash
# 标准化 ladder 臂：双向同牌山、按块并行、自动出 δ 与 bootstrap CI。
#
# 用法：
#   bash ladder_arm.sh <challenger.pth> <champion.pth> <out_dir> \
#        [--blocks 16] [--sets 500] [--seed-a 400000] [--seed-b 420000] \
#        [--guard-lambda 0] [--guard-risk-mode fixed|legacy] [--gpus 0,1,2,4] [--tag name]
#
# 约定（与 MEMORY §2.4 评测纪律一致）：
#   * 1 块 = 1 个种子 × 500 sets；**1 set = 4 局**（四个席位各坐一次）⇒ 500 sets = 2000 半庄。
#     A 向 = 挑战者当庄家位，B 向 = 轮换；δ = (A_avg_pt − B_avg_pt)/2；
#     **双向同号且 bootstrap CI 下界 > 0 才算赢**。
#   * 默认 16 块×500 = **8000 sets/向**（双向 16000 sets = 64000 半庄）
#     ⇒ 解析 SE≈**0.33**、可分辨 δ≈**0.66**（标定：5000 sets 双向 = 2500/向，SE 0.594）。
#     预算不足时脚本会打印「该预算能分辨的最小 δ」，避免拿 5000 sets 判 +0.3。
#   * 种子块一旦用过即消费；新实验必须换 SEED_A/SEED_B（台账见 MEMORY §4.2 备注）。
set -u -o pipefail

PY=${PY:-/mnt/data8/miniconda3/envs/mortal/bin/python}
CFG=${MORTAL_CFG:-config.toml}
cd "$(dirname "$0")"

[ $# -ge 3 ] || { sed -n 2,16p "$0"; exit 2; }
STATE=$1; BASE=$2; OUT=$3; shift 3
BLOCKS=${BLOCKS:-16}
SETS=${SETS:-500}
SEED_A=${SEED_A:-400000}
SEED_B=${SEED_B:-420000}
GUARD=${GUARD:-0}
RISK_MODE=${RISK_MODE:-fixed}
GPUS_CSV=${GPUS_CSV:-0,1,2,4}
TAG=${TAG:-arm}

while [ $# -gt 0 ]; do
    case "$1" in
        --blocks) BLOCKS=$2; shift 2;;
        --sets) SETS=$2; shift 2;;
        --seed-a) SEED_A=$2; shift 2;;
        --seed-b) SEED_B=$2; shift 2;;
        --guard-lambda) GUARD=$2; shift 2;;
        --guard-risk-mode) RISK_MODE=$2; shift 2;;
        --gpus) GPUS_CSV=$2; shift 2;;
        --tag) TAG=$2; shift 2;;
        *) echo "未知参数: $1" >&2; exit 2;;
    esac
done

[ -f "$STATE" ] || { echo "找不到挑战者 ckpt: $STATE" >&2; exit 1; }
[ -f "$BASE" ]  || { echo "找不到卫冕者 ckpt: $BASE" >&2; exit 1; }

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NGPU=${#GPUS[@]}
mkdir -p "$OUT"

# 预算 → 分辨率（标定：5000 sets 双向 = 2500 sets/向时 SE≈0.594；SE ∝ 1/√N）
"$PY" - "$BLOCKS" "$SETS" <<'EOF'
import sys, math
blocks, sets = int(sys.argv[1]), int(sys.argv[2])
n = blocks * sets                         # 每向 sets（1 set = 4 局）
se = 0.594 * math.sqrt(2500 / n)          # 解析 SE（经验标定，按每向）
min_d = 2 * se                             # 2σ 可分辨的最小 |δ|
print(f'[arm] {blocks} 块×{sets} sets = {n} sets / {4*n} 半庄/向 '
      f'（双向 {2*n} sets / {8*n} 半庄）→ 解析 SE≈{se:.2f}，可分辨 |δ|≈{min_d:.2f} pt')
if n < 8000:
    print(f'[arm] ⚠️ 低于 8000 sets/向（双向 16000）：只能看见 |δ|≥{min_d:.2f}，'
          f'小于该值的效应请勿下结论')
EOF

say () { echo "[$(date '+%F %T')] $*"; }

# $1=gpu $2=tag $3=challenger $4=champion $5=seed $6=out_dir
ladder_job () {
    ( CUDA_VISIBLE_DEVICES=$1 MORTAL_CFG=$CFG "$PY" -u ladder.py "$3" "$4" "$5" "$SETS" \
          "$6/$2" --guard-lambda "$GUARD" --guard-risk-mode "$RISK_MODE" > "$6/$2.log" 2>&1
      echo "[$2] $(grep -a 'avg_pt' "$6/$2.log" | tail -1)" ) &
}

say "==== ladder 臂 $TAG：$(basename "$STATE") vs $(basename "$BASE")（λ=$GUARD, risk=$RISK_MODE，双向 ${BLOCKS}×${SETS} sets）===="
i=0
for dir in A B; do
    say "-- $dir 方向（seed 起 $([ "$dir" = A ] && echo "$SEED_A" || echo "$SEED_B")）--"
    for ((k = 0; k < BLOCKS; k++)); do
        if [ "$dir" = A ]; then
            seed=$((SEED_A + k * 500))
            ladder_job "${GPUS[$((i % NGPU))]}" "A_${seed}" "$STATE" "$BASE" "$seed" "$OUT"
        else
            seed=$((SEED_B + k * 500))
            ladder_job "${GPUS[$((i % NGPU))]}" "B_${seed}" "$BASE" "$STATE" "$seed" "$OUT"
        fi
        i=$((i + 1))
        [ $((i % NGPU)) = 0 ] && wait
    done
    wait
done

say "==== 汇总（δ = (A − B)/2；CI 下界 >0 才算赢）===="
"$PY" ../tools/training/agg_ladder.py --dir "$OUT" --new "$STATE" --old "$BASE" \
    --logs-root "$OUT" --challenger-name challenger 2>&1 | tee "$OUT/summary.log"

say "==== 结论行 ===="
"$PY" - "$OUT/summary.log" <<'EOF'
import re, sys
txt = open(sys.argv[1], encoding='utf-8', errors='ignore').read()
d = re.search(r'δ\(新 − 旧\)\s*=\s*([-+0-9.]+)\s*pt[^\n]*?SE\(δ\)\s*=\s*([-+0-9.]+)[^\n]*?t\s*=\s*([-+0-9.]+)', txt)
b = re.search(r'bootstrap\s*δ\s*=\s*([-+0-9.]+)\s*pt[^\n]*?95%CI\s*=\s*\[\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)', txt)
if not (d and b):
    print('[verdict] 解析失败，请看 summary.log'); sys.exit(1)
delta, se, t = map(float, d.groups())
lo, hi = float(b.group(2)), float(b.group(3))
if lo > 0:
    v = 'WIN（bootstrap CI 下界 > 0）'
elif hi < 0:
    v = 'LOSE（bootstrap CI 上界 < 0）'
else:
    v = 'NO-CONCLUSION（CI 跨 0）'
print(f'[verdict] δ={delta:+.3f} pt  SE={se:.3f}  t={t:+.2f}  bootstrap CI=[{lo:+.2f},{hi:+.2f}]  → {v}')
EOF
say "产物：$OUT/summary.log"
