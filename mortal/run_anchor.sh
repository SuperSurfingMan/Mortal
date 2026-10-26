#!/usr/bin/env bash
# 绝对锚点：Mortal（1 席） vs 3×akochan，多进程并行。
#
# 用法：
#   bash run_anchor.sh [--procs 6] [--games-per-proc 68] [--seed-key 20261026] [--gpus 0,1]
#
# 说明：
#   * 每个进程 = 一个独立的 one_vs_three（4 局并行），进程间用不同 seed_key 换牌山；
#   * games-per-proc 必须是 4 的倍数（1 set = 4 半庄）；
#   * 指标：`challenger rankings` 就是 **Mortal 自己的名次分布**（py_vs_ako），
#     avg_pt 为 uma 口径（90/45/0/−135），与 dataset/index/player_ratings_*.csv 同尺度；
#   * 汇总时把各进程的 rankings 相加，再算总 avg_rank / avg_pt。
set -u -o pipefail

PY=${PY:-/mnt/data8/miniconda3/envs/mortal/bin/python}
cd "$(dirname "$0")"

PROCS=${PROCS:-6}
GPP=${GPP:-68}
SEED_KEY=${SEED_KEY:-20261026}
GPUS_CSV=${GPUS_CSV:-0,1}
TAG=${TAG:-anchor1}

while [ $# -gt 0 ]; do
    case "$1" in
        --procs) PROCS=$2; shift 2;;
        --games-per-proc) GPP=$2; shift 2;;
        --seed-key) SEED_KEY=$2; shift 2;;
        --gpus) GPUS_CSV=$2; shift 2;;
        --tag) TAG=$2; shift 2;;
        *) echo "未知参数 $1" >&2; exit 2;;
    esac
done
[ $((GPP % 4)) = 0 ] || { echo "games-per-proc 必须是 4 的倍数" >&2; exit 2; }
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NGPU=${#GPUS[@]}

OUT=logs/$TAG
mkdir -p "$OUT"
say () { echo "[$(date '+%F %T')] $*"; }
say "锚点启动：$PROCS 进程 × $GPP 局 = $((PROCS*GPP)) 局（Mortal×1 vs akochan×3）"

"$PY" - "$PROCS" "$GPP" "$SEED_KEY" "$TAG" "$NGPU" <<'EOF'
import sys, pathlib
procs, gpp, key, tag, ngpu = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
src = pathlib.Path('config.anchor_ako.toml').read_text()
import re
for i in range(procs):
    s = src
    s = re.sub(r"\[1v3\]\nseed_key = -?\d+\ngames_per_iter = \d+\niters = \d+\nlog_dir = '[^']*'",
               f"[1v3]\nseed_key = {key + i}\ngames_per_iter = 4\niters = {gpp // 4}\n"
               f"log_dir = 'runs/{tag}/p{i}'\nmortal_vs_ako = true", s, count=1)
    s = re.sub(r"^num_workers = 8$", "num_workers = 2", s, count=1, flags=re.M)
    assert 'mortal_vs_ako = true' in s, 'patch 失败'
    pathlib.Path(f'config.{tag}_p{i}.toml').write_text(s)
print('configs:', ', '.join(f'config.{tag}_p{i}.toml' for i in range(procs)))
EOF

for ((i = 0; i < PROCS; i++)); do
    gpu=${GPUS[$((i % NGPU))]}
    sess="anchor_${TAG}_p${i}"
    tmux kill-session -t "$sess" 2>/dev/null
    tmux new -d -s "$sess" "cd $(pwd) && CUDA_VISIBLE_DEVICES=$gpu MORTAL_CFG=config.${TAG}_p${i}.toml \
        $PY -u one_vs_three.py > $OUT/p${i}.log 2>&1; echo RC=\$? >> $OUT/p${i}.log"
    say "  $sess  GPU=$gpu  log=$OUT/p${i}.log"
done

say "已起 $PROCS 个进程；汇总：python agg_anchor.py --dir $OUT"
