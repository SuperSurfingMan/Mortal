#!/usr/bin/env bash
# Oracle Guiding + 退火 的最小实现编排（2026-11-03）
#
# 阶段：
#   surgery   把 S6c/956k 扩成 +217 行 oracle 输入的起手 ckpt，并做透明性校验
#   smoke     956k + 1k 步冒烟（验证 dataloader/α/保存/载入/Holdout 全链路）
#   probe     956k + 100k 步探针（α: 1.0 → 0），产出 holdout 门禁数字
#   all       surgery → smoke → probe
#
# 用法（tmux 承载，GPU2）：
#   tmux new -d -s og "bash /mnt/data8/mortal/Mortal/mortal/run_og.sh all"
#   STAGE=smoke bash run_og.sh
#   注意：脚本运行期间不要改本文件（bash 增量读取会报假语法错）。
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
CFG=config.og_probe.toml
CFG_SMOKE=config.og_smoke.toml
BASE=checkpoints/s6c/mortal.pth            # S6c/956k 父代
INIT=checkpoints/og/mortal.pth             # 手术产物 + 探针 state_file
INIT_SMOKE=checkpoints/og_smoke/mortal.pth
LOGS=logs/og
STAGE=${STAGE:-all}

cd "$ROOT/mortal" || { echo "缺少 $ROOT/mortal"; exit 1; }
mkdir -p "$LOGS" checkpoints/og checkpoints/og_smoke

export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

say () { echo "[$(date '+%F %T')] $*"; }
want () { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
holdout () {  # $1=cfg $2=ckpt $3=tag
    MORTAL_CFG="$1" "$PY" -u holdout_metrics.py "$2" --n 24 --workers 8 2>&1 \
        | tee "$LOGS/holdout_$3.log"
}
metric_line () { grep -a 'top1=' "$1" | tail -1 | sed 's/^[^ ]* *//'; }
digest_line () { grep -a 'holdout_sample_digest' "$1" | tail -1; }

alpha_test () {
    say '==== α 调度自检 ===='
    "$PY" - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from train import _oracle_alpha
cfg = {'alpha_start': 1.0, 'alpha_end': 0.0,
       'warmup_steps': 10000, 'anneal_steps': 50000}
vals = [round(_oracle_alpha(cfg, s), 3)
        for s in (0, 9999, 10000, 35000, 60000, 100000)]
assert vals == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0], vals
print('alpha schedule OK:', vals)
PYEOF
    [ $? = 0 ] || { echo 'α 调度错误，终止'; exit 1; }
}

surgery () {
    say '==== 1) 首层手术（S6c/956k → +217 行 oracle，零初始化）===='
    [ -f "$BASE" ] || { echo "缺少父代 $BASE"; exit 1; }
    if [ ! -f "$INIT" ]; then
        "$PY" -u ../tools/expand_ckpt_oracle.py "$BASE" "$INIT" 2>&1 | tee "$LOGS/surgery.log"
        [ "${PIPESTATUS[0]}" = 0 ] || { echo '手术失败，终止'; exit 1; }
    else
        echo "已存在 $INIT，跳过手术"
    fi
}

transparent () {
    say '==== 2) 透明性校验（α=0 时 holdout 必须与 S6c/956k 逐位相同）===='
    holdout "$CFG" "$BASE" base
    holdout "$CFG" "$INIT" init
    local l_base l_init d_base d_init
    l_base=$(metric_line "$LOGS/holdout_base.log")
    l_init=$(metric_line "$LOGS/holdout_init.log")
    d_base=$(digest_line "$LOGS/holdout_base.log")
    d_init=$(digest_line "$LOGS/holdout_init.log")
    echo "  S6c/956k : $l_base"
    echo "  OG 初始化: $l_init"
    [ "$d_base" = "$d_init" ] || { echo "FAIL 抽样摘要不同：$d_base / $d_init"; exit 1; }
    [ "$l_base" = "$l_init" ] || { echo 'FAIL 手术 ckpt 与父代 holdout 不一致 → 管线不透明'; exit 1; }
    say '透明性校验通过（top1/dqn_loss/agari 全部一致）'
}

smoke () {
    say '==== 3) 冒烟（956k + 1k 步）===='
    cp -f "$INIT" "$INIT_SMOKE"
    local t0=$(date +%s)
    MORTAL_CFG="$CFG_SMOKE" "$PY" -u train.py > "$LOGS/smoke.log" 2>&1
    local dt=$(( $(date +%s) - t0 ))
    grep -aE 'obs shape|oracle guiding|oracle alpha|training budget reached|non-finite|Traceback|Error' \
        "$LOGS/smoke.log" | tail -10
    grep -aq 'training budget reached' "$LOGS/smoke.log" || { echo '冒烟未跑到预算，终止'; exit 1; }
    grep -aq 'oracle alpha' "$LOGS/smoke.log" || { echo '没有 alpha 日志，检查调度是否生效'; exit 1; }
    say "冒烟完成，用时 ${dt}s"
    holdout "$CFG" "$INIT_SMOKE" smoke
}

probe () {
    say '==== 4) 100k 探针 ===='
    [ -f "$INIT" ] || { echo "缺少 $INIT（先跑 STAGE=surgery）"; exit 1; }
    local t0=$(date +%s)
    MORTAL_CFG="$CFG" "$PY" -u train.py > "$LOGS/probe_train.log" 2>&1
    local dt=$(( $(date +%s) - t0 ))
    grep -aE 'obs shape|oracle guiding|oracle alpha|training budget reached|non-finite|Traceback|Error' \
        "$LOGS/probe_train.log" | tail -12
    grep -aq 'training budget reached' "$LOGS/probe_train.log" || { echo '探针未跑到预算，终止'; exit 1; }
    grep -a 'behavior:' "$LOGS/probe_train.log" | tail -2
    say "探针训练完成，用时 ${dt}s"

    holdout "$CFG" "$INIT" probe
    # 门禁：α=0 时 top1 不退（−0.3pp 内）
    "$PY" - "$LOGS/holdout_probe.log" "$LOGS/holdout_base.log" 0.003 <<'PYEOF'
import re, sys

def top1(path):
    txt = open(path, encoding='utf-8', errors='ignore').read()
    m = re.findall(r'(?:^|\s)top1=([0-9.]+)', txt, flags=re.M)
    return float(m[-1])

probe, base, tol = top1(sys.argv[1]), top1(sys.argv[2]), float(sys.argv[3])
print(f'probe top1={probe:.4f}  baseline top1={base:.4f}  delta={probe - base:+.4f}  容差={tol:.4f}')
if probe < base - tol:
    print('GATE FAIL: α=0 的 holdout top1 明显退步')
    sys.exit(1)
print('GATE OK: α=0 的 holdout 无显著退步 → 可上 16k 标准臂')
PYEOF
    local gate=$?
    grep -a 'dqn_loss' "$LOGS/holdout_probe.log" | tail -1
    [ $gate = 0 ] || { echo '探针门禁未过：停在这里，不要上臂'; exit 1; }
    say '探针门禁通过 → 下一步 16k 标准臂（需另立种子块）'
}

alpha_test
want surgery && { surgery; transparent; }
want smoke && smoke
want probe && probe
say '完成'
