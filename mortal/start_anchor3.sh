#!/usr/bin/env bash
# anchor3 一键启动（2026-11-04）
#
# 流程：预检 → phase 1（S6c + 守卫 λ=10，1600 局）→ 汇总
#              → phase 2（S6c 无守卫，800 局，同协议对照）→ 汇总
#
# 用法：
#   bash mortal/start_anchor3.sh                  # 自动进 tmux 会话 anchor3
#   CHECK_ONLY=1 bash mortal/start_anchor3.sh     # 只做预检，不启动
#   DRY_RUN=1 bash mortal/start_anchor3.sh        # 预检 + 计划（不进 tmux、不启动）
#   FOREGROUND=1 bash mortal/start_anchor3.sh     # 不进 tmux（调试用）
#
# 可覆盖的环境变量：
#   GPUS=0,1,3,4    物理卡（脚本已强制 CUDA_DEVICE_ORDER=PCI_BUS_ID）
#   PROCS=8         进程数（每进程 4 局并行）
#   GPP1=200        阶段 1 每进程局数（必须是 4 的倍数 → 8×200 = 1600）
#   GPP2=100        阶段 2 每进程局数（8×100 = 800）
#   PHASE=all|guard|noguard
#   SEED1=1040000   phase 1 种子块起点（每个进程 +i）
#   SEED2=1060000   phase 2 种子块起点
#   TAG1=anchor3 / TAG2=anchor3_noguard
#   KILL=1          启动前清掉同名残留会话
#   MIN_FREE_MB=2500 每张卡的最低空闲显存（低于则拒绝启动）
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
MORTAL=$ROOT/mortal
PY=${PY:-/mnt/data8/miniconda3/envs/mortal/bin/python}

GPUS=${GPUS:-0,1,3,4}
PROCS=${PROCS:-8}
GPP1=${GPP1:-200}
GPP2=${GPP2:-100}
PHASE=${PHASE:-all}
SEED1=${SEED1:-1040000}
SEED2=${SEED2:-1060000}
TAG1=${TAG1:-anchor3}
TAG2=${TAG2:-anchor3_noguard}
KILL=${KILL:-0}
MIN_FREE_MB=${MIN_FREE_MB:-2500}

# 判定臂必须固定 PCI 顺序：否则 CUDA_VISIBLE_DEVICES=0 会指到最快的 RTX 8000，
# 而不是 nvidia-smi 的 0 号 P100（2026-11-04 实测踩到）。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MORTAL_ROOT=$ROOT
export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=8

say () { echo "[$(date '+%F %T')] $*"; }
fail () { echo "[FAIL] $*" >&2; exit 1; }

# ---- 自嵌套 tmux（CHECK_ONLY / FOREGROUND 除外）----
if [ "${CHECK_ONLY:-0}" != 1 ] && [ "${FOREGROUND:-0}" != 1 ] && [ -z "${TMUX:-}" ]; then
    tmux kill-session -t anchor3 2>/dev/null
    exec tmux new -s anchor3 \
        "cd '$ROOT' && GPUS='$GPUS' PROCS='$PROCS' GPP1='$GPP1' GPP2='$GPP2' \
         PHASE='$PHASE' SEED1='$SEED1' SEED2='$SEED2' TAG1='$TAG1' TAG2='$TAG2' \
         KILL='$KILL' bash '$MORTAL/start_anchor3.sh'"
fi

cd "$MORTAL" || fail "缺少 $MORTAL"

# ---- 预检 ----
say "==== 预检 ===="
[ -f checkpoints/s6c/mortal.pth ] || fail "缺少 checkpoints/s6c/mortal.pth（S6c/956k）"
[ -f config.anchor3_guard.toml ] || fail "缺少 config.anchor3_guard.toml"
[ -f config.anchor3_noguard.toml ] || fail "缺少 config.anchor3_noguard.toml"
[ -f "$ROOT/tools/akochan_src/ako/system.exe" ] || fail "缺少 akochan 垫片 system.exe"
[ -f one_vs_three.py ] && [ -f agg_anchor.py ] || fail "缺少 one_vs_three.py / agg_anchor.py"
[ "$GPUS" = "0,1,3,4" ] || say "注意：GPUS=$GPUS（非默认 0,1,3,4）"
[ $((GPP1 % 4)) = 0 ] || fail "GPP1=$GPP1 必须是 4 的倍数"
[ $((GPP2 % 4)) = 0 ] || fail "GPP2=$GPP2 必须是 4 的倍数"

for g in ${GPUS//,/ }; do
    free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | awk -F', ' -v g="$g" '$1==g{print $2}')
    [ -n "$free" ] || fail "GPU $g 不存在"
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        fail "GPU $g 仅剩 ${free} MiB 空闲（要求 ≥ ${MIN_FREE_MB}）——换 GPUS 或等空闲"
    fi
    say "  GPU $g 空闲 ${free} MiB ✓"
done

stale=$(tmux ls 2>/dev/null | grep -Ec "^anchor_(${TAG1}|${TAG2})_p" || true)
if [ "${stale:-0}" -gt 0 ]; then
    if [ "$KILL" = 1 ]; then
        say "清理残留会话 $stale 个（KILL=1）"
        tmux ls 2>/dev/null | grep -E "^anchor_(${TAG1}|${TAG2})_p" | cut -d: -f1 \
            | while read -r s; do tmux kill-session -t "$s" 2>/dev/null; done
    else
        fail "有 $stale 个残留 anchor 会话；先停掉或加 KILL=1"
    fi
fi

b1=$((GPP1 / 4)); b2=$((GPP2 / 4))
say "计划："
[ "$PHASE" = all ] || [ "$PHASE" = guard ] && \
    say "  phase1 $TAG1 : $PROCS 进程 × $GPP1 局 = $((PROCS*GPP1)) 局（guard λ=10, seed $SEED1）≈ ${b1}×63min ≈ $((b1*63/60))h"
[ "$PHASE" = all ] || [ "$PHASE" = noguard ] && \
    say "  phase2 $TAG2 : $PROCS 进程 × $GPP2 局 = $((PROCS*GPP2)) 局（guard λ=0, seed $SEED2）≈ ${b2}×63min ≈ $((b2*63/60))h"
say "  模型：checkpoints/s6c/mortal.pth（S6c/956k）；对手：3×akochan"
case "$PHASE" in all|guard|noguard) ;; *) fail "PHASE 必须是 all|guard|noguard（当前 $PHASE）";; esac
[ "${CHECK_ONLY:-0}" = 1 ] && { say "CHECK_ONLY=1，预检通过，不启动"; exit 0; }
[ "${DRY_RUN:-0}" = 1 ] && { say "DRY_RUN=1，预检通过，跳过实例"; exit 0; }

# ---- 运行 ----
LAUNCH_LOG="logs/anchor3_launch_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LAUNCH_LOG") 2>&1
say "启动日志：$MORTAL/$LAUNCH_LOG"

run_phase () {  # $1=cfg $2=tag $3=seed $4=gpp $5=label
    say "==== $5 开始：$4 局/进程 × $PROCS 进程（tag=$2）===="
    BASE_CFG="$1" bash run_anchor.sh \
        --procs "$PROCS" --games-per-proc "$4" \
        --seed-key "$3" --gpus "$GPUS" --tag "$2" \
        || fail "$5 启动失败"
    while tmux ls 2>/dev/null | grep -q "^anchor_$2_p"; do sleep 60; done
    "$PY" agg_anchor.py --dir "logs/$2" | tee "logs/$2/summary.txt"
    local done=$(grep -ah "challenger rankings" logs/$2/p*.log 2>/dev/null | wc -l)
    say "==== $5 结束：完成 $done 批（每批 4 局 ≈ $((done*4)) 局）===="
}

if [ "$PHASE" = all ] || [ "$PHASE" = guard ]; then
    run_phase config.anchor3_guard.toml "$TAG1" "$SEED1" "$GPP1" "phase1 guard λ=10"
fi
if [ "$PHASE" = all ] || [ "$PHASE" = noguard ]; then
    run_phase config.anchor3_noguard.toml "$TAG2" "$SEED2" "$GPP2" "phase2 noguard"
fi

say "==== 全部完成 ===="
for t in "$TAG1" "$TAG2"; do
    [ -f "logs/$t/summary.txt" ] && { echo "--- $t ---"; cat "logs/$t/summary.txt"; }
done
