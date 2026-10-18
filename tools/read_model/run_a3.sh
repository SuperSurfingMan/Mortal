#!/usr/bin/env bash
# A3 规则法基线（读牌模型「vs 人类规则」参照）一键评估 + 三方对决。
#
# 定位（MEMORY §3.12 下一步 / 方案 §8.1 第 2 条）：纯 numpy、零学习的人类规则读牌器
# （现物/筋/壁 + 立直枚举 + 见逃排除），给 A2b（注入基线）提供增量参照。
# 本脚本做两件事：
#   1) a3_eval.py     —— 与 evaluate.py 同口径的离线诊断（20000 固定评估集）
#   2) compare_a3.py  —— 均匀二项 vs A2b vs A3 规则法，bootstrap 200 CI
#
# 纪律：权重标定只用非评估月（calibrate_a3.py，默认 202508）；评估与对决只用
# 202609 固定评估集（与 A2b 同种子同样本），不碰训练/共享树语义。
set -euo pipefail

REPO=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
LOGDIR=$REPO/mortal/logs/read_model
LOG=${LOG:-$LOGDIR/a3_eval_compare.log}
OUT=$REPO/mortal/read_model_a3
NET=${NET:-$REPO/mortal/read_model_v2s_a2b/best.pth}
SAMPLES=${SAMPLES:-20000}
NBOOT=${NBOOT:-200}
CALIB_ROOT=${CALIB_ROOT:-$REPO/dataset/mjai/houou4/202508}
mkdir -p "$LOGDIR" "$OUT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # 与 read_v2s_a2b 同位，勿扰 GPU2 主训
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
{
echo "=== A3 规则法基线 start $(date -u +%FT%TZ) | samples=$SAMPLES boot=$NBOOT ==="

echo "--- [0] 权重标定复跑（非评估月 $CALIB_ROOT，出处记录）---"
"$PY" -u tools/read_model/calibrate_a3.py --root "$CALIB_ROOT" --files 12 --samples 4000

echo "--- [1] A3 离线诊断（evaluate.py 同口径）---"
"$PY" -u tools/read_model/a3_eval.py --samples "$SAMPLES" --out "$OUT/eval_report.json"

echo "--- [2] 三方对决（uniform vs A2b vs A3-rule，bootstrap CI）---"
"$PY" -u tools/read_model/compare_a3.py --net "$NET" --samples "$SAMPLES" --n-boot "$NBOOT"

echo "=== A3 规则法基线 done $(date -u +%FT%TZ) ==="
} 2>&1 | tee -a "$LOG"
