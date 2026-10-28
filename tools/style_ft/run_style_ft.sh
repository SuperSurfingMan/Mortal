#!/usr/bin/env bash
# 风格微调全链驱动（2026-10-27）：白名单 → 配置 → KL 单测 → 训练 →
# snapshot 早停选优 → 最优快照评测 → 人类画像 → 对比报告 → ladder。
#
# 用法（repo root 下）:
#   bash tools/style_ft/run_style_ft.sh <player> <parent_ckpt> <holdout_seasons> [slug] [kl_weight] [trainable]
# 示例:
#   bash tools/style_ft/run_style_ft.sh 岡田紗佳 mortal/checkpoints/s6c/mortal.pth \
#        2025-26,2026-27 okada_sayaka 1.5
#   bash tools/style_ft/run_style_ft.sh 岡田紗佳 mortal/checkpoints/s6c/mortal.pth \
#        2025-26,2026-27 okada_sayaka_lb 0.75 last_blocks
set -euo pipefail

PLAYER="${1:?用法: run_style_ft.sh <player> <parent_ckpt> <holdout_seasons> [slug] [kl_weight] [trainable]}"
PARENT="${2:?缺父代 ckpt}"
HOLDOUT="${3:?缺 holdout 赛季（逗号分隔，如 2025-26,2026-27）}"
SLUG="${4:-$(echo "$PLAYER" | iconv -f utf-8 -t ascii//TRANSLIT 2>/dev/null | tr -cd 'a-zA-Z' | tr 'A-Z' 'a-z' | head -c 12 || echo player)}"
KL_WEIGHT="${5:-1.5}"
# trainable: auto（默认，按局数自动）/ head / last_blocks / full。
# 换档位请换 slug（如 <name>_lb），避免覆盖上一档的 checkpoints 与 report。
TRAINABLE="${6:-auto}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_DIR="runs/style/${SLUG}"
mkdir -p "$RUN_DIR"
echo "== [1/8] 白名单与数据统计 =="
python tools/style_ft/make_player_whitelist.py \
    --player "$PLAYER" \
    --index dataset/mjai/mleague/player_index.json \
    --holdout-seasons "$HOLDOUT" \
    --slug "$SLUG" 2>&1 | tee "$RUN_DIR/01_whitelist.log"

echo "== [2/8] KL 单测（P2 验收：-inf mask 无 NaN） =="
python tools/style_ft/test_style_kl.py 2>&1 | tee "$RUN_DIR/02_kl_test.log"

echo "== [3/8] 生成训练配置 =="
# 档位选择：auto 按 stats.json 的 train_games（<400 局 → head）；可被第 6 参数
# 显式覆盖（如 last_blocks 二次实验）。last_blocks 拟合容量大，步数放宽到 1200
# （6 个快照，snapshot 早停兜底过拟合），其余档 800。
TRAIN_GAMES=$(python -c "import json; print(json.load(open('dataset/index/style/${SLUG}_stats.json'))['train_games'])")
if [ "$TRAINABLE" = "auto" ]; then
    TRAINABLE=last_blocks
    if [ "$TRAIN_GAMES" -lt 400 ]; then TRAINABLE=head; fi
fi
MAX_STEPS=800
if [ "$TRAINABLE" = "last_blocks" ]; then MAX_STEPS=1200; fi
python tools/style_ft/make_style_cfg.py \
    --parent "$PARENT" \
    --stats "dataset/index/style/${SLUG}_stats.json" \
    --slug "$SLUG" \
    --kl-weight "$KL_WEIGHT" \
    --trainable "$TRAINABLE" \
    --max-train-steps "$MAX_STEPS" 2>&1 | tee "$RUN_DIR/03_cfg.log"

echo "== [4/8] 暖启 + 训练（cd mortal; MORTAL_CFG） =="
# train.py 从 state_file 加载权重暖启（train.py:296）——先把父代 ckpt 复制到位
# （同 s6c 的 B756k → checkpoints/s6c/mortal.pth 手术惯例）；reset_optimizer 令牌
# 丢弃父代优化器动量，best_perf 换代自动重置。
mkdir -p "mortal/checkpoints/style/${SLUG}" "mortal/indexes/style"
cp "$PARENT" "mortal/checkpoints/style/${SLUG}/mortal.pth"
(cd mortal && MORTAL_CFG="config.style_${SLUG}.toml" python train.py) \
    2>&1 | tee "$RUN_DIR/04_train.log"

echo "== [5/8] snapshot 扫描（早停选优曲线） =="
python tools/style_ft/eval_agreement.py \
    --snapshots "mortal/checkpoints/style/${SLUG}/mortal_step_*.pth" \
    --stats "dataset/index/style/${SLUG}_stats.json" \
    --device cuda:0 \
    --out "$RUN_DIR/05_agreement_snapshots.json" 2>&1 | tee "$RUN_DIR/05_snapshots.log"

echo "== [6/8] 最优快照 + 父代 holdout 一致率 =="
BEST_SNAP=$(python - "$RUN_DIR/05_agreement_snapshots.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
best = max(r['ckpts'].items(), key=lambda kv: kv[1]['overall'])
print(best[0])
PY
)
cp "$BEST_SNAP" "mortal/checkpoints/style/${SLUG}/best_snapshot.pth"
python tools/style_ft/eval_agreement.py \
    --ckpts "mortal/checkpoints/style/${SLUG}/best_snapshot.pth" "$PARENT" \
    --stats "dataset/index/style/${SLUG}_stats.json" \
    --device cuda:0 \
    --out "$RUN_DIR/06_agreement.json" 2>&1 | tee "$RUN_DIR/06_agreement.log"

echo "== [7/8] 人类画像 + 对比报告 =="
python tools/style_ft/human_profile.py \
    --stats "dataset/index/style/${SLUG}_stats.json" \
    --out "$RUN_DIR/07_human_profile.json" 2>&1 | tee "$RUN_DIR/07_profile.log"

echo "== [8/8] ladder（风格 vs 父代，500 sets） =="
(cd mortal && python ladder.py \
    "checkpoints/style/${SLUG}/best_snapshot.pth" \
    "$(echo "$PARENT" | sed 's|^mortal/||')" \
    30000 500 "runs/style/${SLUG}/ladder" --no-amp) \
    2>&1 | tee "$RUN_DIR/08_ladder.log"

python tools/style_ft/compare_style.py \
    --agreement "$RUN_DIR/06_agreement.json" \
    --human "$RUN_DIR/07_human_profile.json" \
    --ladder "mortal/$RUN_DIR/ladder/run_meta.json" \
    --out "$RUN_DIR/report.md"

echo "完成：$RUN_DIR/report.md（判定见 Style_Finetune_Plan 判定标准节）"