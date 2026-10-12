# 自建玩家强度榜 + 白名单覆盖度分析（纯 CPU，不占 GPU）
# 用法: tmux new -d -s whitelist 'bash /mnt/data8/mortal/Mortal/tools/player_stats/build_whitelist.sh'
set -euo pipefail
# v2（10-11）：修正计分口径——每局 start_kyoku 重置 + reach_accepted 扣 1000 供托（对齐 libriichi/src/stat.rs），
#               旧 v1 名单存在“偏向高频立直玩家”的系统偏差，故重建为 _v2 文件名并存。
# v3（10-12）：数据侧 P0 落地后重建——2026 自建牌谱原先缺 start_game.names，白名单模式下
#               0 席位（13 万局静默失效）；现已由 tools/tenhou_data/repair_names.py 补回并
#               经 validate_logs + verify_names 验收，故强度榜窗口改为
#               「HF 2022-2025 + 修复树 202601-202608」并输出 _v3 系列；v2 产物保持不动以备复现。
#               注意：202609/202610 是 holdout/判定月，不参与建榜（避免把评测月信息带进名单）。

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
G1='dataset/mjai/houou4/202[2-5]??/*.json.gz'          # HF 历史（本就有 names）
G2='dataset/mjai/houou4_names/20260[1-8]/*.json.gz'    # 修复后的 2026 训练窗（补丁树）

cd "$ROOT"
mkdir -p mortal/logs

echo '[1/2] 扫全窗口牌谱，按长期成绩建玩家强度榜 ...'
"$PY" tools/player_stats/scan_player_ratings.py \
  --globs "$G1" "$G2" \
  --out dataset/index/player_ratings_2022_2026H1_v3.csv \
  --names-out dataset/index/top_players_400_v3.txt --top 400 \
  --names-out-extra dataset/index/top_players_200_v3.txt --top-extra 200 \
  --min-games 400 --min-avg-pt 2 \
  --workers 28 --progress-every 200000

echo '[2/2] 白名单覆盖度分析（决定 top-N 该取多大）...'
"$PY" tools/player_stats/whitelist_coverage.py \
  --ratings dataset/index/player_ratings_2022_2026H1_v3.csv \
  --globs "$G1" "$G2" \
  --min-games 400 --sizes 100,200,400,600,900 --workers 28

echo 'ALL DONE'
echo '产物: dataset/index/player_ratings_2022_2026H1_v3.csv / top_players_{400,200}_v3.txt'
