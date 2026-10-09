# 自建玩家强度榜 + 白名单覆盖度分析（纯 CPU，不占 GPU）
# 用法: tmux new -d -s whitelist 'bash /mnt/data8/mortal/Mortal/tools/player_stats/build_whitelist.sh'
set -euo pipefail

ROOT=/mnt/data8/mortal/Mortal
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
G1='dataset/mjai/houou4/202[2-5]??/*.json.gz'
G2='dataset/mjai/houou4/20260[1-8]/*.json.gz'

cd "$ROOT"
mkdir -p mortal/logs

echo '[1/2] 扫全窗口牌谱，按长期成绩建玩家强度榜 ...'
"$PY" tools/player_stats/scan_player_ratings.py \
  --globs "$G1" "$G2" \
  --out dataset/index/player_ratings_2022_2026H1.csv \
  --names-out dataset/index/top_players_400.txt \
  --top 400 --min-games 400 --min-avg-pt 2 \
  --workers 28 --progress-every 200000

echo '[2/2] 白名单覆盖度分析（决定 top-N 该取多大）...'
"$PY" tools/player_stats/whitelist_coverage.py \
  --ratings dataset/index/player_ratings_2022_2026H1.csv \
  --globs "$G1" "$G2" \
  --min-games 400 --sizes 100,200,400,600,900 --workers 28

echo 'ALL DONE'
echo '产物: dataset/index/player_ratings_2022_2026H1.csv / top_players_400.txt'
