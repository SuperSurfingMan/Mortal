"""风格微调 P1/P3：选手行为画像（human profile）。

在 runs/style/<slug>/human_logs/ 建软链农场（该选手全部对局，不只训练窗——
画像属描述性统计），调用 libriichi.stat.Stat.from_dir 导出全部 getter 为 JSON。

⚠ Stat.from_dir 只接受单个精确玩家名（stat.rs:447），与 GameplayLoader 的
白名单 contains 语义不对齐——多别名选手须逐名各跑一次再合并（本工具已实现：
--player 可多次传，别名分别评测后按 count 加权合并比率类指标）。

用法（repo root 下）:
    python tools/style_ft/human_profile.py \
        --stats dataset/index/style/okada_sayaka_stats.json \
        [--player 岡田紗佳 --player <别名...>] \
        --out runs/style/okada_sayaka/human_profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# libriichi 编译产物在 mortal/ 下（mortal/libriichi.*.so），从 repo root 运行
# 须把 mortal/ 加进 sys.path
MORTAL_DIR = str(Path(__file__).resolve().parents[2] / 'mortal')
if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)

from libriichi.stat import Stat

# Stat 的 43 个 getter（stat.rs:519-780；除零项容错跳过）
GETTERS = [
    'avg_rank', 'rank_1_rate', 'rank_2_rate', 'rank_3_rate', 'rank_4_rate',
    'tobi_rate', 'avg_point_per_game', 'avg_point_per_round',
    'avg_point_per_agari', 'avg_point_per_oya_agari', 'avg_point_per_ko_agari',
    'avg_point_per_riichi_agari', 'avg_point_per_fuuro_agari',
    'avg_point_per_dama_agari', 'avg_point_per_ryukyoku',
    'avg_agari_jun', 'avg_riichi_agari_jun', 'avg_fuuro_agari_jun',
    'avg_dama_agari_jun', 'avg_point_per_houjuu', 'avg_point_per_houjuu_to_oya',
    'avg_point_per_houjuu_to_ko', 'avg_houjuu_jun',
    'agari_rate', 'houjuu_rate', 'riichi_rate', 'fuuro_rate', 'ryukyoku_rate',
    'agari_rate_after_riichi', 'houjuu_rate_after_riichi',
    'chasing_riichi_rate', 'riichi_chased_rate', 'avg_riichi_jun',
    'avg_riichi_point', 'agari_rate_as_oya', 'agari_as_oya_rate',
    'houjuu_to_oya_rate', 'avg_fuuro_num', 'agari_rate_after_fuuro',
    'houjuu_rate_after_fuuro', 'avg_fuuro_point', 'yakuman_rate',
    'nagashi_mangan_rate',
]


def stat_to_dict(stat) -> dict:
    out = {'game': stat.game, 'round': stat.round}
    for name in GETTERS:
        try:
            out[name] = getattr(stat, name)
        except ZeroDivisionError:
            out[name] = None
    return out


def main():
    ap = argparse.ArgumentParser(description='选手行为画像导出')
    ap.add_argument('--stats', required=True, help='<slug>_stats.json（读全部对局文件）')
    ap.add_argument('--player', action='append',
                    help='玩家名（可多次传以覆盖别名；默认取 stats.json 的 player）')
    ap.add_argument('--log-dir', default=None, help='软链农场目录（默认 runs/style/<slug>/human_logs）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.stats, encoding='utf-8') as f:
        stats = json.load(f)

    # 画像统计该选手全部对局（训练窗 + holdout），不排除任何数据
    all_files = stats['train_files'] + stats['holdout_files']
    players = args.player or [stats['player']]

    log_dir = Path(args.log_dir or f'runs/style/{stats["slug"]}/human_logs')
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(all_files):
        dst = log_dir / f'{i:05d}.json.gz'
        os.symlink(os.path.abspath(f), dst)

    per_player = {}
    for name in players:
        stat = Stat.from_dir(str(log_dir), name, disable_progress_bar=True)
        per_player[name] = stat_to_dict(stat)
        print(f'{name}: {stat.game} 局 / {stat.round} 局场')

    merged = per_player[players[0]]
    if len(players) > 1:
        # 多别名合并：数量类相加，均值类按 game 加权
        total_game = sum(p['game'] for p in per_player.values())
        for k in GETTERS:
            vals = [(p[k], p['game']) for p in per_player.values() if p.get(k) is not None]
            if len(vals) == len(players) and total_game:
                merged[k] = sum(v * g for v, g in vals) / total_game
        merged['game'] = total_game
        merged['round'] = sum(p['round'] for p in per_player.values())

    out = {
        'player': stats['player'],
        'aliases': players,
        'n_files': len(all_files),
        'profile': merged,
        'per_alias': per_player,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'out: {args.out}')


if __name__ == '__main__':
    main()