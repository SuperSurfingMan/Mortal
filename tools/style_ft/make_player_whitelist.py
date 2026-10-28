"""风格微调 P1：为单个 M-League 选手生成白名单与数据统计（2026-10-27）。

M-League 数据源有现成的 player_index.json（fetch_mleague.py 产物，含每位选手
全部对局的文件路径与赛季/日期/阶段），无需解压扫描——**且**数据集中存在 35 个
gzip 损坏文件（_invalid_files.txt，头部不可读，train.py 的索引扫描遇到即崩），
用索引直接给「精确文件列表」可同时绕开损坏文件与 holdout 排除两件事。

用法（repo root 下）:
    python tools/style_ft/make_player_whitelist.py \
        --player 岡田紗佳 \
        --index dataset/mjai/mleague/player_index.json \
        --holdout-seasons 2025-26,2026-27 \
        --slug okada_sayaka

输出:
    dataset/index/style/<slug>.txt          白名单（一行一名，train.py 直接读）
    dataset/index/style/<slug>_stats.json   分赛季统计 + 训练/holdout 文件清单 + 档位建议
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def tier_of(train_games: int) -> str:
    """数百局量级的可训档位（Style_Finetune_Plan v2）："""
    if train_games >= 1500:
        return 'full（压力测试档）'
    if train_games >= 400:
        return 'last_blocks（默认主力档）'
    if train_games >= 200:
        return 'head'
    return 'abandon（<200 局：不建议权重微调，改调推理采样温度）'


def main():
    ap = argparse.ArgumentParser(description='M-League 单选手白名单与数据统计')
    ap.add_argument('--player', required=True, help='选手实名（须与牌谱 start_game.names 完全一致）')
    ap.add_argument('--index', default='dataset/mjai/mleague/player_index.json')
    ap.add_argument('--holdout-seasons', default='', help='逗号分隔的 holdout 赛季（如 2025-26,2026-27）')
    ap.add_argument('--slug', required=True, help='短名（用于白名单/配置/checkpoint 目录命名）')
    ap.add_argument('--out-dir', default='dataset/index/style')
    args = ap.parse_args()

    index_path = Path(args.index)
    with open(index_path, encoding='utf-8') as f:
        index = json.load(f)

    entry = index.get('players', {}).get(args.player)
    if entry is None:
        known = sorted(index.get('players', {}).keys())
        raise SystemExit(f'选手 {args.player!r} 不在 {index_path} 中（共 {len(known)} 位选手）')

    games = entry['games']
    holdout_seasons = {s.strip() for s in args.holdout_seasons.split(',') if s.strip()}

    # g['file'] 相对 repo root（player_index.json _meta 的 file_root 字段）——
    # 用脚本自身位置推 repo root，不依赖 cwd。首版误用 index_path.parent.parent
    # 拼出 dataset/mjai/dataset/mjai/... 的错误路径：统计数字（读自索引）不受
    # 影响，但 train_files 全部失效（globs 空列表 → 训练零数据），且下方
    # invalid 交集检查沦为恒空假阴性。现附存在性断言，此类错误提前失败。
    repo_root = Path(__file__).resolve().parents[2]
    train_files, holdout_files = [], []
    by_season_train, by_season_holdout = Counter(), Counter()
    by_stage = Counter()
    missing = []
    for g in games:
        by_stage[g.get('stage', '?')] += 1
        f_abs = str((repo_root / g['file']).resolve())
        if not Path(f_abs).exists():
            missing.append(f_abs)
        if g['season'] in holdout_seasons:
            holdout_files.append(f_abs)
            by_season_holdout[g['season']] += 1
        else:
            train_files.append(f_abs)
            by_season_train[g['season']] += 1
    if missing:
        raise SystemExit(f'索引指向不存在的文件（{len(missing)} 个，如 {missing[:3]}）——'
                         f'请检查 player_index.json 的 file_root 假设')

    # 损坏文件核对（若交集非空，后续训练/评测会在读取时崩，提前失败）
    invalid = set()
    invalid_path = index_path.parent / '_invalid_files.txt'
    if invalid_path.exists():
        with open(invalid_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                invalid.add(str(Path(line.split('\t')[0]).resolve()))
    clash = sorted(set(train_files + holdout_files) & invalid)
    if clash:
        raise SystemExit(f'选手对局与 _invalid_files.txt 交集非空（{len(clash)} 个，如 {clash[:3]}）')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    whitelist = out_dir / f'{args.slug}.txt'
    with open(whitelist, 'w', encoding='utf-8') as f:
        f.write(args.player + '\n')

    stats = {
        'player': args.player,
        'slug': args.slug,
        'source_index': str(index_path),
        'total_games': len(games),
        'by_stage': dict(by_stage),
        'by_season_train': dict(sorted(by_season_train.items())),
        'by_season_holdout': dict(sorted(by_season_holdout.items())),
        'train_games': len(train_files),
        'holdout_games': len(holdout_files),
        'train_files': train_files,
        'holdout_files': holdout_files,
        'holdout_seasons': sorted(holdout_seasons),
        'invalid_clash': clash,
        'tier': tier_of(len(train_files)),
        # 每半庄 ~160 决策点（dataloader 实测每局均值 16.00 × ~10 局），
        # enable_augmentation=true 时样本翻倍
        'est_train_samples': len(train_files) * 160,
        'est_holdout_samples': len(holdout_files) * 160,
    }
    stats_path = out_dir / f'{args.slug}_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f'whitelist: {whitelist}（1 名: {args.player}）')
    print(f'stats:     {stats_path}')
    print(f'total={stats["total_games"]}  train={stats["train_games"]}  '
          f'holdout={stats["holdout_files"] and stats["holdout_games"]}  tier={stats["tier"]}')
    print(f'分赛季 train: {stats["by_season_train"]}')
    print(f'分赛季 holdout: {stats["by_season_holdout"]}')


if __name__ == '__main__':
    main()