"""风格微调 P2 排障：定位「动作不在合法 mask 内」的坏样本（M-League 数据）。

train.py:657 的断言 masks[range, actions].all() 在天凤 houou4 数据从未触发，
M-League 转换数据第 9 个 batch 触发 → 个别文件存在坏决策点。本脚本逐文件
复现训练同款加载管线（GameplayLoader + file_batch_size=16），找出：

    ① 违规发生在哪个文件 / 哪个决策点 / 什么动作
    ② 违规只出现在 augmented=True 还是原始数据也有

    → 原始也有：坏文件，从 stats.json/toml 的文件列表剔除
    → 仅增强有：Rust 增强路径边界 bug（不改 Rust 硬约束）→ 关增强绕开

用法（repo root 下）:
    python tools/style_ft/diag_mleague.py --stats dataset/index/style/okada_sayaka_stats.json
    python tools/style_ft/diag_mleague.py --stats dataset/index/style/okada_sayaka_stats.json --augmented
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# libriichi 编译产物在 mortal/ 下（mortal/libriichi.*.so），从 repo root 运行
# 须把 mortal/ 加进 sys.path（与 eval_agreement.py 同款处理）
MORTAL_DIR = str(Path(__file__).resolve().parents[2] / 'mortal')
if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)

from libriichi.dataset import GameplayLoader


def action_class(a: int) -> str:
    if a <= 36:
        return 'discard'
    if a == 37:
        return 'riichi'
    if 38 <= a <= 42:
        return 'meld'
    if a == 43:
        return 'agari'
    if a == 44:
        return 'ryukyoku'
    return 'pass'


def main():
    ap = argparse.ArgumentParser(description='定位 mask 违规样本')
    ap.add_argument('--stats', required=True)
    ap.add_argument('--augmented', action='store_true', help='跑增强口径（默认原始口径）')
    ap.add_argument('--version', type=int, default=4)
    ap.add_argument('--file-batch', type=int, default=16)
    ap.add_argument('--max-print', type=int, default=40)
    args = ap.parse_args()

    with open(args.stats, encoding='utf-8') as f:
        stats = json.load(f)
    files = stats['train_files'] + stats['holdout_files']
    print(f'mode: {"augmented" if args.augmented else "raw"}  files: {len(files)}')

    loader = GameplayLoader(
        version=args.version,
        oracle=False,
        player_names=[stats['player']],
        excludes=None,
        augmented=args.augmented,
        always_include_kan_select=True,
    )

    bad_files = Counter()
    bad_actions = Counter()
    n_print = 0
    n_total_samples = 0
    n_bad = 0
    for start in range(0, len(files), args.file_batch):
        chunk = files[start:start + args.file_batch]
        data = loader.load_gz_log_files(chunk)
        for file_games, fname in zip(data, chunk):
            for gi, game in enumerate(file_games):
                actions = np.asarray(game.take_actions(), dtype=np.int64)
                if len(actions) == 0:
                    continue
                masks = np.stack(game.take_masks())
                # take_at_kyoku() 返回 bytes（每字节一个局编号），须 frombuffer
                # 而非 np.asarray（后者会把 bytes 当字符串解析而炸）
                at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
                n_total_samples += len(actions)
                bad = ~masks[np.arange(len(actions)), actions]
                if not bad.any():
                    continue
                idxs = np.nonzero(bad)[0]
                n_bad += len(idxs)
                bad_files[fname] += len(idxs)
                for i in idxs:
                    bad_actions[int(actions[i])] += 1
                    if n_print < args.max_print:
                        legal = np.nonzero(masks[i])[0].tolist()
                        print(f'  BAD {Path(fname).name} game#{gi} kyoku#{int(at_kyoku[i])} '
                              f'sample#{int(i)}: action={int(actions[i])} ({action_class(int(actions[i]))}) '
                              f'n_legal={len(legal)} legal[:12]={legal[:12]}')
                        n_print += 1

    print(f'\n== 汇总（{"augmented" if args.augmented else "raw"}）==')
    print(f'samples: {n_total_samples:,}  bad: {n_bad}  ({n_bad / max(n_total_samples, 1):.5%})')
    print(f'坏文件数: {len(bad_files)}')
    for f, n in bad_files.most_common():
        print(f'  {Path(f).name}: {n} 个坏决策点')
    print('坏动作分布:', {f'{a}({action_class(a)})': n for a, n in sorted(bad_actions.items())})


if __name__ == '__main__':
    main()