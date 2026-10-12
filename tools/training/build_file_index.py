#!/usr/bin/env python3
"""按当前训练配置构建带 manifest 的 file_index。

用途：在多个并行实验启动前只构建一次相同数据池，避免各进程同时扫描/写入同一个索引。
运行目录为 ``mortal/``，通过 ``MORTAL_CFG`` 选择配置；输出路径默认取
``[dataset].file_index``。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from glob import glob
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
MORTAL = REPO / 'mortal'
sys.path.insert(0, str(MORTAL))
os.chdir(MORTAL)

from config import config  # noqa: E402
from train import build_index_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description='构建当前配置的带 manifest file_index')
    ap.add_argument('--force', action='store_true', help='覆盖已有索引')
    args = ap.parse_args()

    ds = config['dataset']
    output = Path(ds['file_index'])
    if output.exists() and not args.force:
        raise SystemExit(f'索引已存在（如需覆盖请加 --force）: {output}')

    files: list[str] = []
    for pattern in ds['globs']:
        files.extend(glob(pattern, recursive=True))
    files.sort(reverse=True)
    if not files:
        raise SystemExit(f'globs 没有匹配文件: {ds["globs"]}')

    players: set[str] = set()
    for filename in ds.get('player_names_files', []):
        with open(filename, encoding='utf-8') as f:
            players.update(line.strip() for line in f if line.strip())

    names_coverage = None
    if players:
        filtered: list[str] = []
        with_names = without_names = 0
        for filename in files:
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                start = json.loads(next(f))
            names = start.get('names') or []
            if any(names):
                with_names += 1
            else:
                without_names += 1
            if not set(names).isdisjoint(players):
                filtered.append(filename)
        files = filtered
        names_coverage = {'with_names': with_names, 'without_names': without_names}

    manifest = build_index_manifest(
        config,
        players,
        enable_augmentation=bool(ds.get('enable_augmentation', False)),
        augmented_first=bool(ds.get('augmented_first', False)),
    )
    manifest['n_files'] = len(files)
    manifest['names_coverage'] = names_coverage
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'file_list': files, 'manifest': manifest}, output)
    print(f'写入 {output}: n_files={len(files):,}, players={len(players):,}, '
          f'names_coverage={names_coverage}', flush=True)


if __name__ == '__main__':
    main()
