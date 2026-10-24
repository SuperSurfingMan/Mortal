#!/usr/bin/env python3
"""构建「现窗口 + 旧年份按比例混入」的 file_index（数据扩窗探针）。

设计（单变量：只加旧年份数据，不改现窗口的选择标准）：
- 现窗口直接沿用既有索引（默认 B/756k 的 `reward_ablation_r4_v2.pth`，已按 top400_v2 过滤）；
- 旧年份（默认 2019-2021）按**该年代自建的名单**过滤（top_players_400_2019_2021.txt），
  再按文件名 sha1 确定性抽 `--ratio` —— 同一 glob 集合下结果可复现，换比例只改参数；
- 两段互不重叠、各自用各自年代的名单：训练时 `player_names` 取两份名单的**并集**，
  loader 的二次过滤不会放宽现窗口（现窗口文件全部含 v2 玩家），也不会漏掉旧年份文件。

依据（2026-10-24 年审计，`mortal/logs/data_audit/year_style.json`）：
2019-2021 的立直/副露/和了率与 2022 相差 ~3%（2013 才差 ~10%），且旧名单只覆盖
旧年 57~67% 的文件 —— 故探针取 2019-2021、ratio=0.2、用年代名单。

用法（仓库根目录）：
    MORTAL_CFG=config.mix_v1.toml \
    python tools/data_audit/build_mix_index.py --ratio 0.2 --force
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
MORTAL = REPO / 'mortal'
sys.path.insert(0, str(MORTAL))
os.chdir(MORTAL)

from config import config                      # noqa: E402
from train import build_index_manifest         # noqa: E402

DEFAULT_OLD_GLOBS = [
    str(REPO / 'dataset/mjai/houou4/2019??/*.json.gz'),
    str(REPO / 'dataset/mjai/houou4/202[01]??/*.json.gz'),
]


def _first_names(path):
    """读 gz 的第一行（start_game）拿 names；失败返回空列表。"""
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            start = json.loads(next(f))
        return [n for n in (start.get('names') or []) if n]
    except Exception:  # noqa: BLE001
        return []


def _kept(job):
    path, names = job
    return path, bool(set(_first_names(path)) & names)


def _bucket(path, modulus):
    """文件名的稳定桶号（换机器/换顺序都不变）。"""
    return int(hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:8], 16) % modulus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base-index', default=str(MORTAL / 'indexes/reward_ablation_r4_v2.pth'),
                    help='现窗口索引（B/756k 的 top400_v2 索引）')
    ap.add_argument('--old-globs', nargs='+', default=DEFAULT_OLD_GLOBS)
    ap.add_argument('--old-names', default=str(REPO / 'dataset/index/top_players_400_2019_2021.txt'))
    ap.add_argument('--ratio', type=float, default=0.2, help='旧年份保留比例（按文件名哈希）')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    ds = config['dataset']
    out = Path(ds['file_index'])
    if out.exists() and not args.force:
        raise SystemExit(f'索引已存在（如需覆盖请加 --force）: {out}')

    base = torch.load(args.base_index, weights_only=True)
    base_files = list(base['file_list'])
    base_set = set(base_files)
    print(f'现窗口（{Path(args.base_index).name}）: {len(base_files):,} 文件', flush=True)

    old_names = {ln.strip() for ln in open(args.old_names, encoding='utf-8') if ln.strip()}
    names_files = {ln.strip() for f in ds.get('player_names_files', [])
                   for ln in open(f, encoding='utf-8') if ln.strip()}
    assert old_names <= names_files, (
        'config 的 player_names_files 必须包含旧年份名单（manifest 才能与实际选择一致）')

    old_all: list[str] = []
    for pat in args.old_globs:
        old_all.extend(glob(pat, recursive=True))
    old_all = [p for p in old_all if p not in base_set]
    print(f'旧年份候选: {len(old_all):,} 文件（{len(args.old_globs)} 个 glob）', flush=True)

    kept: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path, ok in ex.map(_kept, ((p, old_names) for p in old_all), chunksize=256):
            if ok:
                kept.append(path)
    print(f'旧名单（{len(old_names)} 人）覆盖: {len(kept):,}/{len(old_all):,} '
          f'= {len(kept) / max(len(old_all), 1):.4f}', flush=True)

    modulus = 1_000_000
    thresh = int(round(args.ratio * modulus))
    sampled = [p for p in kept if _bucket(p, modulus) < thresh]
    print(f'按 sha1 抽 {args.ratio:.1%}: {len(sampled):,} 文件', flush=True)

    files = sorted(set(base_files) | set(sampled), reverse=True)
    add = len(files) - len(base_files)
    print(f'合计: {len(files):,} 文件（+{add:,} = +{add / max(len(base_files), 1):.2%}）', flush=True)

    manifest = build_index_manifest(
        config, names_files,
        enable_augmentation=bool(ds.get('enable_augmentation', False)),
        augmented_first=bool(ds.get('augmented_first', False)),
    )
    manifest['n_files'] = len(files)
    manifest['names_coverage'] = None          # 本工具未重扫现窗口
    manifest['mix'] = {
        'base_index': Path(args.base_index).name,
        'base_files': len(base_files),
        'old_globs': list(args.old_globs),
        'old_names': Path(args.old_names).name,
        'old_candidates': len(old_all),
        'old_kept_by_names': len(kept),
        'ratio': args.ratio,
        'old_added': len(sampled),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'file_list': files, 'manifest': manifest}, out)
    print(f'写入 {out}（n_files={len(files):,}）', flush=True)
    print(json.dumps(manifest['mix'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
