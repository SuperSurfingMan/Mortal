#!/usr/bin/env python3
"""导出逐牌存在性（tile presence）token 缓存。

只保留 token 特征与标签，不保存 1012×34 的 obs，样本约 8.5KB。
文件选择/抽样逻辑与 ``tools/read_model/local_shape/features.py`` 完全一致，
因此 202609 eval 的 40000 样本与 A2c 诊断缓存逐位对齐（``--check-alloc`` 校验）。

用法：
  # train / val（202608，按文件 80/20 切分）
  python tools/read_model/tile_presence/export.py --mode trainval \
      --glob 'dataset/mjai/houou4_names/202608/**/*.json.gz' \
      --train-n 40000 --val-n 8000 --out tools/read_model/tile_presence/.cache

  # eval（202609，与 A2c 诊断缓存同源）
  python tools/read_model/tile_presence/export.py --mode eval \
      --glob 'dataset/mjai/houou4/202609/**/*.json.gz' \
      --n 40000 --seed 20261016 --out tools/read_model/tile_presence/.cache \
      --check-alloc /tmp/read_model_diag_cache_a2c/cache.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO / 'mortal'), str(REPO / 'tools' / 'read_model')):
    if p not in sys.path:
        sys.path.insert(0, p)

import dataset as D  # noqa: E402
import tokens as TK  # noqa: E402


def _files(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f'没有文件匹配：{pattern}')
    return files


def _collect(files: list[str], n: int, seed: int, tag: str) -> dict:
    """按 build_fixed_eval_set 的顺序收集 n 条紧凑样本（不保留 obs）。"""
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)
    buf: list[dict] = []
    t0 = time.time()
    for fi, f in enumerate(files):
        try:
            samples, _ = D.replay_file(f, augment=False)
        except Exception as exc:
            print(f'[{tag}] skip {f}: {exc}', flush=True)
            continue
        for s in samples:
            buf.append(TK.build_sample(s))
        if (fi + 1) % 10 == 0:
            print(f'[{tag}] files={fi+1} samples={len(buf)} '
                  f'{time.time()-t0:.0f}s', flush=True)
        if len(buf) >= n:
            break
    if not buf:
        raise RuntimeError(f'{tag}: 没有样本')
    if len(buf) > n:
        idx = np.linspace(0, len(buf) - 1, n).astype(int)
        buf = [buf[i] for i in idx]
    out = TK.stack(buf)
    print(f'[{tag}] n={len(buf)} ({time.time()-t0:.0f}s)', flush=True)
    return out


def _save(out_dir: Path, split: str, arrs: dict, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'{split}.npz', **arrs)
    meta = {**meta, 'n': int(arrs['y'].shape[0])}
    (out_dir / f'{split}.meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    print(f'[out] {out_dir / (split + ".npz")} n={meta["n"]}', flush=True)


def build_trainval(args) -> None:
    files = _files(args.glob)
    rng = random.Random(args.file_seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac))
    val_files, train_files = files[:n_val], files[n_val:]
    for split, fl, n, seed in (('train', train_files, args.train_n, args.seed),
                               ('val', val_files, args.val_n, args.seed + 1)):
        arrs = _collect(fl, n, seed, split)
        _save(Path(args.out), split, arrs, {
            'split': split, 'glob': args.glob, 'seed': seed,
            'file_seed': args.file_seed, 'val_frac': args.val_frac,
            'n_files': len(fl), 'feature_version': 1,
        })


def build_eval(args) -> None:
    files = _files(args.glob)
    arrs = _collect(files, args.n, args.seed, 'eval')
    if args.check_alloc:
        ref = np.load(args.check_alloc, allow_pickle=True)
        ref_alloc = ref['alloc'][:arrs['alloc'].shape[0]].astype(np.uint8)
        if not np.array_equal(ref_alloc, arrs['alloc']):
            bad = np.argwhere(ref_alloc != arrs['alloc'])[:5]
            raise SystemExit(f'alloc 与参考 cache 不一致（前 5 个：{bad.tolist()}）'
                             ' ⇒ 评估集顺序不同，不能直接与 A2c 比较')
        print('[check] alloc 与参考 cache 逐位一致', flush=True)
    _save(Path(args.out), 'eval', arrs, {
        'split': 'eval', 'glob': args.glob, 'seed': args.seed,
        'n_files': len(files), 'feature_version': 1,
        'check_alloc': args.check_alloc,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('trainval', 'eval'), required=True)
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--train-n', type=int, default=40000)
    ap.add_argument('--val-n', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--file-seed', type=int, default=20261103)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--check-alloc', default='')
    args = ap.parse_args()
    print(json.dumps(vars(args), ensure_ascii=False), flush=True)
    t0 = time.time()
    if args.mode == 'eval':
        build_eval(args)
    else:
        build_trainval(args)
    print(f'[done] {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
