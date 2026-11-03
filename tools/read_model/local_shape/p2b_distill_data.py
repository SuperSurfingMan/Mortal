#!/usr/bin/env python3
"""P2b：为 train（202608）的立直目标生成 `behave` 参考软标签。

只对**立直目标**做参考采样：
  pool = unseen；tenpai 手牌 = sample_tenpai_hands（1/#待牌 纠偏）
  行为权重 = StaticLik(该家从本局开始到快照的舍牌, beta=1)
  soft[21,27] = 加权后的局部三张窗口计数形状

输出 `distill_train.npz`：idx[n_targets]（对应 train cache 下标）、j、soft[m,21,27]、
n_ref。idx/j 与 `features.py --mode trainval` 的 train cache 严格对齐（同文件切分 +
同 `build_fixed_eval_set` 采样逻辑），生成后会抽检 alloc 逐位一致。

用法：
  python tools/read_model/local_shape/p2b_distill_data.py \
      --train-glob 'dataset/mjai/houou4_names/202608/**/*.json.gz' \
      --train-cache tools/read_model/local_shape/.cache_p2/train.npz \
      --out tools/read_model/local_shape/.cache_p2/distill_train.npz \
      --max-snapshots 1500 --max-targets 3000 --n-ref 512
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'mortal'))
sys.path.insert(0, str(REPO / 'tools' / 'read_model' / 'diagnostics'))
sys.path.insert(0, str(REPO / 'tools' / 'read_model'))

import local_shape as LS  # noqa: E402
import dataset as D  # noqa: E402
import bayes_bound as BB  # noqa: E402


def build_samples_with_meta(files, n: int, seed: int):
    """与 train.build_fixed_eval_set 同序，但保留 _file/_observer/_event 元数据。"""
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)
    collected, used = [], []
    for f in files:
        samples, _ = D.replay_file(f, augment=False)
        used.append(f)
        for s in samples:
            s['_file'] = f
        collected.extend(samples)
        if len(collected) >= n:
            break
    if len(collected) > n:
        idx = np.linspace(0, len(collected) - 1, n).astype(int)
        collected = [collected[i] for i in idx]
    if not collected:
        raise RuntimeError('train cache 采样为空')
    print(f'[data] {len(collected)} samples from {len(used)} files', flush=True)
    return collected


def local_soft(h: np.ndarray, w: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """h [N,34], w [N] → soft [21,27]。"""
    soft = np.zeros((len(centers), LS.NUM_PATTERNS), np.float16)
    for ci, (a, b, c) in enumerate(centers):
        pidx = LS.pattern_index(h[:, [a, b, c]])
        cnt = np.bincount(pidx, weights=w, minlength=LS.NUM_PATTERNS)
        tot = cnt.sum()
        soft[ci] = (cnt / tot if tot > 0 else cnt).astype(np.float16)
    return soft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-glob', required=True)
    ap.add_argument('--train-cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-n', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--file-seed', type=int, default=20261102)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--max-snapshots', type=int, default=1500)
    ap.add_argument('--max-targets', type=int, default=3000)
    ap.add_argument('--n-ref', type=int, default=512)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--max-k', type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(args.train_glob, recursive=True))
    rng = random.Random(args.file_seed)
    rng_ref = np.random.default_rng(args.seed + 17)
    rng.shuffle(files)
    n_val_files = max(1, int(len(files) * args.val_frac))
    train_files = files[n_val_files:]
    samples = build_samples_with_meta(train_files, args.train_n, args.seed)
    train_cache = np.load(args.train_cache, allow_pickle=True)
    alloc_ref = train_cache['alloc']
    print(f'[cache] train alloc {alloc_ref.shape}  {time.time()-t0:.0f}s', flush=True)

    # 抽检采样顺序
    for i in (0, len(samples) // 2, len(samples) - 1):
        if not np.array_equal(samples[i]['alloc'].astype(np.int8), alloc_ref[i]):
            raise SystemExit(f'采样顺序与 train cache 不一致（idx={i}）')

    centers = LS.centers_3()
    events_cache = {}
    spans_cache = {}
    idxs, js, softs, nrefs = [], [], [], []
    n_snap = 0
    for i, s in enumerate(samples):
        ri = np.asarray(s['riichi']) > 0.5
        if not ri.any():
            continue
        n_snap += 1
        if n_snap > args.max_snapshots:
            break
        fpath = s['_file']
        if fpath not in events_cache:
            events_cache[fpath] = BB.load_events(fpath)
            spans_cache[fpath] = BB.kyoku_spans(events_cache[fpath])
        events, spans = events_cache[fpath], spans_cache[fpath]
        kyoku = int(s['_kyoku'])
        if kyoku <= 0 or kyoku > len(spans):
            continue
        lo, _hi = spans[kyoku - 1]
        seats = [(int(s['_observer']) + rel) % 4 for rel in (1, 2, 3)]
        pool = np.rint(np.asarray(s['unseen'])).astype(np.int64)
        for j in range(BB.TARGETS):
            if not ri[j]:
                continue
            size = 13 - 3 * int(s['meld_counts'][j])
            n_sets = (size - 1) // 3
            lik = BB.StaticLik(
                BB.extract_discards(events, lo, int(s['_event']), seats[j],
                                    args.max_k), args.beta)
            try:
                h, w = BB.sample_tenpai_hands(rng_ref, pool, n_sets, args.n_ref)
                lp = lik.logprob(h).astype(np.float64)
                fin = np.isfinite(lp)
                if fin.any():
                    w = w * np.exp(lp - lp[fin].max())
                    w[~fin] = 0.0
                if not np.isfinite(w).all() or w.sum() <= 0:
                    w = np.ones(len(h), np.float64)
                w = w / w.sum()
            except Exception as exc:  # shanten_fast 的极少数边界手牌
                print(f'[skip] idx={i} j={j} {type(exc).__name__}: {exc}',
                      flush=True)
                continue
            softs.append(local_soft(h, w, centers))
            idxs.append(i)
            js.append(j)
            nrefs.append(len(h))
            if len(idxs) >= args.max_targets:
                break
        if len(idxs) >= args.max_targets:
            break
        if n_snap % 50 == 0:
            print(f'[ref] snapshots={n_snap} targets={len(idxs)} '
                  f'{time.time()-t0:.0f}s', flush=True)

    if not idxs:
        raise SystemExit('没有生成任何立直目标；检查 riichi 字段/采样')
    out = {
        'idx': np.asarray(idxs, np.int64),
        'j': np.asarray(js, np.int8),
        'soft': np.stack(softs).astype(np.float16),
        'n_ref': np.asarray(nrefs, np.int32),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    meta = {
        'n_targets': len(idxs), 'n_snapshots': n_snap,
        'n_ref': args.n_ref, 'beta': args.beta, 'max_k': args.max_k,
        'train_cache': args.train_cache, 'train_glob': args.train_glob,
        'seed': args.seed, 'file_seed': args.file_seed,
        'val_frac': args.val_frac,
    }
    p.with_suffix('.meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    print(f'[out] {p}  targets={len(idxs)} snapshots={n_snap} '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
