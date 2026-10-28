#!/usr/bin/env python3
"""人类标签噪声地板（BC 天花板）测量。

思路：**同一状态上人类可以给出不同动作**，这部分分歧是不可约的（标签噪声）。
本工具用「状态键 → 动作分布」的重复组估计两条量：

1. `agreement_repeat`：在**同一键**上随机取两个人类动作，相同的概率
   `Σ_a n_a(n_a−1) / (n(n−1))`（只在 n≥2 的组上统计）；
2. `ceiling_bound`：把「只出现过一次的状态」乐观地视为完全一致，
   `ceiling = 1 − w + w·agreement_repeat`，其中 `w = Σ n(n−1) / N(N−1)`。

键的两种粒度：
- `--key exact`：自家手牌(34) + 已见牌(34) + 轮目 + 自家/三家立直 + 可打牌掩码 + 庄家局面；
- `--key coarse`：自家手牌 + 轮目 + 可打牌掩码（丢掉已见牌与立直信息 ⇒ 重复组更多但更粗）。

只统计**打牌决策**（action ∈ [0,34)），因为 holdout top1 就是在这个口径上算的。

用法（repo/mortal 下）：
    python ../tools/label_ceiling.py --files 20000 --workers 24 --key exact
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

TILES_SEEN_ROW = 835
RIVER_BASES = (132, 327, 522)
OFF_RIICHI = 7
BANK = 34


def state_keys(obs: np.ndarray, own: np.ndarray, key: str) -> list[bytes]:
    """把 (n,1012,34) 的 obs 压成每样本一个固定长度 bytes 键。"""
    n = obs.shape[0]
    hand = (np.rint(obs[:, :4].sum(1)) > 0).astype(np.uint8)          # 手牌集合
    seen = np.rint(obs[:, TILES_SEEN_ROW] * 4).astype(np.uint8)        # 已见张数
    turn = np.rint(seen.sum(1) / 2).astype(np.uint8)                   # 轮目代理
    riichi = np.zeros(n, np.uint8)
    for j, base in enumerate(RIVER_BASES):
        r = np.zeros(n, np.uint8)
        for i in range(24):
            r |= (obs[:, base + 8 * i + OFF_RIICHI].sum(1) > 0.5).astype(np.uint8)
        riichi |= (r << j)
    mask = np.packbits(own.astype(np.uint8), axis=1)                   # 可打牌掩码 (n,5)
    out = []
    for i in range(n):
        if key == 'exact':
            b = (hand[i].tobytes() + seen[i].tobytes() + bytes([turn[i], riichi[i]])
                 + mask[i].tobytes())
        else:
            b = hand[i].tobytes() + bytes([turn[i]]) + mask[i].tobytes()
        out.append(b)
    return out


def worker(args):
    files, key, max_per_file = args
    from libriichi.dataset import GameplayLoader
    ld = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None,
                        augmented=False, always_include_kan_select=True)
    hashes, acts = [], []
    for f in files:
        try:
            chunks = list(ld.load_gz_log_files([f]))
        except Exception:
            continue
        for games in chunks:
            for game in games:
                try:
                    obs = np.stack([np.asarray(o, np.float32) for o in game.take_obs()])
                    actions = np.asarray(game.take_actions())
                except Exception:
                    continue
                m = (actions < BANK)
                if not m.any():
                    continue
                sub_obs = obs[m]
                sub_act = actions[m]
                if len(sub_act) > max_per_file:
                    idx = np.random.default_rng(0).choice(len(sub_act), max_per_file, replace=False)
                    sub_obs, sub_act = sub_obs[idx], sub_act[idx]
                own = np.rint(sub_obs[:, :4].sum(1)) > 0
                for b, a in zip(state_keys(sub_obs, own, key), sub_act):
                    hashes.append(int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'little'))
                    acts.append(int(a))
    return np.asarray(hashes, np.uint64), np.asarray(acts, np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', type=int, default=20000, help='抽样文件数')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--key', choices=['exact', 'coarse'], default='exact')
    ap.add_argument('--shard', type=int, default=200, help='每个任务的文 件数')
    ap.add_argument('--max-per-file', type=int, default=400)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--glob', default='../dataset/mjai/houou4/2025*/**/*.json.gz')
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob, recursive=True))
    random.Random(a.seed).shuffle(files)
    files = files[:a.files]
    print(f'[ceiling] key={a.key} files={len(files)} workers={a.workers}', flush=True)

    shards = [(files[i:i + a.shard], a.key, a.max_per_file) for i in range(0, len(files), a.shard)]
    hs, as_ = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (h, act) in enumerate(ex.map(worker, shards)):
            hs.append(h); as_.append(act)
            if (k + 1) % 10 == 0:
                print(f'  {k+1}/{len(shards)} shards, {sum(len(x) for x in hs):,} 决策, {time.time()-t0:.0f}s', flush=True)
    H = np.concatenate(hs); A = np.concatenate(as_)
    del hs, as_
    N = len(H)
    print(f'[ceiling] 总决策 {N:,}（{len(files)} 文件）', flush=True)

    order = np.argsort(H, kind='stable')
    Hs, As = H[order], A[order]
    del H, A
    # 组边界
    brk = np.flatnonzero(np.diff(Hs) != 0) + 1
    starts = np.concatenate([[0], brk]); ends = np.concatenate([brk, [N]])
    sizes = ends - starts
    print(f'[ceiling] 唯一键 {len(sizes):,}；重复组(≥2) {int((sizes>=2).sum()):,}；'
          f'覆盖样本 {int(sizes[sizes>=2].sum()):,}（{sizes[sizes>=2].sum()/N:.2%}）', flush=True)

    # 只在重复组内统计：Σ n_a(n_a−1) / n(n−1)
    num = 0.0; den = 0.0
    for s, e in zip(starts[sizes >= 2], ends[sizes >= 2]):
        cnt = np.bincount(As[s:e], minlength=BANK)
        n = e - s
        num += float((cnt * (cnt - 1)).sum())
        den += float(n * (n - 1))
    agreement = num / den if den else float('nan')
    w = den / (N * (N - 1))
    ceiling = (1 - w) + w * agreement
    print(f'[result] 重复对占比 w = {w:.4%}')
    print(f'[result] 重复组内一致率 agreement_repeat = {agreement:.4f}')
    print(f'[result] 乐观天花板 ceiling_bound = {ceiling:.4f}（把单例状态视为完全一致）')
    print(f'[ref]    B/756k 的 holdout top1 = 0.8127；S6c/956k = 0.8167')
    return 0


if __name__ == '__main__':
    sys.exit(main())
