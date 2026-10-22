#!/usr/bin/env python3
"""P1 第 1 步：读牌危险度特征预计算（2 行/决策点）。

对主模型 ``GameplayLoader(version=4)`` 的每个决策点 obs：

1. **从 obs 反解** A2c 前向所需的辅助字段（已逐位验证，见 ``verify_obs_aux``）：
   - ``own_hand`` = ``obs[0:4].any(0)``（自家手牌 4 行 count one-hot）→ 100%
   - ``unseen``   = ``4 - round(obs[835]*4)``（tiles_seen 平面是 count/4）→ 100%
   - ``river``    = 三家 kawa 的 sutehai 通道（base 132/327/522，8ch×24）→ 100%
   - ``riichi``   = 三家 kawa 的 is_riichi 通道（idx+7）→ 99.83%（仅诊断用，
     ``compute_wait_bias`` 不消费）
2. 跑 A2c（``tl_tokens=0``；P0-D 实测 alloc_ce +2.5%、AUC −0.025，能力基本保留）。
3. 输出 **2 行** fp16 特征：

   - row0 = ``Σ_j P(听_j)·P(待牌_j=t)`` —— 三家 union 危险度（34）
   - row1 = ``P(至少一家听牌)`` —— 34 列同值

存储：``<out-dir>/<stem>.npz``，键 ``g0..g3`` = ``(n_steps, 2, 34) fp16``，
另存 ``n_games``/``meta``。dataloader 按 ``(file, game_idx)`` 查表并
``np.concatenate([obs, feat], axis=1)``（obs 1012 → 1014）。

用法::

    # 冒烟（500 文件）
    python tools/read_model/precompute_danger.py --max-files 500 \
        --out-dir mortal/danger_cache_smoke
    # 正式（跨月均匀抽样 20000 文件 ≈ 1400 万决策点）
    python tools/read_model/precompute_danger.py --max-files 20000 \
        --out-dir mortal/danger_cache
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 8 进程并行时，每个进程默认会用满 56 核（PyTorch intra-op + rayon），
# 造成调度震荡（实测 8 进程总吞吐从 32 降到 7.4 文件/s）。这里单线程化。
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'mortal'))    # libriichi.so 所在
sys.path.insert(0, str(HERE))               # 读牌工具优先（防同名 model.py，坑 69）

# 同名 model.py 冲突（坑 69）：显式占住读牌版本
import importlib.util as _ilu  # noqa: E402


def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_read_model = _load_module('model', HERE / 'model.py')
import train as T  # noqa: E402  读牌训练器（forward_batch/wait_bias_of）

A2C_CKPT = Path('/mnt/data8/mortal/Mortal/mortal/read_model_v2s_a2c/best.pth')
NUM_TILES = 34
OBS_ROWS = 1012
TILES_SEEN_ROW = 835                 # v4：tiles_seen = count/4
RIVER_BASES = (132, 327, 522)        # 下家/对家/上家

# ---- 与 obs_repr.rs v4 的逐位契约（实测，勿改）----
CH_KAWA = 8
N_KAWA_SLOTS = 24                    # take(6) + rev().take(18)
OFF_SUTEHAI = 3
OFF_RIICHI = 7


def obs_aux(obs: np.ndarray):
    """从 v4 obs 反解 (own_hand, unseen, river, riichi)。

    ``obs``: ``(B, 1012, 34)`` float32 → ``own (B,34) bool``、
    ``unseen (B,34) f32``、``river (B,3,34) bool``、``riichi (B,3) f32``。
    """
    obs = np.asarray(obs, np.float32)
    if obs.ndim == 2:
        obs = obs[None]
    B = obs.shape[0]
    own = obs[:, 0:4, :].any(1) > 0.5
    unseen = 4.0 - np.round(obs[:, TILES_SEEN_ROW, :] * 4.0)
    river = np.zeros((B, 3, NUM_TILES), bool)
    riichi = np.zeros((B, 3), np.float32)
    for j, base in enumerate(RIVER_BASES):
        for i in range(N_KAWA_SLOTS):
            row = base + CH_KAWA * i
            has = obs[:, row + OFF_SUTEHAI, :].sum(1) > 0.5
            tid = obs[:, row + OFF_SUTEHAI, :].argmax(1)
            river[np.where(has)[0], j, tid[has]] = True
            riichi[:, j] = np.maximum(
                riichi[:, j],
                (obs[:, row + OFF_RIICHI, :].sum(1) > 0.5).astype(np.float32))
    return own, unseen, river, riichi


def load_a2c(device):
    import model_v2 as M2
    st = torch.load(A2C_CKPT, map_location='cpu', weights_only=False)
    cfg = st['config']
    net = M2.build_model_v2s(conv_channels=cfg['conv_channels'],
                             num_blocks=cfg['num_blocks'],
                             with_aux=not cfg['no_aux'],
                             wait_head=cfg.get('wait_head', 'flat'),
                             tl_mode=cfg.get('tl_mode', 'pooled'),
                             next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
                             tile_order=cfg.get('tile_order', False)).to(device).eval()
    net.load_state_dict(st['model'])
    return net


def a2c_danger(net, obs, device, bs=2048):
    """obs (N,1012,34) → feat (N,2,34) float32（union 危险度 + P(任一进听)）。"""
    n = obs.shape[0]
    out = np.zeros((n, 2, NUM_TILES), np.float32)
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        o = torch.from_numpy(obs[sl]).float().to(device)
        own, unseen, river, riichi = obs_aux(obs[sl])
        sub = {
            'obs': o,
            'unseen': torch.from_numpy(unseen).to(device),
            'own_hand': torch.from_numpy(own.astype(np.float32)).to(device),
            'river': torch.from_numpy(river).to(device),
            'post_riichi_river': torch.from_numpy(river).to(device),
            'riichi': torch.from_numpy(riichi).to(device),
            'tl_tokens': torch.zeros(o.shape[0], 3, 26, 43, device=device),
            'tl_mask': torch.zeros(o.shape[0], 3, 26, dtype=torch.bool, device=device),
        }
        with torch.no_grad():
            wp = T.forward_batch(net, sub, device)['wait_probs']   # (B,3,35)
        p_tenpai = (1.0 - wp[..., NUM_TILES]).clamp(0, 1)          # (B,3)
        cond = wp[..., :NUM_TILES]                                 # (B,3,34)
        risk = (p_tenpai[..., None] * cond).sum(1)                 # (B,34)
        p_any = 1.0 - torch.prod(1.0 - p_tenpai, dim=1)            # (B,)
        out[sl, 0, :] = risk.float().cpu().numpy()
        out[sl, 1, :] = p_any.float().cpu().numpy()[:, None]
    return out


def file_stem(path: str) -> str:
    name = Path(path).name
    return name[:-len('.json.gz')] if name.endswith('.json.gz') else Path(name).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--globs', nargs='+', default=[
        '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/20260[1-8]/**/*.json.gz'])
    ap.add_argument('--out-dir', default='/mnt/data8/mortal/Mortal/mortal/danger_cache')
    ap.add_argument('--max-files', type=int, default=0, help='0=全部；否则固定抽样')
    ap.add_argument('--shuffle-seed', type=int, default=20261022)
    ap.add_argument('--batch-files', type=int, default=20)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-augment', action='store_true', default=True)
    ap.add_argument('--always-include-kan-select', action='store_true', default=True)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--shard', type=int, default=0,
                    help='分片序号（0..num-shards-1）；8 进程并行时用')
    ap.add_argument('--num-shards', type=int, default=1,
                    help='分片总数；>1 时按 shuffle 后的文件列表步长切分，互不重叠')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for g in args.globs:
        files.extend(glob.glob(g, recursive=True))
    files = sorted(set(files))
    random.Random(args.shuffle_seed).shuffle(files)
    if args.max_files > 0:
        files = files[:args.max_files]
    if args.num_shards > 1:
        files = files[args.shard::args.num_shards]
    print(f'[pre] shard {args.shard}/{args.num_shards}: {len(files)} 文件 → {out_dir}',
          flush=True)

    from libriichi.dataset import GameplayLoader
    device = torch.device(args.device)
    net = load_a2c(device)
    done = skipped = failed = 0
    t0 = time.time()
    loader = GameplayLoader(version=4, oracle=False, player_names=None,
                            excludes=None, augmented=False,
                            always_include_kan_select=args.always_include_kan_select)
    for start in range(0, len(files), args.batch_files):
        chunk = files[start:start + args.batch_files]
        todo = []
        for f in chunk:
            p = out_dir / f'{file_stem(f)}.npz'
            if p.exists() and not args.overwrite:
                skipped += 1
            else:
                todo.append(f)
        if not todo:
            continue
        try:
            data = loader.load_gz_log_files(todo)
        except Exception as exc:
            print(f'[pre] batch fail {todo[0]}...: {exc}', flush=True)
            failed += len(todo)
            continue
        for f, games in zip(todo, data):
            try:
                g_feats = []
                for game in games:
                    obs = np.stack(game.take_obs())            # (n,1012,34) f32
                    g_feats.append(a2c_danger(net, obs, device))
                payload = {f'g{i}': v.astype(np.float16)
                           for i, v in enumerate(g_feats)}
                payload['n_games'] = np.int64(len(g_feats))
                payload['meta'] = np.array(json.dumps({
                    'file': f, 'sizes': [int(v.shape[0]) for v in g_feats]}))
                np.savez_compressed(out_dir / f'{file_stem(f)}.npz', **payload)
                done += 1
            except Exception as exc:
                failed += 1
                print(f'[pre] skip {f}: {exc}', flush=True)
        el = time.time() - t0
        print(f'[pre] {start + len(chunk)}/{len(files)}  done={done} '
              f'skip={skipped} fail={failed}  {el:.0f}s '
              f'({(done + skipped) / max(el, 1e-9):.2f} file/s)', flush=True)
    print(f'[pre] 完成 done={done} skipped={skipped} failed={failed} '
          f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
