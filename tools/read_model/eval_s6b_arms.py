#!/usr/bin/env python3
"""S6b 四臂判决：在 202609 固定评估集上比 **决策级** 指标（不是 holdout top1）。

对每个臂（A0 基线 / A1 真特征 / A2 安慰剂 / A3 限域）：
- 主模型 top-1 舍牌（0..33，mask=own_hand）
- ``deal_top1``：该牌是否落在「三家有效荣和集」真值里（越低越好）
- ``fixable``：主模型打危险牌且 A2c 有安全替代（可修正点比例）
- ``agree_a2c``：与 A2c 最安全牌的一致率

注入特征由 **A2c 实时前向**生成（``tl_tokens=0``，与 ``precompute_danger.py``
同口径），因此不依赖 202609 的预计算 cache；A3 在这里同样按 ``danger_turn_min``
把早期巡目置零。

用法::

    python tools/read_model/eval_s6b_arms.py \
        --ckpt A0=mortal/checkpoints/inject_s6b/A0.pth \
        --ckpt A1=... --ckpt A2=... --ckpt A3=... \
        --n 20000 --out tools/read_model/diagnostics/results/s6b_arms_20k.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))
sys.path.insert(0, str(HERE))
import _common as C  # noqa: E402
from guard_risk import risk_from_wait_probs  # noqa: E402

# 复用 P0 的数据/推理工具（inject_p0 会显式加载读牌 model 与 mortal_model）
_spec = importlib.util.spec_from_file_location('inject_p0_mod', str(HERE / 'inject_p0.py'))
P0 = importlib.util.module_from_spec(_spec)
sys.modules['inject_p0_mod'] = P0
_spec.loader.exec_module(P0)

D = P0.D
T = P0.T
DEV = P0.DEV
NUM_TILES = P0.NUM_TILES
A2C_CKPT = Path('/mnt/data8/mortal/Mortal/mortal/read_model_v2s_a2c/best.pth')


def load_a2c():
    st = torch.load(A2C_CKPT, map_location='cpu', weights_only=False)
    cfg = st['config']
    net = P0.M2.build_model_v2s(conv_channels=cfg['conv_channels'],
                                num_blocks=cfg['num_blocks'],
                                with_aux=not cfg['no_aux'],
                                wait_head=cfg.get('wait_head', 'flat'),
                                tl_mode=cfg.get('tl_mode', 'pooled'),
                                next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
                                tile_order=cfg.get('tile_order', False)).to(DEV).eval()
    net.load_state_dict(st['model'])
    return net


def load_mortal(ckpt):
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = st['config']
    ver = cfg['control'].get('version', 4)
    extra = int(cfg['control'].get('extra_channels', 0))
    net = P0.MM.Brain(version=ver, extra_channels=extra, **cfg['resnet']).to(DEV).eval()
    net.load_state_dict(st['mortal'])
    dqn = P0.MM.DQN(version=ver).to(DEV).eval()
    dqn.load_state_dict(st['current_dqn'])
    turn_min = int(cfg.get('dataset', {}).get('danger_turn_min', 0))
    return net, dqn, extra, turn_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[],
                    help='name=path（可重复）；name 用作表格行')
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--out', default=str(C.OUT_DIR / 's6b_arms_20k.json'))
    args = ap.parse_args()
    arms = []
    for spec in args.ckpt:
        name, _, path = spec.partition('=')
        arms.append((name, path))
    if not arms:
        raise SystemExit('至少一个 --ckpt name=path')

    files = sorted(glob.glob(
        '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609/**/*.json.gz', recursive=True))
    print(f'[arms] {len(arms)} 臂 | n={args.n} seed={args.seed}', flush=True)

    # ---- 第一遍：固定评估集 + A2c 实时危险度 + 真值 ----
    a2c = load_a2c()
    feats, allocs, waits, owns, turns = [], [], [], [], []
    total = 0
    for buf, src_buf in P0.iter_eval_chunks(files, args.n, args.seed, args.chunk):
        take = min(len(buf), args.n - total)
        if take <= 0:
            break
        buf = buf[:take]
        b = T.collate(buf)
        AP, WP = P0.a2c_forward(a2c, b, args.chunk, zero_tl=True)
        p_t = (1.0 - WP[..., NUM_TILES]).clamp(0, 1)
        # 2026-11-02：修正双乘；历史 s6b_arms_20k.json 是旧口径，
        # 重跑数值会变，不能与旧 JSON 直接横比。
        risk = risk_from_wait_probs(WP)                               # (n,34)
        p_any = 1.0 - torch.prod(1.0 - p_t, dim=1)                   # (n,)
        feats.append(torch.stack([risk, p_any[:, None].expand(-1, NUM_TILES)],
                                 dim=1).numpy().astype(np.float32))
        allocs.append(b['alloc'].numpy())
        waits.append(b['wait'].numpy())
        owns.append(b['own_hand'].numpy())
        turns.append(b['turn'].numpy())
        total += take
        print(f'[data] {total}/{args.n}', flush=True)
        if total >= args.n:
            break

    FEAT = np.concatenate(feats, 0)          # (total,2,34)
    WAIT = np.concatenate(waits, 0)          # (total,3,35)
    OWN = np.concatenate(owns, 0).astype(bool)
    TURN = np.concatenate(turns, 0)
    danger = (WAIT[..., :NUM_TILES] > 0).any(1)                   # (n,34)
    risk = FEAT[:, 0].copy()
    risk_masked = np.where(OWN, risk, np.inf)
    a2c_safest = risk_masked.argmin(-1)
    a2c_deal = danger[np.arange(total), a2c_safest]
    print(f'[A2c] safest deal {a2c_deal.mean():.4f}', flush=True)

    results = {}
    for name, path in arms:
        net, dqn, extra, turn_min = load_mortal(path)
        qs = []
        pos = 0
        for buf, src_buf in P0.iter_eval_chunks(files, args.n, args.seed, args.chunk):
            take = min(len(buf), args.n - pos)
            if take <= 0:
                break
            buf = buf[:take]
            b = T.collate(buf)
            obs = b['obs'].numpy()
            if extra > 0:
                f = FEAT[pos:pos + take].copy()
                if turn_min > 0:
                    f[TURN[pos:pos + take] < turn_min] = 0
                obs = np.concatenate([obs, f], axis=1)
            obs_t = torch.from_numpy(obs).float()
            own_t = b['own_hand'].to(DEV) > 0
            with torch.no_grad():
                phi = net(obs_t.to(DEV))
                mask = torch.zeros(obs_t.shape[0], P0.ACTION_SPACE,
                                   dtype=torch.bool, device=DEV)
                mask[:, :NUM_TILES] = own_t
                q = dqn(phi, mask)
            qs.append(q[:, :NUM_TILES].float().cpu().numpy())
            pos += take
            if pos >= args.n:
                break
        Q = np.concatenate(qs)[:total]
        Q[~OWN] = -1e9
        top1 = Q.argmax(-1)
        deal_top1 = danger[np.arange(total), top1]
        agree = (top1 == a2c_safest)
        fixable = deal_top1 & (~a2c_deal)
        results[name] = {
            'n': int(total), 'extra_channels': extra, 'turn_min': turn_min,
            'deal_top1': float(deal_top1.mean()),
            'agree_a2c': float(agree.mean()),
            'fixable': float(fixable.mean()),
            'n_fixable': int(fixable.sum()),
            'ckpt': path,
        }
        print(f'[arm] {name}: deal_top1 {deal_top1.mean():.4f}  '
              f'agree {agree.mean():.4f}  fixable {fixable.mean():.4f} '
              f'(n={int(fixable.sum())})', flush=True)

    report = {'meta': {'n': total, 'seed': args.seed,
                       'protocol': 'A2c 实时前向（tl=0）；固定评估集同 P0'},
              'a2c_safest_deal': float(a2c_deal.mean()),
              'arms': results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\n[out] {args.out}')
    for name, r in results.items():
        print(f'  {name}: deal_top1 {r["deal_top1"]:.4f}  '
              f'agree {r["agree_a2c"]:.4f}  fixable {r["fixable"]:.4f}')


if __name__ == '__main__':
    main()
