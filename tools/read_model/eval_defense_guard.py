#!/usr/bin/env python3
"""推理期防守守卫的单步诊断：λ 扫描（选 λ 用，不是判决）。

口径与 P0-B 一致：202609 固定 20000 样本（seed 20261016）、A2c ``tl=0`` 实时
危险度、主模型 B/756k 冻结。对每个 λ 计算：

- ``change``     相对 λ=0 的打牌动作变化率（只统计原本就是打牌的决策）
- ``deal_top1``  选中牌是某家实际和了牌的比例（放铳率，P0-B 同口径）
- ``risk_top1``  选中牌的平均 A2c 危险度
- ``a2c_agree``  与 A2c「最安全牌」的一致率
- ``q_gap``      top1−top2 的 Q 差（判断 λ 的量纲是否匹配）

**单步指标只用于选 λ 与确认行为方向**（放铳率下降必然伴随进攻下降，而单步
看不到和了收益）；是否真赢必须看 ladder（双向 ≥5000 sets）。

用法::

    python tools/read_model/eval_defense_guard.py --n 20000 \
        --lambdas 0,2,5,10,20
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
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / 'diagnostics'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'mortal'))

import _common as C  # noqa: E402

_spec = importlib.util.spec_from_file_location('inject_p0_mod', HERE / 'inject_p0.py')
P0 = importlib.util.module_from_spec(_spec)
sys.modules['inject_p0_mod'] = P0
_spec.loader.exec_module(P0)

D = P0.D
T = P0.T
NUM_TILES = P0.NUM_TILES
ACTION_SPACE = P0.ACTION_SPACE
BASE_CKPT = REPO / 'mortal/checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth'
A2C_CKPT = REPO / 'mortal/read_model_v2s_a2c/best.pth'


def load_mortal(ckpt: Path):
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = st['config']
    ver = cfg['control'].get('version', 4)
    net = P0.MM.Brain(version=ver, **cfg['resnet']).to(P0.DEV).eval()
    net.load_state_dict(st['mortal'])
    dqn = P0.MM.DQN(version=ver).to(P0.DEV).eval()
    dqn.load_state_dict(st['current_dqn'])
    return net, dqn, int(st.get('steps', 0))


def load_a2c():
    st = torch.load(A2C_CKPT, map_location='cpu', weights_only=False)
    cfg = st['config']
    net = P0.M2.build_model_v2s(
        conv_channels=cfg['conv_channels'], num_blocks=cfg['num_blocks'],
        with_aux=not cfg['no_aux'], wait_head=cfg.get('wait_head', 'flat'),
        tl_mode=cfg.get('tl_mode', 'pooled'),
        next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
        tile_order=cfg.get('tile_order', False),
    ).to(P0.DEV).eval()
    net.load_state_dict(st['model'])
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--lambdas', default='0,2,5,10,20')
    ap.add_argument('--ckpt', default=str(BASE_CKPT))
    ap.add_argument('--out', default=str(C.OUT_DIR / 'defense_guard_sweep_20k.json'))
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(',') if x.strip()]

    files = sorted(glob.glob(
        str(REPO / 'dataset/mjai/houou4/202609/**/*.json.gz'), recursive=True))
    net, dqn, steps = load_mortal(Path(args.ckpt))
    a2c = load_a2c()
    print(f'[guard] base step={steps} n={args.n} lambdas={lambdas}', flush=True)

    Q, OWN, RISK, DANGER = [], [], [], []
    total = 0
    for buf, _src in P0.iter_eval_chunks(files, args.n, args.seed, args.chunk):
        take = min(len(buf), args.n - total)
        if take <= 0:
            break
        buf = buf[:take]
        b = T.collate(buf)
        _AP, WP = P0.a2c_forward(a2c, b, args.chunk, zero_tl=True)
        p_t = (1.0 - WP[..., NUM_TILES]).clamp(0, 1)
        risk = (p_t[..., None] * WP[..., :NUM_TILES]).sum(1).float().cpu().numpy()  # (n,34)
        danger = (b['wait'][..., :NUM_TILES] > 0).any(1).cpu().numpy()               # (n,34)
        with torch.no_grad():
            phi = net(b['obs'].to(P0.DEV))
            own = b['own_hand'].to(P0.DEV) > 0
            mask = torch.zeros(phi.shape[0], ACTION_SPACE, dtype=torch.bool,
                               device=P0.DEV)
            mask[:, :NUM_TILES] = own
            q = dqn(phi, mask)
        Q.append(q[:, :NUM_TILES].float().cpu().numpy())
        OWN.append(own.cpu().numpy())
        RISK.append(risk)
        DANGER.append(danger)
        total += take
        print(f'[data] {total}/{args.n}', flush=True)
        if total >= args.n:
            break

    Q = np.concatenate(Q)[:total].astype(np.float64)      # (n,34) masked Q
    OWN = np.concatenate(OWN)[:total].astype(bool)
    RISK = np.concatenate(RISK)[:total].astype(np.float64)
    DANGER = np.concatenate(DANGER)[:total].astype(bool)
    Q[~OWN] = -1e9
    n = total
    ar = np.arange(n)

    # A2c 最安全牌（只在自己手牌里选）
    risk_masked = np.where(OWN, RISK, np.inf)
    a2c_safest = risk_masked.argmin(-1)
    a2c_deal = DANGER[ar, a2c_safest]

    q0 = Q.argmax(-1)
    top2 = np.sort(Q, -1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    d0 = DANGER[ar, q0]
    print(f'[base] deal_top1 {d0.mean():.4f}  a2c_deal {a2c_deal.mean():.4f}  '
          f'q_gap med {np.median(gap):.3f}', flush=True)

    rows = []
    for lam in lambdas:
        qq = Q - lam * RISK
        qq[~OWN] = -1e9
        d = qq.argmax(-1)
        changed = d != q0
        ch = ar[changed]
        deal = DANGER[ar, d]
        row = {
            'lambda': lam,
            'change': float(changed.mean()),
            'change_n': int(changed.sum()),
            'deal_top1': float(deal.mean()),
            'risk_top1': float(RISK[ar, d].mean()),
            'a2c_agree': float((d == a2c_safest).mean()),
            'safer_ratio': float((RISK[ch, d[changed]] < RISK[ch, q0[changed]]).mean())
                           if changed.any() else 0.0,
        }
        rows.append(row)
        print(f"[lam {lam:>5.1f}] change {row['change']:.4f}  "
              f"deal {row['deal_top1']:.4f}  risk {row['risk_top1']:.4f}  "
              f"a2c {row['a2c_agree']:.4f}  safer {row['safer_ratio']:.3f}", flush=True)

    report = {
        'meta': {
            'n': n, 'seed': args.seed, 'base_ckpt': args.ckpt, 'base_steps': steps,
            'a2c_ckpt': str(A2C_CKPT), 'protocol': 'A2c tl=0 实时危险度；单步诊断',
        },
        'base': {
            'deal_top1': float(d0.mean()),
            'a2c_safest_deal': float(a2c_deal.mean()),
            'q_gap_mean': float(gap.mean()),
            'q_gap_median': float(np.median(gap)),
            'risk_mean': float(RISK.mean()),
            'risk_max': float(RISK.max()),
        },
        'sweep': rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\n[out] {args.out}')


if __name__ == '__main__':
    main()
