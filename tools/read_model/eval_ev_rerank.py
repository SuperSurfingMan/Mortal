#!/usr/bin/env python3
"""EV 重排的单步诊断（选超参用，不是判决）。

口径与 `eval_defense_guard.py` 完全一致：202609 固定 N 样本（seed 20261016）、
A2c ``tl=0`` 实时危险度、主模型 B/756k 冻结。对每个 (β, λ, τ, margin) 计算：

- ``change_vs_guard``  相对**当前线上守卫（λ=10, β=0）**的打牌变化率
- ``deal_top1``        选中牌是某家实际和了牌的比例（P0-B 同口径）
- ``risk_top1``        选中牌的平均 A2c 危险度
- ``ev_top1``          选中牌的 SP 期望价值（ev_norm，0..1）；``d_ev`` 相对守卫
- ``pwin_top1``        选中牌的 SP 和了概率；``d_pwin`` 相对守卫
- ``a2c_agree``        与 A2c「最安全牌」的一致率

⚠️ 单步指标只能证明「方向是否合理」（例如放铳率不升、EV/和了率升），
是否真变强必须看 ladder（双向 ≥5000 sets）；本脚本的输出不构成判决。

用法（mortal/ 里跑）：
    python ../tools/read_model/eval_ev_rerank.py --n 20000 \
        --betas 0,0.1,0.2,0.4,0.8 --lams 10 --margins none,0.3,0.8
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
import ev_rerank as E  # noqa: E402

_spec = importlib.util.spec_from_file_location('inject_p0_mod', HERE / 'inject_p0.py')
P0 = importlib.util.module_from_spec(_spec)
sys.modules['inject_p0_mod'] = P0
_spec.loader.exec_module(P0)

NUM_TILES = P0.NUM_TILES
ACTION_SPACE = P0.ACTION_SPACE
T = P0.T
BASE_CKPT = REPO / 'mortal/checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth'
A2C_CKPT = REPO / 'mortal/read_model_v2s_a2c/best.pth'


def load_nets():
    st = torch.load(BASE_CKPT, map_location='cpu', weights_only=False)
    cfg = st['config']
    ver = cfg['control'].get('version', 4)
    mortal = P0.MM.Brain(version=ver, **cfg['resnet']).to(P0.DEV).eval()
    mortal.load_state_dict(st['mortal'])
    dqn = P0.MM.DQN(version=ver).to(P0.DEV).eval()
    dqn.load_state_dict(st['current_dqn'])
    a2 = torch.load(A2C_CKPT, map_location='cpu', weights_only=False)
    c2 = a2['config']
    a2c = P0.M2.build_model_v2s(
        conv_channels=c2['conv_channels'], num_blocks=c2['num_blocks'],
        with_aux=not c2['no_aux'], wait_head=c2.get('wait_head', 'flat'),
        tl_mode=c2.get('tl_mode', 'pooled'),
        next_tedashi=c2.get('next_tedashi_weight', 0.0) > 0,
        tile_order=c2.get('tile_order', False),
    ).to(P0.DEV).eval()
    a2c.load_state_dict(a2['model'])
    return mortal, dqn, a2c, int(st.get('steps', 0))


def parse_grid(betas, lams, margins):
    out = [(0.0, 0.0, None, 'Q-only'), (0.0, 10.0, None, 'guard λ=10（线上）')]
    for b in betas:
        for lam in lams:
            out.append((b, lam, None, f'β={b} λ={lam}'))
    for mg in margins:
        out.append((0.4, 10.0, mg, f'β=0.4 λ=10 margin={mg}'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--betas', default='0.1,0.2,0.4,0.8')
    ap.add_argument('--lams', default='10')
    ap.add_argument('--margins', default='0.3,0.8')
    ap.add_argument('--out', default=str(C.OUT_DIR / 'ev_rerank_20k.json'))
    args = ap.parse_args()
    betas = [float(x) for x in args.betas.split(',') if x.strip()]
    lams = [float(x) for x in args.lams.split(',') if x.strip()]
    margins = [None if x.strip() == 'none' else float(x)
               for x in args.margins.split(',') if x.strip()]
    grid = parse_grid(betas, lams, margins)

    files = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4/202609/**/*.json.gz'),
                             recursive=True))
    mortal, dqn, a2c, steps = load_nets()
    print(f'[ev] base step={steps} n={args.n} grid={len(grid)}', flush=True)

    acc = {i: dict(n=0, deal=0.0, risk=0.0, ev=0.0, pwin=0.0, a2c=0, change=0)
           for i in range(len(grid))}
    total = 0
    for buf, _src in P0.iter_eval_chunks(files, args.n, args.seed, args.chunk):
        take = min(len(buf), args.n - total)
        if take <= 0:
            break
        buf = buf[:take]
        b = T.collate(buf)
        _AP, WP = P0.a2c_forward(a2c, b, args.chunk, zero_tl=True)
        p_t = (1.0 - WP[..., NUM_TILES]).clamp(0, 1)
        risk = (p_t[..., None] * WP[..., :NUM_TILES]).sum(1).float().cpu().numpy()
        danger = (b['wait'][..., :NUM_TILES] > 0).any(1).cpu().numpy()
        with torch.no_grad():
            phi = mortal(b['obs'].to(P0.DEV))
            own_t = b['own_hand'].to(P0.DEV) > 0
            mask = torch.zeros(phi.shape[0], ACTION_SPACE, dtype=torch.bool,
                               device=P0.DEV)
            mask[:, :NUM_TILES] = own_t
            q_all = dqn(phi, mask)
        q = q_all[:, :NUM_TILES].float().cpu().numpy()
        own = own_t.cpu().numpy()
        obs = b['obs'].float().cpu().numpy()
        n = q.shape[0]
        ar = np.arange(n)
        feats = E.sp_features(obs)
        cov = feats['covered'] & own
        a2c_safest = np.where(own, risk, np.inf).argmin(-1)
        guard = E.rerank_scores(q, obs, risk, beta=0.0, lam=10.0, own=own)

        for i, (beta, lam, margin, _name) in enumerate(grid):
            out = E.rerank_scores(q, obs, risk, beta=beta, lam=lam, tau=0.0,
                                  margin=margin, own=own)
            pick = out['pick']
            a = acc[i]
            a['n'] += n
            a['deal'] += float(danger[ar, pick].sum())
            a['risk'] += float(risk[ar, pick].sum())
            a['ev'] += float(out['ev_chosen'].sum())
            a['pwin'] += float(feats['p_win'][ar, pick].sum())
            a['a2c'] += int((pick == a2c_safest).sum())
            a['change'] += int((pick != guard['pick']).sum())
            if i == 0:
                pass
        total += n
        print(f'[data] {total}/{args.n}', flush=True)
        if total >= args.n:
            break

    rows = []
    for i, (beta, lam, margin, name) in enumerate(grid):
        a = acc[i]
        n = max(a['n'], 1)
        rows.append({
            'name': name, 'beta': beta, 'lam': lam, 'margin': margin, 'n': a['n'],
            'change_vs_guard': a['change'] / n,
            'deal_top1': a['deal'] / n,
            'risk_top1': a['risk'] / n,
            'ev_top1': a['ev'] / n,
            'pwin_top1': a['pwin'] / n,
            'a2c_agree': a['a2c'] / n,
        })
    base = rows[1]  # guard λ=10
    for r in rows:
        r['d_ev'] = r['ev_top1'] - base['ev_top1']
        r['d_pwin'] = r['pwin_top1'] - base['pwin_top1']
        r['d_deal'] = r['deal_top1'] - base['deal_top1']
        print(f"{r['name']:<24s} change {r['change_vs_guard']:.4f}  deal {r['deal_top1']:.4f}"
              f" ({r['d_deal']:+.4f})  risk {r['risk_top1']:.4f}  ev {r['ev_top1']:.4f}"
              f" ({r['d_ev']:+.4f})  pwin {r['pwin_top1']:.4f} ({r['d_pwin']:+.4f})"
              f"  a2c {r['a2c_agree']:.4f}", flush=True)

    report = {
        'meta': {'n': total, 'seed': args.seed, 'base_ckpt': str(BASE_CKPT),
                 'base_steps': steps, 'a2c_ckpt': str(A2C_CKPT),
                 'protocol': 'A2c tl=0；单步诊断（非判决）',
                 'formula': 'Q + beta*ev_norm - lam*max(risk-tau,0)，可选 margin 限域'},
        'rows': rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\n[out] {args.out}')


if __name__ == '__main__':
    main()
