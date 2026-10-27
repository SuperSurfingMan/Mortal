#!/usr/bin/env python3
"""M2：Guard-P2 状态依赖配置的离线 frontier（20k 固定集，选配置用，非判决）。

基线 = 线上固定守卫（λ=10, τ=0）。对每个配置输出：
  change_vs_fixed / deal_top1 / risk_top1 / p_win / ev_norm / a2c_agree

配置维度：
  --v-riichi           逐对手价值权重（放铳给立直家更贵），1.0 = 关
  --use-opp/turn/score 是否启用 w_opp / w_turn / w_score（幅度见 mortal/guard_state.py）
  --lambdas            以 λ0 覆盖（默认 10）
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
import guard_state as G  # noqa: E402

_spec = importlib.util.spec_from_file_location('inject_p0_mod', HERE / 'inject_p0.py')
P0 = importlib.util.module_from_spec(_spec)
sys.modules['inject_p0_mod'] = P0
_spec.loader.exec_module(P0)

NUM_TILES, ACTION_SPACE, T = P0.NUM_TILES, P0.ACTION_SPACE, P0.T
BASE_CKPT = REPO / 'mortal/checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth'
A2C_CKPT = REPO / 'mortal/read_model_v2s_a2c/best.pth'
RIVER_BASES = (132, 327, 522)


def load_nets():
    st = torch.load(BASE_CKPT, map_location='cpu', weights_only=False)
    cfg = st['config']; ver = cfg['control'].get('version', 4)
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
        tile_order=c2.get('tile_order', False)).to(P0.DEV).eval()
    a2c.load_state_dict(a2['model'])
    return mortal, dqn, a2c, int(st.get('steps', 0))


def opp_riichi(obs):
    out = np.zeros((obs.shape[0], 3), np.float32)
    for j, base in enumerate(RIVER_BASES):
        for i in range(24):
            out[:, j] = np.maximum(out[:, j], (obs[:, base + 8 * i + 7].sum(1) > 0.5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--v-riichi', type=float, default=1.3)
    ap.add_argument('--lambdas', default='10')
    ap.add_argument('--out', default=str(C.OUT_DIR / 'guard_state_20k.json'))
    a = ap.parse_args()
    lams = [float(x) for x in a.lambdas.split(',') if x.strip()]
    cfgs = [('fixed λ=10', dict(use_opp=False, use_turn=False, use_score=False))] + [
        ('risk_w(v_r=%.2f)' % a.v_riichi, dict(use_opp=False, use_turn=False, use_score=False)),
        ('w_opp', dict(use_opp=True, use_turn=False, use_score=False)),
        ('w_turn', dict(use_opp=False, use_turn=True, use_score=False)),
        ('w_opp×turn', dict(use_opp=True, use_turn=True, use_score=False)),
        ('w_opp×turn×score', dict(use_opp=True, use_turn=True, use_score=True)),
        ('wscore only', dict(use_opp=False, use_turn=False, use_score=True)),
    ]
    lam = lams[0]
    print(f'[guard-state] n={a.n} λ0={lam} v_riichi={a.v_riichi} cfgs={len(cfgs)}', flush=True)

    files = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4/202609/**/*.json.gz'), recursive=True))
    mortal, dqn, a2c, steps = load_nets()
    acc = {name: dict(n=0, deal=0.0, risk=0.0, pwin=0.0, ev=0.0, a2c=0, chg=0) for name, _ in cfgs}
    total = 0
    for buf, _src in P0.iter_eval_chunks(files, a.n, a.seed, a.chunk):
        take = min(len(buf), a.n - total)
        if take <= 0:
            break
        buf = buf[:take]
        b = T.collate(buf)
        _AP, WP = P0.a2c_forward(a2c, b, a.chunk, zero_tl=True)
        p_t = (1.0 - WP[..., NUM_TILES]).clamp(0, 1)
        risk_j = (p_t[..., None] * WP[..., :NUM_TILES]).float().cpu().numpy()   # (n,3,34)
        danger = (b['wait'][..., :NUM_TILES] > 0).any(1).cpu().numpy()
        with torch.no_grad():
            phi = mortal(b['obs'].to(P0.DEV))
            own_t = b['own_hand'].to(P0.DEV) > 0
            mask = torch.zeros(phi.shape[0], ACTION_SPACE, dtype=torch.bool, device=P0.DEV)
            mask[:, :NUM_TILES] = own_t
            q = dqn(phi, mask)[:, :NUM_TILES].float().cpu().numpy()
        own = own_t.cpu().numpy()
        obs = b['obs'].float().cpu().numpy()
        n = q.shape[0]; ar = np.arange(n)
        rr = opp_riichi(obs)
        v = np.where(rr > 0.5, a.v_riichi, 1.0)                                  # (n,3)
        risk = risk_j.sum(1)
        risk_w = (risk_j * v[:, :, None]).sum(1)
        feats = G.features(obs)
        a2c_safest = np.where(own, risk, np.inf).argmin(-1)
        fixed_pick = np.where(own, q - lam * risk, -np.inf).argmax(-1)

        for name, kw in cfgs:
            base_risk = risk_w if name.startswith('risk_w') else risk
            w = G.weights(feats, **kw)
            score = q - lam * w[:, None] * base_risk
            pick = np.where(own, score, -np.inf).argmax(-1)
            s = acc[name]
            s['n'] += n
            s['deal'] += float(danger[ar, pick].sum())
            s['risk'] += float(risk[ar, pick].sum())
            s['pwin'] += float(E.sp_features(obs)['p_win'][ar, pick].sum())
            s['ev'] += float(E.sp_features(obs)['ev_norm'][ar, pick].sum())
            s['a2c'] += int((pick == a2c_safest).sum())
            s['chg'] += int((pick != fixed_pick).sum())
        total += n
        print(f'[data] {total}/{a.n}', flush=True)
        if total >= a.n:
            break

    rows = []
    for name, kw in cfgs:
        s = acc[name]; n = max(s['n'], 1)
        rows.append(dict(name=name, n=s['n'], change=s['chg'] / n, deal=s['deal'] / n,
                         risk=s['risk'] / n, pwin=s['pwin'] / n, ev=s['ev'] / n,
                         a2c=s['a2c'] / n))
    base = rows[0]
    for r in rows:
        r['d_deal'] = r['deal'] - base['deal']
        r['d_pwin'] = r['pwin'] - base['pwin']
        r['d_ev'] = r['ev'] - base['ev']
        print(f"{r['name']:<20s} chg {r['change']:.4f} deal {r['deal']:.4f} ({r['d_deal']:+.4f}) "
              f"risk {r['risk']:.4f} pwin {r['pwin']:.4f} ({r['d_pwin']:+.4f}) "
              f"ev {r['ev']:.4f} a2c {r['a2c']:.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({'meta': {'n': total, 'seed': a.seed, 'steps': steps, 'lam': lam,
                        'v_riichi': a.v_riichi}, 'rows': rows},
              open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[out]', a.out)


if __name__ == '__main__':
    main()
