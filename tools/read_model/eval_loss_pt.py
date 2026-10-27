#!/usr/bin/env python3
"""小实验：把 risk 换成 `P(deal) × E[loss_pt]`（绝对值，点），再换算回 Q 的单位。

与 Guard-P2 的区别（也是它唯一还没被否证的点）：
  * Guard-P2 用**乘法权重** `v_j`（立直 1.3 / 副露 0.7）与全局 `w_state`，本质是「把 λ 换个大小」；
  * 这里是**加性期望损失**：`loss_j = 3000 + 4000·立直_j + 250·max(0, progress−10)`（点），
    再乘 `c`（点→rank-pt 的一次性标定，取「与固定 λ=10 的平均惩罚相等」）。

输出：总体 frontier + **分层**（按立直家数 / 按 loss 大小三分位）的变化与 deal/p_win 差，
用于判断「是否只是把 λ 调大」还是「在高价值局面才多弃」。
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
LAM = 10.0
BASE_PT, RIICHI_PT, TURN_PT = 3000.0, 4000.0, 250.0


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
    ap.add_argument('--scales', default='1,2,3')
    ap.add_argument('--out', default=str(C.OUT_DIR / 'loss_pt_20k.json'))
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(',') if x.strip()]
    files = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4/202609/**/*.json.gz'), recursive=True))
    mortal, dqn, a2c, steps = load_nets()
    print(f'[loss-pt] n={a.n} λ={LAM} loss_j={BASE_PT}+{RIICHI_PT}·riichi+{TURN_PT}·max(0,prog-10)', flush=True)

    def iter_batches(limit):
        tot = 0
        for buf, _src in P0.iter_eval_chunks(files, limit, a.seed, a.chunk):
            take = min(len(buf), limit - tot)
            if take <= 0:
                break
            buf = buf[:take]
            b = T.collate(buf)
            _AP, WP = P0.a2c_forward(a2c, b, a.chunk, zero_tl=True)
            p_t = (1.0 - WP[..., NUM_TILES]).clamp(0, 1)
            risk_j = (p_t[..., None] * WP[..., :NUM_TILES]).float().cpu().numpy()
            danger = (b['wait'][..., :NUM_TILES] > 0).any(1).cpu().numpy()
            with torch.no_grad():
                phi = mortal(b['obs'].to(P0.DEV))
                own_t = b['own_hand'].to(P0.DEV) > 0
                mask = torch.zeros(phi.shape[0], ACTION_SPACE, dtype=torch.bool, device=P0.DEV)
                mask[:, :NUM_TILES] = own_t
                q = dqn(phi, mask)[:, :NUM_TILES].float().cpu().numpy()
            obs = b['obs'].float().cpu().numpy()
            own = own_t.cpu().numpy()
            rr = opp_riichi(obs)
            feat = G.features(obs)
            late = np.maximum(feat['progress'] - 10.0, 0.0)
            loss = BASE_PT + RIICHI_PT * rr + TURN_PT * late[:, None]
            risk = risk_j.sum(1)
            riskp = (risk_j * loss[:, :, None]).sum(1)
            tot += q.shape[0]
            yield q, own, obs, risk, riskp, danger, rr, tot

    # ① 标定 c：前 2000 样本上「平均惩罚 = λ·平均 risk」
    sr = sp_ = 0.0
    for _q, _own, _obs, risk, riskp, _d, _rr, _t in iter_batches(min(a.n, 2000)):
        sr += float(risk.sum()); sp_ += float(riskp.sum())
    c = LAM * (sr / max(sp_, 1e-9))
    print(f'[calib] c={c:.5f}（点→rank-pt；使平均惩罚与 λ=10 相同）', flush=True)

    # ② 全量评估 + 分层
    acc = {}
    def add(key, d_deal, d_pwin, n):
        s = acc.setdefault(key, dict(n=0, d=0.0, p=0.0))
        s['n'] += n; s['d'] += d_deal; s['p'] += d_pwin

    total = 0
    for q, own, obs, risk, riskp, danger, rr, tot in iter_batches(a.n):
        n = q.shape[0]; ar = np.arange(n)
        sp = E.sp_features(obs)['p_win']
        fixed = np.where(own, q - LAM * risk, -np.inf).argmax(-1)
        df, pf = danger[ar, fixed], sp[ar, fixed]
        add('fixed', float(df.sum()), float(pf.sum()), n)
        for k in scales:
            pick = np.where(own, q - c * k * riskp, -np.inf).argmax(-1)
            dn, pn = danger[ar, pick], sp[ar, pick]
            add(f'scale{k:g}', float(dn.sum()), float(pn.sum()), n)
            add(f'd{k:g}', float(dn.sum() - df.sum()), float(pn.sum() - pf.sum()), n)
        new = np.where(own, q - c * riskp, -np.inf).argmax(-1)
        dn, pn = danger[ar, new], sp[ar, new]
        for k in range(4):
            m = (rr.sum(1) == k)
            if m.any():
                add(f'riichi{k}', float(dn[m].sum() - df[m].sum()),
                    float(pn[m].sum() - pf[m].sum()), int(m.sum()))
        lmax = riskp.max(1)
        qs = np.quantile(lmax, [1/3, 2/3])
        for tag, m in (('low', lmax <= qs[0]), ('mid', (lmax > qs[0]) & (lmax <= qs[1])),
                       ('high', lmax > qs[1])):
            if m.any():
                add(f'loss_{tag}', float(dn[m].sum() - df[m].sum()),
                    float(pn[m].sum() - pf[m].sum()), int(m.sum()))
        total = tot
        if total >= a.n:
            break
    out = {k: dict(n=s['n'], deal=s['d'] / s['n'], pwin=s['p'] / s['n']) for k, s in acc.items()}
    for k in ['fixed'] + [f'scale{x:g}' for x in scales]:
        print(f'{k:<10s} n={out[k]["n"]:>6d}  deal {out[k]["deal"]:.5f} ({(out[k]["deal"]/out["fixed"]["deal"]-1)*100:+.2f}%)  '
              f'pwin {out[k]["pwin"]:.5f} ({(out[k]["pwin"]/out["fixed"]["pwin"]-1)*100:+.2f}%)')
    for k in ('riichi0', 'riichi1', 'riichi2', 'riichi3', 'loss_low', 'loss_mid', 'loss_high'):
        if k in out:
            print(f'{k:<10s} n={out[k]["n"]:>6d}  Δdeal {out[k]["deal"]:+.5f}  Δpwin {out[k]["pwin"]:+.5f}')
    json.dump({'meta': {'n': total, 'c': c, 'lam': LAM, 'seed': a.seed, 'steps': steps}, 'rows': out},
              open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[out]', a.out)


if __name__ == '__main__':
    main()
