#!/usr/bin/env python3
"""档 B 的 teacher：1-ply 结构化 expectimax（SP 进攻 − κ·期望放铳损失），并做离线对照。

teacher（对候选打牌 a ∈ 自家手牌）：

    V(a) = EV_pts(a) − κ · Σ_j risk_j(a) · loss_j
    EV_pts(a) = ev_norm(a) · max_ev(s)                # 来自 v4 obs 的 SP 表（点）
    risk_j(a) = P(听_j) · P(待牌_j = a)                # A2c（zero-tl）
    loss_j    = 3000 + 4000·[立直_j] + 250·max(0, progress−10)

κ 的基准来自守卫 λ=10 的量纲：`κ ≈ λ / E[loss] ≈ 10/4000 = 0.0025`；本脚本扫 {0.001,0.002,0.004}。

输出的四张表回答「teacher 该替换哪些状态」：
1. overall：Q 与人类一致率、teacher 与人类一致率；
2. 按 Q 边际分层（低/中/高）：teacher 站 Q（=模型）还是站人类，还是两者都不是；
3. 三个候选（人类 / Q / teacher）的 SP 期望价值、和了率、危险度；
4. κ 敏感性。

用法（mortal/ 下）：
    python ../tools/ei_teacher.py --n 20000
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

import ev_rerank as E          # noqa: E402
import guard_state as G        # noqa: E402
import s6c_features as S6      # noqa: E402
from defense_guard import DefenseGuard  # noqa: E402
import model as MM            # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

BS = 512
CKPT = 'checkpoints/s6c/mortal.pth'
BASE_PT, RIICHI_PT, TURN_PT = 3000.0, 4000.0, 250.0


def load_net(dev):
    st = torch.load(CKPT, map_location='cpu', weights_only=False)
    ver = st['config']['control'].get('version', 4)
    _, extra = S6.resolve(st['config']['control'])
    net = MM.Brain(version=ver, extra_channels=extra, **st['config']['resnet']).eval().to(dev)
    net.load_state_dict(st['mortal'])
    dqn = MM.DQN(version=ver).eval().to(dev)
    dqn.load_state_dict(st['current_dqn'])
    return net, dqn, extra, int(st.get('steps', 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--month', default='202609')
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--kappas', default='0.001,0.002,0.004')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, dqn, extra, steps = load_net(dev)
    guard = DefenseGuard(lambda_=10.0, device=dev)      # 只用它的 risk()
    kappas = [float(x) for x in a.kappas.split(',') if x.strip()]
    print(f'[teacher] ckpt steps={steps} extra={extra} n={a.n} dev={dev} kappas={kappas}', flush=True)

    files = sorted(glob.glob(f'../dataset/mjai/houou4/{a.month}/**/*.json.gz', recursive=True))
    random.Random(a.seed).shuffle(files)
    ld = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None,
                        augmented=False, always_include_kan_select=True)
    OBS, Y, Q, RISK = [], [], [], []
    got = 0
    for f in files:
        if got >= a.n:
            break
        try:
            chunks = list(ld.load_gz_log_files([f]))
        except Exception:
            continue
        for games in chunks:
            for game in games:
                try:
                    obs = np.stack([np.asarray(o, np.float32) for o in game.take_obs()])
                    acts = np.asarray(game.take_actions())
                except Exception:
                    continue
                m = acts < 34
                if not m.any():
                    continue
                o, y = obs[m], acts[m]
                own = np.rint(o[:, :4].sum(1)) > 0
                x = S6.attach(o, S6.compute(o)) if extra else o
                qs, rs = [], []
                with torch.inference_mode():
                    for i in range(0, len(o), BS):
                        xt = torch.as_tensor(x[i:i + BS]).to(dev)
                        phi = net(xt)
                        mask = torch.zeros(phi.shape[0], MM.ACTION_SPACE, dtype=torch.bool, device=dev)
                        mask[:, :34] = torch.as_tensor(own[i:i + BS]).to(dev)
                        qs.append(dqn(phi, mask)[:, :34].float().cpu())
                        rs.append(guard.risk(torch.as_tensor(o[i:i + BS]).to(dev)).float().cpu())
                OBS.append(o); Y.append(y); Q.append(torch.cat(qs).numpy()); RISK.append(torch.cat(rs).numpy())
                got += len(y)
                if got >= a.n:
                    break
            if got >= a.n:
                break
    OBS = np.concatenate(OBS)[:a.n]; Y = np.concatenate(Y)[:a.n]
    Q = np.concatenate(Q)[:a.n]; RISK = np.concatenate(RISK)[:a.n]
    own = np.rint(OBS[:, :4].sum(1)) > 0
    n = len(Y); ar = np.arange(n)
    print(f'[teacher] 收集 {n:,} 打牌决策', flush=True)

    sp = E.sp_features(OBS)
    feats = G.features(OBS)
    rr = np.zeros((n, 3), np.float32)
    base = 132
    for j in range(3):
        for i in range(24):
            rr[:, j] = np.maximum(rr[:, j], (OBS[:, base + 327 * j + 8 * i + 7].sum(1) > 0.5))
    late = np.maximum(feats['progress'] - 10.0, 0.0)
    loss = BASE_PT + RIICHI_PT * rr + TURN_PT * late[:, None]            # (n,3)
    ev_pts = sp['ev_norm'] * sp['max_ev'][:, None]                        # (n,34) 点
    risk_w = (RISK[:, None, :] * loss[:, :, None]).sum(1)                 # (n,34) 概率×点

    Qm = np.where(own, Q, -1e9)
    order = np.argsort(-Qm, axis=1)
    top1 = order[:, 0]
    margin = Qm[ar, top1] - Qm[ar, order[:, 1]]
    dis = top1 != Y
    print(f'\n[1] Q 与人类一致 {1-dis.mean():.4f}（分歧 {dis.mean():.4f}）')

    # 候选合法性：teacher 只在自家手牌里选；SP 未覆盖的候选记 ev=0（保守）
    valid = own & (sp['covered'] | (ev_pts > 0))
    for kap in kappas:
        V = np.where(valid, ev_pts - kap * risk_w, -1e18)
        T = np.argmax(np.where(own, V, -1e18), axis=1)
        agree_h = float((T == Y).mean())
        dis_idx = np.where(dis)[0]
        side_q = float((T[dis_idx] == top1[dis_idx]).mean())
        side_h = float((T[dis_idx] == Y[dis_idx]).mean())
        qs = np.quantile(margin[dis], [1/3, 2/3])
        parts = []
        for lo, hi, tag in ((-1e9, qs[0], '低'), (qs[0], qs[1], '中'), (qs[1], 1e9, '高')):
            m = dis_idx[(margin[dis_idx] > lo) & (margin[dis_idx] <= hi)]
            if len(m):
                parts.append(f'{tag}(n={len(m)}) 站Q {float((T[m]==top1[m]).mean()):.2f}/站人 {float((T[m]==Y[m]).mean()):.2f}')
        print(f'\n[κ={kap}] teacher 与人类一致 {agree_h:.4f}；在 Q≠人类处：站Q {side_q:.3f} / 站人 {side_h:.3f}')
        print('   ' + '  '.join(parts))

    # 三候选的质量对比（在 Q≠人类 的状态上）
    kap = kappas[len(kappas) // 2]
    V = np.where(valid, ev_pts - kap * risk_w, -1e18)
    T = np.argmax(np.where(own, V, -1e18), axis=1)
    d = np.where(dis)[0]
    print(f'\n[3] 在 {len(d):,} 个 Q≠人类 的状态上（κ={kap}）：')
    print(f'    {"候选":<10}{"SP EV(点)":>12}{"p_win":>9}{"risk":>9}{"risk×loss(点)":>15}')
    for tag, pick in (('人类', Y[d]), ('Q(top1)', top1[d]), ('teacher', T[d])):
        print(f'    {tag:<10}{ev_pts[d, pick].mean():>12.0f}{sp["p_win"][d, pick].mean():>9.4f}'
              f'{RISK[d, pick].mean():>9.4f}{risk_w[d, pick].mean():>15.0f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
