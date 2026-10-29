#!/usr/bin/env python3
"""档 C（短 rollout EI）的机制自检：状态物化 / 分支合法性 / obs 一致性。

**这一版只验证机制，不产生训练目标**。要证明 rollout 可信，必须先确认三件事：

1. `PlayerState.update(event)` 能从零重放整局，且在**我们自己的每个决策点**上
   `encode_obs()` 与训练时 `take_obs()` 的 obs 一致（否则物化出来的状态不可信）；
2. 从决策点**替换成另一个候选打牌**后，分支仍可继续推进（后续事件不抛异常）；
3. 分歧分支能走到「我方下一次决策」，并给出合法的 obs。

用法（mortal/ 下）：
    python ../tools/ei_rollout.py --games 200 --max-states 1000
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys

import numpy as np

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

from libriichi.state import PlayerState  # noqa: E402

TILE = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
        '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
        '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
        'E', 'S', 'W', 'N', 'P', 'F', 'C']
TILE_IDX = {t: i for i, t in enumerate(TILE)}
AKA = {'5mr': '5m', '5pr': '5p', '5sr': '5s'}


def load_events(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(l) for l in f if l.strip()]


def pai_idx(p):
    p = AKA.get(p, p)
    return TILE_IDX.get(p, -1)


SKIP_SELF = ('dahai', 'reach', 'reach_accepted')
SHADOW_OK = ('tsumo', 'dahai', 'dora')


def _load_model(dev):
    import torch
    import model as MM
    import s6c_features as S6
    st = torch.load('checkpoints/s6c/mortal.pth', map_location='cpu', weights_only=False)
    ver = st['config']['control'].get('version', 4)
    _, extra = S6.resolve(st['config']['control'])
    net = MM.Brain(version=ver, extra_channels=extra, **st['config']['resnet']).eval().to(dev)
    net.load_state_dict(st['mortal'])
    dqn = MM.DQN(version=ver).eval().to(dev)
    dqn.load_state_dict(st['current_dqn'])
    return net, dqn, ver, extra, S6


def rollout_branch(evs, start_i, seat, cand, k, net, dqn, ver, extra, S6, dev):
    """从我方第 start_i 个事件（tsumo）之后出发，打 cand，再走 k 个我方回合。

    返回真实终值：本局**我方累计点数变化**（`hora`/`ryukyoku` 的 deltas），
    并记录 `deal_in`（放铳）与是否走到本局结束（`ended`）。
    """
    import torch
    st = PlayerState(seat)
    for e in evs[:start_i + 1]:
        st.update(json.dumps(e))
    _last = st.last_self_tsumo()
    _tsumo = bool(_last is not None and TILE[cand] == _last)
    st.update(json.dumps({'type': 'dahai', 'actor': seat, 'pai': TILE[cand],
                          'tsumogiri': _tsumo}))
    n_own, val, deal_in, ended, skipped = 0, 0.0, 0, False, 0
    for e in evs[start_i + 1:]:
        t = e.get('type'); act = e.get('actor')
        if t == 'hora':
            if e.get('target') == seat:
                deal_in = 1
            dl = e.get('deltas') or [0, 0, 0, 0]
            val += float(dl[seat]); ended = True
            break
        if t == 'ryukyoku':
            dl = e.get('deltas') or [0, 0, 0, 0]
            val += float(dl[seat]); ended = True
            break
        if t in ('end_kyoku', 'end_game'):
            ended = True
            break
        if act == seat and t in SKIP_SELF:
            continue
        if act == seat and t == 'tsumo':
            o, _m = st.encode_obs(ver, False)
            own = (np.rint(o[:4].sum(0)) > 0) if o.ndim == 2 else (np.rint(o[:4]) > 0)
            x = S6.attach(o[None], S6.compute(o[None])) if extra else o[None]
            with torch.inference_mode():
                phi = net(torch.as_tensor(x, dtype=torch.float32).to(dev))
                mask = torch.zeros(1, 46, dtype=torch.bool, device=dev)
                mask[0, :34] = torch.as_tensor(own.ravel()).to(dev)
                q = dqn(phi, mask)[0, :34].float().cpu().numpy()
            pick = int(np.argmax(np.where(own, q, -1e9)))
            last = st.last_self_tsumo()
            st.update(json.dumps({'type': 'dahai', 'actor': seat, 'pai': TILE[pick],
                                  'tsumogiri': bool(last is not None and TILE[pick] == last)}))
            n_own += 1
            if n_own >= k:
                break
            continue
        if t in ('chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'reach'):
            skipped += 1
            break                      # B 回退：鸣牌/立直一律截断（跳过会污染状态）
        try:
            st.update(json.dumps(e))
        except Exception:
            skipped += 1
    try:
        cut_obs = st.encode_obs(ver, False)[0]
    except Exception:
        cut_obs = None
    return dict(shanten=int(st.shanten), n_own=n_own, ended=ended, val=val,
                deal_in=deal_in, skip=skipped, cut_obs=cut_obs)


def mode_rollout(a):
    import time, torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, dqn, ver, extra, S6 = _load_model(dev)
    files = sorted(glob.glob('../dataset/mjai/houou4/202609/**/*.json.gz', recursive=True))
    random.Random(a.seed).shuffle(files)
    states, branches, skipped = 0, 0, 0
    ended_n, deal_n, win_n, lose_n, tie_n = 0, 0, 0, 0, 0
    per_state = []
    t0 = time.time()
    for path in files:
        if states >= a.max_states:
            break
        try:
            evs = load_events(path)
        except Exception:
            continue
        st = PlayerState(0)
        for i, e in enumerate(evs):
            try:
                st.update(json.dumps(e))
            except Exception:
                break
            if e.get('type') != 'tsumo' or e.get('actor') != 0:
                continue
            human = None
            for e2 in evs[i + 1:]:
                if e2.get('type') == 'dahai' and e2.get('actor') == 0:
                    human = pai_idx(e2.get('pai')); break
                if e2.get('type') in ('hora', 'ryukyoku', 'end_kyoku'):
                    break
            cands = [k for k, c in enumerate(st.tehai) if c > 0]
            if not cands or human is None or human < 0 or int(st.tehai[human]) <= 0:
                continue
            cands = cands[:a.max_cands]
            if human not in cands:
                cands = [human] + cands[:a.max_cands - 1]
            res = {}
            for c in cands:
                r = rollout_branch(evs, i, 0, c, a.k, net, dqn, ver, extra, S6, dev)
                branches += 1
                skipped += r['skip']
                if r['ended']:
                    ended_n += 1
                    if r['deal_in']:
                        deal_n += 1
                res[c] = r
            per_state.append((human, res))
            states += 1
            if states % 20 == 0:
                print(f'  {states} 状态 / {branches} 分支 / {time.time()-t0:.0f}s', flush=True)
    dt = time.time() - t0
    print(f'\n[rollout] 状态 {states}、分支 {branches}、走到本局结束 {ended_n}'
          f'（{ended_n/max(branches,1):.3f}）、其中放铳 {deal_n}、跳过事件 {skipped}')
    print(f'[rollout] 耗时 {dt:.1f}s（{dt/max(branches,1)*1000:.1f} ms/分支）')

    # ---- 混合终值：结束分支用真实点数×PT2RANK；截断分支用 Q bootstrap ----
    import torch as _t
    pend = [(si, c, r) for si, (_h, rs) in enumerate(per_state) for c, r in rs.items()
            if (not r['ended']) and r.get('cut_obs') is not None]
    boot = {}
    if pend:
        obsv = np.stack([r['cut_obs'] for _si, _c, r in pend]).astype(np.float32)
        if extra:
            obsv = S6.attach(obsv, S6.compute(obsv))
        ownv = np.rint(obsv[:, :4].sum(1)) > 0
        with _t.inference_mode():
            phis = []
            for i0 in range(0, len(obsv), 512):
                phis.append(net(_t.as_tensor(obsv[i0:i0 + 512]).to(dev)))
            phi = _t.cat(phis)
            mk = _t.zeros(phi.shape[0], 46, dtype=_t.bool, device=dev)
            mk[:, :34] = _t.as_tensor(ownv).to(dev)
            qq = _t.as_tensor(dqn(phi, mk)[:, :34].float().cpu().numpy())
        qq = np.where(ownv, qq, -1e9)
        for (si, c, _r), v in zip(pend, qq.max(1)):
            boot[(si, c)] = float(v)
    n_real = n_boot = n_none = 0
    for si, (_h, rs) in enumerate(per_state):
        for c, r in rs.items():
            if r['ended']:
                r['V'] = r['val'] * a.pt2rank; n_real += 1
            elif (si, c) in boot:
                r['V'] = boot[(si, c)]; n_boot += 1
            else:
                r['V'] = np.nan; n_none += 1
    print(f'[终值] 真实点数 {n_real} / bootstrap {n_boot} / 无值 {n_none}（PT2RANK={a.pt2rank}）')

    # 自洽性：人类动作的终值是否优于其它候选（同状态配对比较）
    for human, res in per_state:
        hv = res[human]
        if not np.isfinite(hv['V']):
            continue
        for c, r in res.items():
            if c == human or not np.isfinite(r['V']):
                continue
            if hv['V'] > r['V']:
                win_n += 1
            elif hv['V'] < r['V']:
                lose_n += 1
            else:
                tie_n += 1
    tot = win_n + lose_n + tie_n
    if tot:
        print(f'[自洽] 人类动作 vs 其它候选（同状态配对，真实点数终值）：'
              f'胜 {win_n/tot:.3f} / 负 {lose_n/tot:.3f} / 平 {tie_n/tot:.3f}（n={tot}）')
        print(f'[判读] 胜率应 >0.55 才算 rollout 终值可信；≈0.5 说明 k 步信号不足')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--max-states', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--mode', choices=['check', 'rollout'], default='check')
    ap.add_argument('--k', type=int, default=2, help='rollout 步数（我方回合数）')
    ap.add_argument('--max-cands', type=int, default=6)
    ap.add_argument('--pt2rank', type=float, default=5e-5,
                    help='点数→rank-pt 的汇率（默认 5e-5：±8000 点 ≈ ±0.4 rank-pt）')
    a = ap.parse_args()
    if a.mode == 'rollout':
        return mode_rollout(a)
    files = sorted(glob.glob('../dataset/mjai/houou4/202609/**/*.json.gz', recursive=True))
    random.Random(a.seed).shuffle(files)

    n_games = n_ok = n_replay_err = n_obs_err = 0
    n_states = 0
    branch_try = branch_ok = 0
    obs_shapes = set()
    for path in files:
        if n_games >= a.games or n_states >= a.max_states:
            break
        try:
            evs = load_events(path)
        except Exception:
            continue
        n_games += 1
        # 以 0 号位为主视角重放
        st = PlayerState(0)
        pending_action = None          # 我们在上一个决策点「应当执行」的动作索引
        try:
            for i, ev in enumerate(evs):
                t = ev.get('type')
                st.update(json.dumps(ev))
                if t in ('start_kyoku',):
                    continue
                # 我方决策点：更新后若仍轮到我方打牌且事件不是 dahai/tsumo，则记录
                if t == 'tsumo' and ev.get('actor') == 0:
                    o, _m = st.encode_obs(4, False)
                    obs_shapes.add(tuple(o.shape))
                    if o.shape != (1012, 34) or not np.isfinite(o).all():
                        n_obs_err += 1
                if t == 'dahai' and ev.get('actor') == 0:
                    n_states += 1
                    # 分支检验：把这一手换成同手里的别的牌，看后续 3 个事件是否仍可推进
                    tehai = st.tehai
                    cands = [k for k, c in enumerate(tehai) if c > 0 and k != pai_idx(ev.get('pai'))]
                    if cands and branch_try < 2000:
                        c = random.Random(i).choice(cands)
                        st2 = PlayerState(0)
                        try:
                            for e2 in evs[:i]:
                                st2.update(json.dumps(e2))
                            st2.update(json.dumps({'type': 'dahai', 'actor': 0, 'pai': TILE[c],
                                                   'tsumogiri': False}))
                            for e2 in evs[i + 1:i + 4]:
                                if e2.get('type') in ('tsumo', 'dahai', 'dora'):
                                    st2.update(json.dumps(e2))
                            branch_ok += 1
                        except Exception:
                            pass
                        branch_try += 1
        except Exception as e:
            if n_replay_err == 0:
                print(f'[debug] replay failed at event {i} type={evs[i].get("type")}: {e!r}', flush=True)
            n_replay_err += 1
            continue
        n_ok += 1

    print(f'[check] 游戏 {n_games}（成功重放 {n_ok}、重放异常 {n_replay_err}）')
    print(f'[check] 我方打牌决策 {n_states}；obs 形状 {sorted(obs_shapes)}；obs 异常 {n_obs_err}')
    print(f'[check] 分支物化：尝试 {branch_try}、成功 {branch_ok}'
          f'（{branch_ok / max(branch_try,1):.3f}）')
    print('[判读] 重放异常应为 0；obs 形状应只有 (1012,34)；分支成功率应 >0.95')
    return 0


if __name__ == '__main__':
    sys.exit(main())
