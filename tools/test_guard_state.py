#!/usr/bin/env python3
"""Guard-P2 局面特征回归（obs 布局 / 分数反推 / 权重单调性）。

用法（mortal/ 里）：python ../tools/test_guard_state.py
"""
import glob
import os
import sys

import numpy as np

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

import guard_state as G  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name:<46s} {detail}')
    if not cond:
        FAILS.append(name)


def load_obs(month='202609', files_n=2):
    from libriichi.dataset import GameplayLoader
    files = sorted(glob.glob(f'../dataset/mjai/houou4/{month}/*.json.gz'))[:files_n]
    ld = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None,
                        augmented=False, always_include_kan_select=True)
    out = []
    for f in ld.load_gz_log_files(files):
        for g in f:
            out.extend(np.asarray(o, np.float32) for o in g.take_obs())
    return np.stack(out)


def test_layout(obs):
    print('== 1. obs 布局 ==')
    rows = list(range(6, 22))
    const = np.array([(obs[:, r].max(1) - obs[:, r].min(1) < 1e-6).mean() for r in rows])
    check('6..21 全为常量行', bool(const.min() > 0.999), f'min={const.min():.3f}')
    check('0..2 非常量（自家手牌）', bool((obs[:, 0].max(1) - obs[:, 0].min(1) > 1e-6).mean() > 0.5))
    rank = obs[:, G.RANK_ROWS:G.RANK_ROWS + 4].min(2)
    check('名次 one-hot 合法', bool((rank.sum(1) > 0.999).mean() > 0.999))
    ky = obs[:, G.KYOKU_ROWS:G.KYOKU_ROWS + 4].min(2)
    check('局数 one-hot 合法', bool((ky.sum(1) > 0.999).mean() > 0.999))
    # 分数两路一致性（raw < 30000 时才可比）
    raw = obs[:, 7:15:2].max(2) * 1e5
    small = obs[:, 8:15:2].max(2) * 3e4
    m = raw < 29_999
    ok = np.abs(raw - small)[m] <= 1.0
    check('分数两路一致（未饱和）', bool(ok.mean() > 0.999), f'{ok.mean():.4f}')
    tot = raw.sum(1)
    # 立直棒（供託）不在 scores 里：和 = 100000 - 1000*kyotaku，故只要求「≤100000 且差为 1000 的倍数」
    gap = 100_000 - tot
    ok = (gap >= -500) & (np.abs(gap / 1000 - np.round(gap / 1000)) < 0.05)
    check('4 家分数和 = 100000 − 1000×供託', bool(ok.mean() > 0.98), f'{ok.mean():.3f}')


def test_features(obs):
    print('== 2. 特征反推 ==')
    f = G.features(obs)
    n = len(obs)
    check('my_rank ∈ [0,3]', bool(((f['my_rank'] >= 0) & (f['my_rank'] <= 3)).all()))
    check('scores[:,0] 就是本家分数', bool(np.allclose(f['scores'][:, 0], f['my_score'])))
    # 名次与「比我分高的人数」一致（平分时允许偏差，故阈值 0.90）
    higher = (f['scores'][:, 1:] > f['my_score'][:, None]).sum(1)
    check('名次 ≈ 比我高分的人数（平分例外）', bool((higher == f['my_rank']).mean() > 0.90),
          f'{(higher == f["my_rank"]).mean():.3f}')
    l0 = f['lead'][f['my_rank'] == 0].mean()
    l2 = f['lead'][f['my_rank'] >= 2].mean()
    check('第 1 名平均 lead > 0，2 名以后 lead = 0', bool(l0 > 0 and abs(l2) < 1e-6),
          f'lead(1st)={l0:.0f} lead(≥2nd)={l2:.0f}')
    check('lead ≥ 0 / deficit ≥ 0', bool((f['lead'] >= 0).all() and (f['deficit'] >= 0).all()))
    check('riichi_cnt ∈ [0,3]', bool(((f['riichi_cnt'] >= 0) & (f['riichi_cnt'] <= 3)).all()))
    dist = np.bincount(f['riichi_cnt'].astype(int), minlength=4)
    check('立直家数分布合理（0 最多）', bool(dist[0] == dist.max()),
          f'dist={dist.tolist()}')
    tb = G.turn_bucket(f['progress'])
    frac = np.bincount(tb, minlength=3) / n
    check('早/中/晚 ≈ 25/50/25', bool(0.15 < frac[0] < 0.40 and 0.30 < frac[1] < 0.65),
          f'{np.round(frac, 3).tolist()}')


def test_weights(feats):
    print('== 3. 权重 ==')
    base = dict(my_score=np.array([25000., 25000., 25000.]),
                lead=np.array([0., 0., 0.]), deficit=np.array([0., 0., 0.]))
    mid = G.PROGRESS_Q[1]
    f = dict(lead=base['lead'], deficit=base['deficit'], my_score=base['my_score'],
             riichi_cnt=np.array([0., 0., 0.]), progress=np.array([mid, mid, mid]))
    check('恒等：0 立直 / 中巡 / 无点差 ⇒ 1.0', bool(np.allclose(G.weights(f), 1.0)))
    f1 = dict(f, riichi_cnt=np.array([1., 2., 3.]))
    w1 = G.weights(f1, use_turn=False)
    check('立直家数单调 ↑', bool(w1[0] < w1[1] < w1[2]), f'{w1.tolist()}')
    fe = dict(f, progress=np.array([0., mid, 100.]))
    wt = G.weights(fe, use_opp=False)
    check('晚巡 > 早巡', bool(wt[2] > wt[0]), f'{wt.tolist()}')
    fs = dict(f, lead=np.array([0., 13000., 0.]), deficit=np.array([0., 0., 13000.]),
              progress=np.array([mid, mid, 100.]))
    ws = G.weights(fs, use_opp=False, use_turn=False)
    check('领先 ⇒ 加权', bool(ws[1] > 1.0), f'{ws[1]:.3f}')
    ws2 = G.weights(fs, use_opp=False)
    check('垫底+晚巡 ⇒ 减权', bool(ws2[2] < 1.0), f'{ws2[2]:.3f}')
    fs2 = dict(fs, lead=np.full(3, 50_000.), riichi_cnt=np.array([3., 3., 3.]),
               progress=np.full(3, 100.))
    check('clamp 上限 2.0', bool(G.weights(fs2).max() <= 2.0 + 1e-6))
    check('单项可关：use_opp=False 时与立直家数无关',
          bool(np.allclose(G.weights(dict(f, riichi_cnt=np.array([0., 3., 1.])), use_opp=False),
                           G.weights(f, use_opp=False))))


def main():
    obs = load_obs()
    print(f'== obs: {obs.shape} ==')
    test_layout(obs)
    feats = G.features(obs)
    test_features(obs)
    test_weights(feats)
    print()
    print('ALL PASS' if not FAILS else f'FAIL {len(FAILS)}: {FAILS}')
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
