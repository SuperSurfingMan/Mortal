#!/usr/bin/env python3
"""EV 重排回归：obs 布局 / SP 特征提取 / 打分与约束。

覆盖：
  1. v4 obs 布局（889/890 常量、max_ev 一致性、SP 覆盖 ⊆ 打牌候选、可用率 ≈ shanten≤3）；
  2. SP 表内部关系（p_win ≤ p_tenpai、ev/p_win ∈ [0,1]、每列单调不增/不减不假设）；
  3. ev_neutral 只补未覆盖项；
  4. rerank_scores 的四个契约：β=λ=0 不动 / 大 β 选高 EV / margin 限域 / 只在自家手牌里选；
  5. EVRerank 钩子输出（argmax = pick，其余置 -1e9）。

用法（mortal/ 里跑）：
    python ../tools/test_ev_rerank.py
"""
import glob
import os
import sys

import numpy as np

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

import ev_rerank as E  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name:<44s} {detail}')
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
    print('== 1. v4 obs 布局 / SP 特征 ==')
    feats = E.sp_features(obs)
    own = np.rint(obs[:, :4].sum(1)).astype(int) > 0
    sh = obs[:, 862:869].min(2).argmax(1)
    const889 = np.all(obs[:, E.SP_MAXEV_1E5].max(1) - obs[:, E.SP_MAXEV_1E5].min(1) < 1e-6)
    const890 = np.all(obs[:, E.SP_MAXEV_3E4].max(1) - obs[:, E.SP_MAXEV_3E4].min(1) < 1e-6)
    check('889/890 是常量行（max_ev）', bool(const889 and const890))
    a = obs[:, E.SP_MAXEV_1E5].max(1).astype(np.float64) * 1e5
    b = obs[:, E.SP_MAXEV_3E4].max(1).astype(np.float64) * 3e4
    # 1e5 列不饱和（<1e5）；3e4 列在 max_ev≥3e4 时饱和 → 只比较未饱和样本
    ok = np.abs(a - b) <= np.maximum(1.0, 1e-3 * np.maximum(a, b))
    check('max_ev 两路一致（未饱和样本）', bool(ok[a < 99_999].mean() > 0.999),
          f'{ok[a < 99_999].mean():.4f}')
    check('ev_norm ∈ [0,1]', bool((feats['ev_norm'] >= -1e-6).all() and (feats['ev_norm'] <= 1 + 1e-6).all()))
    cov = feats['covered']
    check('SP 覆盖 ⊆ 自家手牌', bool((cov & ~own).sum() == 0))
    cand = obs[:, E.SP_DISCARD_CAND] > 0.5
    has_cand = cand.any(1)
    rate = float(cov.any(1).mean())
    hi = float((sh <= 3).mean())
    # SP 只在 shanten≤3 且「能打牌」且有保向听候选时计算 → 覆盖率 < sh≤3 是预期行为。
    # ⚠️ obs 的向听 one-hot 在 3n+2（自摸后）是**摸牌前**的旧值（S6c 已实测），
    #    所以允许 sh==4 出现覆盖（真值 3），但 sh≥5 必须为 0。
    check('SP 覆盖 ⊆ obs向听≤4（允许自摸后滞后一抽）',
          bool((cov.any(1) & (sh >= 5)).sum() == 0),
          f'滞后样本 {int((cov.any(1) & (sh == 4)).sum())} 个')
    sub = has_cand & (sh <= 3)
    sub_rate = float(cov.any(1)[sub].mean()) if sub.any() else 0.0
    check('可打牌且 sh≤3 时覆盖率 ≥ 0.95', sub_rate >= 0.95,
          f'cov={rate:.3f} sh≤3={hi:.3f} 子集内={sub_rate:.3f}')
    both = cov & (feats['p_win'] > 0)
    check('p_win ≤ p_tenpai（覆盖处）', bool((feats['p_win'][both] <= feats['p_tenpai'][both] + 1e-6).all()))
    check('SP 覆盖 ⊆ 打牌候选行 874', bool((cov & ~cand).sum() == 0))
    return feats, own


def test_neutral(feats, own):
    print('== 2. ev_neutral ==')
    ev = E.ev_neutral(feats, own)
    cov = feats['covered'] & own
    check('已覆盖项不被改写', bool(np.allclose(ev[cov], feats['ev_norm'][cov])))
    fill_ok = True
    for i in range(min(len(ev), 200)):
        idx = np.where(cov[i])[0]
        if idx.size:
            want = float(feats['ev_norm'][i][idx].mean())
            got = ev[i][~cov[i]]
            if got.size and not np.allclose(got, want, atol=1e-6):
                fill_ok = False
                break
    check('未覆盖项 = 覆盖项均值', bool(fill_ok))


def fake_obs(ev_by_tile: dict, n=1):
    """构造一条 obs：给定 {牌: ev}，这些牌同时是打牌候选（行 874）。"""
    o = np.zeros((n, E.NATIVE_ROWS, E.NUM_TILES), np.float32)
    for t, v in ev_by_tile.items():
        o[:, E.SP_DISCARD_CAND, t] = 1.0
        for k in range(E.SP_N_TURNS):
            o[:, E.SP_TABLE + 2 * E.SP_N_TURNS + k, t] = v
            o[:, E.SP_TABLE + E.SP_N_TURNS + k, t] = v * 0.2
            o[:, E.SP_TABLE + k, t] = v * 0.4
    return o


def test_scores():
    print('== 3. rerank_scores 契约 ==')
    n = 1
    q = np.zeros((n, 34)); q[0, [0, 5]] = [1.0, 0.9]
    own = np.zeros((n, 34), bool); own[0, [0, 5]] = True
    risk = np.zeros((n, 34))
    # 两张都在 SP 覆盖里，但 5 的 EV 明显更高（均值填充下仍有差异）
    obs = fake_obs({0: 0.2, 5: 0.8})
    r0 = E.rerank_scores(q, obs, risk, beta=0.0, lam=0.0, own=own)
    check('β=λ=0 不动', bool((r0['pick'] == r0['base_pick']).all()))
    r1 = E.rerank_scores(q, obs, risk, beta=1.0, lam=0.0, own=own)
    check('大 β 选高 EV 项（5）', int(r1['pick'][0]) == 5, f"pick={int(r1['pick'][0])}")
    r2 = E.rerank_scores(q, obs, risk, beta=1.0, lam=0.0, margin=0.05, own=own)
    check('margin 限域后不动（Q 差 0.1 > 0.05）', int(r2['pick'][0]) == 0, f"pick={int(r2['pick'][0])}")
    r3 = E.rerank_scores(q, obs, risk, beta=1.0, lam=0.0, margin=0.2, own=own)
    check('margin 放宽后可选 5', int(r3['pick'][0]) == 5)
    risk2 = risk.copy(); risk2[0, 5] = 1.0
    r4 = E.rerank_scores(q, obs, risk2, beta=1.0, lam=10.0, own=own)
    check('风险惩罚压过高 EV 项', int(r4['pick'][0]) == 0, f"pick={int(r4['pick'][0])}")
    own2 = np.zeros((n, 34), bool); own2[0, 0] = True   # 5 不在手牌
    r5 = E.rerank_scores(q, obs, risk, beta=1.0, lam=0.0, own=own2)
    check('只在自家手牌里选', int(r5['pick'][0]) == 0)


def test_hook():
    print('== 4. EVRerank 钩子 ==')
    import torch
    q_out = torch.randn(2, 46)
    masks = torch.zeros(2, 46, dtype=torch.bool); masks[:, :34] = True
    obs = torch.zeros(2, E.NATIVE_ROWS, 34)
    hook = E.EVRerank(lambda o: np.zeros((o.shape[0], 34)), beta=0.0, lam=0.0)
    out = hook(q_out, obs, masks)
    pick = out[:, :34].argmax(-1)
    check('钩子 argmax = 重排 pick', bool((pick == q_out[:, :34].argmax(-1)).all()))
    check('非打牌动作未被改写', bool(torch.equal(out[:, 34:], q_out[:, 34:])))
    check('未选中的打牌动作被压到 -1e9', bool((out[:, :34] < -1e8).sum(1).eq(33).all()))


def main():
    obs = load_obs()
    print(f'== obs: {obs.shape} ==')
    feats, own = test_layout(obs)
    test_neutral(feats, own)
    test_scores()
    test_hook()
    print()
    if FAILS:
        print(f'FAIL: {len(FAILS)} 项 → {FAILS}')
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
