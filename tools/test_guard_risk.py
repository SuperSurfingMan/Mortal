#!/usr/bin/env python3
"""守卫 risk 口径回归：锁死 ``risk = Σ_j wp[j,t]``，防止再双乘 P(听)。

用法（仓库任意目录）::

    python tools/test_guard_risk.py
    CUDA_VISIBLE_DEVICES="" python tools/test_guard_risk.py --real --n 64

``--real`` 会加载 A2c 与 202609 固定评估集，验证真实 wp 上的因子化恒等式、
硬掩码与 legacy 关系。纯单元部分不依赖读牌 ckpt。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MORTAL = REPO / 'mortal'
READ = REPO / 'tools' / 'read_model'
sys.path.insert(0, str(MORTAL))

import torch  # noqa: E402

import guard_risk as GR  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = ''):
    print(f'  [{"OK" if cond else "FAIL"}] {name:<52s} {detail}')
    if not cond:
        FAILS.append(name)


def close(a, b, atol=1e-6) -> bool:
    return bool(torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=atol, rtol=0))


def test_synthetic():
    print('== 1. 合成反例（旧代码必失败）==')
    wp = torch.zeros(1, 3, 35)
    # 三家 P(听)=0.9/0.5/0.01，待牌分别为 5/6/7
    wp[0, 0, 5], wp[0, 0, 34] = 0.9, 0.1
    wp[0, 1, 6], wp[0, 1, 34] = 0.5, 0.5
    wp[0, 2, 7], wp[0, 2, 34] = 0.01, 0.99

    fixed = GR.risk_from_wait_probs(wp)[0]
    legacy = GR.legacy_risk_from_wait_probs(wp)[0]
    check('fixed[5] = 0.9', close(fixed[5], 0.9), f'{fixed[5]:.6f}')
    check('fixed[6] = 0.5', close(fixed[6], 0.5), f'{fixed[6]:.6f}')
    check('fixed[7] = 0.01', close(fixed[7], 0.01), f'{fixed[7]:.6f}')
    check('legacy[5] = 0.81（旧双乘）', close(legacy[5], 0.81), f'{legacy[5]:.6f}')
    check('legacy[6] = 0.25（旧双乘）', close(legacy[6], 0.25), f'{legacy[6]:.6f}')
    check('legacy[7] = 0.0001（旧双乘）', close(legacy[7], 0.0001), f'{legacy[7]:.6f}')

    p_ten = (1.0 - wp[..., 34]).clamp(0, 1)
    expected_legacy = (p_ten.unsqueeze(-1) * wp[..., :34]).sum(-2).clamp(0, 1)
    check('legacy == p_ten * fixed', close(legacy, expected_legacy))

    # 多家同听一张：求和后 clamp 到 1
    wp2 = torch.zeros(1, 3, 35)
    wp2[..., 8] = 0.9
    wp2[..., 34] = 0.1
    check('三家同听 8：fixed 饱和到 1', close(GR.risk_from_wait_probs(wp2)[0, 8], 1.0))
    check('三家同听 8：legacy 也饱和到 1', close(GR.legacy_risk_from_wait_probs(wp2)[0, 8], 1.0))


def test_factorization():
    print('== 2. factorized_wait_probs 恒等式 ==')
    torch.manual_seed(20261102)
    logits = torch.randn(8, 3, 35)
    p_not = torch.sigmoid(logits[..., 34:35])
    cond = torch.softmax(logits[..., :34], dim=-1)
    wp = torch.cat([(1.0 - p_not) * cond, p_not], dim=-1)

    check('wp[:34].sum(-1) == 1 - wp[34]',
          close(wp[..., :34].sum(-1), 1.0 - wp[..., 34]))
    raw = wp[..., :34].sum(-2)
    check('risk == wp[:34].sum(-2)（clamp 前）',
          close(GR.risk_from_wait_probs(wp), raw.clamp(0, 1)))
    check('union risk == Σ_j wp[j,:34]',
          close(GR.risk_from_wait_probs(wp), wp[..., :34].sum(1).clamp(0, 1)))
    for j in range(3):
        check(f'对手 {j} 通道和 == 1 - wp[{j},34]',
              close(wp[:, j, :34].sum(-1), 1.0 - wp[:, j, 34]))
    check('全零行 risk=0 且无 NaN',
          close(GR.risk_from_wait_probs(torch.zeros(1, 3, 35)), torch.zeros(1, 34)))


def test_penalty():
    print('== 3. Q 惩罚公式 ==')
    q = torch.arange(46, dtype=torch.float32).view(1, -1)
    masks = torch.zeros(1, 46, dtype=torch.bool)
    masks[0, :34] = True
    masks[0, 5] = False                    # 5 不可打
    risk = torch.zeros(1, 34)
    risk[0, 5] = 0.4                       # 不可打，不应扣
    risk[0, 6] = 0.02
    q2 = GR.apply_risk_penalty(q, risk, lam=10.0, tau=0.0, masks=masks)

    check('可打牌 6 扣 λ·risk', close(q2[0, 6], q[0, 6] - 0.2), f'{q2[0, 6]:.3f}')
    check('不可打牌 5 不扣', close(q2[0, 5], q[0, 5]), f'{q2[0, 5]:.3f}')
    check('动作 34..45 不变', close(q2[0, 34:], q[0, 34:]))
    check('入参 q 未被原地修改', close(q[0, 6], 6.0))
    check('返回新 tensor', q2 is not q)

    q3 = GR.apply_risk_penalty(q, risk, lam=10.0, tau=0.03, masks=masks)
    check('τ=0.03 时 risk=0.02 不扣', close(q3[0, 6], q[0, 6]))
    q4 = GR.apply_risk_penalty(q, risk, lam=10.0, tau=0.0,
                               masks=torch.zeros(1, 46, dtype=torch.bool))
    check('无 live 动作时原样返回', q4 is q)


def test_real(n: int):
    print(f'== 4. 真实批次（A2c，n={n}）==')
    sys.path.insert(0, str(READ / 'diagnostics'))
    sys.path.insert(0, str(READ))
    import _common as C
    a2c = MORTAL / 'read_model_v2s_a2c' / 'best.pth'
    C.CKPT = a2c
    net = C.load_net(a2c)
    batch, _bn, _nf = C.build_eval(n_eval=n)
    import train as T
    wb = T.wait_bias_of(batch)
    with torch.no_grad():
        out = net(batch['obs'], batch['unseen'], wb, batch['tl_tokens'],
                  batch['tl_mask'], **net.tl_extra(batch))
    wp = out['wait_probs'].float()
    p_ten = 1.0 - wp[..., 34]
    raw = wp[..., :34].sum(-1)

    check('真实批次 wp[:34].sum == p_ten', close(raw, p_ten, atol=1e-5))
    fixed = GR.risk_from_wait_probs(wp)
    legacy = GR.legacy_risk_from_wait_probs(wp)
    raw_union = wp[..., :34].sum(-2)
    check('fixed == wp[:34].sum(1)（clamp 后）', close(fixed, raw_union.clamp(0, 1), atol=1e-5))
    check('legacy == p_ten * fixed（clamp 后）',
          close(legacy, (p_ten.unsqueeze(-1) * wp[..., :34]).sum(1).clamp(0, 1), atol=1e-5))

    bad = batch['river'] | ((batch['unseen'] <= 0) & ~batch['own_hand'])[:, None, :]
    vals = wp[..., :34][bad]
    check('河牌 / 不可表达牌 wp 严格为 0',
          bool(vals.numel() == 0 or vals.max() < 1e-7),
          f'max={float(vals.max()) if vals.numel() else 0:.2e}')
    check('无 NaN/Inf', bool(torch.isfinite(wp).all() and torch.isfinite(fixed).all()))


def test_sources():
    print('== 5. 调用方单一真源 ==')
    files = [
        MORTAL / 'defense_guard.py',
        READ / 'precompute_danger.py',
        READ / 'eval_defense_guard.py',
        READ / 'eval_ev_rerank.py',
        READ / 'eval_guard_state.py',
        READ / 'eval_loss_pt.py',
        READ / 'eval_s6b_arms.py',
    ]
    for f in files:
        txt = f.read_text(encoding='utf-8')
        check(f'{f.name} 引用 risk_from_wait_probs', 'risk_from_wait_probs' in txt)

    bad_patterns = (
        'p_tenpai[..., None] * wp[..., :NUM_TILES]',
        'p_t[..., None] * WP[..., :NUM_TILES]',
    )
    for f in files:
        txt = f.read_text(encoding='utf-8')
        hit = [p for p in bad_patterns if p in txt]
        check(f'{f.name} 无旧双乘表达式', not hit, ','.join(hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', action='store_true', help='额外跑 A2c 真实批次')
    ap.add_argument('--n', type=int, default=64)
    args = ap.parse_args()

    test_synthetic()
    test_factorization()
    test_penalty()
    test_sources()
    if args.real:
        test_real(args.n)

    print()
    if FAILS:
        print(f'FAILED: {len(FAILS)} 项')
        for x in FAILS:
            print(f'  - {x}')
        sys.exit(1)
    print('ALL PASS')


if __name__ == '__main__':
    main()
