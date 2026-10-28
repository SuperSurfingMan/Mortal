"""P2 冒烟验收：bc_style KL 项的合成 batch 单测（防 -inf 相减 NaN）。

train.py 的 KL 实现是「先做差再 masked_fill」：
    diff = (logp - logp_base).masked_fill(~masks, 0.)
    kl   = (diff * logp.exp()).sum(-1)
裸乘版（(p * (logp - logp_base)).sum(-1)）在非法动作位会产生
-inf − (−inf) = NaN，且 0 × NaN = NaN 污染整个 sum——本测试用含 -inf mask
的合成 logits 同时验证：①无 NaN；②与逐项循环参考实现数值一致；③logit_temp 缩放后仍正确。

用法: python tools/style_ft/test_style_kl.py
"""
from __future__ import annotations

import torch

B, A = 256, 46


def make_masked_logits(seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, A, generator=g) * 3.0
    # 每行随机 3~40 个合法动作位（与真实对局量级一致）
    n_legal = torch.randint(3, 41, (B,), generator=g)
    mask = torch.zeros(B, A, dtype=torch.bool)
    for i in range(B):
        idx = torch.randperm(A, generator=g)[:int(n_legal[i])]
        mask[i, idx] = True
    return logits.masked_fill(~mask, -float('inf')), mask


def kl_train_impl(q_out, q_base, masks, temp):
    q_out = q_out / temp
    q_base = q_base / temp
    logp = q_out.log_softmax(-1)
    with torch.no_grad():
        logp_base = q_base.log_softmax(-1)
    diff = (logp - logp_base).masked_fill(~masks, 0.)
    return (diff * logp.exp()).sum(-1)


def kl_reference(q_out, q_base, masks, temp):
    """逐项循环参考实现（只在合法位上累加，天然无 NaN）。"""
    q_out = (q_out / temp).softmax(-1)
    q_base = (q_base / temp).log_softmax(-1)
    out = torch.zeros(B)
    for i in range(B):
        s = 0.0
        for a in range(A):
            if masks[i, a]:
                s += float(q_out[i, a]) * (float(torch.log(q_out[i, a])) - float(q_base[i, a]))
        out[i] = s
    return out


def run_case(temp):
    q_out, mask = make_masked_logits(1)
    # ⚠ base 必须与 student 共享同一合法动作集（同一 masks）：训练中
    # q_out = dqn(phi, masks) 与 q_base = frozen_dqn(..., masks) 传同一个
    # masks，DQN.forward 的 masked_fill(~mask, -inf)（model.py:238）保证两者
    # -inf 位完全一致，diff 在合法位恒为有限值。首版测试误让 q_base 自带
    # 独立随机 mask → 合法位 logp_base = -inf → diff = +inf → isfinite
    # 失败——属测试口径错误，train.py 的 KL 实现无此问题。
    g = torch.Generator().manual_seed(2)
    q_base = (torch.randn(B, A, generator=g) * 3.0).masked_fill(~mask, -float('inf'))
    kl = kl_train_impl(q_out, q_base, mask, temp)
    ref = kl_reference(q_out, q_base, mask, temp)
    assert torch.isfinite(kl).all(), f'temp={temp}: KL 含非有限值！'
    # KL 非负（合法位上 p*(logp−logp_b) 的期望按 p 加权 = KL(p||p_b) ≥ 0）
    assert (kl >= -1e-5).all(), f'temp={temp}: KL 出现负值（数值异常）'
    diff = (kl - ref).abs().max().item()
    assert diff < 1e-4, f'temp={temp}: 与参考实现偏差 {diff:.2e}'
    print(f'temp={temp}: max|kl−ref|={diff:.2e}  finite=OK  nonneg=OK  '
          f'mean_kl={kl.mean().item():.4f}')


if __name__ == '__main__':
    run_case(1.0)
    run_case(0.7)
    print('test_style_kl: 全部通过')