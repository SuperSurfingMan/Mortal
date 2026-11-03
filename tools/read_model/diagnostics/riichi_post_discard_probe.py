#!/usr/bin/env python3
"""立直前后打牌信息的利用方式：A2c 是否捕捉到「立直后只能摸切」？

## 问题

立直后立直家只能摸切，打出的牌除了「不和这张牌」之外几乎无信息；立直前的
手切则携带选牌/留牌信息。两者应当被区别对待。

代码层面的事实（2026-11-03 核对）：

- `post_riichi_river` 在 `constraints.compute_wait_bias` 里是**死参数**
  （docstring 标「诊断用」，函数体只引用 `unseen`/`river`）；
- `tl_tokens` 的 43 维 = 牌 one-hot(34) + **摸切(1)** + 动作(7) + 巡目(1)，
  **没有「立直后」标志**；该标志只作为侧车 `flags[...,0]` 传给 tile 模式，
  而 **A2c 是 pooled 模式，只吃 tokens+mask，不吃侧车**。

所以模型只能通过「reach token 之后全是 tsumogiri」间接推断，且依赖 attention
pooling 去记这个规则。本脚本用消融直接测它学到没有。

## 五个变体

| 变体 | 改动 | 若模型已区分，预期 |
|---|---|---|
| `base` | 无 | — |
| `zero_tg` | 抹掉摸切标志（`tok[...,34]=0`） | 立直层几乎不变（摸切是常态）|
| `drop_post` | 立直后 token 的 mask 置 0 | **立直层变化小**（那些牌无信息）|
| `drop_pre` | 立直前 token 的 mask 置 0 | **立直层变化大**（手切有信息）|
| `drop_all` | 全 mask（参照） | 最大变化 |

判据：`R = Δ(drop_post) / Δ(drop_pre)`，在 riichi 层。
`R << 1` ⇒ 模型确实更依赖立直前的信息（正确）；`R ≈ 1` ⇒ 一视同仁（未区分）。

用法：
    python tools/read_model/diagnostics/riichi_post_discard_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RM = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RM))

import _common as C  # noqa: E402
C.bootstrap_paths()

import bayes_bound as BB  # noqa: E402
import train as T  # noqa: E402


def _post_reach_mask(tokens: np.ndarray, mask: np.ndarray):
    """返回 (reach 之后的 token mask, 是否有 reach)。"""
    act = tokens[..., 35:42].argmax(-1)               # 0=dahai, 1=reach
    reach = (act == 1) & mask                          # [n,3,L]
    has = reach.any(-1)
    r_idx = np.where(has, reach.argmax(-1), 10 ** 6)   # [n,3]
    pos = np.arange(mask.shape[-1])[None, None, :]
    post = (pos > r_idx[..., None]) & mask
    return post, has, reach


def _aggregate(md, masks):
    return BB.aggregate(masks, md)


#: bayes_bound.STRATA 的键名（不是简写 riichi/clean_t8）
LAYER_KEY = {
    'all': 'all',
    'riichi': 'riichi=1',
    'nonriichi': 'riichi=0',
    'clean_t8': 'riichi=0&called=0&turn>=8',
}


def run_variant(net, batch, bn, masks, name, mutate, device, chunk=2048):
    b = dict(batch)
    mutate(b)
    n = int(b['obs'].shape[0])
    wb = T.wait_bias_of(b)
    A, W = [], []
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            sl = slice(lo, hi)
            o = net(b['obs'][sl].to(device), b['unseen'][sl].to(device),
                    wb[sl].to(device), b['tl_tokens'][sl].to(device),
                    b['tl_mask'][sl].to(device))
            A.append(torch.softmax(o['alloc_logits'].float(), -1).cpu().numpy())
            W.append(o['wait_probs'].float().cpu().numpy())
    alloc_p = np.concatenate(A)
    wait_p = np.concatenate(W)
    sh = np.zeros(alloc_p.shape[:2] + (8,))
    sh[..., 0] = 1.0 - wait_p[..., BB.NO_WAIT]
    sh[..., 1] = wait_p[..., BB.NO_WAIT]
    p_ten = 1.0 - wait_p[..., BB.NO_WAIT]
    md = BB.metrics_per_target(alloc_p, wait_p, wait_p, sh, p_ten, bn)
    agg = _aggregate(md, masks)
    a_all = agg['all']['alloc_ce']
    a_ri = agg.get(LAYER_KEY['riichi'], {}).get('alloc_ce', float('nan'))
    print(f'[run] {name}  all={a_all:.4f}  riichi={a_ri:.4f}', flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(C.CKPT))
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--out', default=str(HERE / 'results' / 'riichi_post_probe.json'))
    ap.add_argument('--chunk', type=int, default=2048)
    args = ap.parse_args()

    t0 = time.time()
    device = C.DEV
    batch, bn, n_files = C.build_eval(args.n)
    n = int(batch['obs'].shape[0])
    print(f'[data] n={n} files={n_files}  {time.time()-t0:.0f}s', flush=True)

    net = C.load_net(Path(args.ckpt))
    net.eval()
    print(f'[net] {args.ckpt} tl_mode={getattr(net, "tl_mode", "pooled")}', flush=True)

    riichi = bn['riichi'] > 0.5
    called = bn['called'] > 0.5
    turn = bn['turn'].astype(np.int64)
    masks = BB.strata_masks(riichi, called, turn)

    tokens = bn['tl_tokens']
    tmask = bn['tl_mask'].astype(bool)
    post, has, reach = _post_reach_mask(tokens, tmask)
    # 只看「已立直」的目标：立直判定用 reach token 存在
    riichi_j = riichi & has
    print(f'[mask] 有 reach token 的目标 {int(has.sum())} / {has.size}；'
          f'riichi 层 {int(riichi.sum())}', flush=True)
    n_post = post.sum(-1)
    print(f'[mask] 立直后 token 数：mean={n_post[has].mean():.2f} '
          f'max={int(n_post.max())}', flush=True)

    # ---- 五个变体 ----
    def noop(b):
        return None

    def zero_tg(b):
        b['tl_tokens'] = b['tl_tokens'].clone()
        b['tl_tokens'][..., 34] = 0.0

    def drop_post(b):
        b['tl_mask'] = b['tl_mask'].clone()
        b['tl_mask'][torch.from_numpy(post)] = False

    def drop_pre(b):
        # 立直前的全部有效 token（含 reach 本身）——与 drop_post 构成对照
        pre = tmask & ~post
        b['tl_mask'] = b['tl_mask'].clone()
        b['tl_mask'][torch.from_numpy(pre)] = False

    def drop_all(b):
        b['tl_mask'] = torch.zeros_like(b['tl_mask'])

    variants = [('base', noop), ('zero_tg', zero_tg),
                ('drop_post', drop_post), ('drop_pre', drop_pre),
                ('drop_all', drop_all)]
    res = {}
    for name, fn in variants:
        res[name] = run_variant(net, batch, bn, masks, name, fn, device,
                                args.chunk)

    # ---- 汇总 ----
    def d(a, b, layer, key='alloc_ce'):
        k = LAYER_KEY[layer]
        return (res[a].get(k, {}).get(key, float('nan'))
                - res[b].get(k, {}).get(key, float('nan')))

    def v(name, layer, key):
        return res[name].get(LAYER_KEY[layer], {}).get(key, float('nan'))

    rows = []
    for layer in ('all', 'riichi', 'nonriichi', 'clean_t8'):
        for key in ('alloc_ce', 'wait_ce', 'hit1'):
            r = {
                'layer': layer, 'metric': key,
                'base': v('base', layer, key),
                'zero_tg': d('zero_tg', 'base', layer, key),
                'drop_post': d('drop_post', 'base', layer, key),
                'drop_pre': d('drop_pre', 'base', layer, key),
                'drop_all': d('drop_all', 'base', layer, key),
            }
            rows.append(r)

    # 关键比值：立直层上 post/pre 的敏感度
    ri = [r for r in rows if r['layer'] == 'riichi']
    ratio = {}
    for r in ri:
        den = r['drop_pre']
        ratio[r['metric']] = (r['drop_post'] / den) if abs(den) > 1e-9 else None

    findings = [
        f"立直后 token 数 mean={n_post[has].mean():.2f}（max {int(n_post.max())}）；"
        f"有 reach token 的目标 {int(has.sum())}。",
        f"riichi 层 alloc_ce：base {v('base', 'riichi', 'alloc_ce'):.4f}；"
        f"Δzero_tg {d('zero_tg', 'base', 'riichi'):+.4f}、"
        f"Δdrop_post {d('drop_post', 'base', 'riichi'):+.4f}、"
        f"Δdrop_pre {d('drop_pre', 'base', 'riichi'):+.4f}、"
        f"Δdrop_all {d('drop_all', 'base', 'riichi'):+.4f}。",
        f"riichi 层 wait_ce：base {v('base', 'riichi', 'wait_ce'):.4f}；"
        f"Δzero_tg {d('zero_tg', 'base', 'riichi', 'wait_ce'):+.4f}、"
        f"Δdrop_post {d('drop_post', 'base', 'riichi', 'wait_ce'):+.4f}、"
        f"Δdrop_pre {d('drop_pre', 'base', 'riichi', 'wait_ce'):+.4f}。",
        f"敏感度比 R = Δpost/Δpre（alloc_ce）= "
        f"{ratio.get('alloc_ce') if ratio.get('alloc_ce') is None else round(ratio['alloc_ce'], 3)}；"
        f"（wait_ce）= "
        f"{ratio.get('wait_ce') if ratio.get('wait_ce') is None else round(ratio['wait_ce'], 3)}。",
        f"nonriichi 层 alloc_ce：base {v('base', 'nonriichi', 'alloc_ce'):.4f}；"
        f"Δzero_tg {d('zero_tg', 'base', 'nonriichi'):+.4f}、"
        f"Δdrop_all {d('drop_all', 'base', 'nonriichi'):+.4f}。",
    ]

    r_post = d('drop_post', 'base', 'riichi')
    r_pre = d('drop_pre', 'base', 'riichi')
    r_tg = d('zero_tg', 'base', 'riichi')
    if abs(r_pre) < 1e-6:
        verdict = 'NO-EFFECT（立直层对时序 token 整体不敏感，无法判定）'
    elif abs(r_post) <= 0.25 * abs(r_pre):
        verdict = ('CAPTURED（模型更依赖立直前的手切：drop_post 影响 '
                   f'≤25% of drop_pre）⇒ 已在隐式区分')
    elif abs(r_post) >= 0.75 * abs(r_pre):
        verdict = ('NOT-CAPTURED（立直后与立直前同等重要）⇒ 模型把摸切当手切读，'
                   '没有得到「立直后无信息」这一先验')
    else:
        verdict = 'PARTIAL（介于两者之间，需看 wait_ce 是否一致）'

    out = {'meta': {'ckpt': str(args.ckpt), 'n': n,
                    'tl_mode': getattr(net, 'tl_mode', 'pooled'),
                    'n_reach': int(has.sum()),
                    'post_tokens_mean': float(n_post[has].mean())},
           'variants': res, 'rows': rows, 'ratio': ratio,
           'verdict': {'decision': verdict, 'findings': findings}}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f'\n[verdict] {verdict}')
    for line in findings:
        print(' -', line)
    print(f'[out] {p}')
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
