#!/usr/bin/env python3
"""立直层温度校准：A2c 的 27 类后验在立直层是否「过度自信」，能回收多少 CE。

## 问题

P0-a 发现：`riichi/all_centers` 上 A2c 的 27 类联合 CE **2.494** 比「每中心先验」
**2.429** 还差 0.065 nats，但 top1（0.2322 vs 0.1943）更好 —— 典型的**过度自信**：
排序对、概率幅度错。

问题是：这 0.065 nats 里有多少是**校准误差**（温度缩放能回收），多少是
**分布形状错误**（温度缩放是单调变换，救不了）？

## 方法

对每个 cell 的后验做 `q ∝ p^(1/T)`，在**同一批样本**上最小化 CE 拟合 T：

- `T > 1` ⇒ 原分布过尖（过度自信），温度缩放能降 CE；
- `T ≈ 1` ⇒ 已经校准好，CE 差来自形状而非幅度；
- `T < 1` ⇒ 原分布过平（信心不足）。

温度缩放是**单调变换** ⇒ top1 / top3 完全不变，所以本脚本只回答
「概率幅度对不对」，不改变任何排序指标。

除了 per-cell（诊断用），另外拟合一条**可落地**的分组：
`T_riichi` / `T_nonriichi`（推理时只知道「对手是否立直」这一个 bit），
并在其它 act 上检验它的泛化（避免 per-cell 过拟合）。

## 产物

- ``results/RIICHI_CALIB_REPORT.md``
- ``results/riichi_calib.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DIAG = REPO / 'tools' / 'read_model' / 'diagnostics'
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(HERE))

import eval_local_shape as P0  # noqa: E402
import local_shape as LS  # noqa: E402

LAYERS = ('all', 'riichi', 'nonriichi', 'called', 'turn>=8', 'clean_t8')
ACTIVES = ('all_centers', 'active_any', 'active_both')
T_RANGE = (0.2, 10.0)
# T 只有 1 个参数；大 cell（最大 169 万）下子采样到 50k 足以拟合，
# 否则 41 个网格点 × 4500 万元素的 exp 会跑几十分钟。
FIT_MAX_N = 50000
RNG = np.random.default_rng(20261104)


def _logp(p):
    return np.log(np.clip(p.astype(np.float32), 1e-12, None))


def _softmax_T(logp, T):
    z = logp / T
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def _ce(logp, y, T):
    q = _softmax_T(logp, T)
    return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean())


def _golden(f, lo, hi, tol=1e-3, max_iter=80):
    """一维黄金分割搜索（conda 环境无 scipy，自己实现）。"""
    phi = (5 ** 0.5 - 1) / 2
    a, b = float(lo), float(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2, min(fc, fd)


def fit_T(logp, y):
    """在 [T_RANGE] 上最小化 CE；先粗网格定位，再黄金分割细化。"""
    if len(y) > FIT_MAX_N:
        sel = RNG.choice(len(y), FIT_MAX_N, replace=False)
        logp, y = logp[sel], y[sel]
    grid = np.exp(np.linspace(np.log(T_RANGE[0]), np.log(T_RANGE[1]), 41))
    ces = [_ce(logp, y, t) for t in grid]
    i = int(np.argmin(ces))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    T, ce = _golden(lambda t: _ce(logp, y, t), lo, hi)
    if ce > ces[i]:
        return float(grid[i]), float(ces[i])
    return float(T), float(ce)


def prior_ce(y, center_idx, n_class=27):
    """每中心先验的 CE（与 P0-a 同口径：按中心分组，cond 样本内拟合）。"""
    ce = np.zeros(len(y), np.float64)
    for ci in range(int(center_idx.max()) + 1):
        sel = center_idx == ci
        if not sel.any():
            continue
        cnt = np.bincount(y[sel], minlength=n_class).astype(np.float64)
        pr = cnt / max(cnt.sum(), 1.0)
        ce[sel] = -np.log(np.clip(pr[y[sel]], 1e-12, None))
    return float(ce.mean())


def collect(P, z, model, layer, act, masks, active, yall, max_n=None):
    """取一个 cell 的 (logp, y, center_idx)。

    cond 口径与 P0-a **完全一致**：筛 `y != 0` **且** 把第 0 类（EMPTY）概率置 0
    后重归一——少了后者会把 CE 抬高 ~0.7 nats（首版就错在这里）。
    """
    idx = np.argwhere(masks[layer][..., None] & active[act])   # (i,j,ci)
    if len(idx) == 0:
        return None
    y = yall[idx[:, 0], idx[:, 1], idx[:, 2]]
    sel = y != 0
    if not sel.any():
        return None
    idx, y = idx[sel], y[sel]
    if max_n and len(y) > max_n:
        keep = RNG.choice(len(y), max_n, replace=False)
        idx, y = idx[keep], y[keep]
    p = P[model][idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
    p[:, 0] = 0.0
    p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
    return _logp(p), y, idx[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', default='/tmp/ls_cond_probs.npz')
    ap.add_argument('--cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    ap.add_argument('--models', default='a2c,p2c')
    ap.add_argument('--out', default=str(HERE / 'results'))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    d = np.load(args.probs)
    P = {m: d[m] for m in models if m in d.files}
    print(f'[probs] {args.probs}  models={list(P)}', flush=True)

    z = P0.load_cache(Path(args.cache), 40000)
    centers = LS.centers_3()
    masks = P0.strata(z)
    any_, both = P0.active_masks(z, centers)
    active = {'all_centers': np.ones_like(any_, bool),
              'active_any': any_, 'active_both': both}
    yall = np.stack([LS.true_pattern(z['alloc'], c) for c in centers], -1)

    def prior_of(layer, act):
        """全量（未子采样）的每中心先验 CE，与 P0-a 同口径。"""
        idx = np.argwhere(masks[layer][..., None] & active[act])
        y = yall[idx[:, 0], idx[:, 1], idx[:, 2]]
        sel = y != 0
        return prior_ce(y[sel], idx[sel, 2]) if sel.any() else float('nan')

    cells = {}
    for model in P:
        cells[model] = {}
        for layer in LAYERS:
            cells[model][layer] = {}
            for act in ACTIVES:
                r = collect(P, z, model, layer, act, masks, active, yall,
                            max_n=FIT_MAX_N)
                if r is None:
                    continue
                logp, y, ci = r
                ce0 = _ce(logp, y, 1.0)
                T, ce1 = fit_T(logp, y)
                cells[model][layer][act] = {
                    'n': int(len(y)), 'ce': ce0, 'T': T, 'ce_T': ce1,
                    'delta': ce1 - ce0, 'prior_ce': prior_of(layer, act),
                    'top1': float((_softmax_T(logp, 1.0).argmax(1) == y).mean()),
                    'subsampled': bool(len(y) >= FIT_MAX_N),
                }
                print(f'[{model}] {layer:9s} {act:13s} n={len(y):>7d} '
                      f'CE {ce0:.4f} → {ce1:.4f} (T={T:.3f})  '
                      f'prior {cells[model][layer][act]["prior_ce"]:.4f}', flush=True)

    # ---- 可落地的分组校准：T_riichi / T_nonriichi（all_centers 拟合）----
    deploy = {}
    for model in P:
        dep = {}
        for layer in ('riichi', 'nonriichi'):
            r = collect(P, z, model, layer, 'all_centers', masks, active, yall,
                        max_n=FIT_MAX_N)
            if r is None:
                continue
            logp, y, _ = r
            T, _ = fit_T(logp, y)
            dep[layer] = {'T': T, 'n_fit': int(len(y))}
            # 用这个 T 去量其它 act 的收益（检验泛化，不是过拟合）
            for act in ACTIVES:
                rr = collect(P, z, model, layer, act, masks, active, yall,
                             max_n=FIT_MAX_N)
                if rr is None:
                    continue
                lp, yy, _ = rr
                dep[layer][act] = {'ce': _ce(lp, yy, 1.0),
                                   'ce_T': _ce(lp, yy, T),
                                   'n': int(len(yy))}
        # 交叉应用：用另一层的 T 看是否更差（层特异性检验）
        if 'riichi' in dep and 'nonriichi' in dep:
            for src, dst in (('riichi', 'nonriichi'), ('nonriichi', 'riichi')):
                rr = collect(P, z, model, dst, 'all_centers', masks, active, yall,
                             max_n=FIT_MAX_N)
                if rr is None:
                    continue
                lp, yy, _ = rr
                dep[f'{dst}_with_T_{src}'] = {
                    'ce': _ce(lp, yy, 1.0),
                    'ce_T': _ce(lp, yy, dep[src]['T']),
                    'T_used': dep[src]['T'], 'n': int(len(yy))}
        deploy[model] = dep

    out = {'meta': {'probs': args.probs, 'cache': args.cache,
                    'models': list(P), 'T_range': list(T_RANGE)},
           'cells': cells, 'deploy': deploy}
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    with open(o / 'riichi_calib.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    write_report(o / 'RIICHI_CALIB_REPORT.md', out)
    print(f'\n[out] {o}/riichi_calib.json\n[out] {o}/RIICHI_CALIB_REPORT.md')


def _f(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '—'


def write_report(path, res):
    L = ['# 立直层温度校准：A2c 的 27 类后验\n']
    L.append('问题：P0-a 发现 `riichi/all_centers` 上 A2c 的 27 类 CE（2.494）比'
             '每中心先验（2.429）还差，但 top1 更好 ⇒ 典型**过度自信**。\n')
    L.append('方法：`q ∝ p^(1/T)`，在同批样本上最小化 CE 拟合 T。'
             '温度缩放是**单调变换** ⇒ top1/top3 不变，只回答「概率幅度对不对」。\n')
    for model in res['cells']:
        L.append(f'\n## {model.upper()}：per-cell 温度\n')
        L.append('| 层 | 子集 | n | prior CE | CE(T=1) | T* | CE(T*) | ΔCE | 过尖? |')
        L.append('|---|---|---:|---:|---:|---:|---:|---:|---|')
        for layer in LAYERS:
            for act in ACTIVES:
                c = res['cells'][model].get(layer, {}).get(act)
                if not c:
                    continue
                sharp = ('是' if c['T'] > 1.05 else
                         '否（过平）' if c['T'] < 0.95 else '已校准')
                L.append(f"| {layer} | {act} | {c['n']} | {_f(c['prior_ce'])} | "
                         f"{_f(c['ce'])} | **{_f(c['T'], 3)}** | {_f(c['ce_T'])} | "
                         f"{_f(c['delta'], 4)} | {sharp} |")
        L.append('')
    L.append('\n## 可落地：分组校准（只看「对手是否立直」一个 bit）\n')
    for model, dep in res['deploy'].items():
        L.append(f'\n**{model.upper()}**\n')
        L.append('| 拟合 | T | 子集 | n | CE(T=1) | CE(T) | Δ |')
        L.append('|---|---:|---|---:|---:|---:|---:|')
        for layer in ('riichi', 'nonriichi'):
            if layer not in dep:
                continue
            T = dep[layer]['T']
            for act in ACTIVES:
                d = dep[layer].get(act)
                if not d:
                    continue
                L.append(f"| {layer} 上拟合 | {_f(T, 3)} | {act} | {d['n']} | "
                         f"{_f(d['ce'])} | {_f(d['ce_T'])} | "
                         f"{_f(d['ce_T'] - d['ce'], 4)} |")
        for k in ('nonriichi_with_T_riichi', 'riichi_with_T_nonriichi'):
            d = dep.get(k)
            if not d:
                continue
            L.append(f"| 交叉：{k} | {_f(d['T_used'], 3)} | all_centers | {d['n']} | "
                     f"{_f(d['ce'])} | {_f(d['ce_T'])} | "
                     f"{_f(d['ce_T'] - d['ce'], 4)} |")
        L.append('')
    path.write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
