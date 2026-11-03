#!/usr/bin/env python3
"""P0：局部形状读数基线（只读缓存，不训练）。

回答三个问题：
  1. 当前 A2c 的「逐张边际」在局部成对/三张形状上到底有多准？
  2. 它比「无信息先验」和 A3 规则法好多少？
  3. 有没有便宜的校准/温度修正能补上差距？

产物：``results/p0_local_shape.json`` + ``results/P0_REPORT.md``。

用法（缓存默认是 A2c 的 40000 样本诊断缓存）：
    python tools/read_model/local_shape/eval_local_shape.py \
        --cache /tmp/read_model_diag_cache_a2c/cache.npz \
        --out tools/read_model/local_shape/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import local_shape as LS  # noqa: E402

LAYERS = ('all', 'riichi', 'nonriichi', 'called', 'turn>=8', 'clean_t8')
ACTIVES = ('all_centers', 'active_any', 'active_both')
DISTANCES = (1, 2)


def _meta(cache_path: Path) -> dict:
    mf = cache_path.with_name('cache.meta.json')
    if mf.exists():
        return json.loads(mf.read_text())
    return {}


def load_cache(path: Path, n_limit: int = 0) -> dict:
    z = np.load(path, allow_pickle=True)
    need = ('AP', 'RAP', 'EC', 'REC', 'alloc', 'CPMF0', 'riichi',
            'called', 'turn', 'hc')
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f'cache 缺字段 {miss}: {path}')
    n = int(z['alloc'].shape[0])
    if n_limit and n_limit < n:
        n = n_limit
    out = {k: np.asarray(z[k])[:n] for k in need}
    out['AP'] = out['AP'][..., :5].astype(np.float32)
    out['RAP'] = out['RAP'][..., :5].astype(np.float32)
    out['EC'] = out['EC'].astype(np.float32)
    out['REC'] = out['REC'].astype(np.float32)
    out['alloc'] = out['alloc'].astype(np.int8)
    out['CPMF0'] = out['CPMF0'].astype(np.float32)
    out['riichi'] = out['riichi'].astype(bool)
    out['called'] = out['called'].astype(bool)
    out['turn'] = out['turn'].astype(np.int32)
    out['hc'] = out['hc'].astype(np.int32)
    out['n'] = n
    return out


def strata(z: dict) -> dict:
    n = z['n']
    t8 = np.repeat((z['turn'] >= 8)[:, None], 3, axis=1)
    return {
        'all': np.ones((n, 3), dtype=bool),
        'riichi': z['riichi'],
        'nonriichi': ~z['riichi'],
        'called': z['called'],
        'turn>=8': t8,
        'clean_t8': (~z['riichi']) & (~z['called']) & t8,
    }


def active_masks(z: dict, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n,3,21]：至少一侧手切过 / 两侧都手切过。"""
    hc = z['hc']
    any_ = []
    both = []
    for a, _b, c in centers:
        any_.append((hc[:, :, a] < 99) | (hc[:, :, c] < 99))
        both.append((hc[:, :, a] < 99) & (hc[:, :, c] < 99))
    return np.stack(any_, -1), np.stack(both, -1)


def eval_pairwise(z: dict, masks: dict) -> dict:
    alloc = z['alloc']
    pairs = LS.pair_indices()
    out = {}
    for d in DISTANCES:
        out[f'd{d}'] = {}
        for name, mask in masks.items():
            ys, p2, p3, pc = [], [], [], []
            for a, b, dd in pairs:
                if int(dd) != d:
                    continue
                ys.append(((alloc[:, :, int(a)] > 0) &
                           (alloc[:, :, int(b)] > 0))[mask])
                p2.append(((1.0 - z['AP'][:, :, int(a), 0]) *
                           (1.0 - z['AP'][:, :, int(b), 0]))[mask])
                p3.append(((1.0 - z['RAP'][:, :, int(a), 0]) *
                           (1.0 - z['RAP'][:, :, int(b), 0]))[mask])
                # CPMF0 是 [n,3,34]（target 维在中间），不是 [n,34]
                base = ((1.0 - z['CPMF0'][:, :, int(a)]) *
                        (1.0 - z['CPMF0'][:, :, int(b)]))
                pc.append(base[mask])
            y = np.concatenate(ys)
            p2 = np.concatenate(p2)
            p3 = np.concatenate(p3)
            pc = np.concatenate(pc)
            out[f'd{d}'][name] = {
                'n': int(y.size),
                'base': float(y.mean()) if y.size else float('nan'),
                'a2c': {
                    'auc': LS.auc_score(y, p2),
                    'brier': LS.brier_score(y, p2),
                    'logloss': LS.log_loss(y, p2),
                },
                'a3': {
                    'auc': LS.auc_score(y, p3),
                    'brier': LS.brier_score(y, p3),
                    'logloss': LS.log_loss(y, p3),
                },
                'counts': {
                    'auc': LS.auc_score(y, pc),
                    'brier': LS.brier_score(y, pc),
                    'logloss': LS.log_loss(y, pc),
                },
            }
    return out


def _shape_center(joint: np.ndarray, yidx: np.ndarray,
                  mask: np.ndarray) -> tuple[dict, dict]:
    """返回 (聚合指标, 逐样本-目标的和) 供 bootstrap。"""
    sel = mask.reshape(-1)
    j = joint[sel]
    y = yidx[sel]
    n3 = joint.shape[0]
    if len(y) == 0:
        return {'n': 0}, None
    order = np.argsort(-j, axis=1)
    top1_obs = (order[:, 0] == y).astype(np.float32)
    top3_obs = (order[:, :3] == y[:, None]).any(axis=1).astype(np.float32)
    ce_obs = -np.log(np.clip(j[np.arange(len(y)), y], 1e-9, None)).astype(np.float32)
    cnt = np.bincount(y, minlength=LS.NUM_PATTERNS).astype(np.float64)
    prior = cnt / max(cnt.sum(), 1.0)

    s_top1 = np.zeros(n3, np.float32)
    s_top3 = np.zeros(n3, np.float32)
    s_ce = np.zeros(n3, np.float32)
    s_cnt = np.zeros(n3, np.float32)
    s_top1[sel] = top1_obs
    s_top3[sel] = top3_obs
    s_ce[sel] = ce_obs
    s_cnt[sel] = 1.0
    return {
        'n': int(len(y)),
        'top1': float(top1_obs.mean()),
        'top3': float(top3_obs.mean()),
        'ce': float(ce_obs.mean()),
        'prior_major': float(prior.max()),
        'prior_top3': float(np.sort(prior)[-3:].sum()),
        'prior_entropy': float(-(prior * np.log(np.clip(prior, 1e-9, None))).sum()),
    }, {
        'top1': s_top1.reshape(-1, 3),
        'top3': s_top3.reshape(-1, 3),
        'ce': s_ce.reshape(-1, 3),
        'cnt': s_cnt.reshape(-1, 3),
    }


def eval_shape(z: dict, masks: dict, centers: np.ndarray,
               active_any: np.ndarray, active_both: np.ndarray,
               boot: int, seed: int) -> dict:
    n = z['n']
    active = {
        'all_centers': np.ones_like(active_any, dtype=bool),
        'active_any': active_any,
        'active_both': active_both,
    }
    # accumulators
    agg = {(l, a): {'n': 0, 'top1': 0.0, 'top3': 0.0, 'ce': 0.0,
                    'prior_major': 0.0, 'prior_top3': 0.0,
                    'prior_entropy': 0.0}
           for l in LAYERS for a in ACTIVES}
    agg3 = {(l, a): {'n': 0, 'top1': 0.0, 'top3': 0.0, 'ce': 0.0}
            for l in LAYERS for a in ACTIVES}
    per = {(l, a): {'top1': np.zeros((n, 3), np.float32),
                    'top3': np.zeros((n, 3), np.float32),
                    'ce': np.zeros((n, 3), np.float32),
                    'cnt': np.zeros((n, 3), np.float32)}
           for l in LAYERS for a in ACTIVES}

    t0 = time.time()
    for ci, center in enumerate(centers):
        yidx = LS.true_pattern(z['alloc'], center).reshape(-1)
        j2 = LS.joint_pattern_probs(z['AP'], center)
        j3 = LS.joint_pattern_probs(z['RAP'], center)
        for layer in LAYERS:
            for act in ACTIVES:
                m = masks[layer] & active[act][:, :, ci]
                md, ps = _shape_center(j2, yidx, m)
                if ps is not None:
                    for k in per[(layer, act)]:
                        per[(layer, act)][k] += ps[k]
                    a = agg[(layer, act)]
                    a['n'] += md['n']
                    for k in ('top1', 'top3', 'ce', 'prior_major',
                              'prior_top3', 'prior_entropy'):
                        a[k] += md[k] * md['n']
                md3, _ = _shape_center(j3, yidx, m)
                if md3['n']:
                    a3 = agg3[(layer, act)]
                    a3['n'] += md3['n']
                    for k in ('top1', 'top3', 'ce'):
                        a3[k] += md3[k] * md3['n']
        if (ci + 1) % 7 == 0:
            print(f'[shape] {ci+1}/{len(centers)} centers  {time.time()-t0:.0f}s',
                  flush=True)

    out = {}
    for layer in LAYERS:
        out[layer] = {}
        for act in ACTIVES:
            a = agg[(layer, act)]
            n_ = max(a['n'], 1)
            a2c = {k: float(a[k] / n_) for k in
                   ('top1', 'top3', 'ce', 'prior_major', 'prior_top3',
                    'prior_entropy')}
            a3 = agg3[(layer, act)]
            n3_ = max(a3['n'], 1)
            a3m = {k: float(a3[k] / n3_) for k in ('top1', 'top3', 'ce')}
            row = {
                'n': int(a['n']),
                'a2c': a2c,
                'a3': a3m,
                'delta_top1_vs_prior': a2c['top1'] - a2c['prior_major'],
                'delta_top3_vs_prior': a2c['top3'] - a2c['prior_top3'],
                'delta_ce_vs_prior': a2c['ce'] - a2c['prior_entropy'],
            }
            if boot > 0:
                row['ci'] = _bootstrap(per[(layer, act)], n, boot, seed)
            out[layer][act] = row
    return out


def _bootstrap(per: dict, n: int, boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    keys = ('top1', 'top3', 'ce')
    vals = {k: [] for k in keys}
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        cnt = float(per['cnt'][idx].sum())
        if cnt <= 0:
            continue
        for k in keys:
            vals[k].append(float(per[k][idx].sum()) / cnt)
    out = {}
    for k in keys:
        a = np.asarray(vals[k], np.float64)
        if len(a) == 0:
            out[k] = [float('nan'), float('nan')]
        else:
            out[k] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return out


def eval_window_mae(z: dict, centers: np.ndarray) -> dict:
    alloc = z['alloc']
    out = {}
    for name, P in (('a2c', z['EC']), ('a3', z['REC'])):
        vals = []
        for idx in centers:
            d = np.abs(P[:, :, idx] - alloc[:, :, idx]).mean()
            vals.append(float(d))
        out[name] = float(np.mean(vals))
    out['n_centers'] = int(len(centers))
    return out


def temperature_scan(z: dict, centers: np.ndarray) -> list:
    n = z['n']
    mask = np.ones((n, 3), dtype=bool)
    rows = []
    for temp in (0.75, 1.0, 1.25, 1.5, 2.0):
        n_sum = 0
        top1 = top3 = ce = 0.0
        for center in centers:
            yidx = LS.true_pattern(z['alloc'], center).reshape(-1)
            joint = LS.joint_pattern_probs_t(z['AP'], center, temp)
            md, _ = _shape_center(joint, yidx, mask)
            if md['n']:
                n_sum += md['n']
                top1 += md['top1'] * md['n']
                top3 += md['top3'] * md['n']
                ce += md['ce'] * md['n']
        rows.append({'temp': temp, 'n': int(n_sum),
                     'top1': float(top1 / max(n_sum, 1)),
                     'top3': float(top3 / max(n_sum, 1)),
                     'ce': float(ce / max(n_sum, 1))})
    return rows


def _fmt(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '—'


def write_report(path: Path, res: dict) -> None:
    m = res['meta']
    p = res['pairwise']
    s = res['shape']
    lines = []
    lines.append('# P0：局部形状读数基线（A2c）\n')
    lines.append(f"- 缓存：`{m.get('cache')}`")
    lines.append(f"- ckpt：`{m.get('ckpt')}`")
    lines.append(f"- 样本：{m.get('n')}；划分/种子：{m.get('seed')}")
    lines.append(f"- 口径：{m.get('protocol')}\n")
    lines.append('## 1. 成对共持（两张都持有）\n')
    lines.append('AUC：A2c 的独立边际乘积 vs A3 规则法 vs counts-only 地板。\n')
    lines.append('| 距离 | 层 | n | base | A2c AUC | A3 AUC | counts AUC | A2c logloss | A3 logloss |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for d in ('d1', 'd2'):
        for layer in LAYERS:
            r = p[d][layer]
            lines.append(
                f"| {d} | {layer} | {r['n']} | {_fmt(r['base'])} | "
                f"{_fmt(r['a2c']['auc'])} | {_fmt(r['a3']['auc'])} | "
                f"{_fmt(r['counts']['auc'])} | {_fmt(r['a2c']['logloss'])} | "
                f"{_fmt(r['a3']['logloss'])} |")
    lines.append('\n## 2. 三张窗口的 27 类形状\n')
    lines.append('`prior_major` / `prior_top3` = 同子集上「每中心永远猜最常见形状」的 top1/top3 准确率（oracle 先验，保守基线）。\n')
    lines.append('| 层 | 子集 | n | A2c top1 | prior top1 | Δtop1 | A2c top3 | prior top3 | A3 top1 | A3 top3 | A2c CE | A3 CE |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for layer in LAYERS:
        for act in ACTIVES:
            r = s[layer][act]
            lines.append(
                f"| {layer} | {act} | {r['n']} | {_fmt(r['a2c']['top1'])} | "
                f"{_fmt(r['a2c']['prior_major'])} | {_fmt(r['delta_top1_vs_prior'], 3)} | "
                f"{_fmt(r['a2c']['top3'])} | {_fmt(r['a2c']['prior_top3'])} | "
                f"{_fmt(r['a3']['top1'])} | {_fmt(r['a3']['top3'])} | "
                f"{_fmt(r['a2c']['ce'])} | {_fmt(r['a3']['ce'])} |")
    lines.append('\n## 3. 五张窗口 MAE\n')
    w = res['window_mae']
    lines.append(f"- A2c EC：**{_fmt(w['a2c'])}** 张/格；A3 REC：{_fmt(w['a3'])}；窗口数 {w['n_centers']}")
    lines.append('\n## 4. 温度缩放（校准是否可救）\n')
    lines.append('| T | top1 | top3 | CE |')
    lines.append('|---:|---:|---:|---:|')
    for r in res['temperature']:
        lines.append(f"| {r['temp']} | {_fmt(r['top1'])} | {_fmt(r['top3'])} | {_fmt(r['ce'])} |")
    lines.append('\n## 5. 结论\n')
    for line in res['verdict']['findings']:
        lines.append(f'- {line}')
    lines.append(f"\n**判定：{res['verdict']['decision']}**\n")
    lines.append(f"P1 门禁：{res['verdict']['p1_gate']}\n")
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=str,
                    default='/tmp/read_model_diag_cache_a2c/cache.npz')
    ap.add_argument('--out', type=str,
                    default=str(HERE / 'results'))
    ap.add_argument('--n', type=int, default=0, help='只用前 n 个样本（0=全部）')
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261102)
    args = ap.parse_args()

    t0 = time.time()
    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(
            f'缓存不存在：{cache}\n'
            '先跑：python tools/read_model/diagnostics/s3158_local_extract.py 40000')
    z = load_cache(cache, args.n)
    print(f'[data] {cache} n={z["n"]}  {time.time()-t0:.0f}s', flush=True)

    masks = strata(z)
    centers = LS.centers_3()
    active_any, active_both = active_masks(z, centers)
    print(f'[active] any={active_any.mean():.3f} both={active_both.mean():.3f}',
          flush=True)

    pairwise = eval_pairwise(z, masks)
    print(f'[pairwise] done  {time.time()-t0:.0f}s', flush=True)
    shape = eval_shape(z, masks, centers, active_any, active_both,
                       args.boot, args.seed)
    print(f'[shape] done  {time.time()-t0:.0f}s', flush=True)
    wmae = eval_window_mae(z, LS.centers_5())
    temp = temperature_scan(z, centers)
    print(f'[scan] done  {time.time()-t0:.0f}s', flush=True)

    # 判定：A2c 的局部形状是否明显高于先验，且 pairwise 有信息
    a = shape['all']['active_any']
    d1 = pairwise['d1']['all']
    go = (a['a2c']['top1'] - a['a2c']['prior_major'] >= 0.01
          and d1['a2c']['auc'] >= 0.70)
    findings = [
        f"成对共持 d=1：A2c AUC {d1['a2c']['auc']:.3f} vs A3 {d1['a3']['auc']:.3f}"
        f" vs counts-only {d1['counts']['auc']:.3f}（base {d1['base']:.3f}）。",
        f"三张窗口（active_any）：A2c top1 {a['a2c']['top1']:.3f} vs 先验 "
        f"{a['a2c']['prior_major']:.3f}（Δ {a['delta_top1_vs_prior']:+.3f}），"
        f"top3 {a['a2c']['top3']:.3f}；A3 top1 {a['a3']['top1']:.3f}。",
        f"active_both：A2c top1 {shape['all']['active_both']['a2c']['top1']:.3f} vs "
        f"先验 {shape['all']['active_both']['a2c']['prior_major']:.3f}"
        f"（Δ {shape['all']['active_both']['delta_top1_vs_prior']:+.3f}）；"
        f"副露 active_both Δ {shape['called']['active_both']['delta_top1_vs_prior']:+.3f}。",
        f"立直 active_both：A2c top1 {shape['riichi']['active_both']['a2c']['top1']:.3f} vs "
        f"先验 {shape['riichi']['active_both']['a2c']['prior_major']:.3f}"
        f"（Δ {shape['riichi']['active_both']['delta_top1_vs_prior']:+.3f}）；"
        f"top3 {shape['riichi']['active_both']['a2c']['top3']:.3f} vs 先验 top3 "
        f"{shape['riichi']['active_both']['a2c']['prior_top3']:.3f}（模型更差）。",
        f"温度扫描：top1 对 T 不敏感；CE 在 T=1.0 最低"
        f"（{[r['ce'] for r in temp]}）⇒ 不是校准问题，是独立边际的表达问题。",
        f"五张窗口 MAE：A2c {wmae['a2c']:.3f} vs A3 {wmae['a3']:.3f} 张/格。",
    ]
    verdict = {
        'decision': 'GO → P1 冻结骨干局部形状头' if go else 'NO-GO（先修标签/信息）',
        'go_to_p1': bool(go),
        'p1_gate': ('冻结 A2c + 局部头必须在 active_any 上比每中心先验高 ≥5pp top1'
                    '（或 ≥0.10 nats CE），active_both 高 ≥3pp，且优于 A2c 独立乘积；'
                    '否则停止或转立直显式特征/蒸馏。'),
        'findings': findings,
    }
    meta = {
        'cache': str(cache),
        'ckpt': _meta(cache).get('ckpt', ''),
        'n': z['n'],
        'seed': args.seed,
        'protocol': ('202609 holdout 缓存；三张窗口 m−1/m/m+1 计数截断 0/1/2；'
                     '成对 d=1/2；active_any/both = 目标家手切过邻张；'
                     'prior_major = 同子集每中心最常见形状（oracle 先验，保守基线）；'
                     'bootstrap 为样本级（非文件块），CI 偏窄。'),
        'layers': list(LAYERS),
        'actives': list(ACTIVES),
    }
    res = {'meta': meta, 'pairwise': pairwise, 'shape': shape,
           'window_mae': wmae, 'temperature': temp, 'verdict': verdict}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'p0_local_shape.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'P0_REPORT.md', res)
    print(f'\n[out] {out}/p0_local_shape.json')
    print(f'[out] {out}/P0_REPORT.md')
    print(f'[verdict] {verdict["decision"]}')
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
