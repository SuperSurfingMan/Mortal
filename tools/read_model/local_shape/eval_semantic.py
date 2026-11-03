#!/usr/bin/env python3
"""语义标签路线验证（零训练）：10 类局部结构角色值不值得训练。

## 为什么不能只比"准确率"

`center` 本来就是已知的（21 个中心分开建模），所以语义标签**不增加信息**：
它只是把 21×27 的标签空间（有效 567 类）压缩到 21×10（有效约 200 类）。

真正的收益只能是**跨中心共享参数**：若 10 类语义在各中心上分布更一致，
就可以用「共享判别结构 + 位置偏置」代替 21 个独立头，等效样本量 ×21，
方差下降。反之若压缩丢信息，则不如不换。

## 四组诊断

| 维度 | 指标 | 含义 |
|---|---|---|
| V0 标签结构 | 每中心熵、有效类别数 | 标签有多集中 |
| V0 跨中心一致性 | KL(P_center ‖ P_pooled) | **共享可行性**（越小越能共享）|
| V1 共享代价 | CE(shared_prior) − CE(per_center_prior) | 压缩丢多少（27 类应显著大于 10 类）|
| V2 可预测性 | A2c 相对 prior 的 CE 降低 / top1 增益 | 信号是否更容易读出 |
| V3 上限 | counts 精确后验的同类指标 | 贝叶斯可达性 |

## 预注册门禁（GO 才训练）

1. `share_penalty_10 <= 0.05` nats（27 类基准约 0.16）；
2. `ce_gain_ratio_10 >= ce_gain_ratio_27`（相对信息量不降）；
3. `top1_norm_gain_10 >= 1.5 × top1_norm_gain_27`（归一化增益至少 1.5 倍）。

三条全过 → 训练「跨中心共享语义头」；否则停，不改标签。

用法：
    python tools/read_model/local_shape/eval_semantic.py \\
        --cache /tmp/read_model_diag_cache_a2c/cache.npz \\
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

import bayes_ceiling as BC  # noqa: E402
import eval_local_shape as P0  # noqa: E402
import local_shape as LS  # noqa: E402
import semantic_label as SL  # noqa: E402

LAYERS = P0.LAYERS
ACTIVES = P0.ACTIVES
N10 = SL.NUM_SEMANTIC


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _center_metrics(p_flat, y_flat, mask_flat, n_class, nonempty=False,
                    cond=False):
    """p_flat [n*3,K], y_flat [n*3], mask_flat [n*3]。

    三种口径：
      micro      全样本（被 EMPTY 主导）
      nonempty   y≠0，但 p 仍是全类分布（EMPTY 的概率质量会抢走 argmax）
      cond       y≠0 且 p **条件重归一化**（把第 0 类置 0 再归一）——
                 这才是「已知窗口有牌，问是什么结构」的公平口径。
    """
    p = p_flat
    if cond:
        p = np.array(p_flat, dtype=np.float64, copy=True)
        p[:, 0] = 0.0
        s = p.sum(1, keepdims=True)
        p = np.where(s > 0, p / np.maximum(s, 1e-12), 1.0 / n_class)
    sel = mask_flat & (y_flat != 0) if (nonempty or cond) else mask_flat
    if not sel.any():
        return None
    pp = np.clip(p[sel], 1e-12, 1.0)
    yy = y_flat[sel]
    idx = np.arange(len(yy))
    order = np.argsort(-pp, axis=1)
    cnt = np.bincount(yy, minlength=n_class).astype(np.float64)
    pr = cnt / max(cnt.sum(), 1.0)
    return {
        'n': int(len(yy)),
        'top1': float((order[:, 0] == yy).mean()),
        'top3': float((order[:, :3] == yy[:, None]).any(1).mean()),
        'ce': float(-np.log(pp[idx, yy]).mean()),
        'prior_major': float(pr.max()),
        'prior_top3': float(np.sort(pr)[-3:].sum()),
        'prior_entropy': _entropy(pr),
        'margin': float((np.sort(pp, 1)[:, -1] - np.sort(pp, 1)[:, -2]).mean()),
    }


def _pool_prior(y_flat, mask_flat, n_class):
    yy = y_flat[mask_flat]
    cnt = np.bincount(yy, minlength=n_class).astype(np.float64)
    return cnt / max(cnt.sum(), 1.0)


def _shared_ce(p_prior, y_flat, mask_flat):
    yy = y_flat[mask_flat]
    if len(yy) == 0:
        return float('nan')
    return float(-np.log(np.clip(p_prior[yy], 1e-12, None)).mean())


def _kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float((p * np.log(p / q)).sum())


def run(z, masks, centers, active_any, active_both, CU, N):
    n = z['n']
    active = {'all_centers': np.ones_like(active_any, bool),
              'active_any': active_any, 'active_both': active_both}
    ks = [(l, a) for l in LAYERS for a in ACTIVES]

    acc = {k: {tag: {'n': 0, 'top1': 0.0, 'top3': 0.0, 'ce': 0.0,
                     'prior_major': 0.0, 'prior_top3': 0.0,
                     'prior_entropy': 0.0, 'margin': 0.0, 'shared_ce': 0.0}
               for tag in ('a2c27', 'a2c10', 'counts27', 'counts10',
                           'a2c27_ne', 'a2c10_ne', 'counts27_ne', 'counts10_ne',
                           'a2c27_cond', 'a2c10_cond', 'counts27_cond',
                           'counts10_cond')}
           for k in ks}
    # 每中心每层每子集的标签计数（算跨中心 KL 与共享代价；只存计数，近似零内存）
    cf = {(l, a): {'y27': np.zeros((len(centers), 27)),
                   'y10': np.zeros((len(centers), N10))}
          for l in LAYERS for a in ACTIVES}
    center_meta = []

    t0 = time.time()
    for ci, center in enumerate(centers):
        y27 = LS.true_pattern(z['alloc'], center).reshape(-1)      # [n*3]
        abc = np.clip(z['alloc'][:, :, list(center)], 0, 2)
        y10 = SL.semantic_index(abc, center).reshape(-1)           # [n*3]
        M = SL.mapping_matrix(center)

        p27_a2c = LS.joint_pattern_probs(z['AP'], center)          # [n*3,27]
        p10_a2c = p27_a2c @ M

        p27_cnt = np.stack([BC.exact_counts_shape(CU, N, z['hand'][:, j], center)
                            for j in range(3)], axis=1).reshape(-1, 27)
        p10_cnt = p27_cnt @ M

        masks_flat = {k: masks[k].reshape(-1) for k in LAYERS}
        act_flat = {a: active[a][:, :, ci].reshape(-1) for a in ACTIVES}

        for layer in LAYERS:
            for act in ACTIVES:
                m = masks_flat[layer] & act_flat[act]
                k = (layer, act)
                for tag, p, y, nc in (('a2c27', p27_a2c, y27, 27),
                                      ('a2c10', p10_a2c, y10, N10),
                                      ('counts27', p27_cnt, y27, 27),
                                      ('counts10', p10_cnt, y10, N10)):
                    for ne, cd, sfx in ((False, False, ''),
                                        (True, False, '_ne'),
                                        (True, True, '_cond')):
                        md = _center_metrics(p, y, m, nc, nonempty=ne, cond=cd)
                        if md is None:
                            continue
                        a = acc[k][tag + sfx]
                        a['n'] += md['n']
                        for f in ('top1', 'top3', 'ce', 'prior_major',
                                  'prior_top3', 'prior_entropy', 'margin'):
                            a[f] += md[f] * md['n']
                # 标签计数（用于跨中心 KL / 共享代价）
                yy27 = y27[m]
                yy10 = y10[m]
                if len(yy27):
                    cf[k]['y27'][ci] += np.bincount(yy27, minlength=27)
                    cf[k]['y10'][ci] += np.bincount(yy10, minlength=N10)
        center_meta.append({'center': [int(x) for x in center],
                            'y10_entropy': _entropy(
                                np.bincount(y10, minlength=N10) / max(len(y10), 1))})
        if (ci + 1) % 7 == 0:
            print(f'[sem] {ci+1}/{len(centers)} centers  {time.time()-t0:.0f}s',
                  flush=True)

    # ---- 汇总 ----
    out = {}
    for layer in LAYERS:
        out[layer] = {}
        for act in ACTIVES:
            k = (layer, act)
            row = {}
            for tag, a in acc[k].items():
                n_ = max(a['n'], 1)
                row[tag] = {f: float(a[f] / n_) for f in
                            ('top1', 'top3', 'ce', 'prior_major', 'prior_top3',
                             'prior_entropy', 'margin')}
                row[tag]['n'] = int(a['n'])
            # 共享代价与跨中心 KL（精确解析：按中心加权）
            for tag, nc, key in (('27', 27, 'y27'), ('10', N10, 'y10')):
                C = cf[k][key]
                tot = C.sum(0)
                if tot.sum() <= 0:
                    continue
                pooled = tot / tot.sum()
                n_c = C.sum(1)
                w = n_c / n_c.sum()
                Pc = C / np.maximum(n_c[:, None], 1.0)
                lp = np.log(np.clip(Pc, 1e-12, None))
                lq = np.log(np.clip(pooled, 1e-12, None))
                shared_ce = float((w * (-(Pc * lq).sum(1))).sum())
                kl = float((w * (Pc * (lp - lq)).sum(1)).sum())
                row[f'shared_ce_{tag}'] = shared_ce
                row[f'kl_center_{tag}'] = kl
                row[f'share_penalty_{tag}'] = shared_ce - row[
                    'a2c27' if tag == '27' else 'a2c10']['prior_entropy']
            out[layer][act] = row
    return out, center_meta


def load_27_baseline(path):
    """从 P0 json 读 27 类基准（per-center prior / A2c / counts 地板）。"""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out = {}
    for layer in LAYERS:
        out[layer] = {}
        for act in ACTIVES:
            r = (d.get('shape') or {}).get(layer, {}).get(act)
            if not r:
                continue
            a = r.get('a2c') or {}
            out[layer][act] = {
                'top1': a.get('top1'), 'top3': a.get('top3'), 'ce': a.get('ce'),
                'prior_major': a.get('prior_major'),
                'prior_top3': a.get('prior_top3'),
                'prior_entropy': a.get('prior_entropy'),
            }
    return out


def global_prior_27(heads_dir):
    """P1 的 prior_eval 是全局 27 类先验（作为 27 类共享代价的基准）。"""
    for name in ('p2b/p2b_distill.json', 'p1/p1_local_head.json'):
        p = heads_dir / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out = {}
        for layer in LAYERS:
            out[layer] = {}
            for act in ACTIVES:
                r = (d.get('eval') or {}).get(layer, {}).get(act, {})
                pr = r.get('prior_eval')
                if pr:
                    out[layer][act] = {'ce': pr.get('ce'), 'top1': pr.get('top1')}
        if out.get('all', {}).get('active_any'):
            return out
    return {}


def _f(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '—'


def write_report(path, res):
    L = ['# 语义标签路线验证（零训练）\n']
    m = res['meta']
    L.append(f"- 缓存：`{m['cache']}`（n={m['n']}）")
    L.append(f"- 标签：27 类计数形状 vs **10 类语义结构角色**（`semantic_label.py`）")
    L.append(f"- 映射自检：每行恰一 1 = {m['coverage']['all_rows_ok']}；"
             f"有效类数 {m['coverage']['min_classes_used']}~"
             f"{m['coverage']['max_classes_used']}")
    L.append('- 关键前提：`center` 已知 ⇒ 语义标签**不增加信息**，'
             '收益只能来自「跨中心共享参数」；因此核心指标是**共享代价**与**增益率**。\n')

    L.append('## 1. 标签结构（V0）\n')
    L.append('| 层 | 子集 | 27类熵 | 10类熵 | 27类有效类数 | 10类有效类数 |')
    L.append('|---|---|---:|---:|---:|---:|')
    for layer in ('all', 'riichi', 'clean_t8'):
        for act in ACTIVES:
            r = res['table'][layer][act]
            h27 = r['a2c27']['prior_entropy']
            h10 = r['a2c10']['prior_entropy']
            L.append(f"| {layer} | {act} | {_f(h27)} | {_f(h10)} | "
                     f"{_f(float(np.exp(h27)), 1)} | {_f(float(np.exp(h10)), 1)} |")

    L.append('\n## 2. 共享可行性（V0/V1）\n')
    L.append('`KL(中心‖池化)`：10 类语义在各中心间的分布差异（越小越能共享）；'
             '`共享代价` = 用全局先验替代每中心先验的 CE 增量。\n')
    L.append('| 层 | 子集 | KL_center(10类) | 共享代价(10类) | 共享代价(27类) |')
    L.append('|---|---|---:|---:|---:|')
    for layer in ('all', 'riichi', 'clean_t8'):
        for act in ACTIVES:
            r = res['table'][layer][act]
            L.append(f"| {layer} | {act} | {_f(r.get('kl_center_10'))} | "
                     f"{_f(r.get('share_penalty_10'))} | "
                     f"{_f(r.get('share_penalty_27'))} |")

    L.append('\n## 3. 可预测性（V2/V3，A2c 与 counts 上限）\n')
    L.append('`micro` = 全样本（被 EMPTY 主导，仅作参考）；`nonempty` = 窗口非空但 p 未重归一；')
    L.append('`cond` = 非空 + 把 EMPTY 置 0 重归一（**公平口径**，对应「已知有牌问结构」）。')
    L.append('`归一化增益` = (top1 − prior) / (1 − prior)；`CE 降幅` = '
             '(CE_prior − CE) / CE_prior；两者跨标签可比。\n')
    L.append('| 标签 | 层/子集 | 口径 | n | prior top1 | 模型 top1 | 归一化增益 | '
             'prior CE | 模型 CE | CE 降幅 |')
    L.append('|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for tag in ('a2c27', 'a2c10', 'counts27', 'counts10'):
        for layer in ('all', 'riichi', 'clean_t8'):
            for act in ('active_any', 'active_both'):
                for sfx, lab in (('', 'micro'), ('_ne', 'nonempty'),
                                 ('_cond', 'cond')):
                    r = res['table'][layer][act][tag + sfx]
                    if not r.get('n'):
                        continue
                    pr = r['prior_major']
                    g = (r['top1'] - pr) / max(1.0 - pr, 1e-9)
                    cr = (r['prior_entropy'] - r['ce']) / max(
                        r['prior_entropy'], 1e-9)
                    L.append(f"| {tag} | {layer}/{act} | {lab} | {r['n']} | "
                             f"{_f(pr)} | {_f(r['top1'])} | {_f(g, 4)} | "
                             f"{_f(r['prior_entropy'])} | {_f(r['ce'])} | "
                             f"{_f(cr, 4)} |")

    L.append('\n## 4. 门禁判定\n')
    for line in res['verdict']['findings']:
        L.append(f'- {line}')
    L.append(f"\n**判定：{res['verdict']['decision']}**\n")
    if res['verdict'].get('train_plan'):
        L.append('## 5. 训练方案（仅在 GO 时执行）\n')
        for line in res['verdict']['train_plan']:
            L.append(f'- {line}')
        L.append('')
    path.write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--out', default=str(HERE / 'results'))
    ap.add_argument('--p0-json',
                    default=str(HERE / 'results' / 'p0_local_shape.json'))
    ap.add_argument('--n-centers', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    z = P0.load_cache(Path(args.cache), args.n)
    with np.load(Path(args.cache), allow_pickle=False) as zz:
        for k in ('unseen', 'hand'):
            z[k] = np.asarray(zz[k])[:z['n']]
    z['unseen'] = z['unseen'].astype(np.float64)
    z['hand'] = z['hand'].astype(np.int64)
    print(f'[data] n={z["n"]}', flush=True)

    centers = LS.centers_3()
    if args.n_centers:
        centers = centers[:args.n_centers]
    cov = SL.coverage_check(LS.centers_3())
    print(f'[tag] 映射自检 rows_ok={cov["all_rows_ok"]} '
          f'classes={cov["min_classes_used"]}~{cov["max_classes_used"]}', flush=True)
    if not cov['all_rows_ok']:
        raise SystemExit('语义映射不是每行恰一 1，拒绝继续')

    masks = P0.strata(z)
    a_any, a_both = P0.active_masks(z, centers)
    CU, N = BC.build_count_tables(z['unseen'])
    table, cmeta = run(z, masks, centers, a_any, a_both, CU, N)

    base27 = load_27_baseline(Path(args.p0_json))

    # 用 P0 的 27 类基准补 counts 地板对照
    for layer in LAYERS:
        for act in ACTIVES:
            r = table[layer][act]
            b = base27.get(layer, {}).get(act)
            if b:
                r['ref27'] = b

    # --- 门禁（用 all/active_any 与 all/active_both 的**非空条件**判定）---
    def gain(tag, layer, act):
        r = table[layer][act][tag]
        pr, t1, ce, h = (r['prior_major'], r['top1'], r['ce'],
                         r['prior_entropy'])
        return ((t1 - pr) / max(1.0 - pr, 1e-9),
                (h - ce) / max(h, 1e-9))

    g27_any, c27_any = gain('a2c27_cond', 'all', 'active_any')
    g10_any, c10_any = gain('a2c10_cond', 'all', 'active_any')
    g27_both, c27_both = gain('a2c27_cond', 'all', 'active_both')
    g10_both, c10_both = gain('a2c10_cond', 'all', 'active_both')
    sp10 = table['all']['active_any'].get('share_penalty_10')
    sp27 = table['all']['active_any'].get('share_penalty_27')
    kl10 = table['all']['active_any'].get('kl_center_10')

    ok1 = bool(kl10 is not None and np.isfinite(kl10) and kl10 <= 0.05)
    ok2 = bool(c10_any >= c27_any)
    ok3 = bool(g10_any >= 1.5 * g27_any)
    go = ok1 and ok2 and ok3

    ne27 = table['all']['active_any']['a2c27_ne']
    ne10 = table['all']['active_any']['a2c10_ne']
    cd27 = table['all']['active_any']['a2c27_cond']
    cd10 = table['all']['active_any']['a2c10_cond']
    findings = [
        f"**口径修正（关键）**：`active_any` 下最大类是 **EMPTY（空窗口）**，占 "
        f"{_f(table['all']['active_any']['a2c27']['prior_major'])} ⇒ micro top1 被"
        f"「永远猜空」主导。非空占 "
        f"{_f(ne27['n'] / max(table['all']['active_any']['a2c27']['n'], 1))}"
        f"（n={ne27['n']}）。",
        f"**A2c 的非空失效**：非空条件下 A2c 归一化增益 "
        f"{_f(gain('a2c27_ne', 'all', 'active_any')[0], 4)}（top1 "
        f"{_f(ne27['top1'])} vs 先验 {_f(ne27['prior_major'])}）⇒ 它的独立边际乘积"
        f"主要在判「窗口有没有牌」，而非「是什么形状」。",
        f"**条件重归一化后**（将 EMPTY 置 0；这才是公平口径）：27 类 A2c top1 "
        f"{_f(cd27['top1'])} vs 先验 {_f(cd27['prior_major'])}，归一化增益 "
        f"{_f(g27_any, 4)}；10 类 A2c top1 {_f(cd10['top1'])} vs 先验 "
        f"{_f(cd10['prior_major'])}，归一化增益 {_f(g10_any, 4)}。",
        f"标签结构（非空）：27 类熵 {_f(ne27['prior_entropy'])} → 10 类 "
        f"{_f(ne10['prior_entropy'])}；有效类别数 "
        f"{_f(float(np.exp(ne27['prior_entropy'])), 1)} → "
        f"{_f(float(np.exp(ne10['prior_entropy'])), 1)}。",
        f"跨中心一致性：KL 27 类 {_f(table['all']['active_any'].get('kl_center_27'))}"
        f" → 10 类 {_f(kl10)}；共享代价 27 类 {_f(sp27)} → 10 类 {_f(sp10)} nats。",
        f"CE 降幅（cond，all/active_any）：27 类 {_f(c27_any, 4)} → 10 类 "
        f"{_f(c10_any, 4)}；active_both：{_f(c27_both, 4)} → {_f(c10_both, 4)}。",
        f"门禁：KL≤0.05 = {ok1}；CE 降幅不降 = {ok2}；归一化增益 ≥1.5× = {ok3} "
        f"⇒ {'GO' if go else 'NO-GO'}。",
    ]

    train_plan = [
        '**目标**：训练「跨中心共享语义头」，验证压缩是否带来方差收益。',
        '**输入**：`h`（五列 480）+ `e_T`（64）+ `s_belief`（46）+ `g`（96）'
        '+ **位置嵌入**（21 中心 one-hot 或 3 维 (花色, 位置) embedding，32 维）。',
        '**结构**：共享 MLP（2×512, GELU, dropout 0.1）→ 10 类 logits；'
        '位置嵌入只进入第一层（作为 bias 注入），不设 21 个独立输出层。',
        '**参数量**：约 0.9M（对比 P1 的 21×27 独立头 1.35M），'
        '且 10 类共享 ⇒ 每类有效样本 ×21。',
        '**训练**：冻结 A2c 骨干（同 P1），train 202608 / val 5k，'
        'epoch≤20，早停 patience 3，CE + label smoothing 0.02。',
        '**门禁（预注册）**：all/active_any top1 ≥ 0.5755（A2c 0.5746 + 1pp）'
        '或 CE ≤ 1.33（A2c 1.432 − 0.10）；且 active_both top1 ≥ 0.745'
        '（A2c 0.7389 + 0.6pp）；且 alloc CE 不回退。',
        '**失败即停**：若 20 epoch 内未过门禁，语义标签路线关闭，'
        '不再尝试 5 类/分解软标签。',
    ]

    res = {
        'meta': {'cache': args.cache, 'n': int(z['n']),
                 'centers': len(centers), 'coverage': cov,
                 'layers': list(LAYERS), 'actives': list(ACTIVES)},
        'table': table, 'center_meta': cmeta,
        'verdict': {'decision': ('GO → 训练跨中心共享语义头' if go
                                 else 'NO-GO：语义标签不带来可测增益，不训练'),
                    'gate': {'kl_ok': ok1, 'ce_ok': ok2, 'gain_ok': ok3,
                             'kl10': kl10,
                             'share_penalty_27': sp27},
                    'findings': findings,
                    'train_plan': train_plan if go else []},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'semantic_label.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'SEMANTIC_LABEL_REPORT.md', res)
    print(f'[out] {out}/semantic_label.json')
    print(f'[out] {out}/SEMANTIC_LABEL_REPORT.md')
    print(f'[verdict] {res["verdict"]["decision"]}')
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
