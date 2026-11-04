#!/usr/bin/env python3
"""按 cond 口径重算 P1/P2a/P2b/P2c 的局部形状指标。

## 为什么要重算

`active_any` 下最大类是 **EMPTY（空窗口）**，占 56.04%。此前 P0~P2c 的
micro top1 全部被「永远猜空」主导：A2c 只比先验高 1.4pp，被解读为
「信号稀薄」。`eval_semantic.py` 发现换成 cond 口径（筛非空 + 把 EMPTY 置 0
重归一）后，A2c 的真实增益是 +8.0pp / 归一化 11.1%。

本脚本把四个局部头放回同一口径下比较。

## 三种口径

| 口径 | 样本 | 概率 | 对应问题 |
|---|---|---|---|
| micro | 全部 | 原样 | 原先的报告口径（被 EMPTY 主导）|
| nonempty | y≠0 | 原样 | 「已知非空」但概率未条件化（EMPTY 抢 argmax）|
| **cond** | y≠0 | EMPTY 置 0 重归一 | **「已知有牌，问是什么结构」的公平口径** |

## 用法

    python tools/read_model/local_shape/eval_cond_recheck.py \\
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

import features as FEAT  # noqa: E402  必须在 _common 之后
import p1_local_head as P1  # noqa: E402
import p2_riichi_head as P2  # noqa: E402
import p2c_unfreeze as PC  # noqa: E402

import torch  # noqa: E402

LAYERS = P1.LAYERS if hasattr(P1, 'LAYERS') else (
    'all', 'riichi', 'nonriichi', 'called', 'turn>=8', 'clean_t8')
ACTIVES = ('all_centers', 'active_any', 'active_both')
N_CLASS = 27
MODES = ('micro', 'nonempty', 'cond')


def _masks(f: P1.Feat):
    return P1._masks(f)


def _active(f: P1.Feat):
    return {'all_centers': np.ones_like(f.any_, bool),
            'active_any': f.any_, 'active_both': f.both}


# --------------------------------------------------------------------------- #
# 各模型的 27 类概率 [n,3,21,27]
# --------------------------------------------------------------------------- #
def probs_a2c(f: P1.Feat, cache: dict):
    """A2c 的独立边际乘积（逐中心）。"""
    import local_shape as LS
    n = f.n
    out = np.zeros((n, 3, LS.NUM_PATTERNS, len(f.centers)), np.float16)
    AP = cache['AP']
    for ci, c in enumerate(f.centers):
        j = LS.joint_pattern_probs(AP, c)                  # [n*3,27]
        out[:, :, :, ci] = j.reshape(n, 3, LS.NUM_PATTERNS)
    return np.transpose(out, (0, 1, 3, 2))                 # [n,3,21,27]


def probs_head(head, feat, batch=2048):
    """P1/P2a/P2b：用 LocalDS 取 x，前向得 [n,3,21,27]。"""
    n = feat.n
    out = np.zeros((n, 3, 21, N_CLASS), np.float16)
    dl = P1._loader(feat, batch, False, 0)
    head.eval()
    with torch.no_grad():
        for x, _y, _w, idx in dl:
            logits = head(x.float().to(P1.DEV))            # [B,3,27]
            p = torch.softmax(logits, -1).cpu().numpy()
            for b, flat in enumerate(idx.numpy()):
                i, ci = divmod(int(flat), 21)
                out[i, :, ci] = p[b]
    return out


def probs_p2c(net, head, batch: dict, chunk=1024):
    """P2c：端到端（含解冻骨干）。"""
    n = batch['obs'].shape[0]
    out = np.zeros((n, 3, 21, N_CLASS), np.float16)
    net.eval()
    head.eval()
    meta = PC._meta(batch)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            sub = {k: v[lo:hi] for k, v in batch.items()}
            sm = {k: v[lo:hi] for k, v in meta.items()}
            h, g, e_T, s_b = PC._forward_features(net, sub, sm)
            x = PC._local_input(net, sub, sm, h, g, e_T, s_b)
            b = x.shape[0]
            logits = head(x.reshape(-1, PC.TARGETS, x.shape[-1])).reshape(
                b, PC.TARGETS, PC.N_CENTERS, N_CLASS)
            out[lo:hi] = torch.softmax(logits, -1).cpu().numpy().astype(np.float16)
            if (lo // chunk) % 10 == 0:
                print(f'  [p2c] {hi}/{n}', flush=True)
    return out


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _cond(p):
    q = p.astype(np.float32).copy()
    q[..., 0] = 0.0
    s = q.sum(-1, keepdims=True)
    return np.where(s > 0, q / np.maximum(s, 1e-12), 1.0 / N_CLASS)


def evaluate(P, Y, masks, active, tag):
    """P [n,3,21,27] float16；Y [n,3,21] int；返回 {layer:{act:{mode:{...}}}}。"""
    n = Y.shape[0]
    Pc = _cond(P) if 'cond' in MODES else None
    out = {}
    for layer in LAYERS:
        out[layer] = {}
        print(f'  [{tag}] {layer} …', flush=True)
        for act in ACTIVES:
            base = masks[layer][..., None] & active[act]      # [n,3,1]&[n,3,21]
            cell = {}
            for mode in MODES:
                m = base & (Y != 0) if mode != 'micro' else base
                if not m.any():
                    cell[mode] = {'n': 0}
                    continue
                src = Pc if mode == 'cond' else P
                pp = src[m].astype(np.float32)                # [N,27]
                yy = Y[m]
                idx = np.arange(len(yy))
                order = np.argsort(-pp, 1)
                ce = -np.log(np.clip(pp[idx, yy], 1e-12, None))
                # 先验（该中心该层该 act 的标签频率）
                # 注意：这里按 **中心** 分 21 组（把三个目标家合并）；
                # `bayes_ceiling.py` / `p0a_bayes_cond.py` 的 prior 按
                # **(中心, 目标家)** 分 63 组（与 micro prior 同口径）。
                # 细分组更强 ⇒ 63 组的 prior top1 更高（小样本 cell 上差
                # 可达 0.3pp），两者的 norm_gain 不可混用。
                yi = np.argwhere(m)                           # [N,3] → (i,j,ci)
                pri_ce = np.zeros(len(yy), np.float32)
                pri_top1 = np.zeros(len(yy), np.float32)
                pri_top3 = np.zeros(len(yy), np.float32)
                for ci in range(21):
                    sel = yi[:, 2] == ci
                    if not sel.any():
                        continue
                    yc = yy[sel]
                    cnt = np.bincount(yc, minlength=N_CLASS).astype(np.float64)
                    pr = cnt / max(cnt.sum(), 1.0)
                    pri_ce[sel] = -np.log(np.clip(pr[yc], 1e-12, None))
                    pri_top1[sel] = pr.max()
                    pri_top3[sel] = np.sort(pr)[-3:].sum()
                cell[mode] = {
                    'n': int(len(yy)),
                    'top1': float((order[:, 0] == yy).mean()),
                    'top3': float((order[:, :3] == yy[:, None]).any(1).mean()),
                    'ce': float(ce.mean()),
                    'prior_top1': float(pri_top1.mean()),
                    'prior_top3': float(pri_top3.mean()),
                    'prior_ce': float(pri_ce.mean()),
                    'norm_gain': float((((order[:, 0] == yy).astype(float)
                                         - pri_top1)).mean()
                                       / max(1.0 - pri_top1.mean(), 1e-9)),
                    'ce_reduction': float((pri_ce.mean() - ce.mean())
                                          / max(pri_ce.mean(), 1e-9)),
                }
            out[layer][act] = cell
    return out


# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '—'


def write_report(path, res):
    L = ['# cond 口径重算：P1/P2a/P2b/P2c 在「已知窗口非空」下的真实表现\n']
    m = res['meta']
    L.append(f"- 评估集：202609 holdout n={m['n']}（alloc 与 P0 缓存校验 "
             f"{'PASS' if m['alloc_match'] else 'FAIL'}）")
    L.append(f"- 设备：{m['device']}；耗时 {m['seconds']:.0f}s")
    L.append('- 三种口径：`micro` 全样本；`nonempty` 筛 y≠0 但不重归一；'
             '`cond` 筛 y≠0 且把 EMPTY 置 0 重归一（**公平口径**）。\n')

    L.append('## 1. 主表：all/active_any（cond 口径）\n')
    L.append('| 模型 | n | top1 | top3 | CE | prior top1 | 归一化增益 | CE 降幅 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for name in ('prior', 'a2c', 'p1', 'p2a', 'p2b', 'p2c'):
        r = res['table'][name]['all']['active_any']['cond']
        L.append(f"| {name} | {r['n']} | {_f(r['top1'])} | {_f(r['top3'])} | "
                 f"{_f(r['ce'])} | {_f(r['prior_top1'])} | "
                 f"{_f(r['norm_gain'])} | {_f(r['ce_reduction'])} |")

    L.append('\n## 2. 三口径对比（all/active_any）\n')
    L.append('| 模型 | micro top1 | nonempty top1 | cond top1 | cond 归一化增益 |')
    L.append('|---|---:|---:|---:|---:|')
    for name in ('a2c', 'p1', 'p2a', 'p2b', 'p2c'):
        c = res['table'][name]['all']['active_any']
        L.append(f"| {name} | {_f(c['micro']['top1'])} | "
                 f"{_f(c['nonempty']['top1'])} | {_f(c['cond']['top1'])} | "
                 f"{_f(c['cond']['norm_gain'])} |")

    L.append('\n## 3. 立直层（cond 口径）\n')
    L.append('| 模型 | riichi/any top1 | riichi/both top1 | riichi/both CE | '
             'riichi/both 归一化增益 |')
    L.append('|---|---:|---:|---:|---:|')
    for name in ('a2c', 'p1', 'p2a', 'p2b', 'p2c'):
        a = res['table'][name]['riichi']['active_any']['cond']
        b = res['table'][name]['riichi']['active_both']['cond']
        L.append(f"| {name} | {_f(a['top1'])} | {_f(b['top1'])} | {_f(b['ce'])} | "
                 f"{_f(b['norm_gain'])} |")

    L.append('\n## 4. clean_t8/active_both（cond 口径）\n')
    L.append('| 模型 | top1 | CE | prior top1 | 归一化增益 | CE 降幅 |')
    L.append('|---|---:|---:|---:|---:|---:|')
    for name in ('a2c', 'p1', 'p2a', 'p2b', 'p2c'):
        r = res['table'][name]['clean_t8']['active_both']['cond']
        L.append(f"| {name} | {_f(r['top1'])} | {_f(r['ce'])} | "
                 f"{_f(r['prior_top1'])} | {_f(r['norm_gain'])} | "
                 f"{_f(r['ce_reduction'])} |")

    L.append('\n## 4b. 分层胜负（cond，相对 A2c；正=A2c 负=局部头）\n')
    L.append('| 模型 | all/any top1 | all/both top1 | riichi/both top1 | '
             'clean_t8/both top1 | all/any CE |')
    L.append('|---|---:|---:|---:|---:|---:|')
    a2 = res['table']['a2c']
    for name in ('p1', 'p2a', 'p2b', 'p2c'):
        t = res['table'][name]
        cells = []
        for layer, act in (('all', 'active_any'), ('all', 'active_both'),
                           ('riichi', 'active_both'),
                           ('clean_t8', 'active_both')):
            cells.append(t[layer][act]['cond']['top1']
                         - a2[layer][act]['cond']['top1'])
        dce = a2['all']['active_any']['cond']['ce'] \
            - t['all']['active_any']['cond']['ce']
        L.append(f"| {name} | " + ' | '.join(f'{c:+.4f}' for c in cells)
                 + f" | {dce:+.4f} |")

    L.append('\n## 5. 结论\n')
    for line in res['verdict']['findings']:
        L.append(f'- {line}')
    L.append(f"\n**判定：{res['verdict']['decision']}**\n")
    path.write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    ap.add_argument('--ckpt', default=str(FEAT.REPO / 'mortal' / 'read_model_v2s_a2c' / 'best.pth'))
    ap.add_argument('--out', default=str(HERE / 'results'))
    ap.add_argument('--probs-cache', default='/tmp/ls_cond_probs.npz',
                    help='概率缓存（避免重跑推理）')
    ap.add_argument('--reuse-probs', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    print(f'[data] 构建固定评估集 n≤{args.n} …', flush=True)
    batch, bn, n_files = FEAT.C.build_eval(args.n)
    n = int(batch['obs'].shape[0])
    print(f'[data] n={n} files={n_files}  {time.time()-t0:.0f}s', flush=True)

    # 与 P0 缓存校验
    z = np.load(args.cache, allow_pickle=True)
    ref = np.asarray(z['alloc'])[:n].astype(np.int64)
    alloc = np.asarray(bn['alloc'])[:n].astype(np.int64)
    alloc_match = bool(np.array_equal(ref, alloc))
    print(f'[check] alloc 与 P0 缓存 {"PASS" if alloc_match else "FAIL"}', flush=True)
    if not alloc_match:
        raise SystemExit('评估集顺序漂移，拒绝继续')

    # ---- A2c 骨干导出特征（一次前向） ----
    net = FEAT._load_net(Path(args.ckpt))
    print(f'[feat] 导出 A2c 特征 …', flush=True)
    cache, _meta = FEAT._extract(net, batch, FEAT.C.DEV, save_ap=True)
    f1 = P1.Feat(cache)
    f2 = P2.P2Feat(cache)
    print(f'[feat] h={f1.h.shape} d_in={f1.d_in}/{f2.d_in}  '
          f'{time.time()-t0:.0f}s', flush=True)

    masks = _masks(f1)
    active = _active(f1)
    Y = f1.pat                                             # [n,3,21]
    print(f'[y] shape={Y.shape}  空窗口占比='
          f'{float((Y == 0).mean()):.4f}', flush=True)

    # ---- 各模型概率 ----
    pc = Path(args.probs_cache)
    P = {}
    if args.reuse_probs and pc.exists():
        d = np.load(pc)
        P = {k: d[k] for k in d.files}
        print(f'[probs] 复用 {pc}  keys={list(P)}', flush=True)
    else:
        P['a2c'] = probs_a2c(f1, cache)
        print(f'[a2c] {time.time()-t0:.0f}s', flush=True)

        st = torch.load(HERE / 'results' / 'p1' / 'head.pth', map_location=P1.DEV,
                        weights_only=False)
        h1 = P1.LocalShapeHead(st['config']['d_in'], st['config']['hidden']).to(P1.DEV)
        h1.load_state_dict(st['state_dict'])
        P['p1'] = probs_head(h1, f1)
        print(f'[p1] {time.time()-t0:.0f}s', flush=True)

        for name, rel in (('p2a', 'p2/p2_head.pth'), ('p2b', 'p2b/p2b_head.pth')):
            st = torch.load(HERE / 'results' / rel, map_location=P1.DEV,
                            weights_only=False)
            hd = P2.P2Head(st['config']['d_in'], st['config']['hidden'],
                           st['config'].get('dropout', 0.1)).to(P1.DEV)
            hd.load_state_dict(st['state_dict'])
            P[name] = probs_head(hd, f2)
            print(f'[{name}] {time.time()-t0:.0f}s', flush=True)

        stc = torch.load(HERE / 'results' / 'p2c' / 'p2c_best.pth',
                         map_location=P1.DEV, weights_only=False)
        net2 = FEAT.C.load_net(Path(args.ckpt))
        net2.load_state_dict(stc['net'])
        w = stc['head']['fc1.weight']
        hd = P2.P2Head(int(w.shape[1]), int(w.shape[2]), 0.0).to(P1.DEV)
        hd.load_state_dict(stc['head'])
        P['p2c'] = probs_p2c(net2, hd, batch)
        print(f'[p2c] {time.time()-t0:.0f}s', flush=True)
        np.savez_compressed(pc, **P)
        print(f'[probs] 写入 {pc}', flush=True)

    # ---- 指标 ----
    table = {}
    for name, pr in P.items():
        table[name] = evaluate(pr, Y, masks, active, name)
        print(f'[eval] {name} done  {time.time()-t0:.0f}s', flush=True)

    # prior 行（不依赖模型）：用 micro 的 prior 指标
    table['prior'] = {}
    for layer in LAYERS:
        table['prior'][layer] = {}
        for act in ACTIVES:
            src = table['a2c'][layer][act]
            table['prior'][layer][act] = {
                m: {'n': src[m]['n'],
                    'top1': src[m]['prior_top1'], 'top3': src[m]['prior_top3'],
                    'ce': src[m]['prior_ce'],
                    'prior_top1': src[m]['prior_top1'],
                    'prior_top3': src[m]['prior_top3'],
                    'prior_ce': src[m]['prior_ce'],
                    'norm_gain': 0.0, 'ce_reduction': 0.0}
                for m in MODES if src.get(m, {}).get('n')}

    # ---- 判定 ----
    def g(name, layer, act, mode='cond'):
        return table[name][layer][act][mode]

    a = g('a2c', 'all', 'active_any')
    best = max(('p1', 'p2a', 'p2b', 'p2c'),
               key=lambda k: g(k, 'all', 'active_any')['norm_gain'])
    bb = g(best, 'all', 'active_any')

    def d_top1(layer, act):
        return g(best, layer, act)['top1'] - g('a2c', layer, act)['top1']

    def d_ce(layer, act):
        return (g('a2c', layer, act)['ce'] - g(best, layer, act)['ce'])

    d_any, d_both = d_top1('all', 'active_any'), d_top1('all', 'active_both')
    d_ri, d_cl = d_top1('riichi', 'active_both'), d_top1('clean_t8', 'active_both')
    ce_any, ce_both = d_ce('all', 'active_any'), d_ce('all', 'active_both')
    ce_ri, ce_cl = d_ce('riichi', 'active_both'), d_ce('clean_t8', 'active_both')

    findings = [
        f"cond 口径下 A2c（独立边际乘积）all/active_any：top1 "
        f"{_f(a['top1'])} vs 先验 {_f(a['prior_top1'])}，归一化增益 "
        f"{_f(a['norm_gain'])}；而 micro 口径只有 "
        f"{_f(table['a2c']['all']['active_any']['micro']['norm_gain'])} ⇒ "
        f"micro 确实低估了 A2c。",
        f"最强局部头（{best}，all/active_any）：top1 {_f(bb['top1'])}、top3 "
        f"{_f(bb['top3'])}、CE {_f(bb['ce'])}、归一化增益 {_f(bb['norm_gain'])}、"
        f"CE 降幅 {_f(bb['ce_reduction'])} ⇒ top1 {_f(d_any, 4)} / "
        f"CE {_f(ce_any, 4)} 相对 A2c。",
        f"active_both（cond）：top1 {best} {_f(g(best, 'all', 'active_both')['top1'])} "
        f"vs A2c {_f(g('a2c', 'all', 'active_both')['top1'])}（{_f(d_both, 4)}），"
        f"CE {_f(g(best, 'all', 'active_both')['ce'])} vs "
        f"{_f(g('a2c', 'all', 'active_both')['ce'])}（{_f(ce_both, 4)}）。",
        f"riichi/active_both（cond）：{best} top1 "
        f"{_f(g(best, 'riichi', 'active_both')['top1'])} vs A2c "
        f"{_f(g('a2c', 'riichi', 'active_both')['top1'])}（{_f(d_ri, 4)}），"
        f"CE {_f(g(best, 'riichi', 'active_both')['ce'])} vs "
        f"{_f(g('a2c', 'riichi', 'active_both')['ce'])}（{_f(ce_ri, 4)}）；"
        f"但两者归一化增益均为负（{_f(g(best, 'riichi', 'active_both')['norm_gain'])} / "
        f"{_f(g('a2c', 'riichi', 'active_both')['norm_gain'])}）⇒ 该子集全体失效。",
        f"clean_t8/active_both（cond）：{best} "
        f"{_f(g(best, 'clean_t8', 'active_both')['top1'])} vs A2c "
        f"{_f(g('a2c', 'clean_t8', 'active_both')['top1'])}（{_f(d_cl, 4)}），"
        f"CE {_f(g(best, 'clean_t8', 'active_both')['ce'])} vs "
        f"{_f(g('a2c', 'clean_t8', 'active_both')['ce'])}（{_f(ce_cl, 4)}）。",
    ]

    # 多维判：主口径胜 + 立直胜，但 active_both 系退化 ⇒ 「有条件成立」
    if d_any > 0.005 and ce_any > 0:
        if d_both < -0.005 or d_cl < -0.005:
            decision = (
                f'cond 口径下 {best} 在主口径（all/active_any）胜过 A2c '
                f'{d_any:+.4f} top1 / {ce_any:+.4f} CE，立直 active_both '
                f'{d_ri:+.4f} / {ce_ri:+.4f} ⇒ **原 FAIL 的「top1 未过 A2c」'
                f'确实是 micro 口径造成的**；但 active_both 系反而低 '
                f'{d_both:+.4f}/{d_cl:+.4f} ⇒ 真实情况是「**分层互有胜负**」，'
                f'而非单向超越。叠加 alloc CE 退化（P2c +0.029）这一真实代价，'
                f'原「关闭」结论应改为「**有条件成立**」：局部头可作概率校准/'
                f'立直子集的组件，但不足以支撑通用局部形状头重启 200k。')
        else:
            decision = (f'cond 口径下 {best} 在全部分层均不弱于 A2c '
                        f'（主口径 {d_any:+.4f}、both {d_both:+.4f}、'
                        f'立直 {d_ri:+.4f}、clean {d_cl:+.4f}）⇒ '
                        f'原 FAIL 判定**过严**，局部形状线不宜按原结论关闭')
    elif d_any > -0.005 and ce_any > 0:
        decision = (f'cond 口径下 {best} 的 top1 与 A2c 基本持平'
                    f'（{d_any:+.4f}）但 CE 更好（{ce_any:+.4f}）⇒ '
                    f'局部头的价值在**校准**而非判别；原 FAIL 成立（不重启）')
    else:
        decision = (f'cond 口径下 {best} 仍不优于 A2c'
                    f'（top1 {d_any:+.4f}，CE {ce_any:+.4f}）⇒ '
                    f'原 FAIL/关闭结论成立')

    res = {
        'meta': {'n': int(n), 'n_files': int(n_files),
                 'alloc_match': alloc_match, 'device': P1.DEV,
                 'seconds': float(time.time() - t0),
                 'cache': args.cache, 'ckpt': args.ckpt},
        'table': table,
        'verdict': {'decision': decision, 'findings': findings},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'cond_recheck.json', 'w') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'COND_RECHECK_REPORT.md', res)
    print(f'\n[out] {out}/cond_recheck.json')
    print(f'[out] {out}/COND_RECHECK_REPORT.md')
    print(f'[verdict] {decision}')
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
