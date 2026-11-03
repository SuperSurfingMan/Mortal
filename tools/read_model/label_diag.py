#!/usr/bin/env python3
"""标签诊断：扫全部在读任务的「最大类占比 / 有效类数」，标出口径风险。

## 背景

局部形状（27 类）的 `active_any` 最大类是 EMPTY，占 56.04%，使 micro top1
被「永远猜空」主导：A2c 的归一化增益在 micro 下只有 3.23%，cond 下有 11.08%。

同类问题可能存在于其他读牌任务。本脚本对每个任务的标签计算：

- `p_max`：最大类占比（= 常数预测器的 top1）
- `H`：标签熵；`eff = exp(H)`：有效类别数
- `H_norm = H / ln K`：归一化熵

并给出风险等级与建议口径：

| 等级 | 判定 | 建议 |
|---|---|---|
| HIGH | p_max ≥ 0.40 且 eff < 8 | 平凡类主导；top1 不可作门禁，改 cond / macro |
| MED | p_max ≥ 0.25 或 eff < K/3 | 中度集中；top1 与 macro 双报 |
| LOW | 其余 | top1 可用 |

## 覆盖的任务

| 任务 | K | 来源 | 说明 |
|---|---:|---|---|
| `shape27` | 27 | alloc + 21 中心 | 局部计数形状（含 EMPTY）|
| `shape10` | 10 | semantic_label | 语义结构角色 |
| `count5` | 5 | alloc 摊平 (n×3×34) | 单牌计数 0..4 |
| `presence2` | 2 | alloc>0 摊平 | 单牌在不在（= Phase 3 tile_presence）|
| `tenpai2` | 2 | cache['tenpai'] | 听牌 |
| `shanten8` | 8 | shanten_fast | 向听 0..7 |
| `riichi2` | 2 | cache['riichi'] | 立直 |
| `called2` | 2 | cache['called'] | 副露 |
| `wait36` | 36 | shanten_fast.waits_vec | **多标签**：34 待牌 + 无待牌 |

分层：`all` / `riichi` / `clean_t8`；局部形状额外报 `all_centers / active_any / active_both`。

用法：
    python tools/read_model/label_diag.py \\
        --cache /tmp/read_model_diag_cache_a2c/cache.npz \\
        --out tools/read_model/diagnostics/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE / 'diagnostics'
LS_DIR = HERE / 'local_shape'
sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(LS_DIR))
sys.path.insert(0, str(HERE))

import local_shape as LS  # noqa: E402
import semantic_label as SL  # noqa: E402
import shanten_fast as SF  # noqa: E402

LAYERS = ('all', 'riichi', 'clean_t8')
ACTIVES = ('all_centers', 'active_any', 'active_both')

#: 各任务当前实际使用的指标（人工核对自代码）与是否会被 top1 稀释
METRIC_IN_USE = {
    'shape27': ('top1 + CE', True,
                'P0/P1/P2a/P2b/P2c 门禁用 top1 ⇒ 已确认被 EMPTY 稀释'),
    'shape10': ('top1 + CE', True,
                '语义标签验证用 top1 ⇒ 同样受影响'),
    'count5': ('CE + MAE', False, 'A2c alloc 指标不用 top1 ⇒ 未受影响'),
    'presence2': ('AUC + BCE', False, 'Phase 3 用 AUC/BCE ⇒ 未受影响'),
    'tenpai2': ('AUC', False, 'bayes_bound.tenpai_auc ⇒ 未受影响'),
    'shanten8': ('accuracy（辅助）', True,
                 'shanten_acc 会被 28.6% 多数类抬高，影响有限'),
    'riichi2': ('BCE / AUC', False, '辅助头 ⇒ 未受影响'),
    'called2': ('BCE / AUC', False, '辅助头 ⇒ 未受影响'),
    'wait36': ('hit@1 / hit@5', False, '多标签 ⇒ top1 不适用'),
}


# --------------------------------------------------------------------------- #
def load_cache(path: Path, n: int):
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in ('alloc', 'unseen', 'hand', 'turn', 'riichi', 'called', 'hc',
              'tg', 'tenpai'):
        if k in z.files:
            out[k] = np.asarray(z[k])[:n] if n else np.asarray(z[k])
    out['alloc'] = out['alloc'].astype(np.int64)
    out['hand'] = out['hand'].astype(np.int64)
    out['turn'] = out['turn'].astype(np.int64)
    out['riichi'] = out['riichi'].astype(bool)
    out['called'] = out['called'].astype(bool)
    out['tenpai'] = out['tenpai'].astype(bool)
    out['n'] = out['alloc'].shape[0]
    return out


def masks_of(z):
    n = z['n']
    t8 = np.broadcast_to((z['turn'] >= 8)[:, None], (n, 3))
    return {
        'all': np.ones((n, 3), bool),
        'riichi': z['riichi'],
        'clean_t8': (~z['riichi']) & (~z['called']) & t8,
    }


def active_masks(z, centers):
    hc = z['hc']
    any_ = np.stack([(hc[:, :, a] < 99) | (hc[:, :, c] < 99)
                     for a, _b, c in centers], -1)
    both = np.stack([(hc[:, :, a] < 99) & (hc[:, :, c] < 99)
                     for a, _b, c in centers], -1)
    return any_, both


# --------------------------------------------------------------------------- #
def structure_of(y: np.ndarray, mask: np.ndarray, K: int):
    """y 与 mask 同形（任意维度）；返回标签结构指标。"""
    yy = np.asarray(y)[np.asarray(mask, bool)]
    if yy.size == 0:
        return None
    cnt = np.bincount(yy.astype(np.int64).ravel(), minlength=K).astype(np.float64)
    tot = cnt.sum()
    if tot <= 0:
        return None
    p = cnt / tot
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    eff = float(np.exp(H))
    return {
        'n': int(tot), 'K': int(K),
        'p_max': float(p.max()),
        'p_max_class': int(cnt.argmax()),
        'H': H, 'eff': eff,
        'H_norm': float(H / np.log(K)) if K > 1 else 0.0,
        'n_classes_used': int((cnt > 0).sum()),
    }


def risk_of(r):
    if r is None:
        return 'NA', '—'
    if r['p_max'] >= 0.999:
        return 'DEGENERATE', '标签恒定，该层无信息，不应参与评估'
    if r['p_max'] >= 0.40 and r['eff'] < 8:
        return 'HIGH', 'cond / macro；top1 不可作门禁'
    if r['p_max'] >= 0.25 or r['eff'] < r['K'] / 3.0:
        return 'MED', 'top1 + macro 双报'
    return 'LOW', 'top1 可用'


# --------------------------------------------------------------------------- #
def build_labels(z):
    """返回 {任务: (y, K, axes, extra)}；axes 描述 y 的维度含义。"""
    n = z['n']
    centers = LS.centers_3()
    out = {}

    # 局部形状：21 中心各一个 27 类
    y27 = np.stack([LS.true_pattern(z['alloc'], c) for c in centers], -1)  # [n,3,21]
    out['shape27'] = (y27, 27, 'target_center', None)
    abc = np.clip(z['alloc'][:, :, centers], 0, 2)                         # [n,3,21,3]
    y10 = np.stack([SL.semantic_index(abc[:, :, ci], centers[ci])
                    for ci in range(len(centers))], -1)                    # [n,3,21]
    out['shape10'] = (y10, SL.NUM_SEMANTIC, 'target_center', None)

    # 单牌：摊平到 [n,3,34]
    out['count5'] = (np.clip(z['alloc'], 0, 4), 5, 'target_tile', None)
    out['presence2'] = ((z['alloc'] > 0).astype(np.int64), 2, 'target_tile', None)

    # 目标级
    out['tenpai2'] = (z['tenpai'].astype(np.int64), 2, 'target', None)
    out['riichi2'] = (z['riichi'].astype(np.int64), 2, 'target', None)
    out['called2'] = (z['called'].astype(np.int64), 2, 'target', None)

    # 向听：从 alloc 算
    ld = (z['hand'] // 3).astype(np.int64)                                 # [n,3]
    flat = z['alloc'].reshape(-1, 34).astype(np.int16)
    ld_flat = ld.reshape(-1)
    ok = ld_flat >= 1
    sh = np.zeros(len(flat), np.int64)
    if ok.any():
        sh[ok] = SF.calc_all_vec(flat[ok], ld_flat[ok].astype(int))
    ysh = np.clip(sh.reshape(n, 3), 0, 7)
    out['shanten8'] = (ysh, 8, 'target', None)

    return out


def build_wait(z, batch=20000):
    """待牌是多标签：返回 (多热 [n,3,35], 有效 mask [n,3])。"""
    n = z['n']
    ld = (z['hand'] // 3).astype(np.int64)
    flat = z['alloc'].reshape(-1, 34).astype(np.int16)
    ld_flat = ld.reshape(-1)
    multi = np.zeros((len(flat), 35), bool)
    for lv in range(1, 5):                       # waits_vec 的 len_div3 是标量
        idx_lv = np.flatnonzero(ld_flat == lv)
        for lo in range(0, len(idx_lv), batch):
            sel = idx_lv[lo:lo + batch]
            if len(sel) == 0:
                continue
            waits, ten = SF.waits_vec(flat[sel], int(lv))
            multi[sel, :34] = waits
            multi[sel, 34] = ~ten
    return multi.reshape(n, 3, 35), (ld >= 1)


# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '—'


def write_report(path, res):
    L = ['# 在读任务标签诊断：最大类占比与有效类数\n']
    m = res['meta']
    L.append(f"- 缓存：`{m['cache']}`（n={m['n']}）")
    L.append(f"- 判定：`HIGH` = p_max≥0.40 且 eff<8；`MED` = p_max≥0.25 或 eff<K/3；"
             f"`LOW` = 其余。\n")
    L.append('口径说明：`p_max` = 最大类占比 = 常数预测器的 top1；'
             '`eff` = exp(熵) = 有效类别数；`H_norm` = 熵 / ln K。\n')

    L.append('## 1. 任务级标签结构（all 层，单标签任务）\n')
    L.append('| 任务 | K | n | p_max | 最大类 | 熵 | eff | H_norm | 风险 | 当前指标 | 被 top1 坑 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|')
    for t, r in res['tasks'].items():
        if r.get('multi'):
            continue
        s = r['all']
        if not s:
            continue
        hit = ('**是**' if (r.get('hit_by_top1') and r['risk'] == 'HIGH')
               else ('需双报' if r.get('hit_by_top1') else '否'))
        L.append(f"| {t} | {s['K']} | {s['n']} | {_f(s['p_max'])} | "
                 f"{s['p_max_class']} | {_f(s['H'])} | {_f(s['eff'], 1)} | "
                 f"{_f(s['H_norm'], 3)} | **{r['risk']}** | "
                 f"{r.get('metric_in_use', '—')} | {hit} |")

    L.append('\n## 2. 关键任务的分层\n')
    for t in ('shape27', 'shape10', 'count5', 'presence2', 'tenpai2', 'shanten8'):
        r = res['tasks'].get(t)
        if not r or r.get('multi'):
            continue
        L.append(f"\n**{t}**（{r['risk']}）\n")
        L.append('| 层 / 子集 | n | p_max | 熵 | eff | H_norm |')
        L.append('|---|---:|---:|---:|---:|---:|')
        for key, s in r['slices'].items():
            if not s:
                continue
            L.append(f"| {key} | {s['n']} | {_f(s['p_max'])} | {_f(s['H'])} | "
                     f"{_f(s['eff'], 1)} | {_f(s['H_norm'], 3)} |")

    L.append('\n## 3. 多标签任务（待牌）\n')
    w = res['tasks'].get('wait36')
    if w:
        for key, s in w['slices'].items():
            if not s:
                continue
            L.append(f"- **{key}**：n={s['n']}，平均待牌数 "
                     f"{_f(s['mean_pos'], 2)}，单类最高正例率 "
                     f"{_f(s['p_max'], 3)}（类 {s['p_max_class']}），"
                     f"无待牌率 {_f(s['p_none'], 3)}")
        L.append('\n> 待牌是**多标签**任务，门禁用 hit@1/hit@5 而非 top1；'
                 '本节只报基础统计，不作口径判定。\n')

    L.append('\n## 4. 结论\n')
    for line in res['verdict']['findings']:
        L.append(f'- {line}')
    L.append(f"\n**判定：{res['verdict']['decision']}**\n")
    path.write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--out', default=str(HERE / 'diagnostics' / 'results'))
    args = ap.parse_args()

    t0 = time.time()
    z = load_cache(Path(args.cache), args.n)
    print(f'[data] n={z["n"]}', flush=True)
    centers = LS.centers_3()
    any_, both = active_masks(z, centers)
    m = masks_of(z)
    labels = build_labels(z)
    print(f'[label] {list(labels)}  {time.time()-t0:.0f}s', flush=True)

    tasks = {}
    for name, (y, K, axes, _x) in labels.items():
        entry = {'K': K, 'axes': axes, 'slices': {}}
        if axes == 'target_center':
            for layer in LAYERS:
                for act in ACTIVES:
                    am = {'all_centers': np.ones_like(any_, bool),
                          'active_any': any_, 'active_both': both}[act]
                    mk = m[layer][..., None] & am                    # [n,3,21]
                    entry['slices'][f'{layer}/{act}'] = structure_of(y, mk, K)
        elif axes == 'target':
            for layer in LAYERS:
                entry['slices'][layer] = structure_of(y, m[layer], K)
        else:                                                     # target_tile
            for layer in LAYERS:
                mk = np.broadcast_to(m[layer][..., None], y.shape)
                entry['slices'][layer] = structure_of(y, mk, K)
        entry['all'] = (entry['slices'].get('all/active_any')
                        or entry['slices'].get('all')
                        or entry['slices'].get('all/all_centers'))
        entry['risk'], entry['suggest'] = risk_of(entry['all'])
        mi = METRIC_IN_USE.get(name, ('未核实', True, '需人工确认'))
        entry['metric_in_use'], entry['hit_by_top1'], entry['metric_note'] = mi
        tasks[name] = entry
        print(f'  [{name}] p_max={entry["all"]["p_max"]:.4f} '
              f'eff={entry["all"]["eff"]:.2f} risk={entry["risk"]}', flush=True)

    # 待牌（多标签）
    multi, wmask = build_wait(z)
    print(f'[label] wait  {time.time()-t0:.0f}s', flush=True)
    wslices = {}
    for layer in LAYERS:
        mk = m[layer] & wmask
        if not mk.any():
            continue
        mm = multi[mk]
        cnt = mm.sum(0).astype(np.float64)
        n = int(mm.shape[0])
        pos = mm.sum(1)
        wslices[layer] = {
            'n': n,
            'mean_pos': float(pos.mean()),
            'p_max': float(cnt.max() / n),
            'p_max_class': int(cnt.argmax()),
            'p_none': float(cnt[34] / n),
        }
    tasks['wait36'] = {'K': 35, 'axes': 'multi', 'multi': True,
                       'slices': wslices}
    mi = METRIC_IN_USE['wait36']
    tasks['wait36'].update({'metric_in_use': mi[0], 'hit_by_top1': mi[1],
                            'metric_note': mi[2], 'risk': 'NA'})

    # 结论：区分「标签集中」与「实际被 top1 坑」
    highs = [k for k, v in tasks.items() if v.get('risk') == 'HIGH']
    meds = [k for k, v in tasks.items() if v.get('risk') == 'MED']
    lows = [k for k, v in tasks.items() if v.get('risk') == 'LOW']
    hurt = [k for k in highs + meds
            if tasks[k].get('hit_by_top1') and tasks[k].get('risk') in ('HIGH', 'MED')]
    safe = [k for k in highs + meds if not tasks[k].get('hit_by_top1')]

    # 退化标签：某层上 p_max=1（恒为常数）——该层无信息
    degen = []
    for t, v in tasks.items():
        for key, s in (v.get('slices') or {}).items():
            if s and s.get('p_max', 0) >= 0.999:
                degen.append(f'{t}@{key}')

    findings = [
        f"标签集中的任务（HIGH）：{', '.join(highs) or '无'}；"
        f"MED：{', '.join(meds) or '无'}；LOW：{', '.join(lows) or '无'}；"
        f"退化（p_max=1）：{', '.join(degen) or '无'}。",
        f"**真正被 top1 坑的（集中且现用 top1/accuracy）：{', '.join(hurt) or '无'}**。"
        f"其余 {', '.join(safe) or '无'} 虽集中但用 AUC/CE/MAE/hit@k，**未受影响**。",
    ]
    if degen:
        findings.append(
            f"退化项 {', '.join(degen)}：标签恒定（立直必听牌、向听恒 0）"
            f"⇒ 这些子集上的 tenpai/shanten 指标恒为 100%，**不应参与任何比较**。")
    for t in highs:
        s = tasks[t]['all']
        tag = '⚠ 已受影响' if tasks[t].get('hit_by_top1') else '指标安全'
        findings.append(
            f"`{t}`：K={s['K']}、p_max={s['p_max']:.4f}（类 {s['p_max_class']}）、"
            f"eff={s['eff']:.1f}；当前指标 {tasks[t]['metric_note']} ⇒ **{tag}**。")
    for t in meds:
        s = tasks[t]['all']
        tag = '⚠ 需双报' if tasks[t].get('hit_by_top1') else '指标安全'
        findings.append(
            f"`{t}`：p_max={s['p_max']:.4f}、eff={s['eff']:.1f} ⇒ **{tag}**。")

    if hurt:
        decision = (f'标签集中且**现用** top1/accuracy 的任务：'
                    f'{", ".join(hurt)} ⇒ 这些门禁必须改 cond/macro 或补「非平凡类条件」；'
                    f'其余 {len(safe)} 个高集中任务已用 AUC/CE/MAE/hit@k，无需改口径。')
    else:
        decision = ('未发现「标签集中 + 现用 top1」的组合 ⇒ 无需改口径')
    if degen:
        decision += (f' 另：{len(degen)} 个退化项（{", ".join(degen)}）'
                     f'因标签恒定，已无信息，应直接从评估中移除。')

    res = {'meta': {'cache': args.cache, 'n': int(z['n']),
                    'centers': len(centers),
                    'seconds': float(time.time() - t0)},
           'tasks': tasks,
           'verdict': {'decision': decision, 'findings': findings}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'label_diag.json', 'w') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'LABEL_DIAG_REPORT.md', res)
    print(f'\n[out] {out}/label_diag.json')
    print(f'[out] {out}/LABEL_DIAG_REPORT.md')
    print(f'[verdict] {decision}')
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
