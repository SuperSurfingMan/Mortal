#!/usr/bin/env python3
"""P2b：`behave` 参考蒸馏 + 局部形状头（冻结 A2c）。

在 P2a 的 762 维输入和 P2Head 上，把 train 立直目标的**硬标签**按权重替换为
`behave` 参考软标签（`p2b_distill_data.py` 生成），训练时用软 CE：

    soft = (1-w)·onehot(true counts) + w·behave_ref
    L = −Σ soft·log_softmax(head(x))

其余（train/val/eval cache、门禁、分层）与 P2a 完全一致。

用法：
  python tools/read_model/local_shape/p2b_distill_head.py \
      --cache-dir tools/read_model/local_shape/.cache_p2 \
      --distill tools/read_model/local_shape/.cache_p2/distill_train.npz \
      --out tools/read_model/local_shape/results/p2b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p1_local_head as P1  # noqa: E402
import p2_riichi_head as P2  # noqa: E402

TARGETS = P1.TARGETS
N_CENTERS = P1.N_CENTERS
NUM_CLASSES = P1.NUM_CLASSES
DEV = P1.DEV


def _onehot_soft(pat: np.ndarray) -> np.ndarray:
    """pat [n,3,21] → soft [n,3,21,27] one-hot。"""
    n = pat.shape[0]
    soft = np.zeros((n, TARGETS, N_CENTERS, NUM_CLASSES), np.float16)
    ar = np.arange(n)
    for j in range(TARGETS):
        for ci in range(N_CENTERS):
            soft[ar, j, ci, pat[:, j, ci]] = 1.0
    return soft


class SoftDS(Dataset):
    def __init__(self, f: P2.P2Feat, soft: np.ndarray,
                 active_w=(1.0, 2.0)):
        self.f = f
        self.soft = soft
        self.active_w = active_w

    def __len__(self):
        return self.f.n * N_CENTERS

    def __getitem__(self, idx):
        i, ci = divmod(int(idx), N_CENTERS)
        x = np.stack([self.f.x(i, ci, j) for j in range(TARGETS)], 0)
        y = self.soft[i, :, ci, :].astype(np.float32)
        w = (1.0 + self.active_w[0] * self.f.any_[i, :, ci]
             + self.active_w[1] * self.f.both[i, :, ci]).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(w), torch.tensor(int(idx), dtype=torch.int64))


def soft_loader(f, soft, batch, shuffle, workers):
    ds = SoftDS(f, soft)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=(DEV == 'cuda'),
                      persistent_workers=(workers > 0), drop_last=False)


def _fmt(x, nd=4):
    if x is None:
        return '—'
    try:
        return f'{x:.{nd}f}' if np.isfinite(x) else '—'
    except TypeError:
        return '—'


def write_report(path: Path, res: dict) -> None:
    m = res['meta']
    lines = ['# P2b：`behave` 参考蒸馏 + 局部形状头（冻结 A2c）\n']
    lines += [
        f"- 参考数据：{m['distill_targets']} 个立直目标 / {m['distill_snapshots']} 快照，"
        f"每条 {m['n_ref']} 手；beta={m['beta']}；蒸馏权重 w={m['distill_weight']}",
        f"- 训练：{m['train_n']}（202608）；验证 {m['val_n']}；评估 {m['eval_n']}（202609）",
        f"- 输入 {m['d_in']} 维；头 {m['d_in']}→{m['hidden']}→{m['hidden']}→27；"
        f"dropout {m['dropout']}；设备 {m['device']}",
        f"- best epoch：{m['best_epoch']}\n",
    ]
    lines.append('## 训练曲线\n')
    lines.append('| epoch | train loss | val riichi any top1 | val riichi both top1 | val riichi any CE |')
    lines.append('|---:|---:|---:|---:|---:|')
    for r in res['curve']:
        v = r['val']['riichi']
        lines.append(f"| {r['epoch']} | {_fmt(r['train_loss'])} | "
                     f"{_fmt(v['active_any']['top1'])} | {_fmt(v['active_both']['top1'])} | "
                     f"{_fmt(v['active_any']['ce'])} |")
    lines.append('\n## 评估（202609）\n')
    lines.append('| 层 | 子集 | head top1 | P2a head | P1 head | prior | A2c product | head top3 | A2c top3 | head CE | prior CE | A2c CE |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for layer in ('riichi', 'all', 'nonriichi', 'turn>=8', 'clean_t8'):
        for an in ('all_centers', 'active_any', 'active_both'):
            t = res['eval'][layer][an]
            lines.append(
                f"| {layer} | {an} | {_fmt(t['head']['top1'])} | "
                f"{_fmt(t.get('p2a_head', {}).get('top1'))} | "
                f"{_fmt(t.get('p1_head', {}).get('top1'))} | "
                f"{_fmt(t['prior_train']['top1'])} | {_fmt(t['a2c']['top1'])} | "
                f"{_fmt(t['head']['top3'])} | {_fmt(t['a2c']['top3'])} | "
                f"{_fmt(t['head']['ce'])} | {_fmt(t['prior_train']['ce'])} | "
                f"{_fmt(t['a2c']['ce'])} |")
    lines.append('\n## 判定\n')
    for line in res['verdict']['findings']:
        lines.append(f'- {line}')
    lines.append(f"\n**判定：{res['verdict']['decision']}**\n")
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default=str(HERE / '.cache_p2'))
    ap.add_argument('--distill', default=str(HERE / '.cache_p2' / 'distill_train.npz'))
    ap.add_argument('--out', default=str(HERE / 'results' / 'p2b'))
    ap.add_argument('--p1-json', default=str(HERE / 'results' / 'p1' / 'p1_local_head.json'))
    ap.add_argument('--p2a-json', default=str(HERE / 'results' / 'p2' / 'p2_riichi_head.json'))
    ap.add_argument('--distill-weight', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--active-w-any', type=float, default=1.0)
    ap.add_argument('--active-w-both', type=float, default=2.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = Path(args.cache_dir)
    tr = P2.P2Feat(P1._load(d / 'train.npz'))
    va = P2.P2Feat(P1._load(d / 'val.npz'))
    ev = P2.P2Feat(P1._load(d / 'eval.npz'))
    assert tr.d_in == va.d_in == ev.d_in, (tr.d_in, va.d_in, ev.d_in)

    dz = np.load(args.distill, allow_pickle=True)
    didx, dj, dsoft = dz['idx'], dz['j'], dz['soft'].astype(np.float32)
    n_ref = int(dz['n_ref'].min()) if 'n_ref' in dz.files else -1
    meta_d = {}
    mp = Path(args.distill).with_suffix('.meta.json')
    if mp.exists():
        meta_d = json.loads(mp.read_text())

    soft = _onehot_soft(tr.pat)
    n_rep = 0
    for k in range(len(didx)):
        i, j = int(didx[k]), int(dj[k])
        ref = dsoft[k]
        ref = ref / np.clip(ref.sum(-1, keepdims=True), 1e-9, None)
        soft[i, j] = ((1.0 - args.distill_weight) * soft[i, j]
                      + args.distill_weight * ref).astype(np.float16)
        n_rep += 1
    print(f'[distill] targets={n_rep} weight={args.distill_weight} '
          f'n_ref≈{n_ref} cache={args.distill}', flush=True)
    print(f'[data] train={tr.n} val={va.n} eval={ev.n} d_in={tr.d_in} '
          f'device={DEV}', flush=True)

    head = P2.P2Head(tr.d_in, args.hidden, args.dropout).to(DEV)
    n_params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    tr_dl = soft_loader(tr, soft, args.batch, True, args.workers)
    masks_va, masks_ev = P1._masks(va), P1._masks(ev)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = 0
    bad = 0
    curve = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        tot = n = 0
        for x, y, w, _idx in tr_dl:
            x = x.float().to(DEV)
            y = y.to(DEV)
            w = w.to(DEV)
            logits = head(x)
            logp = F.log_softmax(logits, -1)
            loss = (-(y * logp).sum(-1) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        sched.step()
        v1, v3, vc = P1.predict(head, va, args.batch, args.workers)
        vm = P1._aggregate(v1, v3, vc, va, masks_va)
        score = vm['riichi']['active_any']['top1']
        curve.append({'epoch': epoch, 'train_loss': tot / max(n, 1), 'val': vm})
        print(f"[epoch {epoch:02d}] loss={tot/max(n,1):.4f} "
              f"val_r_any={vm['riichi']['active_any']['top1']:.4f} "
              f"val_r_both={vm['riichi']['active_both']['top1']:.4f} "
              f"ce={vm['riichi']['active_any']['ce']:.4f}  {time.time()-t0:.0f}s",
              flush=True)
        if score > best + 1e-6:
            best, best_epoch, bad = score, epoch, 0
            torch.save({'state_dict': head.state_dict(),
                        'config': {'d_in': tr.d_in, 'hidden': args.hidden,
                                   'dropout': args.dropout,
                                   'distill_weight': args.distill_weight,
                                   'n_targets': TARGETS, 'n_classes': NUM_CLASSES}},
                       out / 'p2b_head.pth')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'[early] stop at epoch {epoch} (best {best_epoch})', flush=True)
                break

    state = torch.load(out / 'p2b_head.pth', map_location=DEV, weights_only=False)
    head.load_state_dict(state['state_dict'])
    head.eval()

    h1, h3, hc = P1.predict(head, ev, args.batch, args.workers)
    a1, a3, ac = P1._a2c_product(ev)
    prior_train = P1._prior_from(tr)
    prior_eval = P1._prior_from(ev)
    p1_, p3_, pc_ = P1._prior_arrays(prior_train, ev)
    q1_, q3_, qc_ = P1._prior_arrays(prior_eval, ev)
    mh = P1._aggregate(h1, h3, hc, ev, masks_ev)
    ma = P1._aggregate(a1, a3, ac, ev, masks_ev)
    mp = P1._aggregate(p1_, p3_, pc_, ev, masks_ev)
    mq = P1._aggregate(q1_, q3_, qc_, ev, masks_ev)

    def _load_eval(path):
        p = Path(path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())['eval']
        except Exception:
            return {}

    p1_eval = _load_eval(args.p1_json)
    p2a_eval = _load_eval(args.p2a_json)
    table = {}
    for layer in mh:
        table[layer] = {}
        for an in mh[layer]:
            row = {'head': mh[layer][an], 'a2c': ma[layer][an],
                   'prior_train': mp[layer][an], 'prior_eval': mq[layer][an]}
            if layer in p1_eval and an in p1_eval[layer]:
                row['p1_head'] = p1_eval[layer][an].get('head', {})
            if layer in p2a_eval and an in p2a_eval[layer]:
                row['p2a_head'] = p2a_eval[layer][an].get('head', {})
            table[layer][an] = row

    rb = table['riichi']['active_both']
    ra = table['riichi']['active_any']
    aa = table['all']['active_any']
    d_both = rb['head']['top1'] - rb['prior_train']['top1']
    ce_both = rb['prior_train']['ce'] - rb['head']['ce']
    gate_riichi = (d_both >= 0.03) or (ce_both >= 0.10)
    beat_a2c = ((rb['head']['top1'] > rb['a2c']['top1']) or
                (rb['head']['ce'] < rb['a2c']['ce']))
    p1_any_top1 = p1_eval.get('all', {}).get('active_any', {}).get('head', {}).get('top1')
    p1_any_ce = p1_eval.get('all', {}).get('active_any', {}).get('head', {}).get('ce')
    if p1_any_top1 is None:
        guard_any = True
    else:
        guard_any = ((aa['head']['top1'] >= p1_any_top1 - 0.005) or
                     (aa['head']['ce'] <= p1_any_ce + 0.05))
    gate = bool(gate_riichi and beat_a2c and guard_any)

    findings = [
        f"riichi active_both：head top1 {rb['head']['top1']:.3f} vs prior "
        f"{rb['prior_train']['top1']:.3f}（Δ {d_both:+.3f}），A2c {rb['a2c']['top1']:.3f}；"
        f"P2a {rb.get('p2a_head', {}).get('top1', float('nan')):.3f}；"
        f"head CE {rb['head']['ce']:.3f} vs prior {rb['prior_train']['ce']:.3f} / "
        f"A2c {rb['a2c']['ce']:.3f}。",
        f"riichi active_any：head {ra['head']['top1']:.3f} vs prior "
        f"{ra['prior_train']['top1']:.3f} vs A2c {ra['a2c']['top1']:.3f}。",
        f"all active_any guard：head {aa['head']['top1']:.3f} / CE {aa['head']['ce']:.3f}"
        f"（P1 head {p1_any_top1 if p1_any_top1 is not None else '—'}）⇒ {guard_any}。",
        f"门禁：riichi_both +3pp/CE−0.10 = {gate_riichi}；beat A2c = {beat_a2c}；"
        f"guard = {guard_any}。",
    ]
    verdict = {
        'decision': ('PASS → P2c 受控解冻（top1 未超 A2c，暂不 200k）' if gate
                     else 'FAIL → 关闭局部形状线（P1/P2a/P2b 三次未过门禁）'),
        'gate': gate,
        'gate_parts': {'riichi_both': bool(gate_riichi),
                       'beat_a2c': bool(beat_a2c),
                       'guard_any': bool(guard_any)},
        'findings': findings,
    }
    meta = {
        'train_n': tr.n, 'val_n': va.n, 'eval_n': ev.n,
        'd_in': tr.d_in, 'hidden': args.hidden, 'dropout': args.dropout,
        'params': int(n_params), 'device': DEV, 'seed': args.seed,
        'best_epoch': best_epoch, 'cache_dir': str(d),
        'distill_targets': int(len(didx)),
        'distill_snapshots': int(meta_d.get('n_snapshots', -1)),
        'n_ref': int(meta_d.get('n_ref', n_ref)),
        'beta': float(meta_d.get('beta', -1)),
        'distill_weight': args.distill_weight,
    }
    res = {'meta': meta, 'curve': curve, 'eval': table, 'verdict': verdict}
    with open(out / 'p2b_distill.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'P2B_REPORT.md', res)
    print(f'\n[out] {out}/p2b_head.pth')
    print(f'[out] {out}/p2b_distill.json')
    print(f'[out] {out}/P2B_REPORT.md')
    print(f"[verdict] {verdict['decision']}")
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
