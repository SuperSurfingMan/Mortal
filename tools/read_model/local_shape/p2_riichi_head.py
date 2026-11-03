#!/usr/bin/env python3
"""P2：立直显式特征 + 局部形状头（冻结 A2c 骨干）。

P1 冻结头 FAIL 后，P2 只加**新信息**：立直宣言牌、宣言巡目、宣言摸切、立直后牌河。
靶子是 P1 最差的 `riichi active_both`（head 0.6412 vs prior 0.6547 vs A2c 0.6686）。

输入 = P1 的 712 维局部特征 + 50 维立直特征：
  decl_onehot(34) + is_decl(1) + dist_to_decl(1) + post_count(1)
  + turns_since(1) + decl_turn(1) + decl_tg(1) + decl_in_window(5) + post_in_window(5)

门禁（预注册）：
  1. riichi active_both top1 比 prior(train) 高 ≥3pp，或 CE 低 ≥0.10；
  2. 且优于 A2c product（top1 或 CE）；
  3. all active_any 不低于 P1 head −0.5pp（防灾难性回归）。
不满足 → 停止；满足 → 再考虑 P2b 解冻/200k。

用法：
  python tools/read_model/local_shape/p2_riichi_head.py \
      --cache-dir tools/read_model/local_shape/.cache_p2 \
      --out tools/read_model/local_shape/results/p2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import local_shape as LS  # noqa: E402
import p1_local_head as P1  # noqa: E402

TARGETS = P1.TARGETS
N_CENTERS = P1.N_CENTERS
NUM_CLASSES = P1.NUM_CLASSES
DEV = P1.DEV


class P2Feat(P1.Feat):
    """P1 特征 + 立直特征。"""

    def __init__(self, cache: dict):
        super().__init__(cache)
        self.post = np.asarray(cache['post_riichi_river']).astype(bool)
        self.decl_tile = np.asarray(cache['riichi_decl_tile']).astype(np.int16)
        self.decl_turn = np.asarray(cache['riichi_decl_turn']).astype(np.int16)
        self.decl_tg = np.asarray(cache['riichi_decl_tg']).astype(np.int8)
        self.decl_onehot = np.zeros((self.n, TARGETS, 34), np.float16)
        for j in range(TARGETS):
            ok = self.decl_tile[:, j] >= 0
            self.decl_onehot[ok, j, self.decl_tile[ok, j]] = 1.0
        self.base_d_in = self.d_in
        self.d_in = self.base_d_in + 50

    def x(self, i: int, ci: int, j: int) -> np.ndarray:
        base = super().x(i, ci, j)
        b = int(self.centers[ci][1])
        suit = b // 9
        decl = int(self.decl_tile[i, j])
        if decl >= 0 and decl // 9 == suit:
            is_decl = 1.0 if decl == b else 0.0
            dist = abs(b - decl) / 9.0
        else:
            is_decl = 0.0
            dist = 1.0
        decl_in = np.zeros(5, np.float16)
        post_in = np.zeros(5, np.float16)
        for k, d in enumerate(range(-2, 3)):
            t = b + d
            if 0 <= t < 27 and t // 9 == suit:
                if decl >= 0 and t == decl:
                    decl_in[k] = 1.0
                post_in[k] = 1.0 if self.post[i, j, t] else 0.0
        post_count = min(float(self.post[i, j].sum()), 13.0) / 13.0
        turns_since = (max(int(self.turn[i]) - int(self.decl_turn[i, j]), 0) / 18.0
                       if self.decl_turn[i, j] > 0 else 0.0)
        decl_turn_n = float(self.decl_turn[i, j]) / 18.0
        decl_tg = float(self.decl_tg[i, j])
        riichi = np.concatenate([
            self.decl_onehot[i, j],
            np.array([is_decl, dist, post_count, turns_since,
                      decl_turn_n, decl_tg], np.float16),
            decl_in, post_in,
        ]).astype(np.float16)
        return np.concatenate([base, riichi]).astype(np.float16)


class P2Head(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = P1.GroupedLinear(TARGETS, d_in, hidden)
        self.fc2 = P1.GroupedLinear(TARGETS, hidden, hidden)
        self.out = P1.GroupedLinear(TARGETS, hidden, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = F.gelu(self.fc2(x))
        x = self.drop(x)
        return self.out(x)


def _fmt(x, nd=4):
    return f'{x:.{nd}f}' if np.isfinite(x) else '—'


def write_report(path: Path, res: dict) -> None:
    m = res['meta']
    lines = ['# P2：立直显式特征 + 局部形状头（冻结 A2c）\n']
    lines += [f"- 训练：{m['train_n']}（202608，按文件切分）；验证 {m['val_n']}；评估 {m['eval_n']}（202609）",
              f"- 输入：P1 {m['base_d_in']} 维 + 立直 50 维 = **{m['d_in']}**；"
              f"头 {m['d_in']}→{m['hidden']}→{m['hidden']}→27，dropout {m['dropout']}",
              f"- 设备：{m['device']}；seed：{m['seed']}；best epoch：{m['best_epoch']}\n"]
    lines.append('## 训练曲线\n')
    lines.append('| epoch | train loss | val riichi any top1 | val riichi both top1 | val riichi any CE |')
    lines.append('|---:|---:|---:|---:|---:|')
    for r in res['curve']:
        v = r['val']['riichi']
        lines.append(f"| {r['epoch']} | {_fmt(r['train_loss'])} | "
                     f"{_fmt(v['active_any']['top1'])} | {_fmt(v['active_both']['top1'])} | "
                     f"{_fmt(v['active_any']['ce'])} |")
    lines.append('\n## 评估（202609）\n')
    lines.append('| 层 | 子集 | head top1 | P1 head | prior(train) | A2c product | head top3 | A2c top3 | head CE | prior(train) CE | A2c CE |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for layer in ('riichi', 'all', 'nonriichi', 'turn>=8', 'clean_t8'):
        for an in ('all_centers', 'active_any', 'active_both'):
            t = res['eval'][layer][an]
            p1v = t.get('p1_head', {}).get('top1')
            lines.append(
                f"| {layer} | {an} | {_fmt(t['head']['top1'])} | "
                f"{_fmt(p1v) if p1v is not None else '—'} | "
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
    ap.add_argument('--out', default=str(HERE / 'results' / 'p2'))
    ap.add_argument('--p1-json', default=str(HERE / 'results' / 'p1' / 'p1_local_head.json'))
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--patience', type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = Path(args.cache_dir)
    tr = P2Feat(P1._load(d / 'train.npz'))
    va = P2Feat(P1._load(d / 'val.npz'))
    ev = P2Feat(P1._load(d / 'eval.npz'))
    assert tr.d_in == va.d_in == ev.d_in, (tr.d_in, va.d_in, ev.d_in)
    print(f'[data] train={tr.n} val={va.n} eval={ev.n} d_in={tr.d_in} '
          f'(base {tr.base_d_in}+50) device={DEV}', flush=True)

    head = P2Head(tr.d_in, args.hidden, args.dropout).to(DEV)
    n_params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    tr_dl = P1._loader(tr, args.batch, True, args.workers)
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
            w = w.float().to(DEV)
            logits = head(x)
            loss = (F.cross_entropy(logits.reshape(-1, NUM_CLASSES),
                                    y.reshape(-1), reduction='none')
                    * w.reshape(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        sched.step()
        v1, v3, vc = P1.predict(head, va, args.batch, args.workers)
        vm = P1._aggregate(v1, v3, vc, va, masks_va)
        # 选型看 riichi active_any（样本多），active_both 只打印
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
                                   'n_targets': TARGETS, 'n_classes': NUM_CLASSES}},
                       out / 'p2_head.pth')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'[early] stop at epoch {epoch} (best {best_epoch})', flush=True)
                break

    state = torch.load(out / 'p2_head.pth', map_location=DEV, weights_only=False)
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

    p1_json = Path(args.p1_json)
    p1_eval = {}
    if p1_json.exists():
        try:
            p1_eval = json.loads(p1_json.read_text())['eval']
        except Exception:
            p1_eval = {}

    table = {}
    for layer in mh:
        table[layer] = {}
        for an in mh[layer]:
            row = {
                'head': mh[layer][an],
                'a2c': ma[layer][an],
                'prior_train': mp[layer][an],
                'prior_eval': mq[layer][an],
            }
            if layer in p1_eval and an in p1_eval[layer]:
                row['p1_head'] = p1_eval[layer][an].get('head', {})
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
        'decision': ('PASS → 可考虑 P2b 蒸馏/解冻/200k' if gate
                     else 'FAIL → 停止 P2a；下一步 P2b behave 蒸馏（唯一新信息源）或关闭局部形状线'),
        'gate': gate,
        'gate_parts': {'riichi_both': bool(gate_riichi),
                       'beat_a2c': bool(beat_a2c),
                       'guard_any': bool(guard_any)},
        'findings': findings,
    }
    meta = {
        'train_n': tr.n, 'val_n': va.n, 'eval_n': ev.n,
        'base_d_in': tr.base_d_in, 'd_in': tr.d_in,
        'hidden': args.hidden, 'dropout': args.dropout,
        'params': int(n_params), 'device': DEV, 'seed': args.seed,
        'best_epoch': best_epoch, 'cache_dir': str(d),
    }
    res = {'meta': meta, 'curve': curve, 'eval': table, 'verdict': verdict}
    with open(out / 'p2_riichi_head.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'P2_REPORT.md', res)
    print(f'\n[out] {out}/p2_head.pth')
    print(f'[out] {out}/p2_riichi_head.json')
    print(f'[out] {out}/P2_REPORT.md')
    print(f"[verdict] {verdict['decision']}")
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
