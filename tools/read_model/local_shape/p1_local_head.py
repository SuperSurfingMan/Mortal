#!/usr/bin/env python3
"""P1：冻结 A2c 骨干，训练局部形状头并做探针判定。

输入：`features.py` 生成的 train/val/eval 缓存。
输出：`head.pth` + `p1_local_head.json` + `P1_REPORT.md`。

口径：
- 头输入（逐 target j、逐中心 m）：
  h 的 m−2..m+2 五列（5C）+ e_T_j + s_belief_j + g
  + target one-hot + turn/riichi/called
  + unseen 五列 + hc/tg/valid 五列（目标家邻张手切/摸切/时间）
- 标签：三张窗口 (m−1,m,m+1) 的计数截断 0/1/2 ⇒ 27 类。
- 训练：202608 按文件 80/20 切 train/val；评估：202609（与 P0 同源）。
- 冻结骨干，只训头；alloc/wait 不变。

用法：
  python tools/read_model/local_shape/p1_local_head.py \
      --cache-dir /tmp/read_model_local_shape_p1 \
      --out tools/read_model/local_shape/results/p1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import local_shape as LS  # noqa: E402

NUM_CLASSES = LS.NUM_PATTERNS
TARGETS = 3
N_CENTERS = 21
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class GroupedLinear(nn.Module):
    """与 model_v2.GroupedLinear 同语义的本地副本（避免 import 主模型同名 module）。"""

    def __init__(self, n_targets: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_targets, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_targets, d_out))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        bound = 1.0 / math.sqrt(d_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bjd,jde->bje', x, self.weight) + self.bias


class LocalShapeHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.fc1 = GroupedLinear(TARGETS, d_in, hidden)
        self.fc2 = GroupedLinear(TARGETS, hidden, hidden)
        self.out = GroupedLinear(TARGETS, hidden, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.out(x)


def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _centers():
    return LS.centers_3()


def _win5(x: np.ndarray, centers: np.ndarray, width: int) -> np.ndarray:
    """x[n,34] → [n,21,5*width]；同花色外补零（unseen 用）。"""
    n = x.shape[0]
    out = np.zeros((n, len(centers), 5 * width), np.float16)
    for ci, (_a, b, _c) in enumerate(centers):
        base = (b // 9) * 9
        for k, d in enumerate(range(-2, 3)):
            t = b + d
            if 0 <= t < 27 and t // 9 == base // 9:
                out[:, ci, k * width:(k + 1) * width] = x[:, t][:, None]
    return out


def _win5_h(x: np.ndarray, centers: np.ndarray, width: int) -> np.ndarray:
    """x[n,C,34] → [n,21,5*C]；同花色外补零。"""
    n = x.shape[0]
    out = np.zeros((n, len(centers), 5 * width), np.float16)
    for ci, (_a, b, _c) in enumerate(centers):
        base = (b // 9) * 9
        for k, d in enumerate(range(-2, 3)):
            t = b + d
            if 0 <= t < 27 and t // 9 == base // 9:
                out[:, ci, k * width:(k + 1) * width] = x[:, :, t]
    return out


def _win5_i16(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """hc/tg int16 [n,3,34] → [n,3,21,5]（同花色外 -1）。"""
    n = x.shape[0]
    out = np.full((n, 3, len(centers), 5), -1, np.int16)
    for ci, (_a, b, _c) in enumerate(centers):
        base = (b // 9) * 9
        for k, d in enumerate(range(-2, 3)):
            t = b + d
            if 0 <= t < 27 and t // 9 == base // 9:
                out[:, :, ci, k] = x[:, :, t]
    return out


def _active(hc: np.ndarray, centers: np.ndarray):
    any_ = np.stack([(hc[:, :, a] < 99) | (hc[:, :, c] < 99)
                     for a, _b, c in centers], -1)
    both = np.stack([(hc[:, :, a] < 99) & (hc[:, :, c] < 99)
                     for a, _b, c in centers], -1)
    return any_, both


class Feat:
    """一个 split 的全部派生数组。"""

    def __init__(self, cache: dict):
        self.cache = cache
        self.h = cache['h']
        self.g = cache['g']
        self.e_T = cache['e_T']
        self.s = cache['s_belief']
        self.alloc = cache['alloc']
        self.unseen = cache['unseen']
        self.turn = cache['turn']
        self.riichi = cache['riichi']
        self.called = cache['called']
        self.hc = cache['hc']
        self.tg = cache['tg']
        self.n = self.h.shape[0]
        self.C = self.h.shape[1]
        self.dT = self.e_T.shape[-1]
        self.S = self.s.shape[-1]
        self.centers = _centers()
        self.h_win = _win5_h(self.h, self.centers, self.C)
        self.u_win = _win5(self.unseen.astype(np.float16), self.centers, 1)
        self.hc_win = _win5_i16(self.hc, self.centers)
        self.tg_win = _win5_i16(self.tg, self.centers)
        self.hc_valid = ((self.hc_win >= 0) & (self.hc_win < 99)).astype(np.float16)
        self.hc_norm = np.where(self.hc_valid > 0,
                                np.clip(self.hc_win, 0, 26) / 26.0, 0.0).astype(np.float16)
        self.tg_flag = (self.tg_win >= 0).astype(np.float16)
        self.pat = np.stack([LS.true_pattern(self.alloc, c) for c in self.centers], -1)
        self.any_, self.both = _active(self.hc, self.centers)
        self.onehot = np.eye(TARGETS, dtype=np.float16)
        self.d_in = (5 * self.C + self.dT + self.S + self.C + 3 + 3 + 20)

    def x(self, i: int, ci: int, j: int) -> np.ndarray:
        return np.concatenate([
            self.h_win[i, ci],
            self.e_T[i, j],
            self.s[i, j],
            self.g[i],
            self.onehot[j],
            np.array([min(float(self.turn[i]), 18.0) / 18.0], np.float16),
            np.array([float(self.riichi[i, j]), float(self.called[i, j])], np.float16),
            self.u_win[i, ci] / 4.0,
            self.hc_norm[i, j, ci],
            self.tg_flag[i, j, ci],
            self.hc_valid[i, j, ci],
        ]).astype(np.float16)


class LocalDS(Dataset):
    def __init__(self, f: Feat, active_w=(2.0, 4.0)):
        self.f = f
        self.active_w = active_w
        self.n = f.n

    def __len__(self):
        return self.n * N_CENTERS

    def __getitem__(self, idx):
        i, ci = divmod(int(idx), N_CENTERS)
        x = np.stack([self.f.x(i, ci, j) for j in range(TARGETS)], 0)
        y = self.f.pat[i, :, ci].astype(np.int64)
        w = (1.0 + self.active_w[0] * self.f.any_[i, :, ci]
             + self.active_w[1] * self.f.both[i, :, ci]).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(w), torch.tensor(int(idx), dtype=torch.int64))


def _loader(f: Feat, batch: int, shuffle: bool, workers: int):
    ds = LocalDS(f)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=(DEV == 'cuda'),
                      persistent_workers=(workers > 0), drop_last=False)


def predict(head: nn.Module, f: Feat, batch: int, workers: int):
    """返回 top1_hit/top3_hit/ce，形状 [n,3,21]。"""
    head.eval()
    top1 = np.zeros((f.n, TARGETS, N_CENTERS), bool)
    top3 = np.zeros((f.n, TARGETS, N_CENTERS), bool)
    ce = np.zeros((f.n, TARGETS, N_CENTERS), np.float32)
    dl = _loader(f, batch, False, workers)
    with torch.no_grad():
        for x, y, _w, idx in dl:
            logits = head(x.float().to(DEV))
            probs = F.softmax(logits, -1)
            pred = probs.argmax(-1)                       # [B,3]
            top3i = probs.topk(3, -1).indices             # [B,3,3]
            ydev = y.to(DEV)
            ce_b = -torch.log(probs.gather(-1, ydev[..., None]).squeeze(-1) + 1e-9)
            pred = pred.cpu().numpy()
            top3i = top3i.cpu().numpy()
            yn = y.numpy()
            cen = ce_b.cpu().numpy()
            for b, flat in enumerate(idx.numpy()):
                i, ci = divmod(int(flat), N_CENTERS)
                top1[i, :, ci] = (pred[b] == yn[b])
                top3[i, :, ci] = (top3i[b] == yn[b][..., None]).any(-1)
                ce[i, :, ci] = cen[b]
    return top1, top3, ce


def _aggregate(top1, top3, ce, f: Feat, layers: dict):
    active = {'all_centers': np.ones_like(f.any_, dtype=bool),
              'active_any': f.any_, 'active_both': f.both}
    out = {}
    for layer, mask in layers.items():
        out[layer] = {}
        for an, am in active.items():
            n_sum = 0
            s1 = s3 = sc = 0.0
            for ci in range(N_CENTERS):
                m = mask & am[:, :, ci]
                if not m.any():
                    continue
                n_sum += int(m.sum())
                s1 += float(top1[:, :, ci][m].sum())
                s3 += float(top3[:, :, ci][m].sum())
                sc += float(ce[:, :, ci][m].sum())
            out[layer][an] = {
                'n': n_sum,
                'top1': s1 / max(n_sum, 1),
                'top3': s3 / max(n_sum, 1),
                'ce': sc / max(n_sum, 1),
            }
    return out


def _prior_arrays(prior: np.ndarray, f: Feat):
    """prior [21,27] → top1/top3/ce 数组 [n,3,21]。"""
    top1 = np.zeros((f.n, TARGETS, N_CENTERS), bool)
    top3 = np.zeros((f.n, TARGETS, N_CENTERS), bool)
    ce = np.zeros((f.n, TARGETS, N_CENTERS), np.float32)
    for ci in range(N_CENTERS):
        p = prior[ci]
        y = f.pat[:, :, ci].reshape(-1)
        a1 = int(np.argmax(p))
        a3 = np.argsort(-p)[:3]
        top1[:, :, ci] = (a1 == y).reshape(f.n, TARGETS)
        top3[:, :, ci] = (a3 == y[:, None]).any(1).reshape(f.n, TARGETS)
        ce[:, :, ci] = (-np.log(p[y] + 1e-9)).reshape(f.n, TARGETS)
    return top1, top3, ce


def _prior_from(f: Feat):
    prior = np.zeros((N_CENTERS, NUM_CLASSES), np.float64)
    for ci in range(N_CENTERS):
        cnt = np.bincount(f.pat[:, :, ci].reshape(-1),
                          minlength=NUM_CLASSES).astype(np.float64)
        prior[ci] = cnt / max(cnt.sum(), 1.0)
    return prior


def _a2c_product(f: Feat):
    """A2c 独立边际乘积的 top1/top3/ce。"""
    top1 = np.zeros((f.n, TARGETS, N_CENTERS), bool)
    top3 = np.zeros((f.n, TARGETS, N_CENTERS), bool)
    ce = np.zeros((f.n, TARGETS, N_CENTERS), np.float32)
    for ci, center in enumerate(f.centers):
        joint = LS.joint_pattern_probs(
            f.cache['AP'].astype(np.float32), center)
        y = f.pat[:, :, ci].reshape(-1)
        order = np.argsort(-joint, 1)
        ptrue = joint[np.arange(len(y)), y]
        top1[:, :, ci] = (order[:, 0] == y).reshape(f.n, TARGETS)
        top3[:, :, ci] = (order[:, :3] == y[:, None]).any(1).reshape(f.n, TARGETS)
        ce[:, :, ci] = (-np.log(np.clip(ptrue, 1e-9, None))).reshape(f.n, TARGETS)
    return top1, top3, ce


def _masks(f: Feat):
    n = f.n
    t8 = np.repeat((f.turn >= 8)[:, None], TARGETS, axis=1)
    return {
        'all': np.ones((n, TARGETS), bool),
        'riichi': f.riichi,
        'nonriichi': ~f.riichi,
        'called': f.called,
        'turn>=8': t8,
        'clean_t8': (~f.riichi) & (~f.called) & t8,
    }


def _fmt(x, nd=4):
    return f'{x:.{nd}f}' if np.isfinite(x) else '—'


def write_report(path: Path, res: dict) -> None:
    lines = ['# P1：冻结骨干局部形状头（探针）\n']
    m = res['meta']
    lines += [f"- 训练：{m['train_n']} 样本（202608，按文件切分）",
              f"- 验证：{m['val_n']} 样本；评估：{m['eval_n']} 样本（202609）",
              f"- 头：{m['d_in']}→{m['hidden']}→{m['hidden']}→27；参数量 {m['params']}",
              f"- 设备：{m['device']}；seed：{m['seed']}\n"]
    lines.append('## 训练曲线\n')
    lines.append('| epoch | train loss | val top1 | val top3 | val CE |')
    lines.append('|---:|---:|---:|---:|---:|')
    for r in res['curve']:
        v = r['val']['all']['all_centers']
        lines.append(f"| {r['epoch']} | {_fmt(r['train_loss'])} | {_fmt(v['top1'])} | "
                     f"{_fmt(v['top3'])} | {_fmt(v['ce'])} |")
    lines.append('\n## 评估（202609）\n')
    lines.append('| 层 | 子集 | head top1 | prior(train) | prior(eval) | A2c product | head top3 | A2c top3 | head CE | prior(train) CE | A2c CE |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for layer in ('all', 'riichi', 'nonriichi', 'called', 'turn>=8', 'clean_t8'):
        for an in ('all_centers', 'active_any', 'active_both'):
            t = res['eval'][layer][an]
            lines.append(
                f"| {layer} | {an} | {_fmt(t['head']['top1'])} | "
                f"{_fmt(t['prior_train']['top1'])} | {_fmt(t['prior_eval']['top1'])} | "
                f"{_fmt(t['a2c']['top1'])} | {_fmt(t['head']['top3'])} | "
                f"{_fmt(t['a2c']['top3'])} | {_fmt(t['head']['ce'])} | "
                f"{_fmt(t['prior_train']['ce'])} | {_fmt(t['a2c']['ce'])} |")
    lines.append('\n## 判定\n')
    for line in res['verdict']['findings']:
        lines.append(f'- {line}')
    lines.append(f"\n**判定：{res['verdict']['decision']}**\n")
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default='/tmp/read_model_local_shape_p1')
    ap.add_argument('--out', default=str(HERE / 'results' / 'p1'))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--active-w-any', type=float, default=2.0)
    ap.add_argument('--active-w-both', type=float, default=4.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = Path(args.cache_dir)
    tr = Feat(_load(d / 'train.npz'))
    va = Feat(_load(d / 'val.npz'))
    ev = Feat(_load(d / 'eval.npz'))
    assert tr.d_in == va.d_in == ev.d_in, (tr.d_in, va.d_in, ev.d_in)
    print(f'[data] train={tr.n} val={va.n} eval={ev.n} d_in={tr.d_in} '
          f'C={tr.C} dT={tr.dT} S={tr.S} device={DEV}', flush=True)

    head = LocalShapeHead(tr.d_in, args.hidden).to(DEV)
    n_params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    tr_dl = _loader(tr, args.batch, True, args.workers)
    masks_va, masks_ev = _masks(va), _masks(ev)

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
        v1, v3, vc = predict(head, va, args.batch, args.workers)
        vm = _aggregate(v1, v3, vc, va, masks_va)
        val = vm['all']['all_centers']
        curve.append({'epoch': epoch, 'train_loss': tot / max(n, 1),
                      'val': vm})
        print(f"[epoch {epoch:02d}] loss={tot/max(n,1):.4f} "
              f"val_top1={val['top1']:.4f} top3={val['top3']:.4f} "
              f"ce={val['ce']:.4f}  {time.time()-t0:.0f}s", flush=True)
        if val['top1'] > best + 1e-6:
            best, best_epoch, bad = val['top1'], epoch, 0
            torch.save({'state_dict': head.state_dict(),
                        'config': {'d_in': tr.d_in, 'hidden': args.hidden,
                                   'n_targets': TARGETS, 'n_classes': NUM_CLASSES}},
                       out / 'head.pth')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'[early] stop at epoch {epoch} (best {best_epoch})', flush=True)
                break

    # 载入 best
    state = torch.load(out / 'head.pth', map_location=DEV, weights_only=False)
    head.load_state_dict(state['state_dict'])
    head.eval()

    # eval 数组
    h1, h3, hc = predict(head, ev, args.batch, args.workers)
    a1, a3, ac = _a2c_product(ev)
    prior_train = _prior_from(tr)
    prior_eval = _prior_from(ev)
    p1, p3, pc = _prior_arrays(prior_train, ev)
    q1, q3, qc = _prior_arrays(prior_eval, ev)

    mh = _aggregate(h1, h3, hc, ev, masks_ev)
    ma = _aggregate(a1, a3, ac, ev, masks_ev)
    mp = _aggregate(p1, p3, pc, ev, masks_ev)
    mq = _aggregate(q1, q3, qc, ev, masks_ev)
    table = {}
    for layer in mh:
        table[layer] = {}
        for an in mh[layer]:
            table[layer][an] = {
                'head': mh[layer][an],
                'a2c': ma[layer][an],
                'prior_train': mp[layer][an],
                'prior_eval': mq[layer][an],
            }

    any_ = table['all']['active_any']
    both = table['all']['active_both']
    d_any = any_['head']['top1'] - any_['prior_train']['top1']
    d_both = both['head']['top1'] - both['prior_train']['top1']
    ce_any = any_['prior_train']['ce'] - any_['head']['ce']
    gate_any = (d_any >= 0.05) or (ce_any >= 0.10)
    gate_both = d_both >= 0.03
    gate_a2c = (any_['head']['top1'] > any_['a2c']['top1']) or \
               (any_['head']['ce'] < any_['a2c']['ce'])
    gate = bool(gate_any and gate_both and gate_a2c)

    findings = [
        f"active_any：head top1 {any_['head']['top1']:.3f} vs prior(train) "
        f"{any_['prior_train']['top1']:.3f}（Δ {d_any:+.3f}）；vs A2c product "
        f"{any_['a2c']['top1']:.3f}；head top3 {any_['head']['top3']:.3f} vs "
        f"A2c {any_['a2c']['top3']:.3f}；head CE {any_['head']['ce']:.3f} vs "
        f"prior {any_['prior_train']['ce']:.3f} / A2c {any_['a2c']['ce']:.3f}。",
        f"active_both：head top1 {both['head']['top1']:.3f} vs prior(train) "
        f"{both['prior_train']['top1']:.3f}（Δ {d_both:+.3f}）；立直 active_both "
        f"Δ {table['riichi']['active_both']['head']['top1'] - table['riichi']['active_both']['prior_train']['top1']:+.3f}。",
        f"门禁：active_any top1+5pp 或 CE−0.10 = {gate_any}；"
        f"active_both top1+3pp = {gate_both}；优于 A2c product = {gate_a2c}。",
    ]
    verdict = {
        'decision': 'PASS → 解冻微调/200k 候选' if gate else 'FAIL → 停止局部头，转立直显式特征/蒸馏',
        'gate': gate,
        'gate_parts': {'active_any': bool(gate_any),
                       'active_both': bool(gate_both),
                       'beat_a2c': bool(gate_a2c)},
        'findings': findings,
    }
    meta = {
        'train_n': tr.n, 'val_n': va.n, 'eval_n': ev.n,
        'd_in': tr.d_in, 'hidden': args.hidden, 'params': int(n_params),
        'C': tr.C, 'dT': tr.dT, 'S': tr.S,
        'device': DEV, 'seed': args.seed, 'best_epoch': best_epoch,
        'cache_dir': str(d),
    }
    res = {'meta': meta, 'curve': curve, 'eval': table, 'verdict': verdict}
    with open(out / 'p1_local_head.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'P1_REPORT.md', res)
    print(f'\n[out] {out}/head.pth')
    print(f'[out] {out}/p1_local_head.json')
    print(f'[out] {out}/P1_REPORT.md')
    print(f"[verdict] {verdict['decision']}")
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
