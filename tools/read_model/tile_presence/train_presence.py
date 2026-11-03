#!/usr/bin/env python3
"""逐牌存在性（tile presence）：训练、评测与门禁判定。

任务：给定观察者自家手牌与全场公开信息，对 3 个目标家各输出 34 类
「这张牌在不在他手里」的概率（二分类，不预测张数）。

对照：
  - prior：train 上同口径的逐牌存在率（无信息基线）；
  - A2c：冻结读牌模型的 `1 - P(count=0)`（强基线，来自 202609 诊断缓存）。

预注册门禁（202609 40000 样本，informative 单元 = unseen>0）：
  1. all：macro AUC ≥ A2c + 0.010，且 logloss ≤ A2c − 0.010；
  2. turn≥8：macro AUC ≥ A2c + 0.010；
  3. riichi：macro AUC ≥ A2c − 0.005（不能明显更差）。
通过 → 可进入注入探针；不通过 → 记录证据后关闭该结构。

用法：
  python tools/read_model/tile_presence/train_presence.py \
      --cache-dir tools/read_model/tile_presence/.cache \
      --out tools/read_model/tile_presence/results --steps 12000
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]

import tokens as TK  # noqa: E402
from model import TilePresenceNet  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
KEYS = ('tile_feat', 'ev_tile', 'ev_feat', 'ev_mask', 'cons', 'ctx', 'y',
        'unseen', 'tenpai', 'turn', 'riichi', 'called', 'meld_counts')
STRATA = ('all', 'turn>=8', 'turn>=12', 'riichi', 'nonriichi', 'called', 'clean_t8')


def _suit_perm() -> np.ndarray:
    """万↔筒置换（与 dataset.augment_perm 同语义）。"""
    p = np.arange(TK.NUM_TILES, dtype=np.int64)
    p[:9] = np.arange(9, 18)
    p[9:18] = np.arange(0, 9)
    return p


class PresDataset(Dataset):
    def __init__(self, path: Path, aug: bool = False, aug_p: float = 0.5,
                 drop: str = ''):
        z = np.load(path)
        self.data = {k: z[k] for k in KEYS}
        self.n = int(self.data['y'].shape[0])
        self.aug = aug
        self.aug_p = aug_p
        self.drop = drop
        self.perm = _suit_perm()

    def __len__(self) -> int:
        return self.n

    def _augment(self, s: dict) -> dict:
        """万↔筒置换：牌轴重排 + 牌 id 重映射 + ctx 里万/筒手牌数互换。"""
        pm = torch.from_numpy(self.perm)
        s['tile_feat'] = s['tile_feat'][:, pm, :]
        s['y'] = s['y'][:, pm]
        s['unseen'] = s['unseen'][pm]
        s['ev_tile'] = pm[s['ev_tile'].long()].to(s['ev_tile'].dtype)
        cons = s['cons'].long()
        s['cons'] = torch.where(cons >= 0, pm[cons.clamp(min=0)], cons).to(torch.int8)
        s['ctx'] = s['ctx'].clone()
        s['ctx'][:, [10, 11]] = s['ctx'][:, [11, 10]]
        return s

    def _drop(self, s: dict) -> dict:
        """特征组消融（训练期，不是事后零化）。"""
        if self.drop == 'events':
            s['ev_feat'] = torch.zeros_like(s['ev_feat'])
            s['ev_mask'] = torch.zeros_like(s['ev_mask'])
        elif self.drop == 'history':
            t = s['tile_feat'].clone()
            t[..., list(range(0, 12)) + list(range(17, 22))] = 0
            s['tile_feat'] = t
        elif self.drop == 'safety':
            t = s['tile_feat'].clone()
            t[..., [17, 18, 19]] = 0
            e = s['ev_feat'].clone()
            e[..., [14, 15, 16, 17]] = 0
            s['tile_feat'], s['ev_feat'] = t, e
        return s

    def __getitem__(self, i: int) -> dict:
        s = {k: torch.from_numpy(np.asarray(self.data[k][i])) for k in KEYS}
        if self.aug and np.random.random() < self.aug_p:
            s = self._augment(s)
        if self.drop:
            s = self._drop(s)
        return s


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _to_dev(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if k in ('tile_feat', 'ev_feat', 'ctx'):
            out[k] = v.to(DEV, non_blocking=True)
        elif k in ('ev_tile', 'ev_mask', 'cons', 'y', 'tenpai', 'unseen',
                   'meld_counts'):
            out[k] = v.to(DEV, non_blocking=True)
        else:
            out[k] = v
    return out


def _rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos < 1 or n_neg < 1:
        return float('nan')
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    cum = np.concatenate([[0.0], np.cumsum(cnt)])
    avg = (cum[:-1] + cum[1:] + 1.0) / 2.0
    ranks = avg[inv]
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _metrics(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> dict:
    """y/p [N,3,34]；m 为 bool 掩码。"""
    yy = y[m].astype(np.float64)
    pp = np.clip(p[m].astype(np.float64), 1e-6, 1 - 1e-6)
    if yy.size == 0:
        return {}
    ll = float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))
    brier = float(np.mean((pp - yy) ** 2))
    pred = pp > 0.5
    acc = float(np.mean(pred == (yy > 0.5)))
    tp = float(np.sum(pred & (yy > 0.5)))
    fp = float(np.sum(pred & (yy < 0.5)))
    fn = float(np.sum(~pred & (yy > 0.5)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
    auc_micro = _rank_auc(yy, pp)

    aucs = []
    for j in range(3):
        for t in range(TK.NUM_TILES):
            mm = m[:, j, t]
            if mm.sum() < 10:
                continue
            a = _rank_auc(y[mm, j, t], p[mm, j, t])
            if np.isfinite(a):
                aucs.append(a)
    return {
        'n_cells': int(yy.size),
        'base_rate': float(yy.mean()),
        'auc_macro': float(np.mean(aucs)) if aucs else float('nan'),
        'auc_n': len(aucs),
        'auc_micro': auc_micro,
        'logloss': ll,
        'brier': brier,
        'acc': acc,
        'f1': f1,
    }


def _masks(unseen: np.ndarray, turn: np.ndarray, riichi: np.ndarray,
           called: np.ndarray) -> dict:
    n = unseen.shape[0]
    base = np.broadcast_to((unseen > 0)[:, None, :], (n, 3, TK.NUM_TILES)).copy()
    out = {'all': base}
    t8 = np.broadcast_to((turn >= 8)[:, None, None], base.shape)
    t12 = np.broadcast_to((turn >= 12)[:, None, None], base.shape)
    out['turn>=8'] = base & t8
    out['turn>=12'] = base & t12
    out['riichi'] = base & riichi[:, :, None]
    out['nonriichi'] = base & ~riichi[:, :, None]
    out['called'] = base & called[:, :, None]
    out['clean_t8'] = base & ~riichi[:, :, None] & ~called[:, :, None] & t8
    return out


@torch.no_grad()
def predict(net: nn.Module, loader: DataLoader) -> dict:
    net.eval()
    p, y, tp = [], [], []
    for batch in loader:
        b = _to_dev(batch)
        out = net(b['tile_feat'], b['ev_tile'], b['ev_feat'], b['ev_mask'],
                  b['ctx'], b['cons'])
        p.append(torch.sigmoid(out['logits']).float().cpu().numpy())
        tp.append(torch.sigmoid(out['tenpai_logit']).float().cpu().numpy())
        y.append(batch['y'].numpy())
    return {'p': np.concatenate(p), 'y': np.concatenate(y),
            'tp': np.concatenate(tp)}


def _meta(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in ('unseen', 'turn', 'riichi', 'called', 'tenpai')}


def evaluate_split(net, path: Path, batch_size: int, workers: int) -> dict:
    ds = PresDataset(path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    out = predict(net, loader)
    meta = _meta(path)
    masks = _masks(meta['unseen'], meta['turn'], meta['riichi'], meta['called'])
    res = {s: _metrics(out['y'], out['p'], masks[s]) for s in STRATA}
    res['tenpai_auc'] = _rank_auc(meta['tenpai'].reshape(-1), out['tp'].reshape(-1))
    res['_pred'] = out['p']
    res['_y'] = out['y']
    return res


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    return f'{x:.{nd}f}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--w-tenpai', type=float, default=0.2)
    ap.add_argument('--w-sum', type=float, default=0.1)
    ap.add_argument('--label-smooth', type=float, default=0.02)
    ap.add_argument('--aug', type=int, default=1)
    ap.add_argument('--aug-p', type=float, default=0.5)
    ap.add_argument('--drop', default='',
                    choices=('', 'events', 'history', 'safety'))
    ap.add_argument('--w-trivial', type=float, default=0.3)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--diag-cache', default='/tmp/read_model_diag_cache_a2c/cache.npz')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = Path(args.cache_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[dev] {DEV}  cache={cache}', flush=True)

    tr_ds = PresDataset(cache / 'train.npz', aug=bool(args.aug), aug_p=args.aug_p,
                        drop=args.drop)
    va_ds = PresDataset(cache / 'val.npz', drop=args.drop)
    print(f'[data] train={len(tr_ds)} val={len(va_ds)}', flush=True)
    g = torch.Generator()
    g.manual_seed(args.seed)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True, generator=g)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                           num_workers=args.workers)

    net = TilePresenceNet(d=args.d, n_blocks=args.blocks, heads=args.heads,
                          dropout=args.dropout).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    print(f'[net] params={n_par/1e6:.2f}M', flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = args.warmup

    def lr_at(step: int) -> float:
        if step < warm:
            return args.lr * (step + 1) / warm
        prog = (step - warm) / max(args.steps - warm, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    best_auc = -1.0
    best_state = None
    best_step = 0
    bad = 0
    hist = []
    loader = _cycle(tr_loader)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step)
        net.train()
        batch = _to_dev(next(loader))
        out = net(batch['tile_feat'], batch['ev_tile'], batch['ev_feat'],
                  batch['ev_mask'], batch['ctx'], batch['cons'])
        y = batch['y'].float()
        y_s = y * (1.0 - args.label_smooth) + 0.5 * args.label_smooth
        valid = (batch['unseen'].float() > 0)[:, None, :].expand_as(y)
        w = torch.where(valid, torch.ones_like(y), torch.full_like(y, args.w_trivial))
        bce = F.binary_cross_entropy_with_logits(out['logits'], y_s, reduction='none')
        l_main = (bce * w).sum() / w.sum()
        l_tp = F.binary_cross_entropy_with_logits(
            out['tenpai_logit'], batch['tenpai'].float())
        hs = 13.0 - 3.0 * batch['meld_counts'].float()
        p = torch.sigmoid(out['logits']) * valid.float()
        l_sum = ((p.sum(-1) - hs) ** 2).mean()
        loss = l_main + args.w_tenpai * l_tp + args.w_sum * l_sum

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % 100 == 0:
            print(f'[step {step}] loss {loss.item():.4f} main {l_main.item():.4f} '
                  f'tp {l_tp.item():.4f} sum {l_sum.item():.4f} '
                  f'lr {lr_at(step):.2e} {time.time()-t0:.0f}s', flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                pv, yv = [], []
                for vb in va_loader:
                    b = _to_dev(vb)
                    o = net(b['tile_feat'], b['ev_tile'], b['ev_feat'],
                            b['ev_mask'], b['ctx'], b['cons'])
                    pv.append(torch.sigmoid(o['logits']).float().cpu().numpy())
                    yv.append(vb['y'].numpy())
            pv = np.concatenate(pv)
            yv = np.concatenate(yv)
            vm = _masks(va_ds.data['unseen'], va_ds.data['turn'],
                        va_ds.data['riichi'], va_ds.data['called'])
            mv = _metrics(yv, pv, vm['all'])
            hist.append({'step': step, 'val': mv})
            print(f'[val {step}] auc_macro {_fmt(mv["auc_macro"])} '
                  f'logloss {_fmt(mv["logloss"])} acc {_fmt(mv["acc"])} '
                  f'{time.time()-t0:.0f}s', flush=True)
            if mv['auc_macro'] > best_auc + 1e-5:
                best_auc = mv['auc_macro']
                best_step = step
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'[early] no gain for {bad} evals, stop at {step}',
                          flush=True)
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save({'state_dict': net.state_dict(), 'args': vars(args),
                'best_step': best_step, 'best_val_auc': best_auc}, out_dir / 'presence.pth')
    print(f'[best] step={best_step} val_auc={best_auc:.4f}', flush=True)

    # ---------------- 终评（202609 40000） ----------------
    ev_path = cache / 'eval.npz'
    res = evaluate_split(net, ev_path, args.batch, args.workers)
    meta = _meta(ev_path)
    masks = _masks(meta['unseen'], meta['turn'], meta['riichi'], meta['called'])
    y_ev, p_model = res['_y'], res['_pred']

    z_tr = np.load(cache / 'train.npz')
    y_tr, un_tr = z_tr['y'], z_tr['unseen']
    m_tr = un_tr > 0
    prior = np.stack([y_tr[..., t][m_tr[..., t]].mean()
                      for t in range(TK.NUM_TILES)])
    p_prior = np.broadcast_to(prior[None, None, :], y_ev.shape)

    p_a2c = None
    if Path(args.diag_cache).exists():
        ref = np.load(args.diag_cache, allow_pickle=True)
        if ref['AP'].shape[0] >= y_ev.shape[0]:
            p_a2c = 1.0 - ref['AP'][:y_ev.shape[0], :, :, 0].astype(np.float32)
        else:
            print(f'[warn] diag cache 只有 {ref["AP"].shape[0]} 样本，跳过 A2c 对照',
                  flush=True)

    models = {'model': p_model, 'prior': p_prior}
    if p_a2c is not None:
        models['a2c'] = p_a2c

    # 推理期手牌数归一：按 13-3*melds 做几轮比例缩放（不动排序，只调跨样本水平）
    hs = (13.0 - 3.0 * np.load(ev_path)['meld_counts'].astype(np.float64))
    p_norm = p_model.astype(np.float64).copy()
    for _ in range(3):
        s = p_norm.sum(-1)
        scale = np.where(s > 1e-6, hs / np.maximum(s, 1e-6), 1.0)
        p_norm = np.clip(p_norm * scale[..., None], 0.0, 1.0)
    models['model_norm'] = p_norm.astype(np.float32)

    def _per_tile(p):
        out = np.full((3, TK.NUM_TILES), np.nan)
        for j in range(3):
            for t in range(TK.NUM_TILES):
                mm = masks['all'][:, j, t]
                if mm.sum() >= 10:
                    out[j, t] = _rank_auc(y_ev[mm, j, t], p[mm, j, t])
        return out.tolist()

    report = {'strata': {}, 'verdict': {}}
    for s in STRATA:
        report['strata'][s] = {name: _metrics(y_ev, p, masks[s])
                               for name, p in models.items()}
    report['tenpai_aux_auc'] = res['tenpai_auc']
    report['per_tile_auc'] = {name: _per_tile(p) for name, p in models.items()}
    report['meta'] = {'n_eval': int(y_ev.shape[0]), 'best_step': best_step,
                      'best_val_auc': best_auc, 'params': n_par,
                      'args': vars(args)}
    report['history'] = hist

    m_all = report['strata']['all']['model']
    a_all = report['strata']['all'].get('a2c', {})
    m_t8 = report['strata']['turn>=8']['model']
    a_t8 = report['strata']['turn>=8'].get('a2c', {})
    m_ri = report['strata']['riichi']['model']
    a_ri = report['strata']['riichi'].get('a2c', {})
    gate = bool(a_all and
                m_all['auc_macro'] >= a_all['auc_macro'] + 0.010 and
                m_all['logloss'] <= a_all['logloss'] - 0.010 and
                m_t8['auc_macro'] >= a_t8['auc_macro'] + 0.010 and
                m_ri['auc_macro'] >= a_ri['auc_macro'] - 0.005)
    report['verdict'] = {
        'gate': gate,
        'decision': ('PASS → 可进入 S6b 注入 / 风险重排探针' if gate
                     else 'FAIL → 记录证据，停止该结构（先看分层差距再决定是否调）'),
        'deltas': {
            'all_auc': m_all['auc_macro'] - a_all.get('auc_macro', float('nan')),
            'all_ll': m_all['logloss'] - a_all.get('logloss', float('nan')),
            't8_auc': m_t8['auc_macro'] - a_t8.get('auc_macro', float('nan')),
            'riichi_auc': m_ri['auc_macro'] - a_ri.get('auc_macro', float('nan')),
            'prior_auc': m_all['auc_macro'] - report['strata']['all']['prior']['auc_macro'],
        },
    }
    report['strata'] = {s: {k: v for k, v in d.items()} for s, d in report['strata'].items()}
    (out_dir / 'tile_presence.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1))

    lines = ['# Tile Presence：逐牌「在不在」二分类\n',
             f'- 最好 val step：{best_step}（val macro AUC {best_auc:.4f}）',
             f'- 参数：{n_par/1e6:.2f}M；eval n={y_ev.shape[0]}',
             f'- 听牌辅助 AUC：{_fmt(res["tenpai_auc"])}\n',
             '| 子集 | n | base | model AUC | A2c AUC | prior AUC | model LL | A2c LL '
             '| model F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in STRATA:
        row = report['strata'][s]
        mm = row['model']
        aa = row.get('a2c', {})
        pp = row['prior']
        lines.append(
            f"| {s} | {mm['n_cells']} | {mm['base_rate']:.3f} | "
            f"{_fmt(mm['auc_macro'])} | {_fmt(aa.get('auc_macro'))} | "
            f"{_fmt(pp.get('auc_macro'))} | {_fmt(mm['logloss'])} | "
            f"{_fmt(aa.get('logloss'))} | {_fmt(mm['f1'])} |")
    lines.append('')
    mn = report['strata']['all'].get('model_norm', {})
    if mn:
        lines.append(f"- 推理期手牌数归一：all AUC {_fmt(mn.get('auc_macro'))}"
                     f"（raw {_fmt(m_all['auc_macro'])}）、logloss {_fmt(mn.get('logloss'))}"
                     f"（raw {_fmt(m_all['logloss'])}）")
    lines.append(f"**判定：{report['verdict']['decision']}**")
    lines.append('')
    d = report['verdict']['deltas']
    lines.append(f"- all AUC Δ vs A2c：{d['all_auc']:+.4f}；"
                 f"logloss Δ：{d['all_ll']:+.4f}；"
                 f"turn≥8 AUC Δ：{d['t8_auc']:+.4f}；"
                 f"立直 AUC Δ：{d['riichi_auc']:+.4f}；"
                 f"vs prior AUC Δ：{d['prior_auc']:+.4f}")
    (out_dir / 'TILE_PRESENCE_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'[out] {out_dir / "tile_presence.json"}', flush=True)


if __name__ == '__main__':
    main()
