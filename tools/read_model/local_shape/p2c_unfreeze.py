#!/usr/bin/env python3
"""P2c：受控解冻探针（最后 2 个 ResBlock + P2 head）。

目标：回答 P2b 的 CE 增益能否在**表示层**变成 top1 增益。
- 冻结 A2c 其余部分（stem/前 10 个 block/timeline/struct/H1/H2/H3）；
- 只解冻 `backbone.blocks[-2:]` + P2 local head；
- 损失 = 局部 27 类 CE（硬标签，active 加权）+ w_alloc·alloc CE（冻结 H1，防漂移）；
- 数据：202608 train（ReadIterDataset，随机增广），202608 20% val，202609 eval；
- 20k 步，batch 512；GPU4 / tmux。

门禁（预注册）：
  1. riichi active_both top1 **严格 > A2c 0.6686**，且 CE ≤ A2c 1.0173；
  2. all active_any top1 ≥ A2c 0.5746 − 0.005；
  3. eval alloc_ce ≤ A2c 0.6516 + 0.005。
通过 → 可考虑 200k；不通过 → 关闭局部形状线。

用法：
  python tools/read_model/local_shape/p2c_unfreeze.py \
      --steps 20000 --batch 512 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'mortal'))
sys.path.insert(0, str(REPO / 'tools' / 'read_model' / 'diagnostics'))
sys.path.insert(0, str(REPO / 'tools' / 'read_model'))

os.environ.setdefault('RM_CKPT', str(REPO / 'mortal' / 'read_model_v2s_a2c' / 'best.pth'))

import _common as C  # noqa: E402
C.bootstrap_paths()

import train as T  # noqa: E402
import dataset as D  # noqa: E402
import features as FEAT  # noqa: E402
import local_shape as LS  # noqa: E402
import p1_local_head as P1  # noqa: E402
import p2_riichi_head as P2  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGETS = P1.TARGETS
N_CENTERS = P1.N_CENTERS
NUM_CLASSES = P1.NUM_CLASSES
A2C_ALLOC_CE = 0.6516
A2C_RB_TOP1 = 0.6686
A2C_RB_CE = 1.0173
A2C_ANY_TOP1 = 0.5746


def _glob_files(pat: str) -> list[str]:
    return sorted(glob.glob(pat, recursive=True))


def _split_files(files: list[str], val_frac: float, seed: int):
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    return files[n_val:], files[:n_val]


def _centers_idx() -> torch.Tensor:
    """[21,3] center tile indices。"""
    return torch.tensor(LS.centers_3(), dtype=torch.long)


def _win_idx() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 idx[21,5]（越界=-1）、clamped[21,5]、valid[21,5]、center_m[21]。"""
    idx = np.full((N_CENTERS, 5), -1, np.int64)
    center_m = np.zeros(N_CENTERS, np.int64)
    for ci, (_a, b, _c) in enumerate(LS.centers_3()):
        center_m[ci] = b
        base = (b // 9) * 9
        for k, d in enumerate(range(-2, 3)):
            t = b + d
            if 0 <= t < 27 and t // 9 == base // 9:
                idx[ci, k] = t
    clamped = np.where(idx >= 0, idx, 0)
    valid = idx >= 0
    return (torch.from_numpy(idx), torch.from_numpy(clamped),
            torch.from_numpy(valid), torch.from_numpy(center_m))


IDX5, IDX5C, VALID5, CENTER_M = _win_idx()
CENTERS3 = _centers_idx()


def _meta(batch: dict) -> dict:
    hc, tg = C.parse_tl(batch['tl_tokens'].numpy(), batch['tl_mask'].numpy())
    decl_tile, decl_turn, decl_tg = FEAT._riichi_decl(batch)
    return {
        'hc': torch.from_numpy(hc).to(DEV),
        'tg': torch.from_numpy(tg).to(DEV),
        'decl_tile': torch.from_numpy(decl_tile).to(DEV),
        'decl_turn': torch.from_numpy(decl_turn).to(DEV),
        'decl_tg': torch.from_numpy(decl_tg).to(DEV),
        'post': batch['post_riichi_river'].to(DEV),
        'riichi': (batch['riichi'] > 0.5).to(DEV),
        'called': (batch['called'] > 0.5).to(DEV),
        'turn': batch['turn'].to(DEV),
        'unseen': batch['unseen'].to(DEV),
        'alloc': batch['alloc'].to(DEV),
    }


def _pattern_labels(alloc: torch.Tensor) -> torch.Tensor:
    """alloc [B,3,34] → [B,3,21] 27 类索引。"""
    c = alloc[:, :, CENTERS3.to(alloc.device)]          # [B,3,21,3]
    c = c.clamp(0, 2)
    return c[..., 0] * 9 + c[..., 1] * 3 + c[..., 2]


def _active(hc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = CENTERS3[:, 0].to(hc.device)
    c = CENTERS3[:, 2].to(hc.device)
    hca = hc[:, :, a]                                    # [B,3,21]
    hcc = hc[:, :, c]
    any_ = (hca < 99) | (hcc < 99)
    both = (hca < 99) & (hcc < 99)
    return any_, both


def _local_input(net, batch: dict, meta: dict, h: torch.Tensor,
                 g: torch.Tensor, e_T: torch.Tensor, s_b: torch.Tensor):
    B, Cc = h.shape[0], h.shape[1]
    dev = h.device
    idx5 = IDX5C.to(dev)                                 # [21,5]
    valid = VALID5.to(dev).to(h.dtype)                   # [21,5]
    center_m = CENTER_M.to(dev)                          # [21]

    hw = h[:, :, idx5]                                    # [B,C,21,5]
    hw = hw * valid[None, None]
    hw = hw.permute(0, 2, 3, 1).reshape(B, N_CENTERS, 5 * Cc)

    u = meta['unseen'][:, idx5].to(h.dtype) * valid[None] / 4.0
    post = meta['post'][:, :, idx5].to(h.dtype) * valid[None, None]

    hc = meta['hc'][:, :, idx5]                           # [B,3,21,5]
    hcv = ((hc >= 0) & (hc < 99)).to(h.dtype) * valid[None, None]
    hcn = torch.clamp(hc, 0, 26).to(h.dtype) / 26.0 * hcv
    tgf = (meta['tg'][:, :, idx5] >= 0).to(h.dtype) * valid[None, None]

    decl = meta['decl_tile']                              # [B,3]
    decl_onehot = F.one_hot(decl.clamp(min=0).long(), 34).to(h.dtype) * (decl >= 0)[..., None]
    is_decl = (center_m[None, None, :] == decl[:, :, None]).to(h.dtype)
    same_suit = (center_m[None, None, :] // 9) == (decl[:, :, None] // 9)
    dist = torch.where((decl[:, :, None] >= 0) & same_suit,
                       (center_m[None, None, :] - decl[:, :, None]).abs().to(h.dtype) / 9.0,
                       torch.ones_like(is_decl))
    decl_in = ((idx5[None, None] == decl[:, :, None, None]) &
               valid[None, None].bool()).to(h.dtype)
    post_count = (meta['post'].sum(-1).clamp(max=13).to(h.dtype) / 13.0)
    dt = meta['decl_turn'].to(h.dtype)
    turns_since = torch.where(dt > 0,
                              (meta['turn'][:, None].to(h.dtype) - dt).clamp(min=0) / 18.0,
                              torch.zeros_like(dt))
    decl_turn_n = dt / 18.0
    decl_tg = meta['decl_tg'].to(h.dtype)

    onehot = torch.eye(TARGETS, device=dev, dtype=h.dtype)[None, :, None, :]
    onehot = onehot.expand(B, TARGETS, N_CENTERS, TARGETS)
    turn_n = (torch.clamp(meta['turn'], max=18).to(h.dtype) / 18.0)[:, None, None, None]
    ri = meta['riichi'].to(h.dtype)[:, :, None, None]
    ca = meta['called'].to(h.dtype)[:, :, None, None]

    x = torch.cat([
        hw[:, None, :, :].expand(B, TARGETS, N_CENTERS, 5 * Cc),
        e_T[:, :, None, :].to(h.dtype).expand(B, TARGETS, N_CENTERS, -1),
        s_b[:, :, None, :].to(h.dtype).expand(B, TARGETS, N_CENTERS, -1),
        g[:, None, None, :].to(h.dtype).expand(B, TARGETS, N_CENTERS, -1),
        onehot,
        turn_n.expand(B, TARGETS, N_CENTERS, 1),
        ri.expand(B, TARGETS, N_CENTERS, 1),
        ca.expand(B, TARGETS, N_CENTERS, 1),
        u[:, None, :, :].expand(B, TARGETS, N_CENTERS, 5),
        hcn, tgf, hcv,
        decl_onehot[:, :, None, :].expand(B, TARGETS, N_CENTERS, 34),
        is_decl[..., None], dist[..., None],
        post_count[:, :, None, None].expand(B, TARGETS, N_CENTERS, 1),
        turns_since[:, :, None, None].expand(B, TARGETS, N_CENTERS, 1),
        decl_turn_n[:, :, None, None].expand(B, TARGETS, N_CENTERS, 1),
        decl_tg[:, :, None, None].expand(B, TARGETS, N_CENTERS, 1),
        decl_in,
        post,
    ], -1)
    return x


def _forward_features(net, batch: dict, meta: dict):
    obs = batch['obs'].to(DEV)
    tl = batch['tl_tokens'].to(DEV)
    tm = batch['tl_mask'].to(DEV)
    h, g = net.backbone(obs)
    with torch.no_grad():
        e_T = net.timeline(tl, tm)
        z = torch.cat([g.detach()[:, None, :].expand(-1, TARGETS, -1), e_T], -1)
        st = net.struct(z)
        s_counts = F.softmax(st['count_logits'], -1).flatten(-2)
        s_form = F.softmax(st['form_logits'], -1)
        if net.struct_shanten_belief:
            s_b = torch.cat([s_counts, s_form,
                             F.softmax(st['shanten_logits'], -1)], -1)
        else:
            s_b = torch.cat([s_counts, s_form], -1)
    return h, g, e_T, s_b


def _alloc_loss(net, h, e_T, s_b, meta):
    B, Cc = h.shape[0], h.shape[1]
    h_j = h.unsqueeze(1).expand(B, TARGETS, Cc, 34).transpose(-1, -2)
    e_col = e_T[:, :, None, :].expand(B, TARGETS, 34, -1)
    s_exp = s_b[:, :, None, :].expand(B, TARGETS, 34, -1)
    x1 = torch.cat([h_j, e_col, s_exp], -1)
    x1 = F.gelu(net.h1(x1))
    logits = net.h1_out(x1)
    k = torch.arange(5, device=DEV)
    cap = meta['unseen'][:, None, :, None]
    logits = logits.masked_fill(k > cap, float('-inf'))
    logp = F.log_softmax(logits, -1)
    tgt = meta['alloc'][..., None]
    ll = -logp.gather(-1, tgt).squeeze(-1)              # [B,3,34]
    valid = (meta['unseen'][:, None, :] > 0).expand_as(ll)
    return (ll * valid).sum() / valid.sum().clamp(min=1)


def _local_loss(head, x, y, w):
    # P2Head 的 GroupedLinear 只支持 [B,J,d]；把 21 个中心摊到 batch 维
    b = x.shape[0]
    logits = head(x.reshape(-1, TARGETS, x.shape[-1])).reshape(
        b, TARGETS, N_CENTERS, NUM_CLASSES)
    ll = F.cross_entropy(logits.reshape(-1, NUM_CLASSES),
                         y.reshape(-1), reduction='none')
    return (ll * w.reshape(-1)).mean()


def _eval_batch(net, head, batch: dict, meta: dict, chunk: int = 1024):
    n = batch['obs'].shape[0]
    top1 = np.zeros((n, TARGETS, N_CENTERS), bool)
    top3 = np.zeros((n, TARGETS, N_CENTERS), bool)
    ce = np.zeros((n, TARGETS, N_CENTERS), np.float32)
    alloc_ce = np.zeros((n, TARGETS, 34), np.float32)
    net.eval()
    head.eval()
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            sub = {k: v[lo:hi] for k, v in batch.items()}
            sm = {k: v[lo:hi] for k, v in meta.items()}
            h, g, e_T, s_b = _forward_features(net, sub, sm)
            x = _local_input(net, sub, sm, h, g, e_T, s_b)
            b = x.shape[0]
            logits = head(x.reshape(-1, TARGETS, x.shape[-1])).reshape(
                b, TARGETS, N_CENTERS, NUM_CLASSES)
            probs = F.softmax(logits, -1)
            y = _pattern_labels(sm['alloc'])
            pred = probs.argmax(-1)
            top1[lo:hi] = (pred == y).cpu().numpy()
            top3[lo:hi] = (probs.topk(3, -1).indices == y[..., None]).any(-1).cpu().numpy()
            ce[lo:hi] = (-torch.log(probs.gather(-1, y[..., None]).squeeze(-1) + 1e-9)
                         ).cpu().numpy()
            # alloc CE（诊断）
            B, Cc = h.shape[0], h.shape[1]
            h_j = h.unsqueeze(1).expand(B, TARGETS, Cc, 34).transpose(-1, -2)
            e_col = e_T[:, :, None, :].expand(B, TARGETS, 34, -1)
            s_exp = s_b[:, :, None, :].expand(B, TARGETS, 34, -1)
            x1 = torch.cat([h_j, e_col, s_exp], -1)
            x1 = F.gelu(net.h1(x1))
            al = net.h1_out(x1)
            k = torch.arange(5, device=DEV)
            al = al.masked_fill(k > sm['unseen'][:, None, :, None], float('-inf'))
            alp = F.log_softmax(al, -1)
            alll = -alp.gather(-1, sm['alloc'][..., None]).squeeze(-1)
            alloc_ce[lo:hi] = alll.cpu().numpy()
    return top1, top3, ce, alloc_ce


def _metrics(top1, top3, ce, meta):
    f = SimpleNamespace(n=top1.shape[0], any_=_active(meta['hc'])[0].cpu().numpy(),
                        both=_active(meta['hc'])[1].cpu().numpy(),
                        riichi=meta['riichi'].cpu().numpy(),
                        called=meta['called'].cpu().numpy(),
                        turn=meta['turn'].cpu().numpy())
    masks = P1._masks(f)
    return P1._aggregate(top1, top3, ce, f, masks)


def _fmt(x, nd=4):
    if x is None:
        return '—'
    try:
        return f'{x:.{nd}f}' if np.isfinite(x) else '—'
    except TypeError:
        return '—'


def write_report(path: Path, res: dict) -> None:
    m = res['meta']
    lines = ['# P2c：受控解冻探针（最后 2 个 ResBlock + H1 + 局部头）\n']
    lines += [f"- 步数：{m['steps']}（batch {m['batch']}）；最好 step {m['best_step']}",
              f"- backbone lr {m['lr_bb']}；H1 lr {m.get('lr_h1', '—')}；"
              f"head lr {m['lr_head']}；alloc 权重 {m['w_alloc']}",
              f"- 设备 {m['device']}；训练/验证/评估：{m['train_files']}/{m['val_files']}/{m['eval_files']} 文件\n"]
    lines.append('## 验证曲线（202608 val）\n')
    lines.append('| step | val riichi any top1 | val riichi both top1 | val riichi any CE |')
    lines.append('|---:|---:|---:|---:|')
    for r in res['curve']:
        v = r['val']['riichi']
        lines.append(f"| {r['step']} | {_fmt(v['active_any']['top1'])} | "
                     f"{_fmt(v['active_both']['top1'])} | {_fmt(v['active_any']['ce'])} |")
    lines.append('\n## 评估（202609）\n')
    lines.append('| 层 | 子集 | P2c top1 | A2c product | P2c top3 | A2c top3 | P2c CE | A2c CE |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for layer in ('riichi', 'all', 'nonriichi', 'turn>=8', 'clean_t8'):
        for an in ('all_centers', 'active_any', 'active_both'):
            t = res['eval'][layer][an]
            lines.append(f"| {layer} | {an} | {_fmt(t['head']['top1'])} | "
                         f"{_fmt(t['a2c']['top1'])} | {_fmt(t['head']['top3'])} | "
                         f"{_fmt(t['a2c']['top3'])} | {_fmt(t['head']['ce'])} | "
                         f"{_fmt(t['a2c']['ce'])} |")
    lines.append(f"\n- eval alloc CE：{_fmt(res['alloc_ce']['p2c'])}（A2c 基线 {A2C_ALLOC_CE}）\n")
    lines.append('## 判定\n')
    for line in res['verdict']['findings']:
        lines.append(f'- {line}')
    lines.append(f"\n**判定：{res['verdict']['decision']}**\n")
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--prefetch', type=int, default=2)
    ap.add_argument('--lr-bb', type=float, default=1e-5)
    ap.add_argument('--lr-head', type=float, default=5e-4)
    ap.add_argument('--lr-h1', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--w-alloc', type=float, default=0.5)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--val-n', type=int, default=5000)
    ap.add_argument('--eval-n', type=int, default=40000)
    ap.add_argument('--out', default=str(HERE / 'results' / 'p2c'))
    ap.add_argument('--p2b-head', default=str(HERE / 'results' / 'p2b' / 'p2b_head.pth'))
    ap.add_argument('--seed', type=int, default=20261103)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_glob = str(REPO / 'dataset' / 'mjai' / 'houou4_names' / '202608' / '**' / '*.json.gz')
    eval_glob = str(REPO / 'dataset' / 'mjai' / 'houou4' / '202609' / '**' / '*.json.gz')
    files = _glob_files(train_glob)
    train_files, val_files = _split_files(files, 0.2, 20261102)
    eval_files = _glob_files(eval_glob)
    print(f'[data] train_files={len(train_files)} val_files={len(val_files)} '
          f'eval_files={len(eval_files)} device={DEV}', flush=True)

    net = C.load_net(Path(os.environ['RM_CKPT']))
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.backbone.blocks[-2:].parameters():
        p.requires_grad_(True)
    # H1（alloc 头）也解冻：h 改变后冻结 H1 无法维持 alloc CE
    for p in list(net.h1.parameters()) + list(net.h1_out.parameters()):
        p.requires_grad_(True)
    # 只解冻权重；BN 保持 eval（running stats），避免 batch 统计造成分布漂移
    net.eval()

    head = P2.P2Head(762, 256, 0.2).to(DEV)
    st = torch.load(args.p2b_head, map_location='cpu', weights_only=False)
    head.load_state_dict(st['state_dict'])
    head.train()
    print(f'[init] p2b head={args.p2b_head}', flush=True)

    opt = torch.optim.AdamW([
        {'params': [p for p in net.backbone.blocks[-2:].parameters() if p.requires_grad],
         'lr': args.lr_bb},
        {'params': [p for p in list(net.h1.parameters()) + list(net.h1_out.parameters())
                    if p.requires_grad], 'lr': args.lr_h1},
        {'params': head.parameters(), 'lr': args.lr_head},
    ], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # val / eval 固定集
    val_batch = T.build_fixed_eval_set(val_files, args.val_n, seed=args.seed)
    val_meta = _meta(val_batch)
    print('[data] val batch ready', flush=True)

    ds = T.ReadIterDataset(train_files, stage=3, augment='random', seed=args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers,
        collate_fn=T.collate,
        prefetch_factor=(args.prefetch if args.workers > 0 else None),
        persistent_workers=(args.workers > 0))
    it = iter(loader)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve = []
    best = -1.0
    best_step = 0
    bad = 0
    t0 = time.time()
    step = 0
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        step += 1
        meta = _meta(batch)
        h, g, e_T, s_b = _forward_features(net, batch, meta)
        x = _local_input(net, batch, meta, h, g, e_T, s_b)
        y = _pattern_labels(meta['alloc'])
        any_, both = _active(meta['hc'])
        w = 1.0 + 1.0 * any_.to(x.dtype) + 2.0 * both.to(x.dtype)
        l_local = _local_loss(head, x, y, w)
        l_alloc = _alloc_loss(net, h, e_T, s_b, meta)
        loss = l_local + args.w_alloc * l_alloc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step <= 5 or step % 100 == 0:
            print(f'[step {step}] loss {loss.item():.4f} local {l_local.item():.4f} '
                  f'alloc {l_alloc.item():.4f} lr {sched.get_last_lr()[0]:.2e} '
                  f'{time.time()-t0:.0f}s', flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            top1, top3, ce, ace = _eval_batch(net, head, val_batch, val_meta)
            vm = _metrics(top1, top3, ce, val_meta)
            v = vm['riichi']
            curve.append({'step': step, 'val': vm})
            print(f"[eval {step}] val_r_any={v['active_any']['top1']:.4f} "
                  f"val_r_both={v['active_both']['top1']:.4f} "
                  f"ce={v['active_any']['ce']:.4f} {time.time()-t0:.0f}s", flush=True)
            score = v['active_any']['top1']
            if score > best + 1e-6:
                best, best_step, bad = score, step, 0
                torch.save({'net': net.state_dict(), 'head': head.state_dict(),
                            'config': {'steps': args.steps, 'w_alloc': args.w_alloc,
                                       'lr_bb': args.lr_bb, 'lr_h1': args.lr_h1,
                                       'lr_head': args.lr_head}},
                           out / 'p2c_best.pth')
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'[early] stop at {step} (best {best_step})', flush=True)
                    break
            net.eval()
            head.train()

    state = torch.load(out / 'p2c_best.pth', map_location=DEV, weights_only=False)
    net.load_state_dict(state['net'])
    head.load_state_dict(state['head'])
    net.eval()
    head.eval()

    eval_batch = T.build_fixed_eval_set(eval_files, args.eval_n, seed=20261016)
    eval_meta = _meta(eval_batch)
    print('[data] eval batch ready', flush=True)
    top1, top3, ce, ace = _eval_batch(net, head, eval_batch, eval_meta)
    em = _metrics(top1, top3, ce, eval_meta)
    valid = np.broadcast_to(
        (eval_meta['unseen'].cpu().numpy()[:, None, :] > 0), ace.shape)
    alloc_ce = float((ace * valid).sum() / valid.sum())

    # A2c product 对照直接取 P2B 报告（同口径，避免再建 40k P2Feat）
    p2b_json = HERE / 'results' / 'p2b' / 'p2b_distill.json'
    if not p2b_json.exists():
        raise SystemExit(f'缺少 {p2b_json}，无法取 A2c 对照')
    p2b_eval = json.loads(p2b_json.read_text())['eval']
    am = {layer: {an: p2b_eval[layer][an]['a2c'] for an in em[layer]}
          for layer in em}

    table = {}
    for layer in em:
        table[layer] = {}
        for an in em[layer]:
            table[layer][an] = {'head': em[layer][an],
                                'a2c': am[layer][an]}
    rb = table['riichi']['active_both']
    aa = table['all']['active_any']
    gate_top1 = rb['head']['top1'] > A2C_RB_TOP1
    gate_ce = rb['head']['ce'] <= A2C_RB_CE
    gate_any = aa['head']['top1'] >= A2C_ANY_TOP1 - 0.005
    gate_alloc = alloc_ce <= A2C_ALLOC_CE + 0.005
    gate = bool(gate_top1 and gate_ce and gate_any and gate_alloc)
    findings = [
        f"riichi active_both：P2c top1 {rb['head']['top1']:.4f} vs A2c {A2C_RB_TOP1:.4f}"
        f"（严格 > = {gate_top1}）；CE {rb['head']['ce']:.4f} vs {A2C_RB_CE:.4f}"
        f"（≤ = {gate_ce}）。",
        f"all active_any：P2c top1 {aa['head']['top1']:.4f} vs A2c {A2C_ANY_TOP1:.4f}"
        f"（−0.5pp 容差 = {gate_any}）。",
        f"eval alloc CE：P2c {alloc_ce:.4f} vs A2c {A2C_ALLOC_CE:.4f}"
        f"（+0.005 容差 = {gate_alloc}）。",
    ]
    verdict = {
        'decision': ('PASS → 可考虑 200k' if gate else 'FAIL → 关闭局部形状线（P1/P2a/P2b/P2c）'),
        'gate': gate,
        'gate_parts': {'riichi_top1': bool(gate_top1), 'riichi_ce': bool(gate_ce),
                       'all_any': bool(gate_any), 'alloc': bool(gate_alloc)},
        'findings': findings,
    }
    meta = {
        'steps': args.steps, 'batch': args.batch, 'best_step': best_step,
        'lr_bb': args.lr_bb, 'lr_h1': args.lr_h1, 'lr_head': args.lr_head,
        'w_alloc': args.w_alloc,
        'device': DEV, 'train_files': len(train_files), 'val_files': len(val_files),
        'eval_files': len(eval_files),
    }
    res = {'meta': meta, 'curve': curve, 'eval': table,
           'alloc_ce': {'p2c': alloc_ce, 'a2c': A2C_ALLOC_CE},
           'verdict': verdict}
    with open(out / 'p2c_unfreeze.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_report(out / 'P2C_REPORT.md', res)
    print(f'\n[out] {out}/p2c_best.pth')
    print(f'[out] {out}/p2c_unfreeze.json')
    print(f'[out] {out}/P2C_REPORT.md')
    print(f"[verdict] {verdict['decision']}")
    for line in findings:
        print(' -', line)
    print(f'[done] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
