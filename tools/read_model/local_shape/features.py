#!/usr/bin/env python3
"""P1 特征缓存：冻结 A2c 骨干，导出局部头所需的中间特征与标签。

产物（每个 split 一个 npz）：
  h       [n,C,34]      fp16  空间流逐列特征（三家共享）
  g       [n,C]         fp16  空间流全局池化
  e_T     [n,3,dT]      fp16  时序流 pooled 嵌入
  s_belief[n,3,S]       fp16  结构瓶颈（counts+form[+shanten]）
  AP      [n,3,34,5]    fp16  A2c alloc_probs（仅 eval 保存）
  EC      [n,3,34]      fp16  A2c exp_counts（仅 eval 保存）
  alloc   [n,3,34]      int8  真值手牌计数
  unseen  [n,34]        fp16  4 - tiles_seen
  turn    [n]           int16
  riichi  [n,3]         bool
  called  [n,3]         bool
  hc      [n,3,34]      int16 目标家手切该牌的最早 token（99=未手切）
  tg      [n,3,34]      int16 目标家摸切该牌的最早 token（99=未摸切）

用法：
  # 训练/验证（202608，按文件 80/20 切分；与 202609 eval 不重叠）
  python tools/read_model/local_shape/features.py --mode trainval \
      --glob 'dataset/mjai/houou4_names/202608/**/*.json.gz' \
      --train-n 20000 --val-n 5000 --out /tmp/read_model_local_shape_p1

  # 评估（202609；与 P0 cache 同源）
  python tools/read_model/local_shape/features.py --mode eval \
      --glob 'dataset/mjai/houou4/202609/**/*.json.gz' \
      --n 40000 --out /tmp/read_model_local_shape_p1 \
      --check-alloc /tmp/read_model_diag_cache_a2c/cache.npz
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

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DIAG = REPO / 'tools' / 'read_model' / 'diagnostics'
sys.path.insert(0, str(DIAG))

# 必须在 import _common 之前固定 ckpt（_common.CKPT 在 import 时读取）
os.environ.setdefault('RM_CKPT', str(REPO / 'mortal' / 'read_model_v2s_a2c' / 'best.pth'))

import _common as C  # noqa: E402

C.bootstrap_paths()

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import train as T  # noqa: E402
import dataset as D  # noqa: E402


def _load_net(ckpt: Path):
    net = C.load_net(ckpt)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _riichi_decl(batch: dict):
    """从 tl_tokens 找「reach token 之后的第一张 dahai」= 立直宣言牌。

    返回 decl_tile[n,3]（-1=无）、decl_turn[n,3]、decl_tg[n,3]。
    reach_accepted 之前的那张 dahai 才是宣言牌；post_riichi 集合不含它。
    """
    tok = batch['tl_tokens'].numpy()
    mask = batch['tl_mask'].numpy()
    turn = (batch['tl_turn'].numpy() if 'tl_turn' in batch
            else np.zeros(mask.shape, np.float32))
    n, J, L, _ = tok.shape
    decl_tile = np.full((n, J), -1, np.int16)
    decl_turn = np.zeros((n, J), np.int16)
    decl_tg = np.zeros((n, J), np.int8)
    for j in range(J):
        act = np.argmax(tok[:, j, :, 35:42], -1)       # 0=dahai, 1=reach
        reach = (act == 1) & mask[:, j]
        for i in np.where(reach.any(1))[0]:
            rk = int(np.argmax(reach[i]))
            for k in range(rk + 1, L):
                if not mask[i, j, k]:
                    continue
                if int(act[i, k]) == 0 and tok[i, j, k, :34].sum() > 0.5:
                    decl_tile[i, j] = int(np.argmax(tok[i, j, k, :34]))
                    decl_turn[i, j] = int(round(float(turn[i, j, k]) * 18.0))
                    decl_tg[i, j] = int(tok[i, j, k, 34] > 0.5)
                    break
    return decl_tile, decl_turn, decl_tg


def _extract(net, batch: dict, device: str, save_ap: bool) -> dict:
    n = int(batch['obs'].shape[0])
    Cc = int(net.C)
    # 先跑一个 batch 拿 dT/S
    with torch.no_grad():
        probe = {k: batch[k][:1].to(device) for k in ('obs', 'tl_tokens', 'tl_mask')}
        h0, g0 = net.backbone(probe['obs'])
        e0 = net.timeline(probe['tl_tokens'], probe['tl_mask'])
        st0 = net.struct(torch.cat([g0[:, None, :].expand(1, 3, Cc), e0], -1))
        s0 = F.softmax(st0['count_logits'], -1).flatten(-2)
        s0 = torch.cat([s0, F.softmax(st0['form_logits'], -1)], -1)
        if net.struct_shanten_belief:
            s0 = torch.cat([s0, F.softmax(st0['shanten_logits'], -1)], -1)
    dT = int(e0.shape[-1])
    S = int(s0.shape[-1])

    h = np.zeros((n, Cc, 34), np.float16)
    g = np.zeros((n, Cc), np.float16)
    e_T = np.zeros((n, 3, dT), np.float16)
    s_belief = np.zeros((n, 3, S), np.float16)
    AP = np.zeros((n, 3, 34, 5), np.float16) if save_ap else None
    EC = np.zeros((n, 3, 34), np.float16) if save_ap else None

    keys = ('obs', 'unseen', 'tl_tokens', 'tl_mask')
    bs = 2048
    t0 = time.time()
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        sub = {k: batch[k][sl].to(device) for k in keys}
        aux = {k: batch[k][sl] for k in
               ('unseen', 'river', 'post_riichi_river', 'riichi', 'own_hand')}
        wb = T.wait_bias_of(aux).to(device)
        with torch.no_grad():
            h_b, g_b = net.backbone(sub['obs'])
            if getattr(net, 'tl_mode', 'pooled') == 'tile':
                e_T_b, e_tile = net.timeline.encode_tile(
                    sub['tl_tokens'], sub['tl_mask'],
                    batch['tl_turn'][sl].to(device), batch['tl_age'][sl].to(device),
                    batch['tl_consumed'][sl].to(device), batch['tl_flags'][sl].to(device))
                e_col = e_tile.transpose(-1, -2)
            else:
                e_T_b = net.timeline(sub['tl_tokens'], sub['tl_mask'])
                e_col = e_T_b[:, :, None, :].expand(-1, 3, 34, -1)
            B = sub['obs'].shape[0]
            z = torch.cat([g_b[:, None, :].expand(B, 3, Cc), e_T_b], -1)
            st = net.struct(z)
            s_counts = F.softmax(st['count_logits'], -1).flatten(-2)
            s_form = F.softmax(st['form_logits'], -1)
            if net.struct_shanten_belief:
                s_b = torch.cat([s_counts, s_form,
                                 F.softmax(st['shanten_logits'], -1)], -1)
            else:
                s_b = torch.cat([s_counts, s_form], -1)

            # H1（与 model_v2.ReadNetV2.forward 同口径）
            h_j = h_b.unsqueeze(1).expand(B, 3, Cc, 34).transpose(-1, -2)
            s_exp = s_b[:, :, None, :].expand(B, 3, 34, -1)
            x1 = torch.cat([h_j, e_col, s_exp], -1)
            x1 = F.gelu(net.h1(x1))
            logits = net.h1_out(x1)
            k_idx = torch.arange(5, device=device, dtype=sub['unseen'].dtype)
            cap = sub['unseen'][:, None, :, None]
            logits = logits.masked_fill(k_idx > cap, float('-inf'))
            ap = torch.nan_to_num(F.softmax(logits, -1), nan=0.0)
            ec = (ap * k_idx).sum(-1)

        h[sl] = h_b.cpu().numpy().astype(np.float16)
        g[sl] = g_b.cpu().numpy().astype(np.float16)
        e_T[sl] = e_T_b.cpu().numpy().astype(np.float16)
        s_belief[sl] = s_b.cpu().numpy().astype(np.float16)
        if save_ap:
            AP[sl] = ap.cpu().numpy().astype(np.float16)
            EC[sl] = ec.cpu().numpy().astype(np.float16)
        if (i // bs) % 5 == 0:
            print(f'[forward] {min(i+bs, n)}/{n}  {time.time()-t0:.0f}s', flush=True)

    hc, tg = C.parse_tl(batch['tl_tokens'].numpy(), batch['tl_mask'].numpy())
    decl_tile, decl_turn, decl_tg = _riichi_decl(batch)
    out = {
        'h': h, 'g': g, 'e_T': e_T, 's_belief': s_belief,
        'alloc': batch['alloc'].numpy().astype(np.int8),
        'unseen': batch['unseen'].numpy().astype(np.float16),
        'turn': batch['turn'].numpy().astype(np.int16),
        'riichi': (batch['riichi'].numpy() > 0.5),
        'called': (batch['called'].numpy() > 0.5),
        'hc': hc, 'tg': tg,
        'post_riichi_river': batch['post_riichi_river'].numpy().astype(bool),
        'riichi_decl_tile': decl_tile,
        'riichi_decl_turn': decl_turn,
        'riichi_decl_tg': decl_tg,
    }
    if save_ap:
        out['AP'] = AP
        out['EC'] = EC
    return out, {'n': n, 'C': Cc, 'dT': dT, 'S': S}


def _files(glob_pat: str) -> list[str]:
    files = sorted(glob.glob(glob_pat, recursive=True))
    if not files:
        raise SystemExit(f'没有文件匹配：{glob_pat}')
    return files


def _set_obs_order(ckpt: str) -> None:
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    D.set_obs_river_order(st.get('config', {}).get('obs_river_order', 'keep'))


def build_eval(args) -> None:
    files = _files(args.glob)
    _set_obs_order(args.ckpt)
    batch = T.build_fixed_eval_set(files, args.n, seed=args.seed)
    net = _load_net(Path(args.ckpt))
    print(f'[net] {args.ckpt}  device={C.DEV}', flush=True)
    out, meta = _extract(net, batch, C.DEV, save_ap=True)
    meta.update({'split': 'eval', 'glob': args.glob, 'seed': args.seed,
                 'n_files': len(files), 'ckpt': str(args.ckpt)})
    if args.check_alloc:
        ref = np.load(args.check_alloc, allow_pickle=True)
        ref_alloc = ref['alloc'][:out['alloc'].shape[0]]
        if not np.array_equal(ref_alloc.astype(np.int8), out['alloc']):
            bad = np.argwhere(ref_alloc.astype(np.int8) != out['alloc'])[:5]
            raise SystemExit(f'alloc 与参考 cache 不一致（前 5 个：{bad.tolist()}）'
                             '⇒ 评估集顺序不同，不能直接比较')
        print('[check] alloc 与参考 cache 逐位一致', flush=True)
    _save(args.out, 'eval', out, meta)


def build_trainval(args) -> None:
    files = _files(args.glob)
    rng = random.Random(args.file_seed)
    rng.shuffle(files)
    n_val_files = max(1, int(len(files) * args.val_frac))
    val_files = files[:n_val_files]
    train_files = files[n_val_files:]
    net = _load_net(Path(args.ckpt))
    _set_obs_order(args.ckpt)
    print(f'[net] {args.ckpt}  device={C.DEV}', flush=True)
    for split, fl, n in (('train', train_files, args.train_n),
                         ('val', val_files, args.val_n)):
        batch = T.build_fixed_eval_set(fl, n, seed=args.seed + (0 if split == 'train' else 1))
        out, meta = _extract(net, batch, C.DEV, save_ap=False)
        meta.update({'split': split, 'glob': args.glob, 'seed': args.seed,
                     'n_files': len(fl), 'ckpt': str(args.ckpt),
                     'file_seed': args.file_seed, 'val_frac': args.val_frac})
        _save(args.out, split, out, meta)


def _save(out_dir: str, split: str, out: dict, meta: dict) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f'{split}.npz', **out)
    (d / f'{split}.meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    print(f'[out] {d / (split + ".npz")}  n={meta["n"]}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('trainval', 'eval'), required=True)
    ap.add_argument('--glob', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default=os.environ['RM_CKPT'])
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--train-n', type=int, default=20000)
    ap.add_argument('--val-n', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--file-seed', type=int, default=20261102)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--check-alloc', default='')
    args = ap.parse_args()
    print(json.dumps(vars(args), ensure_ascii=False), flush=True)
    t0 = time.time()
    if args.mode == 'eval':
        build_eval(args)
    else:
        build_trainval(args)
    print(f'[done] {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
