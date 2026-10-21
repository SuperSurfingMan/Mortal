"""H2 头容量探针 v2（一次性，不入仓）。

问题：A2b 的 H2 输入只有 cat(g, e_T, s_belief, exp_counts)=240 维，**完全没有逐列
特征 h[B,96,34]**，也没有「候选牌 × 时序」的交互。这是「架构把信息抹掉了」还是
「信息本来就不在 h 里」？

做法：冻结 A2b 骨干/时序流/结构流，抽 h 与 z；在非评估月 202608 上训练小头
（80/20 切分做早停，选择规则对所有变体一致），在官方固定评估集
（202609 / 20000）上按官方口径比：
  P0  H2 的信息集（z 240 维）
  P1  + 逐列特征（logit_x += u·h_x + w·(q(z) ⊙ h_x)）
  P2  + A3 规则先验（logit_x += α_j·log q_rule(x)）
  P3  + 逐牌条件化时序注意力（候选牌 x 查询目标家动作序列）
  P4  全开
统一因子化输出：P(not tenpai)=sigmoid(s)（BCE 全目标），P(x|tenpai)=softmax34
（仅有效听牌目标 CE）；再用 p_x=P(t)P(x|t) 复原 35 维与官方指标逐位可比。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import (CKPT, DEV, TRAIN_GLOB, EPS, OUT_DIR, SUIT, VAL_GLOB, boot_ci, build_eval,
                     col_ce, counts_tier, fmt_ci, forward, load_net, parse_tl,
                     suit_window)

import sys, glob, json, time
import numpy as np
import torch
from torch import nn


import train as T
import model_v2 as M2
import rule_baseline as RB

DEV = 'cuda'
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
SEED = 20261019


def load_net():
    st = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = st['config']
    net = M2.build_model_v2s(conv_channels=cfg['conv_channels'],
                             num_blocks=cfg['num_blocks'],
                             with_aux=not cfg['no_aux'],
                             wait_head=cfg.get('wait_head', 'flat'),
                             tl_mode=cfg.get('tl_mode', 'pooled'),
                             next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
                             tile_order=cfg.get('tile_order', False)).to(DEV).eval()
    net.load_state_dict(st['model'])
    if net.tl_mode != 'pooled':
        raise SystemExit('s315_head_probe 手工重建 H1/H2 广播 e_T，仅支持 pooled；'
                         'tile 模式请用 order_metrics / evaluate 的常规通路')
    return net


def extract(net, files, n, tag):
    batch = T.build_fixed_eval_set(files, n)
    m = batch['obs'].shape[0]
    H = np.zeros((m, 96, 34), np.float16)
    G = np.zeros((m, 96), np.float16)
    ET = np.zeros((m, 3, 64), np.float16)
    SB = np.zeros((m, 3, 46), np.float16)
    EC = np.zeros((m, 3, 34), np.float16)
    RW = np.zeros((m, 3, 35), np.float32)
    WB = np.zeros((m, 3, 34), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, m, 1024):
            ch = {k: v[i:i + 1024].to(DEV) for k, v in batch.items()}
            wb = T.wait_bias_of(ch).to(DEV)
            b = ch['obs'].shape[0]
            h, g = net.backbone(ch['obs'])
            e_T = net.timeline(ch['tl_tokens'], ch['tl_mask'])
            z0 = torch.cat([g[:, None, :].expand(b, 3, net.C), e_T], -1)
            st_ = net.struct(z0)
            sb = torch.cat([torch.softmax(st_['count_logits'], -1).flatten(-2),
                            torch.softmax(st_['form_logits'], -1),
                            torch.softmax(st_['shanten_logits'], -1)], -1)
            o = net(ch['obs'], ch['unseen'], wb, ch['tl_tokens'], ch['tl_mask'])
            j = slice(i, i + b)
            H[j] = h.cpu().numpy().astype(np.float16)
            G[j] = g.cpu().numpy().astype(np.float16)
            ET[j] = e_T.cpu().numpy().astype(np.float16)
            SB[j] = sb.cpu().numpy().astype(np.float16)
            EC[j] = o['exp_counts'].cpu().numpy().astype(np.float16)
            RW[j] = o['wait_probs'].cpu().numpy()
            WB[j] = wb.cpu().numpy()
    bn = {k: v.numpy() for k, v in batch.items()}
    t1 = time.time()
    rule = RB.rule_predict(bn)['wait_probs'].astype(np.float32)
    print(f'[{tag}] n={m} 特征 {t1-t0:.0f}s + 规则 {time.time()-t1:.0f}s', flush=True)
    n3 = G.shape[0]
    z = np.concatenate([np.broadcast_to(G.astype(np.float32)[:, None, :], (n3, 3, 96)),
                        ET.astype(np.float32), SB.astype(np.float32),
                        EC.astype(np.float32)], -1)
    return {
        'h': H, 'z': z, 'wait_bias': WB, 'ref_wait': RW, 'rule_wait': rule,
        'tl_tokens': bn['tl_tokens'].astype(np.float16), 'tl_mask': bn['tl_mask'],
        'wait': bn['wait'], 'tenpai': bn['tenpai'], 'riichi': bn['riichi'],
    }


class Probe(nn.Module):
    def __init__(self, dz=240, dcol=96, use_h=False, use_rule=False, use_tl=False,
                 hidden=128, dtl=32):
        super().__init__()
        self.use_h, self.use_rule, self.use_tl = use_h, use_rule, use_tl
        self.trunk = nn.Sequential(nn.Linear(dz, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.tile_logit = nn.Linear(hidden, 34)
        self.notp = nn.Linear(hidden, 1)          # 未听 logit（口径与 A2b 一致）
        if use_h:
            self.q = nn.Linear(hidden, dcol)
            self.u = nn.Linear(dcol, 1)
            self.w = nn.Linear(dcol, 1)
        if use_rule:
            self.alpha = nn.Parameter(torch.zeros(3))
        if use_tl:
            self.tl_k = nn.Linear(43, dtl)
            self.tl_v = nn.Linear(43, dtl)
            self.tl_q = nn.Linear(dcol, dtl)
            self.tl_o = nn.Linear(dtl, 1)
            self.dtl = dtl

    def forward(self, z, h, rule_lp, tl, tlm):
        B = z.shape[0]
        hj = h.permute(0, 2, 1)                              # [B,34,96]
        out = torch.empty(B, 3, 35, device=z.device)
        for j in range(3):
            f = self.trunk(z[:, j, :])
            lg = self.tile_logit(f)
            if self.use_h:
                q = self.q(f)[:, None, :]
                lg = lg + self.u(hj).squeeze(-1) + self.w(hj * q).squeeze(-1)
            if self.use_tl:
                K = self.tl_k(tl[:, j]); V = self.tl_v(tl[:, j])
                Q = self.tl_q(hj)
                a = (Q @ K.transpose(1, 2)) / (self.dtl ** 0.5)
                a = a.masked_fill(~tlm[:, j][:, None, :], -1e9)
                lg = lg + self.tl_o(torch.softmax(a, -1) @ V).squeeze(-1)
            if self.use_rule:
                lg = lg + self.alpha[j] * rule_lp[:, j, :]
            out[:, j, :34] = lg
            out[:, j, 34] = self.notp(f).squeeze(-1)
        return out


def metrics(lg35, F):
    """lg35: 未加 bias 的 [n,3,35]（第 35 = 未听 logit）。按官方口径复算。"""
    wb = np.where(np.isneginf(F['wait_bias']), -1e9, F['wait_bias']).astype(np.float64)
    lg = lg35.astype(np.float64)
    lg[..., :34] += wb
    p_not = 1.0 / (1.0 + np.exp(-lg[..., 34]))              # P(未听)
    p_t = 1.0 - p_not
    mx = lg[..., :34].max(-1, keepdims=True)
    cond = np.exp(lg[..., :34] - mx)
    cond /= cond.sum(-1, keepdims=True)
    p = np.concatenate([p_t[..., None] * cond, p_not[..., None]], -1)

    WT = F['wait'].astype(np.float64)
    tenpai = F['tenpai'] > 0.5
    riichi = F['riichi'] > 0.5
    sup = WT[..., :34] > 0
    eff = sup.any(-1)
    ll = -(WT * np.where(WT > 0, np.log(np.clip(p, 1e-12, None)), 0)).sum(-1)
    ll_c = -(WT[..., :34] * np.where(WT[..., :34] > 0, np.log(np.clip(cond, 1e-12, None)),
                                     0)).sum(-1)
    order = np.argsort(-cond, -1)
    hit1 = np.take_along_axis(sup, order[..., :1], -1)[..., 0]
    hit5 = np.take_along_axis(sup, order[..., :5], -1).any(-1)

    def auc(y, sc):
        y = np.asarray(y).ravel().astype(bool); sc = np.asarray(sc).ravel()
        o = np.argsort(sc, kind='mergesort'); r = np.empty(len(sc))
        r[o] = np.arange(1, len(sc) + 1)
        npos = y.sum(); nneg = len(y) - npos
        return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)

    return {'cond_ce': float(ll_c[eff].mean()),
            'wait_ce_active': float(ll[eff].mean()),
            'hit1': float(hit1[eff].mean()), 'hit5': float(hit5[eff].mean()),
            'riichi_hit1': float(hit1[riichi & eff].mean()),
            'riichi_ce': float(ll[riichi & eff].mean()),
            'tenpai_auc': float(auc(tenpai, p_t)),
            'meanP(t)': float(p_t.mean()),
            'meanP(t)|not': float(p_t[~tenpai].mean())}


def show(name, r):
    print(f'  {name:<20s} ' + '  '.join(f'{k}={v:.4f}' for k, v in r.items()), flush=True)


def train_probe(Ftr, Fva, Fev, flags, name, epochs=30, lr=1e-3, bs=2048):
    torch.manual_seed(SEED)
    pr = Probe(**flags).to(DEV)
    opt = torch.optim.AdamW(pr.parameters(), lr=lr, weight_decay=1e-3)

    def to_dev(F, keys, sel=None):
        out = {}
        for k in keys:
            a = F[k] if sel is None else F[k][sel]
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a.astype(np.float32) if a.dtype == np.float16 else a)
            out[k] = a.to(DEV)
        return out

    keys = ('z', 'h', 'tl_tokens', 'tl_mask', 'wait_bias', 'wait', 'tenpai')
    tr = to_dev(Ftr, keys)
    tr['rule_lp'] = torch.from_numpy(
        np.log(np.clip(Ftr['rule_wait'][..., :34], 1e-9, None))).to(DEV)
    va = to_dev(Fva, keys)
    va['rule_lp'] = torch.from_numpy(
        np.log(np.clip(Fva['rule_wait'][..., :34], 1e-9, None))).to(DEV)
    n = tr['z'].shape[0]
    WB35 = torch.cat([tr['wait_bias'], torch.zeros_like(tr['wait_bias'][:, :, :1])], -1)
    sup = tr['wait'][..., :34] > 0
    eff = sup.any(-1)
    best = (1e9, None, 0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n - bs + 1, bs):
            ix = perm[i:i + bs]
            lg = pr(tr['z'][ix], tr['h'][ix], tr['rule_lp'][ix],
                    tr['tl_tokens'][ix], tr['tl_mask'][ix]) + WB35[ix]
            lc = torch.log_softmax(lg[..., :34], -1)
            w = tr['wait'][ix, ..., :34]
            ce = -(w * torch.where(w > 0, lc, torch.zeros_like(lc))).sum(-1)
            ce = (ce * eff[ix]).sum() / eff[ix].sum().clamp(min=1)
            bce = nn.functional.binary_cross_entropy_with_logits(
                lg[..., 34], 1.0 - tr['tenpai'][ix])
            loss = ce + 0.2 * bce
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            lgv = pr(va['z'], va['h'], va['rule_lp'],
                     va['tl_tokens'], va['tl_mask']).cpu().numpy()
        rv = metrics(lgv, Fva)
        if rv['cond_ce'] < best[0]:
            best = (rv['cond_ce'], {k: v.detach().cpu().clone() for k, v in
                                    pr.state_dict().items()}, ep + 1)
    pr.load_state_dict(best[1])
    ev = to_dev(Fev, keys)
    ev['rule_lp'] = torch.from_numpy(
        np.log(np.clip(Fev['rule_wait'][..., :34], 1e-9, None))).to(DEV)
    with torch.no_grad():
        lge = pr(ev['z'], ev['h'], ev['rule_lp'],
                 ev['tl_tokens'], ev['tl_mask']).cpu().numpy()
    r = metrics(lge, Fev)
    r['best_ep'] = best[2]
    show(name, r)
    return r


def main():
    net = load_net()
    f_tr = sorted(glob.glob(TRAIN_GLOB, recursive=True))
    f_ev = sorted(glob.glob(VAL_GLOB, recursive=True))
    Fall = extract(net, f_tr, N_TRAIN, 'train 202608')
    Fev = extract(net, f_ev, N_EVAL, 'eval 202609')
    del net
    # 80/20 切分做早停（同月同分布，选择规则对所有变体一致）
    m = Fall['z'].shape[0]
    cut = int(m * 0.8)
    Ftr = {k: v[:cut] for k, v in Fall.items()}
    Fva = {k: v[cut:] for k, v in Fall.items()}
    print(f'[split] fit={cut} valid={m-cut}', flush=True)

    print('\n== 参照（同评估集，官方口径） ==', flush=True)
    for nm, key in (('A2b(H2 原样)', 'ref_wait'), ('A3-rule', 'rule_wait')):
        p = Fev[key].astype(np.float64)
        WT = Fev['wait'].astype(np.float64); tenpai = Fev['tenpai'] > 0.5
        riichi = Fev['riichi'] > 0.5
        sup = WT[..., :34] > 0; eff = sup.any(-1)
        ll = -(WT * np.where(WT > 0, np.log(np.clip(p, 1e-12, None)), 0)).sum(-1)
        o_ = np.argsort(-p[..., :34], -1)
        h1 = np.take_along_axis(sup, o_[..., :1], -1)[..., 0]
        h5 = np.take_along_axis(sup, o_[..., :5], -1).any(-1)
        y = tenpai.ravel().astype(bool); sc = (1 - p[..., 34]).ravel()
        oo = np.argsort(sc, kind='mergesort'); r = np.empty(len(sc)); r[oo] = np.arange(1, len(sc) + 1)
        auc = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))
        print(f'  {nm:<20s} ce={ll[eff].mean():.4f} hit1={h1[eff].mean():.4f} '
              f'hit5={h5[eff].mean():.4f} riichi_hit1={h1[riichi & eff].mean():.4f} '
              f'auc={auc:.4f} meanP(t)={sc.mean():.4f}', flush=True)

    out = {}
    variants = [
        ('P0_z_only', dict(use_h=False, use_rule=False, use_tl=False)),
        ('P1_z_plus_h', dict(use_h=True, use_rule=False, use_tl=False)),
        ('P2_z_h_rule', dict(use_h=True, use_rule=True, use_tl=False)),
        ('P3_z_h_tl', dict(use_h=True, use_rule=False, use_tl=True)),
        ('P4_all', dict(use_h=True, use_rule=True, use_tl=True)),
    ]
    for nm, fl in variants:
        print(f'\n== {nm} ==', flush=True)
        out[nm] = train_probe(Ftr, Fva, Fev, fl, nm)
    json.dump(out, open(str(OUT_DIR / 'probe.json'), 'w'), indent=1)
    print('\n[done]')


if __name__ == '__main__':
    main()
