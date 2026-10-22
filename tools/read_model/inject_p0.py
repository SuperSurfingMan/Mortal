#!/usr/bin/env python3
"""S6b 注入 P0 前置实验：主模型 backbone 读牌探针 + 决策分歧 + 决策级 deal 指标。

## 为什么先做这个（而不是直接 30k 注入探针）

读牌模型的输入 obs 与主训练管线是**同一个** ``encode_obs(version=4)``（零新增通道），
所以「注入读牌后验」在信息论上不引入新信息，只引入一个「主模型可能没学到的函数」。
S6a（6 行紧迫度）已经证明「同信息注入」可以是零增量。因此在烧 30k 步之前，
必须先回答三个可离线回答的问题：

- **P0-A 信息是否已在主模型表示里**：冻结主模型 ``Brain``，取 ``obs -> phi(1024)``，
  只在 phi 上训练线性/MLP 探针预测三家暗牌计数与听牌/待牌；与 A2c 在同一批样本上
  的指标对比。探针 ≈ A2c ⇒ 主模型已隐含读牌，注入上界 ≈ 0。
- **P0-B 信息是否有决策价值**：主模型舍牌 top-1 / A2c 最安全牌 / 人类实际舍牌
  三方对比——分歧率、危险度差、真实点炮率差（用 wait 真值）。
- **P0-C 判据是否可信**：``deal_pure`` 在官方 20000 集上 n_deal 仅 1470（≈16 个
  点炮事件），本脚本报**按文件 block bootstrap** 的 CI，并支持 ``--n`` 扩容。

## 协议

- 评估集：202609 holdout，seed 20261016，``train.build_fixed_eval_set`` 同协议
  （顺序截取；n=20000 时与官方集仅差 linspace 抽稀的尾部 ≤1 个文件。A2c 指标在本
  脚本内**同批重算**，不引用官方数字，保证同口径可比）。
- 主模型：判定基线 B/756k（``reward_ablation/B_rank_pt_aligned/mortal.pth``）。
- 探针 split 按**文件**分组（同局样本不跨 split），避免同一局泄漏。
- 流式构建：每 ``--chunk`` 个样本构建即推理，内存峰值 ≈ chunk × 137 KB。
- 只读：不改 ``mortal/``、``libriichi/``、读牌 ckpt；产物写 ``diagnostics/results/``。

用法::

    python tools/read_model/inject_p0.py --n 20000
    python tools/read_model/inject_p0.py --n 100000 --probe-n 20000   # 扩容 deal 判据
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))
sys.path.insert(0, str(HERE))
import _common as C  # noqa: E402  路径层 + bootstrap_paths()

# ⚠ 两个 model.py 同名：``_common.bootstrap_paths()`` 会把 ``mortal/`` 插到 sys.path
# 最前面（脚本目录已存在时它不会再插 tools/read_model），于是裸 ``import model``
# 会命中主模型的 ``mortal/model.py``，而读牌 ``model_v2.py`` 要的是
# ``tools/read_model/model.py`` 里的 ReadBackbone。这里显式占住 'model' 名字，
# 主模型则用独立名字 'mortal_model' 按绝对路径加载。
import importlib.util as _ilu  # noqa: E402


def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_read_model = _load_module('model', HERE / 'model.py')

import dataset as D           # noqa: E402
T = _load_module('read_train', HERE / 'train.py')   # 同上：避免命中 mortal/train.py
import model_v2 as M2         # noqa: E402
from evaluate import auc_score  # noqa: E402

MM = _load_module('mortal_model', C.MORTAL_DIR / 'model.py')

NUM_TILES = D.NUM_TILES
ACTION_SPACE = 46
DEV = torch.device(C.DEV)
A2C_CKPT = Path('/mnt/data8/mortal/Mortal/mortal/read_model_v2s_a2c/best.pth')
BASE_CKPT = Path('/mnt/data8/mortal/Mortal/mortal/checkpoints/'
                 'reward_ablation/B_rank_pt_aligned/mortal.pth')
INF = float('inf')


# --------------------------------------------------------------------------- #
# 数据：流式构建（不把 100k × 137KB 的 obs 全留在内存）
# --------------------------------------------------------------------------- #
def iter_eval_chunks(files: list[str], n: int, seed: int, chunk: int):
    """按官方协议（固定 shuffle + 固定回放）流式产出 (samples, srcs)。"""
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)
    buf, src = [], []
    total = 0
    for f in files:
        try:
            ss, _ = D.replay_file(f, augment=False)
        except Exception as exc:                       # 坏文件不中断
            print(f'[eval] skip {f}: {exc}', flush=True)
            continue
        for s in ss:
            s['_src'] = f
            buf.append(s)
            src.append(f)
            total += 1
        while len(buf) >= chunk:
            yield buf[:chunk], src[:chunk]
            buf, src = buf[chunk:], src[chunk:]
        if total >= n:
            break
    if buf:
        yield buf, src


def a2c_forward(net, batch, bs: int = 1024, zero_tl: bool = False):
    """读牌 ckpt 前向：返回 (alloc_probs[n,3,34,5], wait_probs[n,3,35])。

    ``zero_tl``：把 tl_tokens 与 tl_mask 全部置零（P0-D）。用于回答「如果只拿
    主模型 obs（主训练管道不提供 tl_tokens 侧车）去跑 A2c，还能剩下多少能力」。
    """
    n = batch['obs'].shape[0]
    AP = torch.zeros(n, 3, NUM_TILES, 5)
    WP = torch.zeros(n, 3, NUM_TILES + 1)
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        sub = {k: batch[k][sl] for k in
               ('obs', 'unseen', 'river', 'post_riichi_river', 'riichi',
                'own_hand', 'tl_tokens', 'tl_mask')}
        if zero_tl:
            sub['tl_tokens'] = torch.zeros_like(sub['tl_tokens'])
            sub['tl_mask'] = torch.zeros_like(sub['tl_mask'])
        if getattr(net, 'tl_mode', 'pooled') == 'tile':
            for k in ('tl_turn', 'tl_age', 'tl_consumed', 'tl_flags'):
                sub[k] = batch[k][sl]
        with torch.no_grad():
            o = T.forward_batch(net, sub, DEV)
        AP[sl] = o['alloc_probs'].float().cpu()
        WP[sl] = o['wait_probs'].float().cpu()
    return AP, WP


def mortal_forward(mortal, dqn, batch, bs: int = 512):
    """主模型前向：返回 (phi[n,1024], q[n,46])；mask 只放开舍牌动作 0..33。"""
    n = batch['obs'].shape[0]
    PHI = torch.zeros(n, 1024)
    Q = torch.full((n, ACTION_SPACE), -INF)
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        obs = batch['obs'][sl].to(DEV)
        own = batch['own_hand'][sl].to(DEV) > 0
        mask = torch.zeros(obs.shape[0], ACTION_SPACE, dtype=torch.bool, device=DEV)
        mask[:, :NUM_TILES] = own
        with torch.no_grad():
            phi = mortal(obs)
            q = dqn(phi, mask)
        PHI[sl] = phi.float().cpu()
        Q[sl] = q.float().cpu()
    return PHI, Q


# --------------------------------------------------------------------------- #
# P0-A：冻结 backbone 探针
# --------------------------------------------------------------------------- #
class Probe(nn.Module):
    """``phi(1024) -> 三家暗牌 counts(3,34,5) + 待牌 logits(3,35)``。"""

    def __init__(self, d: int = 1024, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.trunk = nn.Identity() if hidden <= 0 else nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        out_dim = hidden if hidden > 0 else d
        self.alloc = nn.Linear(out_dim, 3 * NUM_TILES * 5)
        self.wait = nn.Linear(out_dim, 3 * (NUM_TILES + 1))

    def forward(self, phi):
        z = self.trunk(phi)
        return (self.alloc(z).view(-1, 3, NUM_TILES, 5),
                self.wait(z).view(-1, 3, NUM_TILES + 1))


def factorized_wait(logits):
    """与 ``model.factorized_wait_probs`` 同语义：p34 = sigmoid(l34)。"""
    p34 = torch.sigmoid(logits[..., NUM_TILES])
    cond = torch.softmax(logits[..., :NUM_TILES], dim=-1)
    p = (1.0 - p34)[..., None] * cond
    return torch.cat([p, p34[..., None]], dim=-1)


def eval_read_metrics(alloc_p, wait_p, alloc_t, wait_t):
    """读牌侧指标（与 evaluate.py / strat_metrics.py 同口径）。"""
    eps = 1e-12
    a_p = np.clip(alloc_p, eps, None)
    a_t = alloc_t.astype(np.int64)
    ce = -np.take_along_axis(np.log(a_p), a_t[..., None], -1)[..., 0]
    top1 = (alloc_p.argmax(-1) == a_t).mean(-1)
    exp_cnt = (alloc_p * np.arange(5)).sum(-1)
    mae = np.abs(exp_cnt - a_t).mean(-1)

    w_p = np.clip(wait_p, eps, None)
    supp = wait_t[..., :NUM_TILES] > 0
    eff = supp.any(-1)
    wce = -(wait_t * np.log(w_p)).sum(-1)
    p_t = np.clip(1.0 - wait_p[..., NUM_TILES], eps, None)
    cond = wait_p[..., :NUM_TILES] / p_t[..., None]
    cond_ce = -(wait_t[..., :NUM_TILES] * np.log(np.clip(cond, eps, None))).sum(-1)
    topk = np.argsort(-wait_p[..., :NUM_TILES], -1)
    hit1 = np.take_along_axis(supp, topk[..., :1], -1)[..., 0]
    hit3 = np.take_along_axis(supp, topk[..., :3], -1).any(-1)
    hit5 = np.take_along_axis(supp, topk[..., :5], -1).any(-1)
    tenpai = wait_t[..., NUM_TILES] < 0.5
    score = 1.0 - wait_p[..., NUM_TILES]

    return {
        'alloc_ce': float(ce.mean()),
        'alloc_top1': float(top1.mean()),
        'alloc_mae': float(mae.mean()),
        'wait_ce': float(wce[eff].mean()) if eff.any() else float('nan'),
        'wait_nlt': float((-np.log(p_t))[eff].mean()) if eff.any() else float('nan'),
        'wait_cond_ce': float(cond_ce[eff].mean()) if eff.any() else float('nan'),
        'hit1': float(hit1[eff].mean()) if eff.any() else float('nan'),
        'hit3': float(hit3[eff].mean()) if eff.any() else float('nan'),
        'hit5': float(hit5[eff].mean()) if eff.any() else float('nan'),
        'tenpai_auc': float(auc_score(tenpai.ravel(), score.ravel())),
        'n_targets': int(tenpai.size),
        'n_active': int(eff.sum()),
    }


def train_probe(phi_tr, alloc_tr, wait_tr, phi_va, alloc_va, wait_va,
                hidden: int = 0, epochs: int = 200, bs: int = 512,
                lr: float = 1e-3, wd: float = 1e-2, dropout: float = 0.0):
    """训练冻结表示上的探针；按 val 的 alloc_ce + wait_ce 早停。"""
    net = Probe(phi_tr.shape[1], hidden, dropout).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    a_tr = torch.as_tensor(alloc_tr, dtype=torch.long, device=DEV)
    w_tr = torch.as_tensor(wait_tr, dtype=torch.float32, device=DEV)
    a_va = torch.as_tensor(alloc_va, dtype=torch.long, device=DEV)
    w_va = torch.as_tensor(wait_va, dtype=torch.float32, device=DEV)
    n = phi_tr.shape[0]
    best = (INF, None)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lg_a, lg_w = net(phi_tr[idx])
            loss = F.cross_entropy(lg_a.reshape(-1, 5), a_tr[idx].reshape(-1))
            loss = loss + (-(w_tr[idx] * torch.log(
                factorized_wait(lg_w).clamp_min(1e-12))).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                lg_a, lg_w = net(phi_va)
                va_a = F.cross_entropy(lg_a.reshape(-1, 5), a_va.reshape(-1))
                va_w = (-(w_va * torch.log(
                    factorized_wait(lg_w).clamp_min(1e-12))).sum(-1)).mean()
                score = float(va_a + va_w)
            if score < best[0]:
                best = (score, {k: v.detach().clone()
                                for k, v in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    return net, best[0]


def _agg(metrics):
    """对一组指标 dict 求 mean / std（忽略 nan）。"""
    keys = metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics], dtype=np.float64)
        ok = np.isfinite(vals)
        out[k] = float(vals[ok].mean()) if ok.any() else float('nan')
    std = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics], dtype=np.float64)
        ok = np.isfinite(vals)
        std[k] = float(vals[ok].std(ddof=1)) if ok.sum() > 1 else float('nan')
    return out, std


def run_probe(phi, alloc, wait, srcs, args):
    """P0-A：**按文件 K-fold** 交叉验证（同局样本不跨 split）。

    单次 split 在文件级分布差异下极不稳定（同场半庄的牌风/牌山相关）：
    实测单 split 的 train hit@1 0.42 / val 0.08。K-fold 报 mean±std，才能区分
    「phi 里没有信息」与「一次 split 刚好泛化失败」。
    """
    srcs = np.asarray(srcs)
    uniq = np.unique(srcs)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    k = max(2, args.probe_folds)
    folds = [f for f in np.array_split(rng.permutation(uniq), k) if len(f)]
    phi_dev = phi.to(DEV)
    print(f'[P0-A] 探针 {len(folds)}-fold by file: {len(uniq)} 文件 / {len(srcs)} 样本',
          flush=True)

    results = {}
    for name, hidden, dropout in (('linear', 0, 0.0), ('mlp128', 128, 0.1)):
        t0 = time.time()
        fold_val, fold_tr = [], []
        for fi, va_files in enumerate(folds):
            va_set = set(va_files.tolist())
            va_idx = np.array([i for i, s in enumerate(srcs) if s in va_set])
            tr_idx = np.array([i for i, s in enumerate(srcs) if s not in va_set])
            net, _ = train_probe(
                phi_dev[torch.as_tensor(tr_idx, device=DEV)],
                alloc[tr_idx], wait[tr_idx],
                phi_dev[torch.as_tensor(va_idx, device=DEV)],
                alloc[va_idx], wait[va_idx],
                hidden=hidden, epochs=args.probe_epochs, lr=args.probe_lr,
                wd=args.probe_wd, dropout=dropout)
            with torch.no_grad():
                lg_a, lg_w = net(phi_dev[torch.as_tensor(va_idx, device=DEV)])
                vp_a = torch.softmax(lg_a, -1).cpu().numpy()
                vp_w = factorized_wait(lg_w).cpu().numpy()
                lg_a_tr, lg_w_tr = net(phi_dev[torch.as_tensor(tr_idx, device=DEV)])
                tp_a = torch.softmax(lg_a_tr, -1).cpu().numpy()
                tp_w = factorized_wait(lg_w_tr).cpu().numpy()
            fold_val.append(eval_read_metrics(vp_a, vp_w,
                                              alloc[va_idx], wait[va_idx]))
            fold_tr.append(eval_read_metrics(tp_a, tp_w,
                                             alloc[tr_idx], wait[tr_idx]))
        vm, vs = _agg(fold_val)
        tm, _ = _agg(fold_tr)
        results[name] = {'folds': fold_val, 'val': vm, 'val_std': vs,
                         'train': tm, 'elapsed_s': round(time.time() - t0, 1)}
        print(f'  [{name:7s}] val alloc_ce {vm["alloc_ce"]:.4f}±{vs["alloc_ce"]:.4f} '
              f'hit1 {vm["hit1"]:.4f}±{vs["hit1"]:.4f} '
              f'AUC {vm["tenpai_auc"]:.4f}±{vs["tenpai_auc"]:.4f} '
              f'(train hit1 {tm["hit1"]:.4f})  {time.time()-t0:.0f}s', flush=True)
    # CV 下每个样本都当过 val；用全量索引让 A2c 对照同分布
    return results, np.arange(len(srcs)), np.arange(len(srcs))


# --------------------------------------------------------------------------- #
# P0-B：决策分歧（主模型 / A2c / 人类）
# --------------------------------------------------------------------------- #
_EV_CACHE: dict[str, list] = {}


def load_events(path: str):
    """mjai 日志是 JSONL（每行一个事件），不是单个 JSON 文档。"""
    if path not in _EV_CACHE:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            _EV_CACHE[path] = [json.loads(ln) for ln in f if ln.strip()]
    return _EV_CACHE[path]


def human_discard(events, ev_idx: int, observer: int):
    """从事件流取观察者摸牌后的下一张舍牌（跳过副露等，遇局末返回 None）。"""
    for k in range(ev_idx + 1, min(ev_idx + 200, len(events))):
        e = events[k]
        et = e.get('type')
        if et == 'dahai' and e.get('actor') == observer:
            return D.tile_id(e['pai'])
        if et in ('hora', 'ryukyoku', 'end_kyoku'):
            return None
    return None


def run_divergence(wait_p, q, danger, own, riichi_any, turn, meta, srcs):
    """P0-B：同一批样本上比 主模型 top-1 / A2c 最安全 / 人类实际。"""
    n = wait_p.shape[0]
    risk = wait_p[..., :NUM_TILES].sum(1)                    # A2c 三家 union 危险度
    a2c_tile = np.where(own, risk, INF).argmin(-1)
    qq = q[:, :NUM_TILES].copy()
    qq[~own] = -INF
    mm_tile = qq.argmax(-1)

    hum = np.full(n, -1, dtype=np.int64)
    for i, (src, ev, obs) in enumerate(zip(meta['src'], meta['event'],
                                           meta['observer'])):
        t = human_discard(load_events(src), ev, obs)
        if t is not None and t < NUM_TILES:
            hum[i] = t
    hum_ok = hum >= 0

    def deal_rate(tile, sel):
        d = danger[np.arange(n), tile]
        return float(d[sel].mean()) if sel.any() else float('nan')

    agree = (mm_tile == a2c_tile)
    delta_risk = risk[np.arange(n), mm_tile] - risk[np.arange(n), a2c_tile]
    mm_deal = danger[np.arange(n), mm_tile]
    a2c_deal = danger[np.arange(n), a2c_tile]
    fixable = (~agree) & mm_deal & (~a2c_deal)
    fix_src = np.asarray(srcs)[fixable]
    fix_files = np.unique(fix_src)
    all_files = len(np.unique(srcs))

    out = {
        'n': int(n),
        'agree_rate': float(agree.mean()),
        'delta_risk_mean': float(delta_risk.mean()),
        'delta_risk_p50': float(np.percentile(delta_risk, 50)),
        'delta_risk_p90': float(np.percentile(delta_risk, 90)),
        'deal_top1': deal_rate(mm_tile, np.ones(n, bool)),
        'deal_a2c_safest': deal_rate(a2c_tile, np.ones(n, bool)),
        'deal_human': deal_rate(hum, hum_ok),
        'n_human': int(hum_ok.sum()),
        'fixable_rate': float(fixable.mean()),
        'n_fixable': int(fixable.sum()),
        'fixable_files': int(len(fix_files)),
        'n_files': int(all_files),
        'fixable_per_file_mean': float(fixable.sum() / max(all_files, 1)),
        'by_scene': {},
    }
    for tag, sel in (('riichi_any', riichi_any), ('no_riichi', ~riichi_any),
                     ('turn>=8', turn >= 8), ('turn<8', turn < 8)):
        if not sel.any():
            continue
        out['by_scene'][tag] = {
            'n': int(sel.sum()),
            'agree_rate': float(agree[sel].mean()),
            'delta_risk_mean': float(delta_risk[sel].mean()),
            'deal_top1': deal_rate(mm_tile, sel),
            'deal_a2c_safest': deal_rate(a2c_tile, sel),
            'deal_human': deal_rate(hum, sel & hum_ok),
            'n_human': int((sel & hum_ok).sum()),
            'fixable_rate': float(fixable[sel].mean()),
            'n_fixable': int(fixable[sel].sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# P0-C：deal_pure + 文件 block bootstrap
# --------------------------------------------------------------------------- #
def deal_pure(wait_p, wait_t, own):
    """strat_metrics.py 口径：逐目标家选自家最安全牌，看点炮率。"""
    n = wait_p.shape[0]
    tile_p = wait_p[..., :NUM_TILES]
    own3 = np.broadcast_to(own[:, None, :], (n, 3, NUM_TILES))
    safest = np.where(own3, tile_p, INF).argmin(-1)          # (n,3)
    supp = wait_t[..., :NUM_TILES] > 0
    deal = np.take_along_axis(supp, safest[..., None], -1)[..., 0]
    p_safe = np.take_along_axis(tile_p, safest[..., None], -1)[..., 0]
    eff = supp.any(-1)
    pure = eff & (p_safe > 0)
    return deal, pure, safest


def block_boot_mean(vals, groups, n_boot=400, seed=0):
    """按文件（cluster）重采样的均值 CI。"""
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[g] for g in pick])
        means[b] = vals[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_deal(wait_p, wait_t, own, riichi_any, turn, srcs):
    deal, pure, safest = deal_pure(wait_p, wait_t, own)
    flat = deal[pure]
    grp = np.broadcast_to(np.asarray(srcs)[:, None], pure.shape)[pure]
    lo, hi = block_boot_mean(flat.astype(np.float64), grp)
    out = {
        'deal_pure': float(flat.mean()),
        'n_deal': int(pure.sum()),
        'n_deal_samples': int(pure.any(1).sum()),
        'ci95_block': [lo, hi],
        'by_scene': {},
    }
    for tag, sel in (('riichi', riichi_any), ('no_riichi', ~riichi_any),
                     ('turn>=8', turn >= 8), ('turn<8', turn < 8)):
        m = pure & sel[:, None]
        if m.any():
            g = np.broadcast_to(np.asarray(srcs)[:, None], m.shape)[m]
            l, h = block_boot_mean(deal[m].astype(np.float64), g)
            out['by_scene'][tag] = {'deal_pure': float(deal[m].mean()),
                                    'n_deal': int(m.sum()),
                                    'ci95_block': [l, h]}
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000,
                    help='评估样本数（官方 20000；扩容用 100000）')
    ap.add_argument('--probe-n', type=int, default=0,
                    help='P0-A/B 用的样本数（0=全部；扩容时建议 20000）')
    ap.add_argument('--chunk', type=int, default=2000, help='流式构建块大小')
    ap.add_argument('--val-glob', default=str(
        Path('/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609/**/*.json.gz')))
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--probe-train-frac', type=float, default=0.7,
                    help='（已由 --probe-folds 取代，保留兼容）')
    ap.add_argument('--skip-tl-probe', action='store_true',
                    help='跳过 P0-D（零 tl_tokens 的读牌精度）')
    ap.add_argument('--probe-folds', type=int, default=5,
                    help='按文件 K-fold 交叉验证折数')
    ap.add_argument('--probe-epochs', type=int, default=120)
    ap.add_argument('--probe-lr', type=float, default=1e-3)
    ap.add_argument('--probe-wd', type=float, default=5e-2)
    ap.add_argument('--out', default=str(C.OUT_DIR / 'p0_inject_20k.json'))
    args = ap.parse_args()

    t0 = time.time()
    print(f'=== S6b P0 实验 | n={args.n} seed={args.seed} dev={DEV} ===', flush=True)
    st = torch.load(A2C_CKPT, map_location='cpu', weights_only=False)
    D.set_obs_river_order(st.get('config', {}).get('obs_river_order', 'keep'))
    files = sorted(glob.glob(args.val_glob, recursive=True))

    # ---- 模型 ----
    cfg = st['config']
    a2c = M2.build_model_v2s(conv_channels=cfg['conv_channels'],
                             num_blocks=cfg['num_blocks'],
                             with_aux=not cfg['no_aux'],
                             wait_head=cfg.get('wait_head', 'flat'),
                             tl_mode=cfg.get('tl_mode', 'pooled'),
                             next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
                             tile_order=cfg.get('tile_order', False)).to(DEV).eval()
    a2c.load_state_dict(st['model'])
    mst = torch.load(BASE_CKPT, map_location='cpu', weights_only=False)
    mcfg = mst['config']
    ver = mcfg['control']['version']
    mortal = MM.Brain(version=ver, **mcfg['resnet']).to(DEV).eval()
    mortal.load_state_dict(mst['mortal'])
    dqn = MM.DQN(version=ver).to(DEV).eval()
    dqn.load_state_dict(mst['current_dqn'])
    print(f'[model] A2c {sum(p.numel() for p in a2c.parameters())/1e6:.2f}M | '
          f'主模型 {sum(p.numel() for p in mortal.parameters())/1e6:.2f}M '
          f'(B/756k, v{ver})', flush=True)

    # ---- 流式推理 ----
    META_KEYS = ('alloc', 'wait', 'own_hand', 'riichi', 'turn')
    buf_meta = {k: [] for k in META_KEYS}
    srcs_all, event_all, obs_all = [], [], []
    AP_list, WP_list, PHI_list, Q_list = [], [], [], []
    AP0_list, WP0_list = [], []
    total = 0
    for buf, src_buf in iter_eval_chunks(files, args.n, args.seed, args.chunk):
        take = min(len(buf), args.n - total)
        if take <= 0:
            break
        buf, src_buf = buf[:take], src_buf[:take]
        b = T.collate(buf)
        ap, wp = a2c_forward(a2c, b, args.batch_size)
        phi, q = mortal_forward(mortal, dqn, b, 512)
        AP_list.append(ap)
        WP_list.append(wp)
        if not args.skip_tl_probe:                      # P0-D：零 tl_tokens
            ap0, wp0 = a2c_forward(a2c, b, args.batch_size, zero_tl=True)
            AP0_list.append(ap0)
            WP0_list.append(wp0)
        PHI_list.append(phi)
        Q_list.append(q)
        for k in META_KEYS:
            buf_meta[k].append(b[k].numpy())
        srcs_all.extend(src_buf)
        event_all.extend([int(s['_event']) for s in buf])
        obs_all.extend([int(s['_observer']) for s in buf])
        total += take
        print(f'[infer] {total}/{args.n}  {time.time()-t0:.0f}s', flush=True)
        if total >= args.n:
            break

    AP = torch.cat(AP_list)
    WP = torch.cat(WP_list)
    PHI = torch.cat(PHI_list)
    Q = torch.cat(Q_list)
    meta = {k: np.concatenate(v) for k, v in buf_meta.items()}
    srcs = np.asarray(srcs_all)
    print(f'[data] {total} 样本 / {len(set(srcs_all))} 文件  '
          f'{time.time()-t0:.0f}s', flush=True)

    # ---- 同批 A2c 指标 ----
    a2c_all = eval_read_metrics(AP.numpy(), WP.numpy(),
                                meta['alloc'], meta['wait'])
    print(f'[A2c] alloc_ce {a2c_all["alloc_ce"]:.4f} '
          f'hit1 {a2c_all["hit1"]:.4f} AUC {a2c_all["tenpai_auc"]:.4f}', flush=True)

    # ---- P0-C deal（提前算：P0-D 也要用）----
    riichi_any = (meta['riichi'] > 0.5).any(1)
    deal = run_deal(WP.numpy(), meta['wait'], meta['own_hand'].astype(bool),
                    riichi_any, meta['turn'], srcs)

    # ---- P0-D：零 tl_tokens 的读牌精度（注入口径可用性）----
    p0d = None
    if not args.skip_tl_probe:
        AP0, WP0 = torch.cat(AP0_list), torch.cat(WP0_list)
        m0 = eval_read_metrics(AP0.numpy(), WP0.numpy(),
                               meta['alloc'], meta['wait'])
        deal0 = run_deal(WP0.numpy(), meta['wait'],
                         meta['own_hand'].astype(bool), riichi_any, meta['turn'], srcs)
        p0d = {'metrics_without_tl': m0,
               'deal_without_tl': deal0['deal_pure'],
               'delta_alloc_ce': m0['alloc_ce'] - a2c_all['alloc_ce'],
               'delta_hit1': m0['hit1'] - a2c_all['hit1'],
               'delta_tenpai_auc': m0['tenpai_auc'] - a2c_all['tenpai_auc'],
               'delta_deal_pure': deal0['deal_pure'] - deal['deal_pure']}
        print(f'[P0-D] 零 tl_tokens: alloc_ce {m0["alloc_ce"]:.4f} '
              f'(Δ{m0["alloc_ce"]-a2c_all["alloc_ce"]:+.4f})  '
              f'hit1 {m0["hit1"]:.4f} (Δ{m0["hit1"]-a2c_all["hit1"]:+.4f})  '
              f'AUC {m0["tenpai_auc"]:.4f} '
              f'(Δ{m0["tenpai_auc"]-a2c_all["tenpai_auc"]:+.4f})', flush=True)

    # ---- P0-A 探针 ----
    pn = args.probe_n if args.probe_n > 0 else total
    pn = min(pn, total)
    probe_res, tr_idx, va_idx = run_probe(
        PHI[:pn], meta['alloc'][:pn], meta['wait'][:pn], srcs[:pn], args)
    probe_res['a2c_same_batch_all'] = a2c_all
    probe_res['a2c_same_batch_probe'] = eval_read_metrics(
        AP[:pn].numpy(), WP[:pn].numpy(), meta['alloc'][:pn], meta['wait'][:pn])
    probe_res['a2c_same_batch_probe_val'] = eval_read_metrics(
        AP[va_idx].numpy(), WP[va_idx].numpy(),
        meta['alloc'][va_idx], meta['wait'][va_idx])

    # ---- P0-B 分歧 ----
    div = run_divergence(
        WP[:pn].numpy(), Q[:pn].numpy(),
        (meta['wait'][:pn][..., :NUM_TILES] > 0).any(1),
        meta['own_hand'][:pn].astype(bool), riichi_any[:pn], meta['turn'][:pn],
        {'src': srcs_all[:pn], 'event': event_all[:pn], 'observer': obs_all[:pn]},
        srcs[:pn])


    report = {
        'meta': {
            'script': 'inject_p0.py',
            'n': total, 'probe_n': pn, 'seed': args.seed,
            'val_glob': args.val_glob,
            'a2c_ckpt': str(A2C_CKPT), 'base_ckpt': str(BASE_CKPT),
            'n_files': len(set(srcs_all)), 'elapsed_s': round(time.time() - t0, 1),
            'protocol': '官方 build_fixed_eval_set 同协议（顺序截取）；'
                        'A2c 指标同批重算；探针按文件分组 split',
        },
        'a2c_same_batch': a2c_all,
        'a2c_without_tl': p0d,
        'p0a_probe': probe_res,
        'p0b_divergence': div,
        'p0c_deal': deal,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'\n[out] {args.out}', flush=True)

    # ---- 判决打印 ----
    av = probe_res['a2c_same_batch_probe_val']
    print('\n================ P0 判决表 ================', flush=True)
    print('P0-A 主模型 backbone 探针（val，按文件 K-fold）', flush=True)
    for name in ('linear', 'mlp128'):
        v = probe_res[name]['val']
        print(f'  {name:7s}: alloc_ce {v["alloc_ce"]:.4f}  hit1 {v["hit1"]:.4f}  '
              f'AUC {v["tenpai_auc"]:.4f}  cond_ce {v["wait_cond_ce"]:.4f}', flush=True)
    print(f'  A2c   : alloc_ce {av["alloc_ce"]:.4f}  hit1 {av["hit1"]:.4f}  '
          f'AUC {av["tenpai_auc"]:.4f}  cond_ce {av["wait_cond_ce"]:.4f}', flush=True)
    print(f'P0-B 分歧: agree {div["agree_rate"]:.3f}  Δrisk {div["delta_risk_mean"]:+.4f}  '
          f'deal(mm/a2c/human) {div["deal_top1"]:.4f}/{div["deal_a2c_safest"]:.4f}/'
          f'{div["deal_human"]:.4f}  fixable {div["fixable_rate"]:.4f}', flush=True)
    print(f'P0-C deal_pure {deal["deal_pure"]:.4f} '
          f'CI{deal["ci95_block"]}  n_deal {deal["n_deal"]}', flush=True)
    print(f'总耗时 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
