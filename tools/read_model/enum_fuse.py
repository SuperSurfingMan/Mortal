"""RM9 第 1 步：立直枚举支撑集 × H2 条件分布的推理期融合评估。

背景
----
A2c 的 H2 是逐牌列 MLP，不显式利用「手牌→待牌」的组合结构；立直层是最大
短板（hit@1 0.1724 < 非立直 0.2234、cond_ce 2.8595 > 非立直 2.6579）。
RM9 的思路：从公开信息枚举立直家**可能听的牌**（`rule_baseline.riichi_possible_waits`，
现物/不可表达/见逃排除 + 和牌形支撑集），作为 H2 条件分布的先验。

本脚本不改模型/训练，只做推理期几何融合：
    log q_fused = (1-α)·log q_model + α·log q_enum   （只在立直家上）
    q_enum = 枚举支撑集上的均匀分布（枚举为空时回退 q_model）
`p_ten`（P(听)）不变；只融合 34 维条件待牌分布。

用法
----
python3 tools/read_model/enum_fuse.py --state mortal/read_model_v2s_a2c/best.pth \
  --samples 20000 --glob 'dataset/mjai/houou4/202609/**/*.json.gz'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mortal'))          # libriichi 扩展（与 evaluate.py 同口径）
sys.path.insert(0, str(ROOT / 'tools' / 'read_model'))

import dataset as D            # noqa: E402
import model_v2 as M2          # noqa: E402
import train as T              # noqa: E402
from rule_baseline import (  # noqa: E402
    YAOCHUU, YAOCHUU_SET, _can_form, riichi_possible_waits)

NUM_TILES = 34
NUM_WAIT_CLASSES = 35


def load_net(state_path: str, device: str):
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    cfg = state.get('config', {})
    D.set_obs_river_order(cfg.get('obs_river_order', 'keep'))
    kind = cfg.get('model', 'v1')
    if kind == 'v2':
        net = M2.build_model_v2(
            conv_channels=cfg.get('conv_channels', 96),
            num_blocks=cfg.get('num_blocks', 12),
            with_aux=not cfg.get('no_aux', False),
            wait_head=cfg.get('wait_head', 'flat'),
            tl_mode=cfg.get('tl_mode', 'pooled'),
            next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
            tile_order=cfg.get('tile_order', False)).to(device)
    else:
        net = M2.build_model_v2s(
            conv_channels=cfg.get('conv_channels', 96),
            num_blocks=cfg.get('num_blocks', 12),
            with_aux=not cfg.get('no_aux', False),
            wait_head=cfg.get('wait_head', 'flat'),
            tl_mode=cfg.get('tl_mode', 'pooled'),
            next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
            tile_order=cfg.get('tile_order', False)).to(device)
    net.load_state_dict(state['model'])
    net.eval()
    print(f'[model] {state_path} ({kind}, step={state.get("step")})', flush=True)
    return net, cfg


def forward_all(net, batch, device, batch_size=2048):
    n = batch['obs'].shape[0]
    outs_w, outs_a = [], []
    for i in range(0, n, batch_size):
        chunk = {k: v[i:i + batch_size].to(device) for k, v in batch.items()}
        with torch.no_grad():
            o = net(chunk['obs'], chunk['unseen'], T.wait_bias_of(chunk).to(device),
                    chunk['tl_tokens'], chunk['tl_mask'], **net.tl_extra(chunk))
        outs_w.append(o['wait_probs'].cpu().numpy().astype(np.float64))
        outs_a.append(o['alloc_probs'].cpu().numpy().astype(np.float64))
    return np.concatenate(outs_w), np.concatenate(outs_a)


def sample_hand(probs, hand_size, unseen, rng):
    """从 alloc 后验 [34,5] 采样一副满足行和/容量约束的手牌计数 [34]。"""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    p = p / p.sum(-1, keepdims=True)
    logp = np.log(p)
    cap = np.minimum(np.asarray(unseen, dtype=np.int64), 4)
    k = np.array([rng.choice(5, p=p[i]) for i in range(NUM_TILES)], dtype=np.int64)
    k = np.minimum(k, cap)
    total = int(k.sum())
    # 行和修正：减/加一张的 log 概率代价最小/增益最大
    while total > hand_size:
        cand = np.where(k > 0)[0]
        if len(cand) == 0:
            break
        cost = logp[cand, k[cand] - 1] - logp[cand, k[cand]]
        i = int(cand[np.argmax(cost)])
        k[i] -= 1
        total -= 1
    while total < hand_size:
        cand = np.where(k < cap)[0]
        if len(cand) == 0:
            break
        gain = logp[cand, np.minimum(k[cand] + 1, 4)] - logp[cand, k[cand]]
        i = int(cand[np.argmax(gain)])
        k[i] += 1
        total += 1
    return k


def hand_can_wait(hand, t, n_sets):
    """单副手牌 hand + 摸牌 t 是否和牌（normal/七对/国士）。"""
    R = [int(x) for x in hand]
    R[t] += 1
    budget = [20000]
    if _can_form(list(R), n_sets, True, budget):
        return True
    if n_sets < 4:
        return False
    if sum(R) == 14 and sum(c // 2 for c in R) == 7 and all(c % 2 == 0 for c in R):
        return True
    if sum(R) == 14 and all(R[x] >= 1 for x in YAOCHUU):
        return True
    return False


def metrics(cond, wait_true, active, mask=None):
    """cond [n,3,34] 条件待牌分布；返回 hit@1/3/5、cond_ce。"""
    sup = wait_true[..., :NUM_TILES] > 0
    order = np.argsort(-cond, axis=-1)
    hit1 = np.take_along_axis(sup, order[..., :1], axis=-1)[..., 0]
    hit3 = np.take_along_axis(sup, order[..., :3], axis=-1).any(-1)
    hit5 = np.take_along_axis(sup, order[..., :5], axis=-1).any(-1)
    cce = -(wait_true[..., :NUM_TILES] * np.log(np.clip(cond, 1e-12, None))).sum(-1)
    m = active if mask is None else (active & mask)
    if not m.any():
        return None
    return dict(n=int(m.sum()),
                hit1=float(hit1[m].mean()), hit3=float(hit3[m].mean()),
                hit5=float(hit5[m].mean()), cond_ce=float(cce[m].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='mortal/read_model_v2s_a2c/best.pth')
    ap.add_argument('--glob', default='dataset/mjai/houou4/202609/**/*.json.gz')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--alphas', default='0,0.1,0.2,0.3,0.5,0.7,1.0')
    ap.add_argument('--enum-source', choices=['support', 'rule', 'sample'], default='support',
                    help='support=枚举支撑集均匀；rule=规则法连续分布；sample=alloc 采样手牌→精确 waits')
    ap.add_argument('--samples-per-hand', type=int, default=8,
                    help='sample 模式下每家采样的手牌副数')
    ap.add_argument('--out', default='mortal/read_model_v2s_a2c/enum_fuse_report.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

    net, cfg = load_net(args.state, args.device)
    files = sorted(D.glob_files(args.glob))
    batch = T.build_fixed_eval_set(files, args.samples, seed=args.seed)
    n = batch['obs'].shape[0]
    print(f'[data] {n} samples, {len(files)} files', flush=True)

    wait_probs, alloc_probs = forward_all(net, batch, args.device, args.batch_size)
    wait_true = batch['wait'].numpy()
    tenpai = batch['tenpai'].numpy() > 0.5
    riichi = batch['riichi'].numpy() > 0.5
    unseen = batch['unseen'].numpy()
    river = batch['river'].numpy()
    own = batch['own_hand'].numpy()
    meld = batch['meld_counts'].numpy()

    p_ten = 1.0 - wait_probs[..., NUM_WAIT_CLASSES - 1]            # [n,3]
    cond0 = wait_probs[..., :NUM_TILES] / np.clip(p_ten[..., None], 1e-12, None)
    cond0 = cond0 / np.clip(cond0.sum(-1, keepdims=True), 1e-12, None)

    # ---- 立直枚举/规则先验（只对 riichi 家）----
    enum = np.zeros((n, 3, NUM_TILES), dtype=np.float64)
    n_riichi = n_nonempty = 0
    sizes = []
    if args.enum_source == 'rule':
        from rule_baseline import rule_predict
        rp = rule_predict(batch)
        enum_cond = np.asarray(rp['wait_probs'][..., :NUM_TILES], dtype=np.float64)
        enum_cond = enum_cond / np.clip(enum_cond.sum(-1, keepdims=True), 1e-12, None)
        enum_ok = enum_cond.sum(-1) > 0
        n_riichi = int(riichi.sum())
        n_nonempty = int((enum_ok & riichi).sum())
        sizes = [int((enum_cond[s, j] > 0).sum())
                 for s in range(n) for j in range(3) if riichi[s, j]]
    elif args.enum_source == 'sample':
        enum_cond = np.zeros((n, 3, NUM_TILES), dtype=np.float64)
        rng = np.random.default_rng(args.seed)
        M = args.samples_per_hand
        for s in range(n):
            for j in range(3):
                if not riichi[s, j]:
                    continue
                n_riichi += 1
                n_melds = int(meld[s, j])
                hand_size = 13 - 3 * n_melds
                n_sets = 4 - n_melds
                freq = np.zeros(NUM_TILES, dtype=np.float64)
                for _ in range(M):
                    hand = sample_hand(alloc_probs[s, j], hand_size, unseen[s], rng)
                    for t in range(NUM_TILES):
                        if hand_can_wait(hand, t, n_sets):
                            freq[t] += 1.0
                if freq.sum() > 0:
                    n_nonempty += 1
                    enum_cond[s, j] = freq / freq.sum()
                    sizes.append(int((freq > 0).sum()))
        enum_ok = enum_cond.sum(-1) > 0
        print(f'[sample] M={M} riichi={n_riichi} nonempty={n_nonempty}', flush=True)
    else:
        for s in range(n):
            for j in range(3):
                if not riichi[s, j]:
                    continue
                n_riichi += 1
                e = riichi_possible_waits(unseen[s], river[s, j], own[s],
                                          n_melds=int(meld[s, j]))
                if e.any():
                    n_nonempty += 1
                    enum[s, j] = e.astype(np.float64)
                    sizes.append(int(e.sum()))
        enum_cond = enum / np.clip(enum.sum(-1, keepdims=True), 1e-12, None)
        enum_ok = enum.sum(-1) > 0                                  # [n,3]
    print(f'[enum:{args.enum_source}] riichi={n_riichi} nonempty={n_nonempty} '
          f'empty={n_riichi - n_nonempty} mean_size={np.mean(sizes) if sizes else 0:.2f}',
          flush=True)

    active = tenpai
    err_a = tenpai & ~riichi                                       # 非立直 active
    report = dict(
        meta=dict(state=args.state, glob=args.glob, samples=int(n), seed=args.seed,
                  alphas=alphas, protocol='推理期几何融合，只在 riichi 家；p_ten 不变'),
        enum=dict(n_riichi=int(n_riichi), n_nonempty=int(n_nonempty),
                  n_empty=int(n_riichi - n_nonempty),
                  mean_size=float(np.mean(sizes)) if sizes else 0.0),
        baseline=dict(all=metrics(cond0, wait_true, active),
                      riichi=metrics(cond0, wait_true, active, mask=riichi),
                      nonriichi=metrics(cond0, wait_true, active, mask=~riichi)),
        alphas=[],
    )

    for a in alphas:
        log_c = np.log(np.clip(cond0, 1e-12, None))
        log_e = np.log(np.clip(enum_cond, 1e-12, None))
        log_f = (1.0 - a) * log_c + a * log_e
        # 只在枚举非空的 riichi 家上融合；其余保持模型分布
        use = enum_ok[..., None]
        fused = np.where(use, np.exp(log_f - log_f.max(-1, keepdims=True)), cond0)
        fused = fused / np.clip(fused.sum(-1, keepdims=True), 1e-12, None)
        report['alphas'].append(dict(
            alpha=a,
            all=metrics(fused, wait_true, active),
            riichi=metrics(fused, wait_true, active, mask=riichi),
            nonriichi=metrics(fused, wait_true, active, mask=~riichi),
            nonriichi_active=metrics(fused, wait_true, active, mask=err_a),
        ))
        r = report['alphas'][-1]['riichi']
        print(f'[α={a:.2f}] riichi hit1={r["hit1"]:.4f} hit3={r["hit3"]:.4f} '
              f'hit5={r["hit5"]:.4f} cond_ce={r["cond_ce"]:.4f} | '
              f'all hit1={report["alphas"][-1]["all"]["hit1"]:.4f} '
              f'cond_ce={report["alphas"][-1]["all"]["cond_ce"]:.4f}', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f'[out] {out}', flush=True)


if __name__ == '__main__':
    main()
