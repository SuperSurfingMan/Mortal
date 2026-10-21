"""立直层专项诊断（零训练）：立直后牌河利用率 + 条件分布温度。

回答两个问题：
1. 立直家 hit@1 / cond_ce 是否随「立直后摸切张数」改善？如果不改善，说明
   模型没有利用立直后牌河（立直后只能摸切，序列直接暴露手牌形状）。
2. 立直家条件分布是否校准/锐度合适？温度 T 扫描只改 cond_ce（排序不变），
   用于判断是「排序错误」还是「概率形状问题」。

用法：
python3 tools/read_model/riichi_diag.py --state mortal/read_model_v2s_a2c/best.pth \
  --samples 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mortal'))
sys.path.insert(0, str(ROOT / 'tools' / 'read_model'))

import dataset as D            # noqa: E402
import train as T              # noqa: E402
from enum_fuse import load_net, forward_all, metrics, NUM_TILES  # noqa: E402


def stats(cond, wait_true, active, mask):
    m = active & mask
    if not m.any():
        return None
    c = np.clip(cond[m], 1e-12, None)
    ent = -(c * np.log(c)).sum(-1)
    return dict(metrics(cond, wait_true, active, mask=mask),
                entropy=float(ent.mean()), max_prob=float(c.max(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='mortal/read_model_v2s_a2c/best.pth')
    ap.add_argument('--glob', default='dataset/mjai/houou4/202609/**/*.json.gz')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--out', default='tools/read_model/diagnostics/results/riichi_diag.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    net, cfg = load_net(args.state, args.device)
    files = sorted(D.glob_files(args.glob))
    batch = T.build_fixed_eval_set(files, args.samples, seed=args.seed)
    n = batch['obs'].shape[0]
    print(f'[data] {n} samples', flush=True)

    wait_probs, _ = forward_all(net, batch, args.device, args.batch_size)
    wait_true = batch['wait'].numpy()
    tenpai = batch['tenpai'].numpy() > 0.5
    riichi = batch['riichi'].numpy() > 0.5
    turn = batch['turn'].numpy()
    post = batch['post_riichi_river'].numpy().sum(-1)          # [n,3] 立直后打牌数
    called = batch['called'].numpy() > 0.5

    p_ten = 1.0 - wait_probs[..., -1]
    cond = wait_probs[..., :NUM_TILES] / np.clip(p_ten[..., None], 1e-12, None)
    cond = cond / np.clip(cond.sum(-1, keepdims=True), 1e-12, None)
    T3 = np.broadcast_to(turn[:, None], (n, 3))

    report = dict(meta=dict(state=args.state, samples=int(n), seed=args.seed),
                  baseline=dict(all=stats(cond, wait_true, tenpai, np.ones((n, 3), bool)),
                                riichi=stats(cond, wait_true, tenpai, riichi),
                                nonriichi=stats(cond, wait_true, tenpai, ~riichi)),
                  riichi_strata={}, temperature=[])

    for name, m in (
        ('post0', riichi & (post == 0)),
        ('post1-2', riichi & (post >= 1) & (post <= 2)),
        ('post3+', riichi & (post >= 3)),
        ('turn<=7', riichi & (T3 <= 7)),
        ('turn>=8', riichi & (T3 >= 8)),
        ('called0', riichi & ~called),
        ('called1', riichi & called),
    ):
        report['riichi_strata'][name] = stats(cond, wait_true, tenpai, m)
        s = report['riichi_strata'][name]
        if s:
            print(f'[riichi:{name}] n={s["n"]} hit1={s["hit1"]:.4f} '
                  f'hit3={s["hit3"]:.4f} cond_ce={s["cond_ce"]:.4f} '
                  f'ent={s["entropy"]:.3f} maxp={s["max_prob"]:.3f}', flush=True)

    # 温度扫描：cond_T ∝ cond^(1/T)（排序不变，只改形状）
    for temp in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
        logc = np.log(np.clip(cond, 1e-12, None)) / temp
        ct = np.exp(logc - logc.max(-1, keepdims=True))
        ct = ct / np.clip(ct.sum(-1, keepdims=True), 1e-12, None)
        s = stats(ct, wait_true, tenpai, riichi)
        report['temperature'].append(dict(T=temp, **s))
        print(f'[T={temp:.2f}] riichi hit1={s["hit1"]:.4f} hit3={s["hit3"]:.4f} '
              f'cond_ce={s["cond_ce"]:.4f} ent={s["entropy"]:.3f}', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f'[out] {out}', flush=True)


if __name__ == '__main__':
    main()
