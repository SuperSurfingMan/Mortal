#!/usr/bin/env python3
"""推理期守卫 ladder 结果汇总：多方向合并 + 95%CI（按半庄近似）。

用法::

    python tools/read_model/agg_guard_ladder.py \
        --arms guard_l20=runs/guard_l20_260000,runs/guard_l20_270000 \
        --arms guard_l50t03=runs/guard_l50t03_260000,runs/guard_l50t03_270000

每个 ``--arms name=dir1,dir2`` 读各目录的 ``run_meta.json``，输出逐方向明细
与合并 avg_pt / CI，便于横向比较不同 λ/τ。

**口径**：pt = `[90,45,0,-135]`（与 ladder.py 一致）；SE 用多项分布近似
（``Σ p_i(pt_i−mean)² / n``），未做组内相关修正（同 set 4 局相关会略抬高
真实方差，故 CI 偏窄——判读时以「跨 0 即不显著」为准）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PTS = np.array([90.0, 45.0, 0.0, -135.0])


def load_dirs(dirs):
    rows = []
    for d in dirs:
        p = Path(d) / 'run_meta.json'
        if not p.exists():
            print(f'[skip] {p} 不存在')
            continue
        m = json.loads(p.read_text())
        r = m['results']
        c = np.array(r['raw_rankings'], float)
        n = float(c.sum())
        p_rank = c / n
        mu = float(r['avg_pt'])
        se = float(np.sqrt(((p_rank * (PTS - mu) ** 2).sum()) / n))
        rows.append({
            'dir': str(d), 'seed': m.get('seed_start'), 'n': int(n),
            'guard_lambda': m.get('guard_lambda'), 'guard_risk_floor': m.get('guard_risk_floor'),
            'avg_rank': float(r['avg_rank']), 'avg_pt': mu, 'se': se,
            'agari': float(r['agari_rate']), 'houjuu': float(r['houjuu_rate']),
            'riichi': float(r['riichi_rate']), 'fuuro': float(r['fuuro_rate']),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', action='append', required=True,
                    help='name=dir1,dir2（可重复）')
    ap.add_argument('--baseline', default='',
                    help='baseline 名称（若有，只用于在表头标注）')
    args = ap.parse_args()

    print(f'{"arm":<12s} {"seed":>7s} {"n":>6s} {"rank":>6s} {"avg_pt":>8s} '
          f'{"agari":>6s} {"houjuu":>6s} {"riichi":>6s}')
    summary = {}
    for spec in args.arms:
        name, _, dirs = spec.partition('=')
        rows = load_dirs([x for x in dirs.split(',') if x])
        if not rows:
            continue
        for r in rows:
            print(f"{name:<12s} {r['seed'] or 0:>7d} {r['n']:>6d} {r['avg_rank']:>6.4f} "
                  f"{r['avg_pt']:>+8.3f} {r['agari']:>6.3f} {r['houjuu']:>6.3f} "
                  f"{r['riichi']:>6.3f}")
        mean = float(np.mean([r['avg_pt'] for r in rows]))
        # 多方向合并：SE = sqrt(Σ se_i²) / k（各方向独立）
        se = float(np.sqrt(sum(r['se'] ** 2 for r in rows))) / len(rows)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        print(f"{name:<12s} {'合并':>7s} {sum(r['n'] for r in rows):>6d} "
              f"{'—':>6s} {mean:>+8.3f} {np.mean([r['agari'] for r in rows]):>6.3f} "
              f"{np.mean([r['houjuu'] for r in rows]):>6.3f} "
              f"{np.mean([r['riichi'] for r in rows]):>6.3f}")
        print(f"             → 95%CI [{lo:+.2f}, {hi:+.2f}]  "
              f"{'显著' if lo > 0 or hi < 0 else '跨 0（不显著）'}\n")
        summary[name] = {'mean': mean, 'se': se, 'lo': lo, 'hi': hi,
                         'agari': float(np.mean([r['agari'] for r in rows])),
                         'houjuu': float(np.mean([r['houjuu'] for r in rows])),
                         'rows': rows}
    if len(summary) > 1:
        print('=== 横向对比 ===')
        for name, s in summary.items():
            print(f"{name:<12s} avg_pt {s['mean']:+.3f}  CI[{s['lo']:+.2f},{s['hi']:+.2f}]  "
                  f"agari {s['agari']:.3f}  houjuu {s['houjuu']:.3f}")


if __name__ == '__main__':
    main()
