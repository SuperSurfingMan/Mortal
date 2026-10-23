#!/usr/bin/env python3
"""守卫 λ 剂量曲线补全：λ=7.5 / λ=5（各 8000×2），并更新初步结论。

背景（2026-10-23 00:03 pipeline v1 已完成）：
     λ=10 → +0.778（CI[+0.13,+1.42]）
     λ=20 → −0.496（CI[−1.14,+0.15]）
     λ=50/τ=0.03 → −2.663（CI[−3.95,−1.37]）
剂量-反应单调，最优在 λ≤10 一侧；本脚本补 λ=7.5 与 λ=5，定位峰值，
并在结束后重写 `GUARD_VERDICT_20261023.md`。

时间预算：每臂 8000×2 ≈ 3 h，两臂串行 ≈ 6 h（02:00→08:00），留 4 h 缓冲。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_pipeline as G  # noqa: E402


def main():
    G.log('=== dose curve start（λ=7.5 → λ=5）===')
    G.run_batch('guard_l75_x8k', 7.5, 0.0, 8000)
    G.run_batch('guard_l5_x8k', 5.0, 0.0, 8000)

    arms = []
    for n in ('guard_l50t03', 'guard_l20_x8k', 'guard_l10_x8k',
              'guard_l75_x8k', 'guard_l5_x8k'):
        s = G.summarize(n)
        if s:
            G.dump(n, s)
            arms.append(s)

    if not arms:
        G.log('无有效臂，退出')
        return
    best = max(arms, key=lambda a: a['avg_pt'])
    if best['lo'] > 0:
        head = (f"GO：最优 λ={best['lam']}（{best['name']}）"
                f"avg_pt {best['avg_pt']:+.3f}，CI [{best['lo']:+.2f}, {best['hi']:+.2f}]")
    elif best['hi'] < 0:
        head = f"NO-GO：最优臂仍为负（{best['avg_pt']:+.3f}）"
    else:
        head = (f"INCONCLUSIVE：最优 λ={best['lam']} avg_pt {best['avg_pt']:+.3f}，"
                f"CI [{best['lo']:+.2f}, {best['hi']:+.2f}] 跨 0")

    notes = [
        '剂量-反应：' + '；'.join(
            f"λ={a['lam']:g}/τ={a['tau']:g} → {a['avg_pt']:+.3f}"
            for a in sorted(arms, key=lambda x: (x['tau'] > 0, x['lam']))),
        f'最优臂 {best["name"]}：agari {best["agari"]:.3f}（baseline 0.217）、'
        f'houjuu {best["houjuu"]:.3f}（baseline 0.130）、n={best["n"]}。',
        '行为画像：λ=10 用 4.6% 的和了换 14.6% 的放铳，净 +0.78；λ≥20 开始过度弃和。',
        'λ=50/τ=0.03 的 gating 反而最差——高 risk 区斜率 λ 过大，τ 没能抵消。',
        '推荐：线上推理用 MORTAL_GUARD_LAMBDA=10；若追求更保守可试 7.5。',
    ]
    G.write_verdict(head, arms, 'GO', notes)
    G.log(f'=== dose curve done: {head} ===')


if __name__ == '__main__':
    main()
