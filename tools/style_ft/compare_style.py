"""风格微调 P3：合并 agreement / human_profile / ladder 结果为 Markdown 对比报告。

一致率差用**配对、局级** bootstrap 95% CI（同一 holdout 上两 ckpt 的逐局一致率
差重采样；~2500 样本上方差才可控的教训见 tools/read_model/compare_bootstrap.py）。

用法（repo root 下）:
    python tools/style_ft/compare_style.py \
        --agreement runs/style/okada_sayaka/agreement.json \
        --human runs/style/okada_sayaka/human_profile.json \
        [--ladder runs/style/okada_sayaka/ladder/run_meta.json] \
        --out runs/style/okada_sayaka/report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOOT_N = 2000
SEED = 20261027

# 8 项核心行为指标（Stat getter 名 → 展示名；v2 审阅后无「默听和率」getter，改 6+2 项）
PROFILE_METRICS = [
    ('riichi_rate', '立直率'),
    ('fuuro_rate', '鸣牌率'),
    ('agari_rate', '和了率'),
    ('houjuu_rate', '放铳率'),
    ('avg_riichi_jun', '平均立直巡目'),
    ('avg_point_per_agari', '平均和了点'),
    ('avg_rank', '平均顺位'),
    ('avg_point_per_round', '每局场均点'),
]


def paired_bootstrap(per_game_a, per_game_b, n_boot=BOOT_N):
    """局级配对 bootstrap：mean(a-b) 的 95% CI（a=风格, b=父代）。"""
    keys = sorted(set(per_game_a) & set(per_game_b))
    if not keys:
        return None
    diffs = np.array([per_game_a[k] - per_game_b[k] for k in keys])
    rng = np.random.default_rng(SEED)
    n = len(diffs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = diffs[idx].mean()
    return {
        'mean_diff': float(diffs.mean()),
        'ci95': [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        'n_games': n,
        'significant': bool(np.quantile(boots, 0.025) > 0 or np.quantile(boots, 0.975) < 0),
    }


def main():
    ap = argparse.ArgumentParser(description='风格微调对比报告')
    ap.add_argument('--agreement', required=True)
    ap.add_argument('--human', required=True)
    ap.add_argument('--ladder', default=None, help='ladder run_meta.json（风格 vs 父代强度）')
    ap.add_argument('--style-key', default=None, help='agreement.json 里风格 ckpt 的键名（默认取第一个）')
    ap.add_argument('--parent-key', default=None, help='父代 ckpt 键名（默认取最后一个）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.agreement, encoding='utf-8') as f:
        agree = json.load(f)
    with open(args.human, encoding='utf-8') as f:
        human = json.load(f)

    ckpts = list(agree['ckpts'].keys())
    style_key = args.style_key or ckpts[0]
    parent_key = args.parent_key or ckpts[-1]
    style_r = agree['ckpts'][style_key]
    parent_r = agree['ckpts'][parent_key]

    lines = []
    lines.append(f'# 风格微调对比报告：{agree["player"]}\n')
    lines.append(f'- holdout：{style_r["n_games"]} 局 / {style_r["n_samples"]:,} 决策点\n')

    lines.append('## 动作一致率（holdout top-1）\n')
    lines.append('| ckpt | overall | discard | riichi | meld | agari | pass |')
    lines.append('|---|---|---|---|---|---|---|')
    for k, r in agree['ckpts'].items():
        def cls(name):
            c = r['by_action_class'].get(name)
            return f'{c["agreement"]:.4f}' if c else '-'
        lines.append(f'| `{Path(k).name}` | **{r["overall"]:.4f}** | {cls("discard")} | '
                     f'{cls("riichi")} | {cls("meld")} | {cls("agari")} | {cls("pass")} |')

    boot = paired_bootstrap(style_r['per_game'], parent_r['per_game'])
    if boot:
        lines.append(f'\n风格 − 父代（配对局级 bootstrap，{boot["n_games"]} 局）：'
                     f'**{boot["mean_diff"]:+.4f}**，95% CI '
                     f'[{boot["ci95"][0]:+.4f}, {boot["ci95"][1]:+.4f}]'
                     f'{"（显著）" if boot["significant"] else "（不显著）"}\n')

    lines.append('## 人类行为画像\n')
    prof = human['profile']
    lines.append(f'样本：{prof["game"]} 半庄 / {prof["round"]} 局场（含全部赛季）\n')
    lines.append('| 指标 | 数值 |')
    lines.append('|---|---|')
    for key, name in PROFILE_METRICS:
        v = prof.get(key)
        lines.append(f'| {name} | {v:.4f} |' if v is not None else f'| {name} | - |')

    if args.ladder and Path(args.ladder).exists():
        with open(args.ladder, encoding='utf-8') as f:
            lad = json.load(f)
        res = lad.get('results', {})
        lines.append('\n## 强度（ladder：风格 vs 父代）\n')
        lines.append(f'- avg_pt：{res.get("avg_pt", float("nan")):+.3f}（判定门限 ≥ −8）')
        lines.append(f'- avg_rank：{res.get("avg_rank", float("nan")):.4f}')
        lines.append(f'- 行为：agari={res.get("agari_rate")} houjuu={res.get("houjuu_rate")} '
                     f'fuuro={res.get("fuuro_rate")} riichi={res.get("riichi_rate")}')
        lines.append(f'- sets：{lad.get("sets")}（seed_start={lad.get("seed_start")}）\n')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'out: {out}')


if __name__ == '__main__':
    main()