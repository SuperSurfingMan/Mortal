"""风格微调选优：多档位（trainable × kl_weight）横向汇总为单页 Markdown。

compare_style.py 出的是「单档 vs 父代」报告；档位/kl 扫需要横向表：一次读多个
runs/style/<slug>/ 目录的 06_agreement.json + ladder/run_meta.json（+ 可选
05_agreement_snapshots.json / 07_human_profile.json），输出：

    一致率（总体/分动作类）+ 配对局级 bootstrap CI（vs 父代）+ ladder avg_pt
    + 行为画像距人类的方向（鸣牌率/立直率/和了率与人类的差）

用法（repo root 下）:
    python tools/style_ft/compare_runs.py \
        --run toi_takaharu --run toi_takaharu_lb --run toi_takaharu_lb05 \
        --human runs/style/toi_takaharu/07_human_profile.json \
        --out runs/style/toi_takaharu/DECISION.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from compare_style import PROFILE_METRICS, paired_bootstrap


def load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def load_run(slug: str) -> dict:
    run_dir = Path('runs/style') / slug
    out = {'slug': slug, 'agreement': load_json(run_dir / '06_agreement.json'),
           'snapshots': load_json(run_dir / '05_agreement_snapshots.json'),
           'human': load_json(run_dir / '07_human_profile.json'),
           'ladder': load_json(Path('mortal/runs/style') / slug / 'ladder' / 'run_meta.json')}
    cfg = Path('mortal') / f'config.style_{slug}.toml'
    if cfg.exists():
        import toml
        with open(cfg, encoding='utf-8') as f:
            c = toml.load(f)
        out['cfg'] = {'kl_weight': c['style']['kl_weight'],
                      'trainable': c['style']['trainable'],
                      'max_train_steps': c['control']['max_train_steps']}
    return out


def main():
    ap = argparse.ArgumentParser(description='多档位风格微调汇总')
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--human', required=True, help='任一次 human_profile.json（玩家画像与档位无关）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    human = load_json(Path(args.human))
    prof = human['profile']
    runs = [load_run(s) for s in args.run]
    parent_keys = []
    for r in runs:
        if r['agreement']:
            parent_keys.append(list(r['agreement']['ckpts'].keys())[-1])

    lines = [f'# 风格微调选优汇总：{human["player"]}\n']

    # ---- 1. holdout 一致率 ----
    lines.append('## holdout 一致率（top-1）\n')
    lines.append('| slug | kl | 档位 | peak step | overall | discard | riichi | meld | agari | pass | vs 父代 [95% CI] |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in runs:
        ag = r['agreement']
        if not ag:
            lines.append(f'| {r["slug"]} | - | - | - | 未完成 | | | | | | |')
            continue
        keys = list(ag['ckpts'].keys())
        style, parent = ag['ckpts'][keys[0]], ag['ckpts'][keys[-1]]
        boot = paired_bootstrap(style['per_game'], parent['per_game'])
        ci = (f'{boot["mean_diff"]:+.4f} [{boot["ci95"][0]:+.4f},{boot["ci95"][1]:+.4f}]'
              f'{"*" if boot["significant"] else ""}') if boot else '-'
        peak = '-'
        if r['snapshots']:
            best = max(r['snapshots']['ckpts'].items(), key=lambda kv: kv[1]['overall'])
            peak = Path(best[0]).stem.replace('mortal_step_00000', '').lstrip('0') or '0'

        def cls(name):
            c = style['by_action_class'].get(name)
            return f'{c["agreement"]:.4f}' if c else '-'
        cfg = r.get('cfg', {})
        lines.append(f'| {r["slug"]} | {cfg.get("kl_weight", "-")} | {cfg.get("trainable", "-")} | {peak} | '
                     f'**{style["overall"]:.4f}** | {cls("discard")} | {cls("riichi")} | {cls("meld")} | '
                     f'{cls("agari")} | {cls("pass")} | {ci} |')
    par = runs[0]['agreement']['ckpts'][list(runs[0]['agreement']['ckpts'].keys())[-1]]
    lines.append(f'| 父代基线 | - | - | - | {par["overall"]:.4f} | '
                 f'{par["by_action_class"].get("discard", {}).get("agreement", float("nan")):.4f} | '
                 f'{par["by_action_class"].get("riichi", {}).get("agreement", float("nan")):.4f} | '
                 f'{par["by_action_class"].get("meld", {}).get("agreement", float("nan")):.4f} | '
                 f'{par["by_action_class"].get("agari", {}).get("agreement", float("nan")):.4f} | '
                 f'{par["by_action_class"].get("pass", {}).get("agreement", float("nan")):.4f} | - |')
    lines.append('')
    if parent_keys:
        lines.append(f'父代 ckpt：`{parent_keys[0]}`；holdout {par["n_games"]} 局 / {par["n_samples"]:,} 决策点。'
                     '`*` = 配对 bootstrap 95% CI 不含 0。\n')

    # ---- 2. snapshot 曲线 ----
    lines.append('## snapshot 曲线（步数 → overall）\n')
    for r in runs:
        if not r['snapshots']:
            continue
        curve = {Path(k).stem.split('_')[-1].lstrip('0') or '0': round(v['overall'], 4)
                 for k, v in sorted(r['snapshots']['ckpts'].items())}
        lines.append(f'- `{r["slug"]}`：{curve}')
    lines.append('')

    # ---- 3. ladder 强度 ----
    lines.append('## 强度与行为（ladder 500 sets vs 父代）\n')
    lines.append('| slug | avg_pt | avg_rank | agari | houjuu | fuuro | riichi |')
    lines.append('|---|---|---|---|---|---|---|')
    for r in runs:
        lad = r['ladder']
        if not lad:
            lines.append(f'| {r["slug"]} | - | - | - | - | - | - |')
            continue
        res = lad.get('results', {})
        lines.append(f'| {r["slug"]} | {res.get("avg_pt", float("nan")):+.3f} | '
                     f'{res.get("avg_rank", float("nan")):.4f} | {res.get("agari_rate", float("nan")):.4f} | '
                     f'{res.get("houjuu_rate", float("nan")):.4f} | {res.get("fuuro_rate", float("nan")):.4f} | '
                     f'{res.get("riichi_rate", float("nan")):.4f} |')
    lines.append('')

    # ---- 4. 人类画像与方向 ----
    lines.append('## 人类画像（全 269 局历史，描述性）\n')
    lines.append('| 指标 | 人类 | 说明 |')
    lines.append('|---|---|---|')
    for key, name in PROFILE_METRICS:
        v = prof.get(key)
        lines.append(f'| {name} | {v:.4f} | |' if v is not None else f'| {name} | - | |')
    lines.append('')
    lines.append('ladder 行的 fuuro/riichi/agari/houjuu 是风格模型实际对局行为，可直接与人类列比对'
                 '（M-League 与天凤对手池不同，仅作方向性参考）。\n')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'out: {out}')


if __name__ == '__main__':
    main()
