#!/usr/bin/env python3
"""便宜前置实验（超参消融）：cql.min_q_weight 1/10 与 aux.next_rank_weight 0。

目的（MEMORY §3.3 / NEXT_STEPS §3.3 顺序 2、3）：在进入 IQL/Bellman 大改前，先确认
"是否只是超参没扫对"。三组均**严格复刻 B 组配方**（数据池 top400_v2、reward
`rank_pt pts=[2,1,0,-3]`、augmentation=true、seed、+111k 步到 756k、reset_optimizer），
每组**只改一个旋钮**，B 组本身（cql=5/aux=0.2）即现成控制组。

判定仍只认双向 ≥5000 sets 全新种子块的 block bootstrap CI；holdout 仅诊断。
产物落在 `mortal/hp_ablation/<tag>/`，与 reward_ablation 完全隔离（不同 state_file /
tensorboard_dir / test_play.log_dir / ladder log_dir）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[2]
MORTAL = REPO / 'mortal'
BASE_CONFIG = MORTAL / 'reward_ablation/B_rank_pt_aligned/config.toml'   # 直接以 B 配方为母版
PARENT = MORTAL / 'checkpoints/round4/mortal.pth'                          # B 的同一父代（645k）
OUT_ROOT = MORTAL / 'hp_ablation'

RESET_TOKEN = 'hp-ablation-v1'
TRAIN_SEED = 20261012         # 与 B 相同 → 数据顺序一致，仅旋钮不同

VARIANTS = {
    'cql_min1':   {'label': 'CQL min_q_weight=1（B=5）',   'set': {'cql': {'min_q_weight': 1.0}}},
    'cql_min10':  {'label': 'CQL min_q_weight=10（B=5）',  'set': {'cql': {'min_q_weight': 10.0}}},
    'aux_next0':  {'label': 'Aux next_rank_weight=0（B=0.2）', 'set': {'aux': {'next_rank_weight': 0.0}}},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def apply_set(cfg: dict, spec: dict) -> None:
    for table, kv in spec['set'].items():
        cfg.setdefault(table, {}).update(kv)


def configure(base: dict, tag: str, spec: dict) -> dict:
    cfg = deepcopy(base)
    control = cfg['control']
    control['state_file'] = f'checkpoints/hp_ablation/{tag}/mortal.pth'
    control['best_state_file'] = f'checkpoints/hp_ablation/{tag}/best.pth'
    control['tensorboard_dir'] = f'runs/hp_ablation/{tag}'
    control['max_train_steps'] = 756000        # 与 B 同深度
    control['snapshot_every'] = 50000
    control['seed'] = TRAIN_SEED
    control['reset_best_perf'] = RESET_TOKEN
    control['reset_optimizer'] = RESET_TOKEN

    cfg['test_play']['log_dir'] = f'runs/hp_ablation/{tag}/test_play'
    cfg['1v3']['log_dir'] = f'runs/hp_ablation/{tag}/ladder'
    cfg['1v3']['challenger']['state_file'] = control['state_file']
    cfg['1v3']['champion']['state_file'] = 'checkpoints/round4/mortal.pth'
    cfg['grp']['state_file'] = 'checkpoints/grp_h128.pth'

    apply_set(cfg, spec)   # 唯一变量：只动 cql 或 aux
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='允许覆盖已有 hp_ablation 目录')
    args = ap.parse_args()
    if not BASE_CONFIG.is_file() or not PARENT.is_file():
        raise SystemExit(f'缺少基准配置或父 checkpoint: {BASE_CONFIG}, {PARENT}')
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()) and not args.force:
        raise SystemExit(f'输出目录已有内容（如需覆盖请加 --force）: {OUT_ROOT}')

    base = toml.load(BASE_CONFIG)
    metadata = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'control': 'reward_ablation/B_rank_pt_aligned (cql=5, aux=0.2), 756k',
        'parent': str(PARENT),
        'parent_sha256': sha256(PARENT),
        'parent_steps': 645000,
        'max_train_steps': 756000,
        'additional_steps': 111000,
        'train_seed': TRAIN_SEED,
        'reset_token': RESET_TOKEN,
        'data_pool': 'B 组一致: R4 v2 (HF 2022-2025 + 原 202601-202608), top_players_400_v2, augmentation=true',
        'reward': 'rank_pt pts=[2,1,0,-3] (与 B 一致，本实验不改奖励)',
        'ladder_seed_blocks': {'A': '140000-', 'B': '150000-'},
        'variants': {},
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tag, spec in VARIANTS.items():
        branch = OUT_ROOT / tag
        branch.mkdir(parents=True, exist_ok=True)
        target_ckpt = MORTAL / f'checkpoints/hp_ablation/{tag}/mortal.pth'
        target_ckpt.parent.mkdir(parents=True, exist_ok=True)
        if target_ckpt.exists() and not args.force:
            raise SystemExit(f'目标 checkpoint 已存在（如需覆盖请加 --force）: {target_ckpt}')
        shutil.copy2(PARENT, target_ckpt)
        cfg = configure(base, tag, spec)
        cfg_path = branch / 'config.toml'
        cfg_path.write_text(toml.dumps(cfg), encoding='utf-8')
        metadata['variants'][tag] = {
            'label': spec['label'],
            'config': str(cfg_path),
            'state_file': str(target_ckpt),
            'state_sha256': sha256(target_ckpt),
            'changed': spec['set'],
        }
    (OUT_ROOT / 'experiment.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
