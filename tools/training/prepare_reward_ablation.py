#!/usr/bin/env python3
"""准备奖励口径消融的独立配置、父 checkpoint 分支和实验元数据。

三组均从同一份 `round4/mortal.pth` 权重/优化器父代复制，使用相同 R4 v2 数据池、
相同模型、seed 和新增步数；仅改变 `env` 奖励口径。由于奖励目标改变，配置带一次性
`reset_optimizer` 令牌：首次启动只继承模型权重，跳过旧 optimizer/scheduler/scaler，
避免把父代目标的动量混入对照。
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
BASE_CONFIG = MORTAL / 'config.round4.toml'
PARENT = MORTAL / 'checkpoints/round4/mortal.pth'
OUT_ROOT = MORTAL / 'reward_ablation'

VARIANTS = {
    'A_rank_pt': {
        'label': '现状等间距顺位奖励',
        'env': {'reward_mode': 'rank_pt', 'reward_scale': 1.0, 'pts': [6.0, 4.0, 2.0, 0.0]},
    },
    'B_rank_pt_aligned': {
        'label': '对齐天凤顺位奖励',
        'env': {'reward_mode': 'rank_pt', 'reward_scale': 1.0, 'pts': [2.0, 1.0, 0.0, -3.0]},
    },
    'C_delta_points': {
        'label': '真实分数差奖励',
        'env': {'reward_mode': 'delta_points', 'reward_scale': 1e-3, 'pts': [6.0, 4.0, 2.0, 0.0]},
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def configure(base: dict, tag: str, spec: dict, root: Path) -> dict:
    cfg = base
    control = cfg['control']
    control['state_file'] = f'checkpoints/reward_ablation/{tag}/mortal.pth'
    control['best_state_file'] = f'checkpoints/reward_ablation/{tag}/best.pth'
    control['tensorboard_dir'] = f'runs/reward_ablation/{tag}'
    control['max_train_steps'] = 756000
    control['snapshot_every'] = 50000
    control['seed'] = 20261012
    control['reset_best_perf'] = 'reward-ablation-v1'
    control['reset_optimizer'] = 'reward-ablation-v1'

    dataset = cfg['dataset']
    # 严格复现 R4 数据池：不把 names 修复或新 v3 榜单混入奖励变量。
    dataset['globs'] = [
        '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202[2-5]??/**/*.json.gz',
        '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/20260[1-8]/**/*.json.gz',
    ]
    dataset['player_names_files'] = ['../dataset/index/top_players_400_v2.txt']
    dataset['file_index'] = 'indexes/reward_ablation_r4_v2.pth'
    dataset['data_version'] = 'reward-ablation-r4-v2'
    dataset['enable_augmentation'] = True
    dataset['augmented_first'] = True
    dataset['num_epochs'] = 1

    cfg['env'].update(spec['env'])
    cfg['test_play']['log_dir'] = f'runs/reward_ablation/{tag}/test_play'
    cfg['baseline']['test']['state_file'] = 'checkpoints/mortal_step_000200000.pth'
    cfg['baseline']['train']['state_file'] = 'checkpoints/mortal_step_000200000.pth'
    cfg['grp']['state_file'] = 'checkpoints/grp_h128.pth'
    cfg['1v3']['log_dir'] = f'runs/reward_ablation/{tag}/ladder'
    cfg['1v3']['ladder'] = {
        'device': 'cuda:0',
        'enable_amp': False,
        'enable_rule_based_agari_guard': True,
    }
    cfg['1v3']['challenger']['state_file'] = control['state_file']
    cfg['1v3']['champion']['state_file'] = 'checkpoints/round4/mortal.pth'
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='允许覆盖已有 reward_ablation 目录')
    args = ap.parse_args()
    if not BASE_CONFIG.is_file() or not PARENT.is_file():
        raise SystemExit(f'缺少基准配置或父 checkpoint: {BASE_CONFIG}, {PARENT}')
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()) and not args.force:
        raise SystemExit(f'输出目录已有内容（如需覆盖请加 --force）: {OUT_ROOT}')

    base = toml.load(BASE_CONFIG)
    parent_hash = sha256(PARENT)
    metadata = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'parent': str(PARENT),
        'parent_sha256': parent_hash,
        'parent_steps': 645000,
        'max_train_steps': 756000,
        'additional_steps': 111000,
        'data_pool': 'R4 v2: HF 2022-2025 + original 202601-202608, top_players_400_v2, augmentation=true',
        'seed': 20261012,
        'variants': {},
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tag, spec in VARIANTS.items():
        branch = OUT_ROOT / tag
        branch.mkdir(parents=True, exist_ok=True)
        target_ckpt = MORTAL / f'checkpoints/reward_ablation/{tag}/mortal.pth'
        target_ckpt.parent.mkdir(parents=True, exist_ok=True)
        if target_ckpt.exists() and not args.force:
            raise SystemExit(f'目标 checkpoint 已存在（如需覆盖请加 --force）: {target_ckpt}')
        shutil.copy2(PARENT, target_ckpt)
        cfg = configure(deepcopy(base), tag, spec, OUT_ROOT)
        cfg_path = branch / 'config.toml'
        cfg_path.write_text(toml.dumps(cfg), encoding='utf-8')
        metadata['variants'][tag] = {
            'label': spec['label'],
            'config': str(cfg_path),
            'state_file': str(target_ckpt),
            'state_sha256': sha256(target_ckpt),
            'env': spec['env'],
        }
    (OUT_ROOT / 'experiment.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
