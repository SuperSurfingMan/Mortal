"""风格微调 P2：以父代 ckpt 内保存的 config 为底，生成 style_<slug> 训练配置。

父代 config 完整保存在 state['config']（train.py:783），以其为底逐项打补丁可
保证 [resnet]/[env]/[baseline]/[test_play]/[grp] 等必读键原样继承（Brain 形状
暖启可加载、GRP 前向可用）。所有补丁键集中在下方 PATCH 部分。

用法（repo root 下）:
    python tools/style_ft/make_style_cfg.py \
        --parent mortal/checkpoints/s6c/mortal.pth \
        --stats dataset/index/style/okada_sayaka_stats.json \
        --slug okada_sayaka \
        [--kl-weight 1.5] [--trainable head] [--max-train-steps 800]

输出: mortal/config.style_<slug>.toml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import toml


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_config(parent_cfg: dict, stats: dict, args) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v))
           for k, v in parent_cfg.items()}
    slug = args.slug

    control = dict(cfg.get('control', {}))
    control.update({
        'objective': 'bc_style',
        'online': False,
        'state_file': f'checkpoints/style/{slug}/mortal.pth',
        'best_state_file': f'checkpoints/style/{slug}/best.pth',
        'tensorboard_dir': f'runs/style/{slug}',
        'reset_optimizer': f'style-{slug}-v1',
        'reset_best_perf': f'style-{slug}-v1',
        # 暖启 ckpt 携带父代主线 steps（如 s6c 的 940k）≥ max_train_steps，会在
        # 预算检查处零步退出（"training budget already reached"）——必须归零
        'reset_steps': True,
        'enable_test_play': False,
        'batch_size': args.batch_size,
        'max_train_steps': args.max_train_steps,
        'save_every': args.save_every,
        'snapshot_every': args.snapshot_every,
        'test_every': args.max_train_steps,      # 仅满足断言 test_every % save_every == 0
        'seed': args.seed,
        'enable_compile': False,
    })
    cfg['control'] = control

    dataset = dict(cfg.get('dataset', {}))
    dataset.update({
        # 精确文件列表（非目录通配）：
        #   ① 绕开 35 个 gzip 损坏文件（_invalid_files.txt，头部不可读，通配扫描会崩）
        #   ② 天然排除 holdout 赛季
        #   ③ manifest 的 globs 键逐文件比对，任何文件变动自动失效重建
        'globs': stats['train_files'],
        'player_names_files': [f'../dataset/index/style/{slug}.txt'],
        'file_index': f'indexes/style/{slug}_train.pth',
        'data_version': f'style-{slug}-v1',
        'file_batch_size': 16,
        'reserve_ratio': 0.0,
        'num_workers': 2,
        'num_epochs': 10,                       # 富余值，实际步数由 max_train_steps 截断
        'enable_augmentation': True,            # 数百局量级下增强翻倍价值大
        'augmented_first': True,
        # M-League 转换缺陷：丢弃「动作不在合法 mask」的坏决策点（如鸣牌后仍标
        # 记立直 action=37；diag_mleague.py 实测 18/33602 = 0.054%）
        'filter_invalid_actions': True,
    })
    cfg['dataset'] = dataset

    cfg['style'] = {
        'kl_weight': args.kl_weight,
        'kl_state_file': args.parent_rel,
        'trainable': args.trainable,
        'unfreeze_last_blocks': args.unfreeze_last_blocks,
        'logit_temp': 1.0,
        'parent_sha256': sha16(args.parent),
    }

    cfg['freeze_bn'] = dict(cfg.get('freeze_bn', {}))
    cfg['freeze_bn']['mortal'] = True           # 数百局下 BN 批统计必被污染，显式覆盖父代

    cfg['cql'] = dict(cfg.get('cql', {}))
    cfg['cql']['min_q_weight'] = 0              # bc_style 不读，置 0 保持语义干净
    cfg['aux'] = dict(cfg.get('aux', {}))
    cfg['aux']['next_rank_weight'] = 0

    optim = dict(cfg.get('optim', {}))
    scheduler = dict(optim.get('scheduler', {}))
    scheduler.update({
        'peak': args.peak_lr,
        'final': 1e-7,
        'warm_up_steps': args.warm_up_steps,
        'max_steps': args.max_train_steps,
    })
    optim['scheduler'] = scheduler
    cfg['optim'] = optim
    return cfg


def check_asserts(cfg: dict) -> None:
    """train.py 启动断言的前置自检（train.py:176-181 同款）。"""
    c = cfg['control']
    save, opt = c['save_every'], c['opt_step_every']
    test, snap, mx = c['test_every'], c['snapshot_every'], c['max_train_steps']
    assert save % opt == 0, f'save_every % opt_step_every: {save} % {opt}'
    assert test % save == 0, f'test_every % save_every: {test} % {save}'
    assert snap % save == 0, f'snapshot_every % save_every: {snap} % {save}'
    assert mx == 0 or mx % save == 0, f'max_train_steps % save_every: {mx} % {save}'


def main():
    ap = argparse.ArgumentParser(description='生成 style_<slug> 训练配置 toml')
    ap.add_argument('--parent', required=True, help='父代 ckpt 路径（repo root 相对）')
    ap.add_argument('--stats', required=True, help='make_player_whitelist.py 产出的 stats.json')
    ap.add_argument('--slug', required=True)
    ap.add_argument('--kl-weight', type=float, default=1.5)
    ap.add_argument('--trainable', choices=('head', 'last_blocks', 'full'), default='head')
    ap.add_argument('--unfreeze-last-blocks', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-train-steps', type=int, default=800)
    ap.add_argument('--save-every', type=int, default=100)
    ap.add_argument('--snapshot-every', type=int, default=200)
    ap.add_argument('--peak-lr', type=float, default=1e-5)
    ap.add_argument('--warm-up-steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20261027)
    args = ap.parse_args()

    with open(args.stats, encoding='utf-8') as f:
        stats = json.load(f)

    state = torch.load(args.parent, map_location=torch.device('cpu'), weights_only=True)
    parent_cfg = state['config']
    if not isinstance(parent_cfg, dict) or 'control' not in parent_cfg:
        raise SystemExit(f'{args.parent} 内无完整 config，无法以父代为底生成')

    args.parent_rel = args.parent.replace('mortal/', '', 1) if args.parent.startswith('mortal/') else args.parent
    cfg = build_config(parent_cfg, stats, args)
    check_asserts(cfg)

    out = Path('mortal') / f'config.style_{args.slug}.toml'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'# 风格模仿微调配置（make_style_cfg.py 生成，2026-10-27）\n')
        f.write(f'# 玩家: {stats["player"]}  训练 {stats["train_games"]} 局 / holdout {stats["holdout_games"]} 局\n')
        f.write(f'# 父代: {args.parent} (sha256:{cfg["style"]["parent_sha256"]})\n')
        f.write(f'# 启动: cd mortal && MORTAL_CFG=config.style_{args.slug}.toml python train.py\n\n')
        f.write(toml.dumps(cfg))

    print(f'config: {out}')
    print(f'  trainable={args.trainable} kl_weight={args.kl_weight} '
          f'max_train_steps={args.max_train_steps} batch={args.batch_size}')


if __name__ == '__main__':
    main()