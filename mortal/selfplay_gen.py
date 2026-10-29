#!/usr/bin/env python3
"""档 A 的第一步：用现役模型**自博弈**生成 mjai 日志（供 MC 回报训练）。

不做任何搜索，只换数据分布：4 席全部由同一份权重驱动，牌山按种子生成，
日志落盘后与人类日志同格式（`GameplayLoader` 可直接读）。

用法（mortal/ 下）：
    python selfplay_gen.py --sets 200 --seed 700000 --out ../logs/selfplay/s6c_700000
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import torch

import prelude  # noqa: F401,E402
import s6c_features  # noqa: E402
from model import Brain, DQN  # noqa: E402
from engine import MortalEngine  # noqa: E402
from libriichi.arena import OneVsThree  # noqa: E402


def load_engine(state_file: str, device: str, name: str):
    st = torch.load(state_file, weights_only=True, map_location=torch.device('cpu'))
    cfg = st['config']
    ver = cfg['control'].get('version', 1)
    extra_features, extra_ch = s6c_features.resolve(cfg['control'])
    brain = Brain(version=ver, conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks'], extra_channels=extra_ch).eval()
    dqn = DQN(version=ver).eval()
    brain.load_state_dict(st['mortal'])
    dqn.load_state_dict(st['current_dqn'])
    eng = MortalEngine(brain, dqn, is_oracle=False, version=ver,
                       device=torch.device(device), enable_amp=False,
                       enable_rule_based_agari_guard=True,
                       extra_features=extra_features, name=name)
    return eng, int(st.get('steps', 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='checkpoints/s6c/mortal.pth')
    ap.add_argument('--sets', type=int, default=100, help='1 set = 4 半庄（同牌山轮座）')
    ap.add_argument('--seed', type=int, default=700000)
    ap.add_argument('--key', type=int, default=0x2000)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='../logs/selfplay/run')
    ap.add_argument('--keep', action='store_true', help='保留已有目录（默认清空）')
    a = ap.parse_args()

    out = pathlib.Path(a.out).resolve()
    if out.is_dir() and not a.keep:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    eng_a, steps = load_engine(a.state, a.device, 'selfplay_a')
    eng_b, _ = load_engine(a.state, a.device, 'selfplay_b')
    print(f'[selfplay] state={a.state} steps={steps} sets={a.sets} seed={a.seed} out={out}', flush=True)

    env = OneVsThree(disable_progress_bar=False, log_dir=str(out))
    env.py_vs_py(challenger=eng_a, champion=eng_b,
                 seed_start=(a.seed, a.key), seed_count=a.sets)
    n = len(list(out.glob('*.json.gz')))
    print(f'[selfplay] 完成：{n} 个日志 → {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
