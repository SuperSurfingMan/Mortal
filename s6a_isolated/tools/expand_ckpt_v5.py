#!/usr/bin/env python3
"""S6a：把 version-4 checkpoint 手术为 version-5（首层 conv 输入通道 1012 -> 1018）。

v5 = v4 + 6 行「逆转紧迫度」obs 特征（`libriichi/consts.rs`: obs_shape(5)=(1018,34)）。
唯一形状变化是 Brain 首层 Conv1d 的输入通道；新通道权重零初始化 —— 当新输入通道
为 0 时前向与 v4 模型逐位等价，因此语义等价于「原模型 + 紧迫度特征从零学起」。
同时把 ckpt 内嵌 config 的 `control.version` 改为 5：`ladder.py` / `mortal.py` 等
按 ckpt 内嵌 config 构建模型，版本不同步会导致加载失败。

用法（任意目录；脚本不依赖 cwd）:
    python tools/training/expand_ckpt_v5.py <src_v4.pth> <dst_v5.pth>
"""
import argparse
import datetime
import sys

import torch

OLD_IN = 1012
NEW_IN = 1018
FIRST_CONV = 'encoder.net.0.weight'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', help='父代 v4 checkpoint')
    ap.add_argument('dst', help='输出 v5 checkpoint')
    ap.add_argument('--old-in', type=int, default=OLD_IN)
    ap.add_argument('--new-in', type=int, default=NEW_IN)
    args = ap.parse_args()

    st = torch.load(args.src, map_location='cpu', weights_only=False)
    version = st.get('config', {}).get('control', {}).get('version')
    if version != 4:
        sys.exit(f'[FAIL] 父代应为 version 4，实际为 {version}: {args.src}')

    sd = st['mortal']
    if FIRST_CONV not in sd:
        sys.exit(f'[FAIL] 缺少 {FIRST_CONV}（state 可能不是 Brain）')
    w = sd[FIRST_CONV]
    if w.dim() != 3 or w.shape[1] != args.old_in:
        sys.exit(f'[FAIL] 首层 conv 形状异常: {tuple(w.shape)}（期望 [*, {args.old_in}, *]）')

    out_ch, _, k = w.shape
    w_new = torch.zeros(out_ch, args.new_in, k, dtype=w.dtype)
    w_new[:, :args.old_in, :] = w
    sd[FIRST_CONV] = w_new

    st['config']['control']['version'] = 5
    st['timestamp'] = datetime.datetime.now().timestamp()
    torch.save(st, args.dst)

    n_new = args.new_in - args.old_in
    print(f'[OK] {args.src} (v4, in={args.old_in}) -> {args.dst} (v5, in={args.new_in})')
    print(f'     首层 {tuple(w.shape)} -> {tuple(w_new.shape)}；新增 {n_new} 通道零初始化；'
          f'steps={st.get("steps")}')


if __name__ == '__main__':
    main()
