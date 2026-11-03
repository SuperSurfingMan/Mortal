#!/usr/bin/env python3
"""Oracle Guiding 的首层手术：给 v4 ckpt 追加 217 行「三家暗牌」输入通道。

与 `tools/expand_ckpt_v6.py`（obs 尾部 +N 行特征）同思路，区别：
- 新增的是 `oracle_obs_shape(version)[0] = 217` 行隐藏信息（三家手牌/摸牌等）；
- 新通道**零初始化**，所以暖启瞬间 `Brain(obs, invisible_obs=0)` 与父代逐位等价；
- `config['resnet']['is_oracle'] = True`，推理侧 engine 会自动喂全零（α=0）。

用法::

    python tools/expand_ckpt_oracle.py \
        mortal/checkpoints/s6c/mortal.pth \
        mortal/checkpoints/og/mortal.pth
"""
import argparse
import datetime
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'mortal'))

import model as M  # noqa: E402
import s6c_features  # noqa: E402
from libriichi.consts import oracle_obs_shape  # noqa: E402

FIRST_CONV = 'encoder.net.0.weight'


def build(cfg, state):
    version = cfg.get('control', {}).get('version', 4)
    # 与 train.py 同口径：extra_channels 由特征族名推导（S6c=2；旧 ckpt=0）
    _, extra = s6c_features.resolve(cfg.get('control', {}))
    net = M.Brain(version=version, extra_channels=extra, **cfg['resnet']).eval()
    net.load_state_dict(state)
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', help='父代 v4 checkpoint（可带 S6c extra_channels）')
    ap.add_argument('dst', help='输出 oracle 起手 ckpt')
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()

    st = torch.load(args.src, map_location='cpu', weights_only=False)
    cfg = st.get('config', {})
    version = int(cfg.get('control', {}).get('version', 4))
    if bool(cfg.get('resnet', {}).get('is_oracle', False)):
        sys.exit('[FAIL] 父代已是 is_oracle=True，拒绝二次手术')

    sd = st['mortal']
    if FIRST_CONV not in sd:
        sys.exit(f'[FAIL] 缺少 {FIRST_CONV}')
    w = sd[FIRST_CONV]
    old_in = int(w.shape[1])
    n_oracle = int(oracle_obs_shape(version)[0])
    out_ch, _, k = w.shape
    new_in = old_in + n_oracle

    w_new = torch.zeros(out_ch, new_in, k, dtype=w.dtype)
    w_new[:, :old_in, :] = w
    sd[FIRST_CONV] = w_new

    cfg.setdefault('resnet', {})['is_oracle'] = True
    st['config'] = cfg
    st['timestamp'] = datetime.datetime.now().timestamp()
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(st, args.dst)
    print(f'[OK] {args.src} (v{version}, in={old_in}) -> {args.dst} '
          f'(in={new_in}, oracle={n_oracle})')
    print(f'     首层 {tuple(w.shape)} -> {tuple(w_new.shape)}；'
          f'steps={st.get("steps")}  params={sum(v.numel() for v in sd.values()):,}')

    if args.no_verify:
        return
    # ---- 数值自检：新通道零权重 ⇒ 无论 oracle 输入是什么，前向与父代逐位一致 ----
    torch.manual_seed(0)
    n = 4
    obs = torch.randn(n, old_in, 34)
    inv = torch.randn(n, n_oracle, 34)
    cfg_parent = {**cfg, 'resnet': {**cfg['resnet'], 'is_oracle': False}}
    sd_old = dict(st['mortal'])
    sd_old[FIRST_CONV] = w
    m0 = build(cfg_parent, sd_old)
    m1 = build(cfg, st['mortal'])
    with torch.no_grad():
        phi0 = m0(obs)
        phi1 = m1(obs, inv)   # Brain 内部会 cat([obs, inv])
    d = (phi0 - phi1).abs().max().item()
    print(f'[verify] phi max|diff| = {d:.3e}  (应 == 0：新通道权重零初始化)')
    if d > 1e-6:
        sys.exit('[FAIL] 手术前后前向不等价')


if __name__ == '__main__':
    main()
