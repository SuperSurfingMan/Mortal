#!/usr/bin/env python3
"""obs 尾部特征注入的首层手术：把 version-4 checkpoint 扩成「+N 行」的注入起手 ckpt。

用途（两个阶段共用同一套手术）：
- S6b（已关闭）：+2 行读牌危险度（`tools/read_model/precompute_danger.py` 预计算）；
- S6c（当前）：+2 行自家速度特征（`mortal/s6c_features.py` 在线计算）。

与 S6a（``s6a_isolated/tools/expand_ckpt_v5.py``）的区别：
- **不改** ``control.version``（obs_shape 仍 (1012,34)；v4 编码零改动），
  而是把外部特征族记在 ``control.extra_features``、行数记在 ``control.extra_channels``，
  ``Brain`` 用 ``obs_shape(version)[0] + extra_channels`` 建首层。
- 首层 ``Conv1d(192, 1012, 3)`` → ``Conv1d(192, 1014, 3)``；新增 2 通道
  **零初始化**，新输入为 0 时前向与父代逐位等价（脚本内置数值自检）。

S6c 的 2 行（见 ``mortal/s6c_features.py``）：
- row 1012 = ``clamp(自家向听, 0, 8) / 8``（34 列同值；自摸后为当前手牌真值）
- row 1013 = ``max_候选(有效牌剩余枚数) / 136``（tiles_seen 实战口径，34 列同值）

用法::

    python tools/expand_ckpt_v6.py \
        mortal/checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth \
        mortal/checkpoints/s6c/B756k_s6c_init.pth
"""
import argparse
import datetime
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'mortal'))

import model as M  # noqa: E402  （mortal/model.py，cwd 无关）

OLD_IN = 1012
FIRST_CONV = 'encoder.net.0.weight'


def build(cfg, state, extra):
    net = M.Brain(version=cfg['control'].get('version', 4),
                  extra_channels=extra, **cfg['resnet']).eval()
    net.load_state_dict(state)
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', help='父代 v4 checkpoint（如 B/756k）')
    ap.add_argument('dst', help='输出注入起手 ckpt')
    ap.add_argument('--extra-channels', type=int, default=2)
    ap.add_argument('--extra-features', default='s6c',
                    help="特征族名（写进 control.extra_features；'' = 只扩通道不标特征）")
    ap.add_argument('--old-in', type=int, default=OLD_IN)
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()

    st = torch.load(args.src, map_location='cpu', weights_only=False)
    cfg = st.get('config', {})
    version = cfg.get('control', {}).get('version', 4)
    if int(cfg.get('control', {}).get('extra_channels', 0)) != 0:
        sys.exit(f'[FAIL] 父代已带 extra_channels={cfg["control"]["extra_channels"]}，拒绝二次手术')

    sd = st['mortal']
    if FIRST_CONV not in sd:
        sys.exit(f'[FAIL] 缺少 {FIRST_CONV}')
    w = sd[FIRST_CONV]
    if w.dim() != 3 or w.shape[1] != args.old_in:
        sys.exit(f'[FAIL] 首层形状异常: {tuple(w.shape)}（期望 [*, {args.old_in}, *]）')

    out_ch, _, k = w.shape
    new_in = args.old_in + args.extra_channels
    w_new = torch.zeros(out_ch, new_in, k, dtype=w.dtype)
    w_new[:, :args.old_in, :] = w
    w_old = w.clone()                      # 自检用（父代首层原样）
    sd[FIRST_CONV] = w_new

    cfg.setdefault('control', {})['extra_channels'] = args.extra_channels
    if args.extra_features:
        cfg['control']['extra_features'] = args.extra_features
    st['config'] = cfg
    st['timestamp'] = datetime.datetime.now().timestamp()
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(st, args.dst)
    print(f'[OK] {args.src} (v{version}, in={args.old_in}) -> {args.dst} '
          f'(in={new_in}, extra={args.extra_channels}, features={args.extra_features!r})')
    print(f'     首层 {tuple(w.shape)} -> {tuple(w_new.shape)}；'
          f'steps={st.get("steps")}  params={sum(v.numel() for v in sd.values()):,}')

    if args.no_verify:
        return
    # ---- 数值自检：新通道权重零初始化 ⇒ 无论新通道输入多少，前向都与父代逐位一致 ----
    torch.manual_seed(0)
    n = 4
    obs0 = torch.randn(n, args.old_in, 34)
    # 故意用非零输入：证明「不依赖新通道输入为 0」，而是首层权重贡献为 0
    obs1 = torch.cat([obs0, torch.rand(n, args.extra_channels, 34) + 0.5], dim=1)
    sd_old = dict(st['mortal'])
    sd_old[FIRST_CONV] = w_old
    m0 = build(cfg, sd_old, 0)
    m1 = build(cfg, st['mortal'], args.extra_channels)
    with torch.no_grad():
        phi0, phi1 = m0(obs0), m1(obs1)
    d = (phi0 - phi1).abs().max().item()
    print(f'[verify] phi max|diff| = {d:.3e}  (应 == 0：新通道权重零初始化)')
    if d > 1e-6:
        sys.exit('[FAIL] 手术前后前向不等价')


if __name__ == '__main__':
    main()
