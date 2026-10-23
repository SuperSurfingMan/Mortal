#!/usr/bin/env python3
"""S6c：首层新通道的「使用度」诊断（S6a 的教训：注入不等于被使用）。

两种口径：
- **权重变化量（推荐，需 --ref）**：`ΔL2[c] = ||w[:, c, :] − w_ref[:, c, :]||₂`，
  比较新通道 ΔL2 落在原始 1012 通道 ΔL2 分布的哪个分位。新通道零初始化 ⇒
  它的 ΔL2 就是「学到的量」；与旧通道同一尺度比较即可判断是否真被使用。
  阈值：两个新通道的 ΔL2 分位都 ≥ p10 才算 OK。
- **绝对 L2（无 --ref）**：新通道当前 L2 与原始通道 L2 的分位比较（阈值 p25）。
  冷启动（刚手术完）必然 WARN，只适合「有没有长大」的粗看。

用法（在 mortal/ 里跑）：
    python ../tools/s6c_channel_usage.py checkpoints/s6c/mortal.pth \
        --ref checkpoints/s6c/B756k_s6c_init.pth
"""
import argparse
import sys
from pathlib import Path

import torch

MORTAL = Path(__file__).resolve().parents[1] / 'mortal'
sys.path.insert(0, str(MORTAL))

import s6c_features  # noqa: E402

FIRST_CONV = 'encoder.net.0.weight'
NATIVE = 1012


def _load(ckpt):
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    spec, extra = s6c_features.resolve(st.get('config', {}).get('control', {}))
    w = st['mortal'][FIRST_CONV].float()
    if w.shape[1] != NATIVE + extra:
        raise SystemExit(f'{ckpt}: 首层 in={w.shape[1]}，与 extra={extra} 不符')
    return st, spec, extra, w


def _chan_l2(w):
    return w.pow(2).sum((0, 2)).sqrt()


def report(ckpt, ref=None):
    st, spec, extra, w = _load(ckpt)
    print(f'{Path(ckpt).name:<26} steps={st.get("steps", "-")!s:<8} extra={extra} ({spec or "none"})')
    if extra == 0:
        return True
    l2 = _chan_l2(w)
    q_l2 = torch.quantile(l2[:NATIVE], torch.tensor([0.10, 0.25, 0.50, 0.90]))
    print(f'    绝对 L2  原始: p10={q_l2[0]:.3e} p25={q_l2[1]:.3e} median={q_l2[2]:.3e}')
    print(f'    绝对 L2  新通道: ' + ' '.join(f'{v:.3e}' for v in l2[NATIVE:].tolist()))

    if ref is None:
        ok = bool((l2[NATIVE:] >= q_l2[1]).all())
        print(f'    [{"OK" if ok else "WARN"}] 绝对 L2 ≥ 原始 p25'
              f'{"" if ok else "（零初始化冷启动必然如此，只看变化量请加 --ref）"}')
        return ok

    st0, _, extra0, w0 = _load(ref)
    if extra0 != extra:
        raise SystemExit(f'--ref 的 extra={extra0} 与本 ckpt 的 {extra} 不一致')
    d = _chan_l2(w - w0)
    old = d[:NATIVE]
    q = torch.quantile(old, torch.tensor([0.10, 0.25, 0.50, 0.90]))
    pct = [float((old < v).float().mean() * 100) for v in d[NATIVE:]]
    print(f'    ΔL2 原始({Path(ref).name}): p10={q[0]:.3e} p25={q[1]:.3e} median={q[2]:.3e} p90={q[3]:.3e}')
    print(f'    ΔL2 新通道: ' + ' '.join(f'{v:.3e}' for v in d[NATIVE:].tolist())
          + '   分位: ' + ' '.join(f'{p:.1f}%' for p in pct))
    ok = all(p >= 10.0 for p in pct)
    print(f'    [{"OK" if ok else "WARN"}] 新通道权重变化量 ≥ 原始 p10'
          f'{"" if ok else "：新通道几乎没在学，等同没注入"}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--ref', default=None, help='零初始化时的 ckpt（如手术产物 B756k_s6c_init.pth）')
    args = ap.parse_args()
    ok = all([report(p, args.ref) for p in args.ckpts])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
