#!/usr/bin/env python3
"""把「高置信分歧」拆成「模型对 / 模型错」：用可观测的手牌级真值裁决。

⚠️ 状态（2026-10-28）：**未跑通**。read-model harness 的样本里没有人类动作张量
（collate 丢掉 `action`），`_event` 也不是动作 id（实测取到的是别的编号，导致 98.7%
被判为「分歧」、n 只有 685）。**修复办法**：改用 `GameplayLoader` 直接取
`game.take_actions()`（见 `tools/label_noise_diag.py` 的做法）作为 y，再从日志
解析本局和了牌作为 `danger` 真值，绕开 read-model harness。


在**模型 top1 ≠ 人类动作**的状态上，按 Q 边际（top1−次优）分成低/中/高三档，比较：

- `deal`：该牌是否为**本局某家最终和了牌**（`wait` 真值）——越低越安全；
- `p_win` / `ev_norm`：v4 obs 里 SP 前瞻给出的和了概率与期望价值——越高越进攻。

若在高置信档里模型的选择**更安全且进攻不差**，说明这些分歧是「人类偏离」而非模型错，
BC 一致率就不是有效目标；反之则说明存在可定向修复的系统性错误。

用法（mortal/ 下）：
    python ../tools/disagree_split.py --n 20000
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'tools' / 'read_model'))
sys.path.insert(0, str(REPO / 'tools' / 'read_model' / 'diagnostics'))
sys.path.insert(0, str(REPO / 'mortal'))

import _common as C  # noqa: E402
import ev_rerank as E  # noqa: E402
import guard_state as G  # noqa: E402

_spec = importlib.util.spec_from_file_location('p0', REPO / 'tools/read_model/inject_p0.py')
P0 = importlib.util.module_from_spec(_spec)
sys.modules['p0'] = P0
_spec.loader.exec_module(P0)

NUM_TILES, ACTION_SPACE, T = P0.NUM_TILES, P0.ACTION_SPACE, P0.T
CKPT = REPO / 'mortal/checkpoints/s6c/mortal.pth'

_SUITS = {'m': 0, 'p': 9, 's': 18}
_HONORS = {'E': 27, 'S': 28, 'W': 29, 'N': 30, 'P': 31, 'F': 32, 'C': 33}


def tile_idx(pai: str):
    """'5m'→4, '5pr'→13, 'E'→27；无法识别返回 None。"""
    if not pai:
        return None
    if pai in _HONORS:
        return _HONORS[pai]
    d = pai[0]
    if not d.isdigit() or len(pai) < 2 or pai[1] not in _SUITS:
        return None
    return _SUITS[pai[1]] + int(d) - 1


def load_nets():
    st = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = st['config']; ver = cfg['control'].get('version', 4)
    _, extra = __import__('s6c_features').resolve(cfg['control'])
    mortal = P0.MM.Brain(version=ver, extra_channels=extra, **cfg['resnet']).to(P0.DEV).eval()
    mortal.load_state_dict(st['mortal'])
    dqn = P0.MM.DQN(version=ver).to(P0.DEV).eval()
    dqn.load_state_dict(st['current_dqn'])
    return mortal, dqn, extra, int(st.get('steps', 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--out', default=str(C.OUT_DIR / 'disagree_split.json'))
    a = ap.parse_args()
    import s6c_features as S6
    files = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4/202609/**/*.json.gz'), recursive=True))
    mortal, dqn, extra, steps = load_nets()
    print(f'[split] ckpt steps={steps} extra={extra} n={a.n}', flush=True)

    rows = []
    total = 0
    for buf, _src in P0.iter_eval_chunks(files, a.n, a.seed, a.chunk):
        take = min(len(buf), a.n - total)
        if take <= 0:
            break
        b = T.collate(buf[:take])
        # 人类动作从 `_event` 解码（read-model 样本不带 action 张量）
        ylist, keep = [], []
        for i, smp in enumerate(buf[:take]):
            ev = smp.get('_event')
            t = int(ev) if isinstance(ev, (int, np.integer)) else None
            if t is not None and 0 <= t < NUM_TILES:
                ylist.append(t); keep.append(i)
        if not keep:
            continue
        keep = np.asarray(keep, int)
        y = np.asarray(ylist, int)
        obs = b['obs'].float().numpy()[keep]
        own = b['own_hand'].numpy()[keep] > 0
        danger = (b['wait'][..., :NUM_TILES] > 0).any(1).numpy()[keep]
        x = S6.attach(obs, S6.compute(obs)) if extra else obs
        with torch.no_grad():
            phi = mortal(torch.as_tensor(x).to(P0.DEV))
            mask = torch.zeros(phi.shape[0], ACTION_SPACE, dtype=torch.bool, device=P0.DEV)
            mask[:, :NUM_TILES] = torch.as_tensor(own).to(P0.DEV)
            q = dqn(phi, mask)[:, :NUM_TILES].float().cpu().numpy()
        sp = E.sp_features(obs)
        order = np.argsort(-np.where(own, q, -1e9), axis=1)
        top1 = order[:, 0]
        margin = np.where(own, q, -1e9)[np.arange(len(q)), top1] - \
                 np.where(own, q, -1e9)[np.arange(len(q)), order[:, 1]]
        disagree = top1 != y
        rows.append(dict(q=q, own=own, y=y, top1=top1, margin=margin,
                         danger=danger, pwin=sp['p_win'], ev=sp['ev_norm'], disagree=disagree))
        total += len(y)
        print(f'[data] {total}/{a.n}', flush=True)
        if total >= a.n:
            break

    y = np.concatenate([r['y'] for r in rows])
    top1 = np.concatenate([r['top1'] for r in rows])
    margin = np.concatenate([r['margin'] for r in rows])
    danger = np.concatenate([r['danger'] for r in rows])
    own = np.concatenate([r['own'] for r in rows])
    pwin = np.concatenate([r['pwin'] for r in rows])
    ev = np.concatenate([r['ev'] for r in rows])
    dis = np.concatenate([r['disagree'] for r in rows])
    ar = np.arange(len(y))
    print(f'\n[总体] n={len(y):,}  一致 {1-dis.mean():.4f}  分歧 {dis.mean():.4f}')
    # 只在分歧样本上，按 margin 三分位分层
    dm = margin[dis]; idx = np.where(dis)[0]
    qs = np.quantile(dm, [1/3, 2/3])
    print(f'  分歧处 margin 中位 {np.median(dm):.3f}  三分位 {qs[0]:.3f} / {qs[1]:.3f}')
    print(f'  {"分层":<6}{"n":>7}{"margin":>9}{"deal(模型)":>11}{"deal(人类)":>11}{"p_win(模)":>10}{"p_win(人)":>10}{"ev(模)":>8}{"ev(人)":>8}')
    for lo, hi, tag in ((-1e9, qs[0], '低'), (qs[0], qs[1], '中'), (qs[1], 1e9, '高')):
        m = idx[(margin[idx] > lo) & (margin[idx] <= hi)]
        print(f'  {tag:<6}{len(m):>7}{np.median(margin[m]):>9.3f}'
              f'{danger[m, top1[m]].mean():>11.4f}{danger[m, y[m]].mean():>11.4f}'
              f'{pwin[m, top1[m]].mean():>10.4f}{pwin[m, y[m]].mean():>10.4f}'
              f'{ev[m, top1[m]].mean():>8.4f}{ev[m, y[m]].mean():>8.4f}')
    # 对照：一致样本
    ag = idx[np.zeros(0, int)] if False else np.where(~dis)[0]
    print(f'  {"一致":<6}{len(ag):>7}{np.median(margin[ag]):>9.3f}'
          f'{danger[ag, top1[ag]].mean():>11.4f}{"—":>11}'
          f'{pwin[ag, top1[ag]].mean():>10.4f}{"—":>10}'
          f'{ev[ag, top1[ag]].mean():>8.4f}{"—":>8}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
