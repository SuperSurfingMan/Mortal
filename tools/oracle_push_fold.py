#!/usr/bin/env python3
"""押し引き的**价值裁决**：对攻 EV 是否盖过弃牌 EV。

对每个「有人听牌」的我方决策状态：

```
atk(a)  = SP 表（ev_norm × max_ev，单位点）
def(a)  = Σ_j waits_j[a] × pt_j          # 全知；立直 5200 / 默听 3500 / 副露 3000
EV_push = max_a [atk(a) − def(a)]
EV_fold = max_{a: def(a)=0} atk(a)       # 安牌里攻击力最高的一张
Δ       = EV_push − EV_fold              # >0 ⇒ 该推；<0 ⇒ 该弃
```

统计 Δ 在「人类推了对攻」「模型推了对攻」这两组上的分布：
若多数 Δ<0 ⇒ 人类在过度对攻，模型可以靠更会弃赚分；反之这条线关闭。

用法（mortal/ 下）：python ../tools/oracle_push_fold.py --games 120
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys

import numpy as np
import torch  # noqa: F401  (ev_rerank 依赖)

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

from libriichi.state import PlayerState  # noqa: E402
import ev_rerank as E  # noqa: E402

TILE = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
        '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
        '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
        'E', 'S', 'W', 'N', 'P', 'F', 'C']
TILE_IDX = {t: i for i, t in enumerate(TILE)}
AKA = {'5mr': '5m', '5pr': '5p', '5sr': '5s'}


def pai_idx(p):
    return TILE_IDX.get(AKA.get(p, p), -1)


def as34(x):
    if isinstance(x, (bytes, bytearray)):
        arr = np.frombuffer(x, dtype=np.uint8).astype(np.int32)
    else:
        arr = np.asarray(list(x) if hasattr(x, '__iter__') else [x], dtype=np.int32).ravel()
    if arr.size < 34:
        arr = np.pad(arr, (0, 34 - arr.size))
    return arr[:34]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=120)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--model', action='store_true')
    a = ap.parse_args()
    files = sorted(glob.glob('../dataset/mjai/houou4/202609/**/*.json.gz', recursive=True))
    random.Random(a.seed).shuffle(files)

    NET = DQN = None
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VER, EXTRA, S6 = 4, 0, None
    if a.model:
        import model as MM
        import s6c_features as _S6
        S6 = _S6
        stk = torch.load('checkpoints/s6c/mortal.pth', map_location='cpu', weights_only=False)
        VER = stk['config']['control'].get('version', 4)
        _, EXTRA = S6.resolve(stk['config']['control'])
        NET = MM.Brain(version=VER, extra_channels=EXTRA, **stk['config']['resnet']).eval().to(DEV)
        NET.load_state_dict(stk['mortal'])
        DQN = MM.DQN(version=VER).eval().to(DEV)
        DQN.load_state_dict(stk['current_dqn'])
        print(f'[push] 模型已加载 steps={stk.get("steps")}', flush=True)

    n_state = n_tp = 0
    deltas, deltas_hpush, deltas_mpush = [], [], []
    h_correct = h_wrong = m_correct = m_wrong = 0
    for path in files[:a.games]:
        try:
            evs = [json.loads(l) for l in gzip.open(path, 'rt') if l.strip()]
        except Exception:
            continue
        sts = [PlayerState(j) for j in range(4)]
        snap = None
        ok = True
        for e in evs:
            t = e.get('type')
            try:
                for st in sts:
                    st.update(json.dumps(e))
            except Exception:
                ok = False
                break
            if t == 'tsumo' and e.get('actor') == 0:
                snap = (np.stack([as34(sts[j].waits) for j in (1, 2, 3)]),
                        as34(sts[0].tehai))
            elif t == 'dahai' and e.get('actor') == 0 and snap is not None:
                w, tehai = snap
                snap = None
                n_state += 1
                if w.sum() == 0:
                    continue
                n_tp += 1
                pt = np.zeros(3, np.float64)
                for k, j in enumerate((1, 2, 3)):
                    try:
                        ri = bool(sts[j].self_riichi_accepted)
                    except Exception:
                        ri = False
                    meld = False
                    for attr in ('pons', 'chis', 'minkans'):
                        try:
                            v = getattr(sts[j], attr)
                            if (v() if callable(v) else v):
                                meld = True
                                break
                        except Exception:
                            pass
                    pt[k] = 5200.0 if ri else (3000.0 if meld else 3500.0)
                defv = (w * pt[:, None]).sum(0).astype(np.float64)
                om34 = (tehai > 0) & (np.arange(34) >= 0)
                own = np.flatnonzero(om34)
                if own.size == 0:
                    continue
                try:
                    o, _m = sts[0].encode_obs(VER, False)
                except Exception:
                    continue
                sp = E.sp_features(o[None] if o.ndim == 2 else o)
                atk = (sp['ev_norm'][0] * sp['max_ev'][0]).astype(np.float64)
                if not np.isfinite(atk).all():
                    continue
                score = np.where(om34, atk - defv, -1e18)
                ev_push = float(score[om34].max())
                safe = own[defv[own] <= 0] if own.size else own
                if safe.size == 0:
                    continue
                ev_fold = float(atk[safe].max())
                d = ev_push - ev_fold
                deltas.append(d)
                c = pai_idx(e.get('pai'))
                if c >= 0:
                    if defv[c] > 0:
                        deltas_hpush.append(d)
                        h_correct += int(d > 0); h_wrong += int(d < 0)
                    if NET is not None:
                        try:
                            om = (np.rint(o[:4].sum(0)) > 0) if o.ndim == 2 else (np.rint(o[:4]) > 0)
                            xx = S6.attach(o[None], S6.compute(o[None])) if EXTRA else o[None]
                            with torch.inference_mode():
                                phi = NET(torch.as_tensor(xx, dtype=torch.float32).to(DEV))
                                mk = torch.zeros(1, 46, dtype=torch.bool, device=DEV)
                                mk[0, :34] = torch.as_tensor(om).to(DEV)
                                qq = DQN(phi, mk)[0, :34].float().cpu().numpy()
                            pick = int(np.argmax(np.where(om, qq, -1e9)))
                            if defv[pick] > 0:
                                deltas_mpush.append(d)
                                m_correct += int(d > 0); m_wrong += int(d < 0)
                        except Exception:
                            pass
        if not ok:
            continue

    def show(tag, arr, corr=None, wrong=None):
        if not arr:
            print(f'{tag}: 无样本')
            return
        x = np.asarray(arr)
        print(f'{tag}: n={len(x)}  中位 Δ={np.median(x):+.0f}  均值 {x.mean():+.0f}  '
              f'Δ>0 比例 {np.mean(x > 0):.3f}  Δ<0 比例 {np.mean(x < 0):.3f}')
        if corr is not None:
            print(f'     —— 其中「该推(Δ>0)」{corr} / 「该弃(Δ<0)」{wrong}')

    print(f'[push] 我方决策 {n_state:,}；有人听牌 {n_tp:,}')
    show('[全部有人听牌]', deltas)
    show('[人类推了对攻]', deltas_hpush, h_correct, h_wrong)
    if a.model:
        show('[模型推了对攻]', deltas_mpush, m_correct, m_wrong)
    return 0


if __name__ == '__main__':
    sys.exit(main())
