#!/usr/bin/env python3
"""全知数据下的「铳牌动作损失」测量（有人听牌时打哪张牌的期望失点）。

对每个我方打牌决策：用 4 个 `PlayerState` 同步重放，取对手真实听牌（全知），
`loss[a] = Σ_j [a ∈ waits_j] · loss_pt`（先统一按每家 3000 点粗算）。

输出：人类在「有人听牌」时的弃牌率、期望失点、以及「明明有安全牌却打危险牌」的比例。

用法（mortal/ 下）：python ../tools/oracle_deal_loss.py --games 120
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

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

from libriichi.state import PlayerState  # noqa: E402

TILE = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
        '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
        '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
        'E', 'S', 'W', 'N', 'P', 'F', 'C']
TILE_IDX = {t: i for i, t in enumerate(TILE)}
AKA = {'5mr': '5m', '5pr': '5p', '5sr': '5s'}


def pai_idx(p):
    return TILE_IDX.get(AKA.get(p, p), -1)


def as34(x):
    """把 waits / tehai 统一成 34 长度的一维 int 数组。"""
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
    a = ap.parse_args()
    files = sorted(glob.glob('../dataset/mjai/houou4/202609/**/*.json.gz', recursive=True))
    random.Random(a.seed).shuffle(files)

    n_state = n_tenpai = 0
    lh, ls = [], []
    fold = push = safe_avail = 0
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
                w = np.stack([as34(sts[j].waits) for j in (1, 2, 3)])   # (3,34) 0/1
                snap = (w, as34(sts[0].tehai))
            elif t == 'dahai' and e.get('actor') == 0 and snap is not None:
                w, tehai = snap
                snap = None
                n_state += 1
                loss = w.sum(0).astype(np.float64) * 3000.0
                if loss.max() <= 0:
                    continue
                n_tenpai += 1
                own = np.flatnonzero(tehai > 0)
                if own.size == 0:
                    continue
                c = pai_idx(e.get('pai'))
                if c < 0:
                    continue
                l_h = float(loss[c])
                l_min = float(loss[own].min())
                lh.append(l_h); ls.append(l_min)
                if l_h == 0:
                    fold += 1
                else:
                    push += 1
                    if l_min == 0:
                        safe_avail += 1
        if not ok:
            continue

    print(f'[oracle] 我方决策 {n_state:,}；其中「有人听牌」{n_tenpai:,}')
    if not lh:
        print('[oracle] 无样本')
        return 0
    lh = np.asarray(lh); ls = np.asarray(ls)
    print(f'[oracle] 人类：打安全牌 {fold/len(lh):.3f} / 打危险牌 {push/len(lh):.3f}')
    print(f'[oracle] 期望失点：人类 {lh.mean():.0f} / 最安全 {ls.mean():.0f} / 可省 {lh.mean()-ls.mean():+.0f}')
    print(f'[oracle] 打危险牌时其实有安全牌可取的比例 {safe_avail/max(push,1):.3f}')
    print(f'[oracle] 零失误下可避免的失点占比 {1 - ls.mean()/max(lh.mean(),1e-9):.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
