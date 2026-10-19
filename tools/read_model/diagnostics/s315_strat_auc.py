"""分层 tenpai AUC：softmax 读法（现行官方口径）vs sigmoid 读法（BCE 真正标定的量）。

若困难分层的 AUC 在正确读法下明显抬升，则「读侧已近公共信息天花板」的结论需要重判。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _common import (CKPT, DEV, EPS, OUT_DIR, SUIT, VAL_GLOB, boot_ci, build_eval,
                     col_ce, counts_tier, fmt_ci, forward, load_net, parse_tl,
                     suit_window)

import sys, glob
import numpy as np
import torch

import train as T
import model_v2 as M2

DEV = 'cuda'
st = torch.load(str(CKPT),
                map_location='cpu', weights_only=False)
cfg = st['config']
net = M2.build_model_v2s(conv_channels=cfg['conv_channels'], num_blocks=cfg['num_blocks'],
                         with_aux=not cfg['no_aux']).to(DEV).eval()
net.load_state_dict(st['model'])
files = sorted(glob.glob(VAL_GLOB,
                         recursive=True))
N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
b = T.build_fixed_eval_set(files, N_EVAL)
n = b['obs'].shape[0]
L34, PN = [], []
with torch.no_grad():
    for i in range(0, n, 2048):
        ch = {k: v[i:i + 2048].to(DEV) for k, v in b.items()}
        wb = T.wait_bias_of(ch).to(DEV)
        o = net(ch['obs'], ch['unseen'], wb, ch['tl_tokens'], ch['tl_mask'])
        L34.append(o['wait_logits'][..., 34].cpu().numpy())
        PN.append(o['wait_probs'][..., 34].cpu().numpy())
l34 = np.concatenate(L34); pn = np.concatenate(PN)
s_soft = 1.0 - pn                        # 现行口径
s_sig = 1.0 - 1.0 / (1.0 + np.exp(-l34))  # BCE 标定口径
ten = b['tenpai'].numpy() > 0.5
ri = b['riichi'].numpy() > 0.5
ca = b['called'].numpy() > 0.5
turn = np.broadcast_to(b['turn'].numpy()[:, None], ten.shape)


def auc(y, s):
    y = np.asarray(y).ravel().astype(bool); s = np.asarray(s).ravel()
    if y.all() or (~y).all():
        return float('nan')
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    ss = s[o]; i = 0
    while i < len(ss):                      # 并列取平均秩
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos = y.sum()
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


strata = {
    'all': np.ones_like(ten),
    'riichi=0': ~ri,
    'called=1': ca,
    'called=0': ~ca,
    'riichi=0&called=0': ~ri & ~ca,
    'riichi=0&called=0&turn>=8': ~ri & ~ca & (turn >= 8),
    'turn0-3': turn <= 3,
    'turn4-7': (turn >= 4) & (turn <= 7),
    'turn8-11': (turn >= 8) & (turn <= 11),
    'turn12+': turn >= 12,
}
print(f"{'分层':<28s}{'n':>7s}{'听牌率':>8s}{'AUC(softmax)':>14s}{'AUC(sigmoid)':>14s}{'Δ':>9s}")
print('-' * 82)
for k, m in strata.items():
    a1, a2 = auc(ten[m], s_soft[m]), auc(ten[m], s_sig[m])
    print(f'{k:<28s}{m.sum():>7d}{ten[m].mean():>8.4f}{a1:>14.4f}{a2:>14.4f}{a2-a1:>+9.4f}')
