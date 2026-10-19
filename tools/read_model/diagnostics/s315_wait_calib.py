"""核实 A2b P(听牌) 过标定的机制：BCE 标定的是 logit（sigmoid），
而指标/注入用的是 softmax 的 1-P(未听)。两者只在 Σexp(其余 34 logit) 恒定时一致。
若 sigmoid(l34) 标定良好而 softmax 版本严重偏，则病灶 = 输出层参数化，而非判别力。
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
L34, ZSUM, WP = [], [], []
with torch.no_grad():
    for i in range(0, n, 2048):
        ch = {k: v[i:i + 2048].to(DEV) for k, v in b.items()}
        wb = T.wait_bias_of(ch).to(DEV)
        o = net(ch['obs'], ch['unseen'], wb, ch['tl_tokens'], ch['tl_mask'])
        L34.append(o['wait_logits'][..., 34].cpu().numpy())
        lg = o['wait_logits'][..., :34].double()
        ZSUM.append(torch.logsumexp(lg, -1).cpu().numpy())
        WP.append(o['wait_probs'][..., 34].cpu().numpy())
l34 = np.concatenate(L34); zsum = np.concatenate(ZSUM); p_not = np.concatenate(WP)
ten = b['tenpai'].numpy() > 0.5
sig = 1.0 / (1.0 + np.exp(-l34))                     # BCE 直接标定的量 = P(未听)
print(f'n={n*3}  tenpai 率={ten.mean():.4f}')
print(f'\n[A] sigmoid(l34)（BCE 的目标量，= P(未听)）')
print(f'    mean={sig.mean():.4f}  vs 实际未听率={1-ten.mean():.4f}  '
      f'比值={sig.mean()/(1-ten.mean()):.3f}')
print(f'    听牌目标 mean={sig[ten].mean():.4f}   非听牌目标 mean={sig[~ten].mean():.4f}')
print(f'[B] softmax 的 P(未听)（指标/注入实际使用的量）')
print(f'    mean={p_not.mean():.4f}  比值={p_not.mean()/(1-ten.mean()):.3f}')
print(f'    听牌目标 mean={p_not[ten].mean():.4f}   非听牌目标 mean={p_not[~ten].mean():.4f}')
print(f'[C] logsumexp(其余 34 logit)：mean={zsum.mean():.3f} sd={zsum.std():.3f} '
      f'p10={np.percentile(zsum,10):.2f} p90={np.percentile(zsum,90):.2f}')
print(f'    → exp(l34)/exp(zsum) 的比值随样本变化 {(np.exp(l34-zsum)).std():.4f}')


def auc(y, s):
    y = np.asarray(y).ravel().astype(bool); s = np.asarray(s).ravel()
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    npos = y.sum()
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


print(f'\n[D] AUC（同一判别力的两种读法）')
print(f'    1-softmax P(未听) : {auc(ten, 1-p_not):.4f}')
print(f'    1-sigmoid(l34)   : {auc(ten, 1-sig):.4f}')
print(f'    差值             : {auc(ten,1-sig)-auc(ten,1-p_not):+.4f}')
print(f'\n[E] sigmoid(l34) 的标定曲线（10 桶）')
for lo in np.linspace(0, 1, 11)[:-1]:
    hi = lo + 0.1
    m = (sig >= lo) & (sig < hi) if hi < 1 else (sig >= lo)
    if m.sum():
        print(f'    P(未听)∈[{lo:.1f},{hi:.1f}) n={m.sum():>6d} 实际未听率={(~ten[m]).mean():.4f}')
