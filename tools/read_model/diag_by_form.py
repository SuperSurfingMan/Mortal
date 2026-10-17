"""按形制（normal/七对/国士）分桶诊断：结构瓶颈是否伤害非 normal 牌型？

回答：7 个结构计数是自由边际直方图（无"必须 3 顺子"约束），但对七对/国士牌型，
counts 标签取的仍是 normal 分解边际 + form 头弱（91.9%=多数类基线）——
需实测这些桶上的 alloc/wait 指标是否明显劣化。

用法：python tools/read_model/diag_by_form.py --samples 20000
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D                                     # noqa: E402
import model as M                                       # noqa: E402
import model_v2 as M2                                   # noqa: E402
import train as T                                       # noqa: E402
from compare_v1_v2 import batch_to_samples, load_net, predict   # noqa: E402

NUM_TILES = 34
FORM_NAMES = {0: 'normal', 1: '七对', 2: '国士'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1', default='mortal/read_model_v1/best.pth')
    ap.add_argument('--v2', default='mortal/read_model_v2/best.pth')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--root', default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))
    samples = batch_to_samples(T.build_fixed_eval_set(files, args.samples))
    n = len(samples)

    # 形制标签（真值）
    form = np.stack([s['struct_form'] for s in samples])          # [n,3]
    sh_true = np.stack([s['shanten'] for s in samples])
    print(f'[data] {n} 样本 = {n*3} 个目标；形制分布：',
          {FORM_NAMES[k]: f'{(form==k).mean():.4f}' for k in range(3)}, flush=True)

    # ---- 标签多样性：证明 counts 不限于"3 顺子" ----
    sc = np.stack([s['struct_counts'] for s in samples])          # [n,3,7,5]
    sc_arg = sc.argmax(-1)                                         # [n,3,7]
    names = ['shuntsu', 'kotsu', 'toitsu', 'ryanmen', 'kanchan', 'penchan', 'isolated']
    print('\n=== 结构计数标签的众数分布（证明无"必须3顺子"约束）===')
    for k, nm in enumerate(names):
        v = sc_arg[:, :, k]
        vals, cnts = np.unique(v, return_counts=True)
        top = sorted(zip(vals.tolist(), cnts.tolist()), key=lambda x: -x[1])[:4]
        print(f'  {nm:9s} 众数: ' + '  '.join(f'{x}:{y/v.size*100:.1f}%' for x, y in top))
    # 零顺子手牌占比（刻子主导/七对/国士的指纹）
    zero_shuntsu = (sc_arg[:, :, 0] == 0)
    print(f'\n  shuntsu 众数=0（无顺子成分）的目标占比: {zero_shuntsu.mean():.4f}')
    m_kotsu = (sc_arg[:, :, 1] >= 2)
    print(f'  kotsu 众数>=2（多刻子）的目标占比: {m_kotsu.mean():.4f}')
    m_ct = (form == 1)
    if m_ct.any():
        print(f'  七对目标中 shuntsu 众数=0 占比: {zero_shuntsu[m_ct].mean():.4f}，'
              f'toitsu 众数>=4 占比: {(sc_arg[:, :, 1][m_ct] if False else sc_arg[:,:,2][m_ct]>=4).mean():.4f}')

    # ---- 模型预测 ----
    net2, _ = load_net(args.v2, device)
    net1, _ = load_net(args.v1, device)
    p2 = predict(net2, samples, device)
    p1 = predict(net1, samples, device)

    H = np.stack([s['alloc'] for s in samples]).astype(float)
    WT = np.stack([s['wait'] for s in samples])
    TP = np.stack([s['tenpai'] for s in samples]) > 0.5
    held = H > 0

    def bucket_metrics(pred, tag):
        E, WP = pred['E'], pred['WP']
        bidx = (np.arange(n)[:, None], np.arange(3)[None])
        # alloc top-1 命中
        top1 = E.argmax(-1)
        top1_hit = held[bidx + (top1,)]                            # [n,3]
        # 向听预测
        sh_pred = pred.get('shanten_pred')
        # wait
        sup = WT[:, :, :NUM_TILES] > 0
        eff = sup.any(-1)
        tile_p = WP[:, :, :NUM_TILES]
        hit1 = np.take_along_axis(sup, np.argsort(-tile_p, -1)[..., :1], -1).squeeze(-1)
        # CE（近似：用概率质量）
        print(f'\n=== {tag}：按形制分桶 ===')
        print(f"{'桶':<10s}{'目标数':>8s}{'top1命中':>10s}{'向听准确':>10s}"
              f"{'wait hit1':>11s}{'听牌率':>8s}")
        for k in range(3):
            m = (form == k)
            if not m.any():
                continue
            sh = (sh_pred == sh_true)[m].mean() if sh_pred is not None else float('nan')
            e = eff & m
            h1 = hit1[m][eff[m]].mean() if eff[m].any() else float('nan')
            print(f'{FORM_NAMES[k]:<10s}{int(m.sum()):>8d}{top1_hit[m].mean():>10.4f}'
                  f'{sh:>10.4f}{h1:>11.4f}{TP[m].mean():>8.4f}')
        # 全量
        sh = (sh_pred == sh_true).mean() if sh_pred is not None else float('nan')
        h1 = hit1[eff].mean() if eff.any() else float('nan')
        print(f'{"全量":<10s}{n*3:>8d}{top1_hit.mean():>10.4f}{sh:>10.4f}{h1:>11.4f}'
              f'{TP.mean():>8.4f}')
        # 按 shanten 分桶（结构估计的另一个切面）
        print(f'\n  {tag} 按向听数分桶（top1 命中 / 向听准确率）:')
        for s_ in range(0, 6):
            m = (sh_true == s_)
            if m.sum() < 100:
                continue
            sa = (sh_pred == sh_true)[m].mean() if sh_pred is not None else float('nan')
            print(f'    shanten={s_}: n={int(m.sum()):>6d}  top1 {top1_hit[m].mean():.4f}  '
                  f'向听准确 {sa:.4f}')

    bucket_metrics(p1, 'v1（无结构瓶颈，对照）')
    bucket_metrics(p2, 'v2（有结构瓶颈）')


if __name__ == '__main__':
    main()
