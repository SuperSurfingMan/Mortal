"""v1 vs v2 对决：同一固定评估集、同一组直观指标，端到端对比。

用法：
    python tools/read_model/compare_v1_v2.py \
        --v1 mortal/read_model_v1/best.pth --v2 mortal/read_model_v2/best.pth \
        --samples 12000
"""
from __future__ import annotations

import argparse
import glob
import random
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
from evaluate import binomial_baseline                 # noqa: E402

NUM_TILES = 34


def batch_to_samples(batch):
    """把 collate 后的 batch（张量字典）拆回逐样本 dict 列表（与 replay_file 输出同构）。"""
    n = batch['obs'].shape[0]
    out = []
    for i in range(n):
        s = {}
        for k, v in batch.items():
            s[k] = v[i].numpy() if torch.is_tensor(v) else v[i]
        out.append(s)
    return out


def load_net(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state.get('config', {})
    kind = cfg.get('model', 'v1')
    if kind == 'v2':
        net = M2.build_model_v2(conv_channels=cfg.get('conv_channels', 96),
                                num_blocks=cfg.get('num_blocks', 12),
                                with_aux=not cfg.get('no_aux', False),
                                wait_head=cfg.get('wait_head', 'flat'))
    elif kind == 'v2s':
        net = M2.build_model_v2s(conv_channels=cfg.get('conv_channels', 96),
                                 num_blocks=cfg.get('num_blocks', 12),
                                 with_aux=not cfg.get('no_aux', False),
                                 wait_head=cfg.get('wait_head', 'flat'))
    else:
        net = M.build_model(conv_channels=cfg.get('conv_channels', 96),
                            num_blocks=cfg.get('num_blocks', 12),
                            with_aux=not cfg.get('no_aux', False),
                            pool=cfg.get('pool', 'mean'))
    net.load_state_dict(state['model'])
    return net.to(device).eval(), cfg


@torch.no_grad()
def predict(net, samples, device, bs=2048):
    E, WP, SH, SC, SF = [], [], [], [], []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        batch = T.collate(chunk)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        wb = T.wait_bias_of(batch).to(device)
        if isinstance(net, M2.ReadNetV2):
            o = net(batch['obs'], batch['unseen'], wb,
                    batch['tl_tokens'], batch['tl_mask'])
        else:
            o = net(batch['obs'], batch['unseen'], wb)
        E.append(o['exp_counts'].cpu().numpy())
        WP.append(o['wait_probs'].cpu().numpy())
        if isinstance(net, M2.ReadNetV2):
            SH.append(o['struct_shanten_logits'].argmax(-1).cpu().numpy())
            SC.append(o['struct_count_logits'].softmax(-1).cpu().numpy())
            SF.append(o['struct_form_logits'].argmax(-1).cpu().numpy())
    out = {'E': np.concatenate(E), 'WP': np.concatenate(WP)}
    if SH:
        out['shanten_pred'] = np.concatenate(SH)
        out['struct_pred'] = np.concatenate(SC)
        out['form_pred'] = np.concatenate(SF)
    return out


def metrics(pred, samples, tag):
    E, WP = pred['E'], pred['WP']
    H = np.stack([s['alloc'] for s in samples]).astype(float)
    U = np.stack([s['unseen'] for s in samples])
    WT = np.stack([s['wait'] for s in samples])
    TP = np.stack([s['tenpai'] for s in samples]) > 0.5
    RI = np.stack([s['riichi'] for s in samples]) > 0.5
    own = np.stack([s['own_hand'] for s in samples])
    n = len(E)

    # ---- alloc ----
    bidx = (np.arange(n)[:, None], np.arange(3)[None])
    p_true_arg = None
    ll = 0.0
    # 直接用概率质量（模型给的是概率）
    # 重新算 log-loss：需要 logits 的 log_softmax；用 log(probs) 近似已足够对比
    logp = np.log(E.clip(1e-9, 1))     # 占位，不用
    # 用 evaluate 的口径需要 logits；这里用期望张数层面的指标
    held = H > 0
    top1 = E.argmax(-1)
    top1_hit = held[bidx + (top1,)].mean()
    order = np.argsort(-E, axis=-1)
    top3_hit = np.take_along_axis(held, order[..., :3], -1).mean()
    U3 = np.broadcast_to(U[:, None, :], (n, 3, 34))
    m1 = (U == 1)                                       # [n,34] 逐牌
    truth_owner = np.where(H.sum(1) > 0, H.argmax(1), 3)   # [n,34] 逐牌归属
    wall_e = U - E.sum(1)
    cand = np.concatenate([E, wall_e[:, None, :]], axis=1)
    pred_owner = cand.argmax(1)                             # [n,34]
    key_acc = (pred_owner == truth_owner)[m1].mean()
    # MAE
    hs = H.sum(-1)
    tl = np.array([float(s['tiles_left']) for s in samples])
    base = binomial_baseline(U, hs, tl)
    E_base = (base * np.arange(5)).sum(-1)
    mae = np.abs(E - H).mean()
    mae_base = np.abs(E_base - H).mean()

    # ---- wait ----
    sup = WT[:, :, :NUM_TILES] > 0
    eff = sup.any(-1)
    tile_p = WP[:, :, :NUM_TILES]
    worder = np.argsort(-tile_p, axis=-1)
    hit1 = np.take_along_axis(sup, worder[..., :1], -1).squeeze(-1)
    hit3 = np.take_along_axis(sup, worder[..., :3], -1).any(-1)
    hit5 = np.take_along_axis(sup, worder[..., :5], -1).any(-1)
    OWN3 = np.broadcast_to(own[:, None, :], (n, 3, 34))
    p_cand = np.where(OWN3, tile_p, np.inf)
    safest = p_cand.argmin(-1)
    p_safe = tile_p[bidx + (safest,)]
    deal_safe = sup[bidx + (safest,)]
    rnd = (sup & OWN3).sum(-1) / np.maximum(OWN3.sum(-1), 1)
    m = eff
    m_pure = m & (p_safe > 0)
    m_ri = RI & eff
    m_ri_pure = m_ri & (p_safe > 0)

    out = {
        'top1_hit': top1_hit, 'top3_hit': top3_hit, 'key_acc': key_acc,
        'mae': mae, 'mae_base': mae_base,
        'wait_hit1': hit1[m].mean(), 'wait_hit3': hit3[m].mean(),
        'wait_hit5': hit5[m].mean(),
        'deal_random': rnd[m].mean(),
        'deal_safest': deal_safe[m].mean(),
        'deal_safest_pure': deal_safe[m_pure].mean() if m_pure.any() else float('nan'),
        'n_pure': int(m_pure.sum()),
        'deal_random_pure': rnd[m_pure].mean() if m_pure.any() else float('nan'),
        'riichi_hit1': hit1[m_ri].mean() if m_ri.any() else float('nan'),
        'riichi_deal_pure': deal_safe[m_ri_pure].mean() if m_ri_pure.any() else float('nan'),
        'riichi_rand_pure': rnd[m_ri_pure].mean() if m_ri_pure.any() else float('nan'),
    }
    # v2 专项：向听预测准确率 + 结构计数分布与真值的 L1
    if 'shanten_pred' in pred:
        sh_true = np.stack([s['shanten'] for s in samples])
        out['shanten_acc'] = float((pred['shanten_pred'] == sh_true).mean())
        sc_true = np.stack([s['struct_counts'] for s in samples])
        out['struct_l1'] = float(np.abs(pred['struct_pred'] - sc_true).sum(-1).mean())
        # 自洽向听：分解直方图暗含的软向听（L=手牌数//3，公开信息推导）
        from structure import MAX_SHANTEN_BUCKET, soft_shanten_from_counts
        len_div3 = (H.sum(-1) // 3).astype(np.float32)               # [n,3]
        with torch.no_grad():
            sh_soft = soft_shanten_from_counts(
                torch.from_numpy(pred['struct_pred']),
                torch.from_numpy(len_div3)).numpy()
        sh_self = np.clip(np.round(sh_soft), 0, MAX_SHANTEN_BUCKET).astype(int)
        out['shanten_self_acc'] = float((sh_self == sh_true).mean())
        out['shanten_consist'] = float((pred['shanten_pred'] == sh_self).mean())
        # 形制准确率（normal 占 ~97%，对照多数类基线看是否退化为常数预测）
        form_true = np.stack([s['struct_form'] for s in samples])
        out['form_acc'] = float((pred['form_pred'] == form_true).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1', required=True, help='基线 checkpoint（列 A）')
    ap.add_argument('--v2', required=True, help='对照 checkpoint（列 B）')
    ap.add_argument('--tag-a', default=None, help='列 A 显示名（默认取 checkpoint 目录名）')
    ap.add_argument('--tag-b', default=None, help='列 B 显示名（默认取 checkpoint 目录名）')
    ap.add_argument('--samples', type=int, default=12000)
    ap.add_argument('--root', default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag_a = args.tag_a or Path(args.v1).parent.name
    tag_b = args.tag_b or Path(args.v2).parent.name

    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))
    # 与 evaluate.py 完全同一固定评估集（固定种子 + 均匀抽稀），保证两模型严格可比
    batch = T.build_fixed_eval_set(files, args.samples)
    samples = batch_to_samples(batch)
    print(f'[data] {len(samples)} 个样本（固定评估集，{args.root}）', flush=True)

    net1, cfg1 = load_net(args.v1, device)
    net2, cfg2 = load_net(args.v2, device)
    print(f'[model] A = {tag_a} ({cfg1.get("model", "v1")}) | '
          f'B = {tag_b} ({cfg2.get("model", "v1")})', flush=True)
    p1 = predict(net1, samples, device)
    p2 = predict(net2, samples, device)
    m1 = metrics(p1, samples, 'v1')
    m2 = metrics(p2, samples, 'v2')

    rows = [
        ('alloc top-1 命中率', 'top1_hit', '.4f'),
        ('alloc top-3 平均命中', 'top3_hit', '.4f'),
        ('关键张归属准确率', 'key_acc', '.4f'),
        ('期望张数 MAE', 'mae', '.4f'),
        ('  基线 MAE', 'mae_base', '.4f'),
        ('wait hit@1', 'wait_hit1', '.4f'),
        ('wait hit@3', 'wait_hit3', '.4f'),
        ('wait hit@5', 'wait_hit5', '.4f'),
        ('立直家 hit@1', 'riichi_hit1', '.4f'),
        ('防守：随机弃牌点炮率', 'deal_random', '.4f'),
        ('防守：模型最安全点炮率', 'deal_safest', '.4f'),
        ('  纯模型场合 随机', 'deal_random_pure', '.4f'),
        ('  纯模型场合 最安全', 'deal_safest_pure', '.4f'),
        ('  立直纯模型 随机', 'riichi_rand_pure', '.4f'),
        ('  立直纯模型 最安全', 'riichi_deal_pure', '.4f'),
        ('向听 头准确率', 'shanten_acc', '.4f'),
        ('向听 自洽准确率', 'shanten_self_acc', '.4f'),
        ('向听 头↔分解一致', 'shanten_consist', '.4f'),
        ('形制准确率', 'form_acc', '.4f'),
        ('结构分布 L1', 'struct_l1', '.4f'),
    ]
    print(f"\n{'指标':<24s} {tag_a:>10s} {tag_b:>10s} {'Δ(B-A)':>10s}")
    print('-' * 58)
    for label, key, fmt in rows:
        a, b = m1.get(key, float('nan')), m2.get(key, float('nan'))
        print(f'{label:<24s} {a:>10{fmt}} {b:>10{fmt}} {b - a:>+10.4f}')


if __name__ == '__main__':
    main()
