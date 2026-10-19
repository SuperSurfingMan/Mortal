"""A3 规则法基线 vs A2b（神经读牌模型）严格对决：同一固定评估集 + bootstrap CI。

回答项目核心问题：「A2b 相对人类规则法的增量有多大、哪些指标真超出规则」。
口径与 compare_bootstrap.py 完全一致（同固定评估集、同 per-sample 统计、
同 bootstrap 重采样），另列均匀二项先验作 alloc 参照。

口径（RM1/RM3，README §11.2）：
  - net 的 wait_probs 已是因子化读法（p34 = sigmoid(l34) = P(未听)），与
    A3 的 wait_probs[34] = 1 − tile_mass 同构 ⇒ tenpai_auc 跨模型可比；
  - wait_ce_active 必须同报拆分列 wait_nlt_active（−logP(听)）与
    wait_cond_ce_active（待牌条件分布 CE）：谁在 P(听) 上更过度自信，
    谁在未拆分的 ce_active 上占便宜（A2 型塌缩曾借此「刷分」）。

用法：
    python tools/read_model/compare_a3.py --net mortal/read_model_v2s_a2b/best.pth \
        --samples 20000 --n-boot 200
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

import rule_baseline as RB                              # noqa: E402
import train as T                                       # noqa: E402
from evaluate import auc_score, binomial_baseline       # noqa: E402
from model import build_model                           # noqa: E402
from model_v2 import build_model_v2, build_model_v2s    # noqa: E402

NUM_TILES = 34
TAG_UNIFORM = 'uniform'
TAG_NET = 'net'
TAG_RULE = 'A3-rule'

#: 指标方向：True = 越大越好；False = 越小越好（CE / 点炮率类）
HIGHER_BETTER = {
    'alloc_ce': False, 'alloc_top1': True,
    'wait_hit1': True, 'wait_hit5': True, 'riichi_hit1': True,
    'wait_ce_active': False, 'wait_nlt_active': False, 'wait_cond_ce_active': False,
    'tenpai_auc': True,
    'deal_pure': False, 'deal_pure_all': False, 'riichi_deal_pure': False,
}


def per_sample_stats(alloc_probs, wait_probs, exp_counts, batch):
    """把预测拆成可 bootstrap 的逐样本统计向量（compare_bootstrap 同口径）。"""
    n = alloc_probs.shape[0]
    H = batch['alloc']
    WT = batch['wait']
    TP = batch['tenpai']
    RI = batch['riichi']
    own = batch['own_hand']
    tenpai = TP > 0.5
    riichi = RI > 0.5
    held = H > 0
    bidx = (np.arange(n)[:, None], np.arange(3)[None])

    # alloc：逐列 CE（每样本对 3*34 列取均值）与「期望张数最高牌是否真持有」
    bidx3 = (np.arange(n)[:, None, None], np.arange(3)[None, :, None],
             np.arange(NUM_TILES)[None, None, :])
    p_true = alloc_probs[bidx3 + (H,)]
    ce = -np.log(p_true.clip(1e-12)).mean(-1).mean(-1)               # [n]
    top1_hit = held[bidx + (exp_counts.argmax(-1),)]                 # [n,3]

    # wait：hit@1/@5（仅有效听牌目标）、AUC 打分、防守模拟
    sup = WT[:, :, :NUM_TILES] > 0
    eff = sup.any(-1)
    tile_p = wait_probs[:, :, :NUM_TILES]
    order = np.argsort(-tile_p, -1)
    hit1 = np.take_along_axis(sup, order[..., :1], -1).squeeze(-1)
    hit5 = np.take_along_axis(sup, order[..., :5], -1).any(-1)
    score = 1.0 - wait_probs[:, :, NUM_TILES]        # P(听)：因子化读法（RM1）
    own3 = np.broadcast_to(own[:, None, :], (n, 3, NUM_TILES))
    p_cand = np.where(own3, tile_p, np.inf)
    safest = p_cand.argmin(-1)
    p_safe = tile_p[bidx + (safest,)]
    deal_safe = sup[bidx + (safest,)]
    pure = eff & (p_safe > 0)
    ri_pure = (riichi & eff) & (p_safe > 0)
    ll_wait = -(WT * np.where(WT > 0, np.log(wait_probs.clip(1e-12)), 0)).sum(-1)
    # RM3：wait CE 拆分 −logP(听) + cond_ce（active 上恒等式 ll = nlt + cond）
    p_t = np.clip(1.0 - wait_probs[:, :, NUM_TILES], 1e-12, None)
    cond = wait_probs[:, :, :NUM_TILES] / p_t[..., None]
    nlt = -np.log(p_t)
    WTt = WT[:, :, :NUM_TILES]
    cond_ce = -(WTt * np.where(cond > 0, np.log(cond.clip(1e-12)), 0)).sum(-1)
    return {
        'alloc_ce': ce, 'top1_hit': top1_hit,
        'hit1': hit1, 'hit5': hit5, 'eff': eff, 'riichi': riichi,
        'score': score, 'tenpai': tenpai,
        'p_safe': p_safe, 'deal_safe': deal_safe, 'pure': pure,
        'ri_pure': ri_pure, 'wait_ce': ll_wait,
        'wait_nlt': nlt, 'wait_cond_ce': cond_ce,
    }


def agg(s, idx):
    """在样本子集 idx 上聚合各指标（bootstrap 每一步调用）。"""
    out = {}
    out['alloc_ce'] = s['alloc_ce'][idx].mean()
    out['alloc_top1'] = s['top1_hit'][idx].mean()
    eff = s['eff'][idx]
    out['wait_hit1'] = s['hit1'][idx][eff].mean() if eff.any() else np.nan
    out['wait_hit5'] = s['hit5'][idx][eff].mean() if eff.any() else np.nan
    ri = s['riichi'][idx] & eff
    out['riichi_hit1'] = s['hit1'][idx][ri].mean() if ri.any() else np.nan
    out['wait_ce_active'] = s['wait_ce'][idx][eff].mean() if eff.any() else np.nan
    out['wait_nlt_active'] = s['wait_nlt'][idx][eff].mean() if eff.any() else np.nan
    out['wait_cond_ce_active'] = s['wait_cond_ce'][idx][eff].mean() if eff.any() else np.nan
    out['tenpai_auc'] = auc_score(s['tenpai'][idx].ravel(), s['score'][idx].ravel())
    pure = s['pure'][idx]
    out['deal_pure'] = s['deal_safe'][idx][pure].mean() if pure.any() else np.nan
    out['deal_pure_all'] = s['deal_safe'][idx][s['eff'][idx]].mean() \
        if s['eff'][idx].any() else np.nan
    ri_pure = s['ri_pure'][idx]
    out['riichi_deal_pure'] = s['deal_safe'][idx][ri_pure].mean() if ri_pure.any() else np.nan
    return out


def load_net_alloc_wait(path, b, device):
    """加载 checkpoint 并前向，返回 (alloc_probs, wait_probs, exp_counts, kind)。"""
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state.get('config', {})
    kind = cfg.get('model', 'v1')
    if kind == 'v2':
        net = build_model_v2(conv_channels=cfg.get('conv_channels', 96),
                             num_blocks=cfg.get('num_blocks', 12),
                             with_aux=not cfg.get('no_aux', False))
    elif kind == 'v2s':
        net = build_model_v2s(conv_channels=cfg.get('conv_channels', 96),
                              num_blocks=cfg.get('num_blocks', 12),
                              with_aux=not cfg.get('no_aux', False))
    else:
        net = build_model(conv_channels=cfg.get('conv_channels', 96),
                          num_blocks=cfg.get('num_blocks', 12),
                          with_aux=not cfg.get('no_aux', False),
                          pool=cfg.get('pool', 'mean'))
    net.load_state_dict(state['model'])
    net = net.to(device).eval()
    n = b['unseen'].shape[0]
    alloc = np.zeros((n, 3, NUM_TILES, 5), dtype=np.float64)
    wait = np.zeros((n, 3, NUM_TILES + 1), dtype=np.float64)
    expc = np.zeros((n, 3, NUM_TILES), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, 2048):
            chunk = {k: (torch.from_numpy(v[i:i + 2048]).to(device)
                         if isinstance(v, np.ndarray) else v[i:i + 2048])
                     for k, v in b.items() if torch.is_tensor(v) or isinstance(v, np.ndarray)}
            wb = T.wait_bias_of(chunk).to(device)
            if kind in ('v2', 'v2s'):
                o = net(chunk['obs'], chunk['unseen'], wb,
                        chunk['tl_tokens'], chunk['tl_mask'])
            else:
                o = net(chunk['obs'], chunk['unseen'], wb)
            alloc[i:i + 2048] = o['alloc_probs'].cpu().numpy()
            wait[i:i + 2048] = o['wait_probs'].cpu().numpy()
            expc[i:i + 2048] = o['exp_counts'].cpu().numpy()
    return alloc, wait, expc, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='mortal/read_model_v2s_a2b/best.pth',
                    help='对照神经网络 checkpoint（A2b 注入基线）')
    ap.add_argument('--root',
                    default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--bseed', type=int, default=20261018)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))
    batch = T.build_fixed_eval_set(files, args.samples)
    b = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in batch.items()}
    n = b['unseen'].shape[0]
    print(f'[data] {n} 个样本（固定评估集，{args.root}）', flush=True)

    # ---- 均匀二项先验（alloc 参照；wait 用支撑集均匀 + 常数基率 0.15）----
    hand_sizes = b['alloc'].sum(-1)
    uni_alloc = binomial_baseline(b['unseen'], hand_sizes, b['tiles_left'])
    expressible = (np.round(b['unseen']).astype(int) > 0) | b['own_hand']
    support = np.broadcast_to(expressible[:, None, :], (n, 3, NUM_TILES)) & ~b['river']
    uni_wait = np.where(support, 1.0, 0.0)
    uni_wait = uni_wait / uni_wait.sum(-1, keepdims=True).clip(min=1e-12) * 0.15
    uni_wait = np.concatenate(
        [uni_wait, np.full((n, 3, 1), 0.85)], axis=-1)
    uni_wait = uni_wait / uni_wait.sum(-1, keepdims=True).clip(min=1e-12)

    out_rule = RB.rule_predict(b)
    net_alloc, net_wait, net_expc, kind = load_net_alloc_wait(args.net, b, device)
    print(f'[model] net = {Path(args.net).parent.name} ({kind})', flush=True)

    uni_expc = (uni_alloc * np.arange(5)).sum(-1)
    stats = {
        TAG_UNIFORM: per_sample_stats(uni_alloc.astype(np.float32),
                                      uni_wait.astype(np.float32), uni_expc, b),
        TAG_NET: per_sample_stats(net_alloc.astype(np.float32),
                                  net_wait.astype(np.float32), net_expc, b),
        TAG_RULE: per_sample_stats(out_rule['alloc_probs'], out_rule['wait_probs'],
                                   out_rule['exp_counts'], b),
    }
    full = {tag: agg(s, np.ones(n, bool)) for tag, s in stats.items()}
    keys = [k for k in full[TAG_RULE] if k in full[TAG_NET]]

    rng = np.random.default_rng(args.bseed)
    boots = {(a, btag): {k: [] for k in keys}
             for a, btag in [(TAG_RULE, TAG_NET), (TAG_RULE, TAG_UNIFORM)]}
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        aggs = {tag: agg(s, idx) for tag, s in stats.items()}
        for a, btag in boots:
            for k in keys:
                boots[(a, btag)][k].append(aggs[a][k] - aggs[btag][k])

    hdr = (f"{'指标':<20s} {TAG_UNIFORM:>9s} {TAG_NET:>9s} {TAG_RULE:>9s} "
           f"{'Δ(rule-net)':>11s} {'95% CI':>19s} {'判定':>8s}")
    print('\n' + hdr)
    print('-' * 96)
    for k in keys:
        d = np.array(boots[(TAG_RULE, TAG_NET)][k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        hb = HIGHER_BETTER.get(k, True)
        if hb:
            sig = 'rule优' if lo > 0 else ('net优' if hi < 0 else '不显著')
        else:
            sig = 'rule优' if hi < 0 else ('net优' if lo > 0 else '不显著')
        print(f'{k:<20s} {full[TAG_UNIFORM].get(k, float("nan")):>9.4f} '
              f'{full[TAG_NET].get(k, float("nan")):>9.4f} '
              f'{full[TAG_RULE].get(k, float("nan")):>9.4f} '
              f'{full[TAG_RULE][k] - full[TAG_NET][k]:>+11.4f}   '
              f'[{lo:+.4f},{hi:+.4f}] {sig:>8s}')
    print('\n（Δ 列 = rule − net；CI 为 rule−net 的 bootstrap 95% 区间；'
          '「不显著」= CI 跨 0）')


if __name__ == '__main__':
    main()
