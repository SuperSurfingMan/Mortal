"""读牌模型离线诊断（方案 §8）。

在固定 holdout 集上输出（全部为概率量诊断，不涉及对局强度）：

  alloc 逐列计数分布：
      - log-loss（总体 + 按巡目分桶）+ top1 准确率 + top1 置信度 ECE
      - 信息增益：模型 log-loss vs 均匀先验基线（见下）的逐列改善
      - 约束层 diagnostics（列守恒/总牌数投影前后的最大违例）
  wait 听牌分布：
      - 听牌样本 log-loss、hit@1 / hit@3（真实可荣集内）
      - 听牌检测 AUC（P(非未听) vs 结构性听牌标签）、振听率
  结构头专项（v2/v2s，向听估计）：
      - 向听头准确率 vs 多数类基线 + 逐桶召回（0..7）
      - 自洽向听（分解直方图暗含的软向听）准确率、头↔分解一致率
      - 形制准确率 + per-class 召回、结构计数 L1
  baseline（方案 §8.1 第 1 条）：每列未见张数按容量独立二项分配——
      P(h_{j,i}=k) = C(u_i, k) p_j^k (1-p_j)^(u_i-k)，
      p_j = hand_j / (Σ_q hand_q + tiles_left)。不使用任何牌河/立直信息。

用法：
    python tools/read_model/evaluate.py --state runs/read_model_v1/best.pth \
        --val-globs '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202510/**/*.json.gz' \
        --samples 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')   # libriichi 扩展
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D                                     # noqa: E402
import model as M                                       # noqa: E402
import model_v2 as M2                                   # noqa: E402
import train as T                                       # noqa: E402
from constraints import apply_constraints, compute_wait_bias, expected_hand_size  # noqa: E402
from structure import MAX_SHANTEN_BUCKET, NUM_SHANTEN_BUCKETS, soft_shanten_from_counts  # noqa: E402

NUM_TILES = D.NUM_TILES
TURN_BUCKETS = [(0, 3, 'turn<=3'), (4, 7, 'turn4-7'), (8, 11, 'turn8-11'),
                (12, 15, 'turn12-15'), (16, 99, 'turn16+')]
FORM_NAMES = {0: 'normal', 1: 'chiitoi', 2: 'kokushi'}


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC（Mann-Whitney U 秩公式；y_true ∈ {0,1}）。"""
    y_true = np.asarray(y_true, dtype=bool)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # 处理并列：平均秩
    s_sorted = y_score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """top1 置信度的期望校准误差。"""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def binomial_baseline(unseen: np.ndarray, hand_sizes: np.ndarray,
                      tiles_left: np.ndarray) -> np.ndarray:
    """方案 §8.1 基线 1：每列未见张数按容量独立二项分配。返回 [B,3,34,5]。

    P(h_{j,i}=k) = C(u_i, k) p_j^k (1-p_j)^(u_i-k)，
    p_j = hand_j / (Σ_q hand_q + tiles_left)（余量归牌山）。
    """
    from math import comb
    B = unseen.shape[0]
    total_hand = hand_sizes.sum(-1, keepdims=True)                      # [B,1]
    denom = (total_hand + tiles_left[:, None]).clip(min=1.0)            # [B,1]
    p = (hand_sizes / denom)[..., None, None]                           # [B,3,1,1]
    u_idx = np.clip(unseen, 0, 4).astype(int)                           # [B,34]
    k = np.arange(5, dtype=np.float64)                                  # [5]
    comb_table = np.array([[float(comb(u, kk)) if kk <= u else 0.0
                            for kk in range(5)] for u in range(5)])     # [5,5]
    cu = comb_table[u_idx][:, None]                                    # [B,1,34,5]
    probs = (cu * np.power(p, k) * np.power(1.0 - p, (u_idx[:, None, :, None] - k).clip(min=0)))
    probs = probs / probs.sum(-1, keepdims=True).clip(min=1e-12)
    return probs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True, help='checkpoint(.pth)，含 model 键')
    ap.add_argument('--val-globs', nargs='+', required=True)
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--out', default=None, help='诊断 JSON 输出路径（默认与 state 同目录）')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(args.state, map_location='cpu', weights_only=False)
    cfg = state.get('config', {})
    model_kind = cfg.get('model', 'v1')
    net = M.build_model(conv_channels=cfg.get('conv_channels', 96),
                        num_blocks=cfg.get('num_blocks', 12),
                        with_aux=not cfg.get('no_aux', False),
                        pool=cfg.get('pool', 'mean')).to(device)
    if model_kind == 'v2':
        net = M2.build_model_v2(conv_channels=cfg.get('conv_channels', 96),
                                num_blocks=cfg.get('num_blocks', 12),
                                with_aux=not cfg.get('no_aux', False)).to(device)
    elif model_kind == 'v2s':
        net = M2.build_model_v2s(conv_channels=cfg.get('conv_channels', 96),
                                 num_blocks=cfg.get('num_blocks', 12),
                                 with_aux=not cfg.get('no_aux', False)).to(device)
    net.load_state_dict(state['model'])
    net.eval()
    print(f'[model] {args.state} ({model_kind}, '
          f'step={state.get("step")}, best_val={state.get("best_val")})', flush=True)

    files = []
    for g in args.val_globs:
        files.extend(D.glob_files(g))
    files = sorted(set(files))
    batch = T.build_fixed_eval_set(files, args.samples)
    n = batch['obs'].shape[0]
    print(f'[data] {n} 个样本', flush=True)

    # ---- 模型前向（分批）----
    alloc_probs_all, wait_probs_all = [], []
    struct_sh_logits_all, struct_count_probs_all, struct_form_logits_all = [], [], []
    is_v2 = model_kind in ('v2', 'v2s')
    for i in range(0, n, args.batch_size):
        chunk = {k: v[i:i + args.batch_size].to(device) for k, v in batch.items()}
        with torch.no_grad():
            if is_v2:
                o = net(chunk['obs'], chunk['unseen'], T.wait_bias_of(chunk).to(device),
                        chunk['tl_tokens'], chunk['tl_mask'])
            else:
                o = net(chunk['obs'], chunk['unseen'], T.wait_bias_of(chunk).to(device))
        alloc_probs_all.append(o['alloc_probs'].cpu().numpy())
        wait_probs_all.append(o['wait_probs'].cpu().numpy())
        if is_v2:
            struct_sh_logits_all.append(o['struct_shanten_logits'].cpu().numpy())
            struct_count_probs_all.append(
                torch.softmax(o['struct_count_logits'], dim=-1).cpu().numpy())
            struct_form_logits_all.append(o['struct_form_logits'].cpu().numpy())
    alloc_probs = np.concatenate(alloc_probs_all)          # [n,3,34,5]
    wait_probs = np.concatenate(wait_probs_all)            # [n,3,35]

    alloc_true = batch['alloc'].numpy()                    # [n,3,34]
    wait_true = batch['wait'].numpy()                      # [n,3,35]
    unseen = batch['unseen'].numpy()
    turn = batch['turn'].numpy()
    tenpai = batch['tenpai'].numpy() > 0.5                 # 结构性听牌
    riichi = batch['riichi'].numpy() > 0.5

    # ---- alloc 诊断 ----
    bidx = (np.arange(n)[:, None, None], np.arange(3)[None, :, None],
            np.arange(NUM_TILES)[None, None, :])
    p_true = alloc_probs[bidx + (alloc_true,)]                          # [n,3,34]
    ll_col = -np.log(p_true.clip(min=1e-12))                            # [n,3,34]
    argmax_count = alloc_probs.argmax(-1)                               # [n,3,34]
    top1_conf = alloc_probs.max(-1)
    correct = argmax_count == alloc_true

    alloc_report = {
        'ce_overall': float(ll_col.mean()),
        'top1_acc': float(correct.mean()),
        'top1_conf': float(top1_conf.mean()),
        'ece_top1': ece(top1_conf.ravel(), correct.ravel()),
    }
    # 基线对照（同一批样本）
    hand_sizes = alloc_true.sum(-1)                                     # [n,3]
    base = binomial_baseline(unseen, hand_sizes, batch['tiles_left'].numpy())
    base_p = base[bidx + (alloc_true,)]
    ll_base = -np.log(base_p.clip(min=1e-12))
    alloc_report['ce_baseline_uniform'] = float(ll_base.mean())
    alloc_report['info_gain_nats'] = float(ll_base.mean() - ll_col.mean())

    # 按巡目分桶
    buckets = {}
    for lo, hi, name in TURN_BUCKETS:
        m = (turn >= lo) & (turn <= hi)
        if m.any():
            buckets[name] = {
                'n': int(m.sum()),
                'ce': float(ll_col[m].mean()),
                'ce_baseline': float(ll_base[m].mean()),
                'top1': float(correct[m].mean()),
            }
    alloc_report['by_turn'] = buckets

    # 约束层诊断（抽 4096 样本）
    sub = min(n, 4096)
    cons = apply_constraints(alloc_probs[:sub], unseen[:sub], hand_sizes[:sub])
    alloc_report['constraints'] = cons['diag']

    # ---- wait 诊断 ----
    active = tenpai                                             # [n,3]
    sup = wait_true[..., :NUM_TILES] > 0                        # 非空可荣集
    eff_active = sup.any(-1)                                    # 有可荣牌的样本
    ll_wait = -(wait_true * np.log(wait_probs.clip(min=1e-12))).sum(-1)   # [n,3]
    tile_probs = wait_probs[..., :NUM_TILES]
    order = np.argsort(-tile_probs, axis=-1)
    hit1 = np.take_along_axis(sup, order[..., :1], axis=-1).squeeze(-1)
    hit3 = np.take_along_axis(sup, order[..., :3], axis=-1).any(-1)
    p_tenpai = 1.0 - wait_probs[..., NUM_TILES]                 # 结构性听牌倾向分

    wait_report = {
        'ce_active': float(ll_wait[eff_active].mean()) if eff_active.any() else None,
        'hit1': float(hit1[eff_active].mean()) if eff_active.any() else None,
        'hit3': float(hit3[eff_active].mean()) if eff_active.any() else None,
        'tenpai_auc': auc_score(tenpai.ravel(), p_tenpai.ravel()),
        'structural_tenpai_frac': float(tenpai.mean()),
        'furiten_frac': float((tenpai & ~eff_active).sum() / max(tenpai.sum(), 1)),
        'riichi_target_frac': float(riichi.mean()),
    }
    # 立直家单独一档（下游最关心）
    m_riichi = riichi & eff_active
    if m_riichi.any():
        wait_report['riichi_hit1'] = float(hit1[m_riichi].mean())
        wait_report['riichi_hit3'] = float(hit3[m_riichi].mean())
        wait_report['riichi_ce'] = float(ll_wait[m_riichi].mean())

    # ---- 结构头专项诊断（v2/v2s）：向听估计是主攻方向 ----
    struct_report = None
    if is_v2:
        sh_true = batch['shanten'].numpy()                          # [n,3] 桶 0..7
        form_true = batch['struct_form'].numpy()                    # [n,3]
        counts_true = batch['struct_counts'].numpy()                # [n,3,7,5]
        sh_head = np.concatenate(struct_sh_logits_all).argmax(-1)   # [n,3]
        count_probs = np.concatenate(struct_count_probs_all)        # [n,3,7,5]
        form_pred = np.concatenate(struct_form_logits_all).argmax(-1)  # [n,3]
        # 自洽向听：分解直方图经可微公式暗含的向听（L = 手牌数//3，公开信息）
        len_div3 = (alloc_true.sum(-1) // 3).astype(np.float32)     # [n,3]
        with torch.no_grad():
            sh_soft = soft_shanten_from_counts(
                torch.from_numpy(count_probs),
                torch.from_numpy(len_div3)).numpy()                 # [n,3]
        sh_self = np.clip(np.round(sh_soft), 0, MAX_SHANTEN_BUCKET).astype(int)

        sh_hist = np.bincount(sh_true.ravel(), minlength=NUM_SHANTEN_BUCKETS)
        majority = int(sh_hist.argmax())
        struct_report = {
            'shanten_label_hist': sh_hist.tolist(),
            'shanten_majority_class': majority,
            'shanten_majority_baseline': float((sh_true == majority).mean()),
            'shanten_head_acc': float((sh_head == sh_true).mean()),
            'shanten_self_acc': float((sh_self == sh_true).mean()),
            'shanten_consistency': float((sh_head == sh_self).mean()),
            'shanten_head_by_bucket': {
                str(k): {
                    'n': int((sh_true == k).sum()),
                    'recall': (float((sh_head[sh_true == k] == k).mean())
                               if (sh_true == k).any() else None),
                } for k in range(NUM_SHANTEN_BUCKETS)
            },
            'shanten_soft_mae': float(np.abs(
                np.clip(sh_soft, 0, MAX_SHANTEN_BUCKET) - sh_true).mean()),
            'form_acc': float((form_pred == form_true).mean()),
            'form_by_class': {
                FORM_NAMES[k]: {
                    'n': int((form_true == k).sum()),
                    'recall': (float((form_pred[form_true == k] == k).mean())
                               if (form_true == k).any() else None),
                } for k in range(len(FORM_NAMES))
            },
            'struct_counts_l1': float(np.abs(count_probs - counts_true).sum(-1).mean()),
        }

    report = {'state': args.state, 'step': state.get('step'),
              'n_samples': int(n), 'alloc': alloc_report, 'wait': wait_report}
    if struct_report is not None:
        report['struct'] = struct_report

    print('\n================ 读牌模型离线诊断 ================')
    print(f"alloc CE        : {alloc_report['ce_overall']:.4f}   "
          f"(基线 {alloc_report['ce_baseline_uniform']:.4f}, "
          f"信息增益 {alloc_report['info_gain_nats']:.4f} nats/列)")
    print(f"alloc top1      : {alloc_report['top1_acc']:.4f}   "
          f"ECE {alloc_report['ece_top1']:.4f}")
    for name, b in buckets.items():
        print(f"  {name:9s}: CE {b['ce']:.4f} vs 基线 {b['ce_baseline']:.4f} "
              f"top1 {b['top1']:.4f} (n={b['n']})")
    print(f"约束层          : {cons['diag']}")
    print(f"wait CE(听牌)   : {wait_report['ce_active']}")
    print(f"wait hit@1/hit@3: {wait_report['hit1']:.4f} / {wait_report['hit3']:.4f}")
    print(f"听牌检测 AUC    : {wait_report['tenpai_auc']:.4f}   "
          f"结构性听牌率 {wait_report['structural_tenpai_frac']:.4f}   "
          f"其中振听(无可荣牌) {wait_report['furiten_frac']:.4f}")
    if 'riichi_hit1' in wait_report:
        print(f"立直家 hit@1/@3 : {wait_report['riichi_hit1']:.4f} / "
              f"{wait_report['riichi_hit3']:.4f} (CE {wait_report['riichi_ce']:.4f})")
    if struct_report is not None:
        print('---- 结构头专项（向听估计）----')
        print(f"向听标签分布    : {struct_report['shanten_label_hist']}  "
              f"多数类 {struct_report['shanten_majority_class']} "
              f"基线 {struct_report['shanten_majority_baseline']:.4f}")
        print(f"向听 头/自洽    : {struct_report['shanten_head_acc']:.4f} / "
              f"{struct_report['shanten_self_acc']:.4f}   "
              f"头↔分解一致率 {struct_report['shanten_consistency']:.4f}   "
              f"软向听 MAE {struct_report['shanten_soft_mae']:.3f}")
        recalls = '  '.join(
            f"{k}:{(v['recall'] if v['recall'] is not None else float('nan')):.3f}"
            for k, v in struct_report['shanten_head_by_bucket'].items())
        print(f"向听逐桶召回    : {recalls}")
        forms = '  '.join(
            f"{nm}:{(v['recall'] if v['recall'] is not None else float('nan')):.3f}"
            for nm, v in struct_report['form_by_class'].items())
        print(f"形制准确/召回   : {struct_report['form_acc']:.4f}   {forms}")
        print(f"结构计数 L1     : {struct_report['struct_counts_l1']:.4f}")
    print('==================================================\n')

    out = args.out or str(Path(args.state).parent / 'eval_report.json')
    with open(out, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'[out] {out}', flush=True)


if __name__ == '__main__':
    main()
