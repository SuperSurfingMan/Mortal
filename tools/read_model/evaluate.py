"""读牌模型离线诊断（方案 §8）。

在固定 holdout 集上输出（全部为概率量诊断，不涉及对局强度）：

  alloc 逐列计数分布：
      - log-loss（总体 + 按巡目分桶）+ top1 准确率 + top1 置信度 ECE
      - 信息增益：模型 log-loss vs 均匀先验基线（见下）的逐列改善
      - 约束层 diagnostics（列守恒/总牌数投影前后的最大违例）
  wait 听牌分布（RM1：wait_probs 为因子化读法，p34 = sigmoid(l34) = P(未听)）：
      - 听牌样本 log-loss，并按 RM3 拆分为 −logP(听) + 条件分布 CE
      - hit@1 / hit@3（真实可荣集内）+ 决策相关分层（turn≥8/立直/鸣牌/干净中盘）
      - 听牌检测 AUC 与标定比值：sigmoid 读法（主口径）+ 旧 softmax35 读法（参照列）
      - 振听率
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


def wait_ce_decomposition(wait_probs: np.ndarray, wait_true: np.ndarray):
    """RM3：把 wait CE 拆成 −logP(听) + 条件分布 CE（README §11.5-⑤）。

    wait_ce_active 混合「听牌判别」与「待牌条件分布」两件事，跨模型比较时
    谁在 P(听) 上更过度自信谁占便宜，必须拆开才可比。因子化读法（RM1）下
    active 目标上恒等：ll_wait = −logP(听) + cond_ce。

    参数：
        wait_probs [n,3,35]：p34 = P(未听)，p[:34] = P(听)·P(x|听)
        wait_true  [n,3,35]：待牌标签（active 目标支撑集上归一；非 active 全 0）
    返回：
        (ll_wait, neg_logp_t, cond_ce)，各 [n,3]
    """
    p_t = np.clip(1.0 - wait_probs[..., NUM_TILES], 1e-12, None)   # P(听)
    cond = wait_probs[..., :NUM_TILES] / p_t[..., None]            # P(x|听)
    ll = -(wait_true * np.log(wait_probs.clip(min=1e-12))).sum(-1)
    nlt = -np.log(p_t)
    wt = wait_true[..., :NUM_TILES]
    cce = -(wt * np.log(cond.clip(min=1e-12))).sum(-1)
    return ll, nlt, cce


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
                                with_aux=not cfg.get('no_aux', False),
                                wait_head=cfg.get('wait_head', 'flat')).to(device)
    elif model_kind == 'v2s':
        net = M2.build_model_v2s(conv_channels=cfg.get('conv_channels', 96),
                                 num_blocks=cfg.get('num_blocks', 12),
                                 with_aux=not cfg.get('no_aux', False),
                                 wait_head=cfg.get('wait_head', 'flat')).to(device)
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
    alloc_probs_all, wait_probs_all, wait_logits_all = [], [], []
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
        wait_logits_all.append(o['wait_logits'].cpu().numpy())
        if is_v2:
            struct_sh_logits_all.append(o['struct_shanten_logits'].cpu().numpy())
            struct_count_probs_all.append(
                torch.softmax(o['struct_count_logits'], dim=-1).cpu().numpy())
            struct_form_logits_all.append(o['struct_form_logits'].cpu().numpy())
    alloc_probs = np.concatenate(alloc_probs_all)          # [n,3,34,5]
    wait_probs = np.concatenate(wait_probs_all)            # [n,3,35] 因子化读法（RM1）
    wait_logits = np.concatenate(wait_logits_all).astype(np.float64)   # [n,3,35]

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

    # ---- wait 诊断（RM1：wait_probs 为因子化读法，p34 = sigmoid(l34) = P(未听)）----
    active = tenpai                                             # [n,3]
    sup = wait_true[..., :NUM_TILES] > 0                        # 非空可荣集
    eff_active = sup.any(-1)                                    # 有可荣牌的样本
    # RM3：CE 拆分（active 上恒等式 ce = −logP(听) + cond_ce）
    ll_wait, neg_logp_t, cond_ce = wait_ce_decomposition(wait_probs, wait_true)
    tile_probs = wait_probs[..., :NUM_TILES]
    order = np.argsort(-tile_probs, axis=-1)
    hit1 = np.take_along_axis(sup, order[..., :1], axis=-1).squeeze(-1)
    hit3 = np.take_along_axis(sup, order[..., :3], axis=-1).any(-1)
    p_tenpai = 1.0 - wait_probs[..., NUM_TILES]                 # P(听)：sigmoid 读法（主口径）

    # 旧 softmax35 读法参照列（RM1 修正前后对照：AUC 0.9461 vs 0.9685、
    # 标定比值 6.26× vs 0.998；l34 无 bias、恒有限，max 平移后 exp 不溢出）
    wl = wait_logits - wait_logits.max(-1, keepdims=True)
    ew = np.exp(wl)
    wp_legacy = ew / ew.sum(-1, keepdims=True)
    p_tenpai_legacy = 1.0 - wp_legacy[..., NUM_TILES]

    tenpai_frac = float(tenpai.mean())
    wait_report = {
        'wait_read': ('factorized（RM1）：p34 = sigmoid(l34) = P(未听)，'
                      'p_x = P(听)·softmax34(l_x)；*_softmax_legacy 为旧读法参照'),
        'ce_active': float(ll_wait[eff_active].mean()) if eff_active.any() else None,
        'neg_logp_t_active': float(neg_logp_t[eff_active].mean()) if eff_active.any() else None,
        'cond_ce_active': float(cond_ce[eff_active].mean()) if eff_active.any() else None,
        'hit1': float(hit1[eff_active].mean()) if eff_active.any() else None,
        'hit3': float(hit3[eff_active].mean()) if eff_active.any() else None,
        'tenpai_auc': auc_score(tenpai.ravel(), p_tenpai.ravel()),
        'tenpai_auc_softmax_legacy': auc_score(tenpai.ravel(), p_tenpai_legacy.ravel()),
        'mean_p_tenpai': float(p_tenpai.mean()),
        'p_tenpai_calib_ratio': float(p_tenpai.mean() / max(tenpai_frac, 1e-9)),
        'p_tenpai_calib_ratio_softmax_legacy':
            float(p_tenpai_legacy.mean() / max(tenpai_frac, 1e-9)),
        'structural_tenpai_frac': tenpai_frac,
        'furiten_frac': float((tenpai & ~eff_active).sum() / max(tenpai.sum(), 1)),
        'riichi_target_frac': float(riichi.mean()),
    }
    # 立直家单独一档（下游最关心）
    m_riichi = riichi & eff_active
    if m_riichi.any():
        wait_report['riichi_hit1'] = float(hit1[m_riichi].mean())
        wait_report['riichi_hit3'] = float(hit3[m_riichi].mean())
        wait_report['riichi_ce'] = float(ll_wait[m_riichi].mean())
        wait_report['riichi_neg_logp_t'] = float(neg_logp_t[m_riichi].mean())
        wait_report['riichi_cond_ce'] = float(cond_ce[m_riichi].mean())

    # RM3：主指标补决策相关分层。全量均值被无信息样本支配（29% 样本 turn≤3、
    # 听牌率 0.6%），必须同时在决策相关子集上报（README §11.2/§12.9）。
    turn3 = np.broadcast_to(turn[:, None], (n, 3))
    called_b = batch['called'].numpy() > 0.5
    strata_defs = {
        'all': np.ones((n, 3), bool),
        'turn>=8': turn3 >= 8,
        'riichi': riichi,
        'called=1': called_b,
        'clean_turn>=8': (turn3 >= 8) & ~riichi & ~called_b,
    }
    strata_out = {}
    for sname, S in strata_defs.items():
        A = S & eff_active
        e = {'n_targets': int(S.sum())}
        if S.any() and 0 < tenpai[S].mean() < 1:
            e['tenpai_rate'] = float(tenpai[S].mean())
            e['tenpai_auc'] = auc_score(tenpai[S].ravel(), p_tenpai[S].ravel())
            e['mean_p_tenpai'] = float(p_tenpai[S].mean())
        if A.any():
            e.update({
                'n_active': int(A.sum()),
                'ce_active': float(ll_wait[A].mean()),
                'neg_logp_t': float(neg_logp_t[A].mean()),
                'cond_ce': float(cond_ce[A].mean()),
                'hit1': float(hit1[A].mean()),
                'hit3': float(hit3[A].mean()),
            })
        strata_out[sname] = e
    wait_report['strata'] = strata_out

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
            # RM3：n/recall 按真值桶（n = 真值个数），n_pred/precision 按预测桶。
            # 两口径同时输出，杜绝 README §11.5-⑦ 那类「recall 还是 precision」
            # 的口径争议（旧报告只有一列，解读歧义无法事后裁决）。
            'shanten_head_by_bucket': {
                str(k): {
                    'n': int((sh_true == k).sum()),
                    'recall': (float((sh_head[sh_true == k] == k).mean())
                               if (sh_true == k).any() else None),
                    'n_pred': int((sh_head == k).sum()),
                    'precision': (float((sh_true[sh_head == k] == k).mean())
                                  if (sh_head == k).any() else None),
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
                    'n_pred': int((form_pred == k).sum()),
                    'precision': (float((form_true[form_pred == k] == k).mean())
                                  if (form_pred == k).any() else None),
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
    print(f"wait CE(听牌)   : {wait_report['ce_active']:.4f} = "
          f"−logP(听) {wait_report['neg_logp_t_active']:.4f} + "
          f"cond {wait_report['cond_ce_active']:.4f}   (RM3 拆分)")
    print(f"wait hit@1/hit@3: {wait_report['hit1']:.4f} / {wait_report['hit3']:.4f}")
    print(f"听牌检测 AUC    : {wait_report['tenpai_auc']:.4f} (sigmoid 读法)   "
          f"旧 softmax35 读法 {wait_report['tenpai_auc_softmax_legacy']:.4f}")
    print(f"听牌标定        : meanP {wait_report['mean_p_tenpai']:.4f} / "
          f"实际 {wait_report['structural_tenpai_frac']:.4f} = "
          f"{wait_report['p_tenpai_calib_ratio']:.4f}   "
          f"(旧读法 {wait_report['p_tenpai_calib_ratio_softmax_legacy']:.4f})")
    print(f"结构性听牌率 {wait_report['structural_tenpai_frac']:.4f}   "
          f"其中振听(无可荣牌) {wait_report['furiten_frac']:.4f}")
    if 'riichi_hit1' in wait_report:
        print(f"立直家 hit@1/@3 : {wait_report['riichi_hit1']:.4f} / "
              f"{wait_report['riichi_hit3']:.4f} (CE {wait_report['riichi_ce']:.4f} = "
              f"−logP(t) {wait_report['riichi_neg_logp_t']:.4f} + "
              f"cond {wait_report['riichi_cond_ce']:.4f})")
    print('---- wait 决策相关分层（RM3）----')
    for sname, e in wait_report['strata'].items():
        if 'ce_active' in e:
            print(f"  {sname:<16s} n_active {e['n_active']:>6d}  "
                  f"ce {e['ce_active']:.4f} (−logP(t) {e['neg_logp_t']:.4f} + "
                  f"cond {e['cond_ce']:.4f})  hit1 {e['hit1']:.4f} "
                  f"hit3 {e['hit3']:.4f}  AUC {e.get('tenpai_auc', float('nan')):.4f}")
        else:
            print(f"  {sname:<16s} n_targets {e['n_targets']:>6d}  (无 active 目标)")
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
            f"/{(v['precision'] if v['precision'] is not None else float('nan')):.3f}"
            for k, v in struct_report['shanten_head_by_bucket'].items())
        print(f"向听逐桶 R/P    : {recalls}")
        forms = '  '.join(
            f"{nm}:{(v['recall'] if v['recall'] is not None else float('nan')):.3f}"
            f"/{(v['precision'] if v['precision'] is not None else float('nan')):.3f}"
            for nm, v in struct_report['form_by_class'].items())
        print(f"形制准确/召回/精确: {struct_report['form_acc']:.4f}   {forms}")
        print(f"结构计数 L1     : {struct_report['struct_counts_l1']:.4f}")
    print('==================================================\n')

    out = args.out or str(Path(args.state).parent / 'eval_report.json')
    with open(out, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'[out] {out}', flush=True)


if __name__ == '__main__':
    main()
