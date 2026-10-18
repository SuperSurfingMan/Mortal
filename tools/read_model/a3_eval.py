"""A3 规则法基线评估：与 evaluate.py 同口径的离线诊断（固定评估集 + 同指标）。

用途：在**与 A2b 完全同一的固定评估集**（202609 holdout，build_fixed_eval_set
同种子，20000 样本）上评估规则法基线，输出与 eval_report.json 同构的报告 +
防守模拟指标（compare_bootstrap 口径）。规则法无训练、无 checkpoint，本脚本
即其「evaluate.py」。

用法：
    python tools/read_model/a3_eval.py --samples 20000
    python tools/read_model/a3_eval.py --samples 20000 --out mortal/read_model_a3/eval_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D                                     # noqa: E402
import rule_baseline as RB                              # noqa: E402
import train as T                                       # noqa: E402
from evaluate import TURN_BUCKETS, auc_score, binomial_baseline, ece  # noqa: E402

NUM_TILES = D.NUM_TILES


def defense_metrics(wait_probs, batch):
    """compare_bootstrap 口径的防守模拟（逐目标家：选自家最安全牌，看点炮率）。

    deal_pure      ：项目口径（仅统计「所选牌概率>0」的场合；对精确 0 概率的
                     规则模型会过滤掉枚举证明安全的牌，系统性偏严）。
    deal_pure_all  ：不过滤口径（所有听牌场合；含 genbutsu/枚举安全牌选择）。
    deal_random    ：随机弃牌基线。
    """
    n = wait_probs.shape[0]
    WT = batch['wait'].numpy() if hasattr(batch['wait'], 'numpy') else batch['wait']
    sup = WT[..., :NUM_TILES] > 0
    eff = sup.any(-1)
    own = batch['own_hand'].numpy() if hasattr(batch['own_hand'], 'numpy') else batch['own_hand']
    riichi = batch['riichi'].numpy() if hasattr(batch['riichi'], 'numpy') else batch['riichi']
    ri = riichi > 0.5
    tile_p = wait_probs[..., :NUM_TILES]
    own3 = np.broadcast_to(own[:, None, :], (n, 3, NUM_TILES))
    p_cand = np.where(own3, tile_p, np.inf)
    safest = p_cand.argmin(-1)
    bidx = (np.arange(n)[:, None], np.arange(3)[None])
    p_safe = tile_p[bidx + (safest,)]
    deal_safe = sup[bidx + (safest,)]
    pure = eff & (p_safe > 0)
    ri_pure = (ri & eff) & (p_safe > 0)
    rnd = (sup & own3).sum(-1) / np.maximum(own3.sum(-1), 1)
    return {
        'deal_pure': float(deal_safe[pure].mean()) if pure.any() else None,
        'n_pure': int(pure.sum()),
        'deal_pure_all': float(deal_safe[eff].mean()) if eff.any() else None,
        'riichi_deal_pure': float(deal_safe[ri_pure].mean()) if ri_pure.any() else None,
        'deal_random': float(rnd[eff].mean()) if eff.any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',
                    default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202609',
                    help='评估月根目录（与 A2b/evaluate.py 同一固定评估集）')
    ap.add_argument('--samples', type=int, default=20000)
    ap.add_argument('--out', default=None,
                    help='报告输出路径（默认 mortal/read_model_a3/eval_report.json）')
    args = ap.parse_args()

    import glob
    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))
    batch = T.build_fixed_eval_set(files, args.samples)
    b = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in batch.items()}
    n = b['unseen'].shape[0]
    print(f'[data] 固定评估集 {n} 样本（{args.root}）', flush=True)

    import time
    t0 = time.time()
    out = RB.rule_predict(b)
    print(f'[rule] rule_predict 用时 {time.time() - t0:.1f}s', flush=True)
    ap_, wp = out['alloc_probs'], out['wait_probs']

    alloc_true = b['alloc']                                  # [n,3,34]
    wait_true = b['wait']                                    # [n,3,35]
    unseen = b['unseen']
    turn = b['turn']
    tenpai = b['tenpai'] > 0.5
    riichi = b['riichi'] > 0.5

    # ---- alloc（与 evaluate.py 同口径）----
    bidx = (np.arange(n)[:, None, None], np.arange(3)[None, :, None],
            np.arange(NUM_TILES)[None, None, :])
    p_true = ap_[bidx + (alloc_true,)]
    ll_col = -np.log(p_true.clip(min=1e-12))
    argmax_count = ap_.argmax(-1)
    top1_conf = ap_.max(-1)
    correct = argmax_count == alloc_true
    alloc_report = {
        'ce_overall': float(ll_col.mean()),
        'top1_acc': float(correct.mean()),
        'top1_conf': float(top1_conf.mean()),
        'ece_top1': ece(top1_conf.ravel(), correct.ravel()),
    }
    hand_sizes = alloc_true.sum(-1)
    base = binomial_baseline(unseen, hand_sizes, b['tiles_left'])
    base_p = base[bidx + (alloc_true,)]
    ll_base = -np.log(base_p.clip(min=1e-12))
    alloc_report['ce_baseline_uniform'] = float(ll_base.mean())
    alloc_report['info_gain_nats'] = float(ll_base.mean() - ll_col.mean())
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
    # 约束层诊断（同一子样本，IPF 投影前后）
    from constraints import apply_constraints
    sub = min(n, 4096)
    cons = apply_constraints(ap_[:sub], unseen[:sub], hand_sizes[:sub])
    alloc_report['constraints'] = cons['diag']

    # ---- wait（与 evaluate.py 同口径）----
    active = tenpai
    sup = wait_true[..., :NUM_TILES] > 0
    eff_active = sup.any(-1)
    ll_wait = -(wait_true * np.where(wait_true > 0, np.log(wp.clip(min=1e-12)), 0)).sum(-1)
    tile_probs = wp[..., :NUM_TILES]
    order = np.argsort(-tile_probs, axis=-1)
    hit1 = np.take_along_axis(sup, order[..., :1], axis=-1).squeeze(-1)
    hit3 = np.take_along_axis(sup, order[..., :3], axis=-1).any(-1)
    hit5 = np.take_along_axis(sup, order[..., :5], axis=-1).any(-1)
    p_tenpai = 1.0 - wp[..., NUM_TILES]
    wait_report = {
        'ce_active': float(ll_wait[eff_active].mean()) if eff_active.any() else None,
        'hit1': float(hit1[eff_active].mean()) if eff_active.any() else None,
        'hit3': float(hit3[eff_active].mean()) if eff_active.any() else None,
        'hit5': float(hit5[eff_active].mean()) if eff_active.any() else None,
        'tenpai_auc': auc_score(tenpai.ravel(), p_tenpai.ravel()),
        'structural_tenpai_frac': float(tenpai.mean()),
        'furiten_frac': float((tenpai & ~eff_active).sum() / max(tenpai.sum(), 1)),
        'riichi_target_frac': float(riichi.mean()),
        'mean_p_tenpai': float(p_tenpai.mean()),
    }
    m_riichi = riichi & eff_active
    if m_riichi.any():
        wait_report['riichi_hit1'] = float(hit1[m_riichi].mean())
        wait_report['riichi_hit3'] = float(hit3[m_riichi].mean())
        wait_report['riichi_ce'] = float(ll_wait[m_riichi].mean())
    # 立直枚举支撑集规模（硬约束强度的诊断）
    enum_sizes = []
    for s in range(n):
        for j in range(3):
            if riichi[s, j]:
                enum_sizes.append(int((wp[s, j, :NUM_TILES] > 0).sum()))
    if enum_sizes:
        wait_report['riichi_enum_support_mean'] = float(np.mean(enum_sizes))
        wait_report['riichi_enum_support_median'] = float(np.median(enum_sizes))

    # ---- 防守模拟 ----
    defense = defense_metrics(wp, b)

    report = {
        'state': 'rule_baseline(A3)',
        'step': None,
        'n_samples': int(n),
        'alloc': alloc_report,
        'wait': wait_report,
        'defense': defense,
    }

    print('\n================ A3 规则法基线离线诊断 ================')
    print(f"alloc CE        : {alloc_report['ce_overall']:.4f}   "
          f"(基线 {alloc_report['ce_baseline_uniform']:.4f}, "
          f"信息增益 {alloc_report['info_gain_nats']:.4f} nats/列)")
    print(f"alloc top1      : {alloc_report['top1_acc']:.4f}   "
          f"ECE {alloc_report['ece_top1']:.4f}")
    for name, bkt in buckets.items():
        print(f"  {name:9s}: CE {bkt['ce']:.4f} vs 基线 {bkt['ce_baseline']:.4f} "
              f"top1 {bkt['top1']:.4f} (n={bkt['n']})")
    print(f"约束层          : {cons['diag']}")
    print(f"wait CE(听牌)   : {wait_report['ce_active']:.4f}")
    print(f"wait hit@1/@3/@5: {wait_report['hit1']:.4f} / {wait_report['hit3']:.4f} "
          f"/ {wait_report['hit5']:.4f}")
    print(f"听牌检测 AUC    : {wait_report['tenpai_auc']:.4f}   "
          f"结构性听牌率 {wait_report['structural_tenpai_frac']:.4f}   "
          f"规则均判听 {wait_report['mean_p_tenpai']:.4f}")
    if 'riichi_hit1' in wait_report:
        print(f"立直家 hit@1/@3 : {wait_report['riichi_hit1']:.4f} / "
              f"{wait_report['riichi_hit3']:.4f} (CE {wait_report['riichi_ce']:.4f})")
    if 'riichi_enum_support_mean' in wait_report:
        print(f"立直枚举支撑集  : 均值 {wait_report['riichi_enum_support_mean']:.1f}  "
              f"中位 {wait_report['riichi_enum_support_median']:.1f} 张听牌候选")
    print(f"防守 deal_pure  : {defense['deal_pure']:.4f} (项目口径, n={defense['n_pure']})  "
          f"deal_pure_all {defense['deal_pure_all']:.4f}  "
          f"立直 {defense['riichi_deal_pure']:.4f}  随机 {defense['deal_random']:.4f}")
    print('======================================================\n')

    out_path = args.out or str(
        Path('/mnt/data8/mortal/Mortal/mortal/read_model_a3/eval_report.json'))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'[out] {out_path}', flush=True)


if __name__ == '__main__':
    main()
