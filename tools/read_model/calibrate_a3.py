"""A3 规则法基线权重标定工具（一次性先验标定，非训练）。

用途：在**非评估月**（默认 202508，评估月 202609 是其后的严格 holdout）上测量
规则法基线各软权重因子的经验比率，为 rule_baseline.py 的常数权重提供出处
（provenance）。标定结论冻结为常数后，评估与对决只在 202609 固定评估集上进行。

测量口径（全部为「成员占比 / 牌种占比」的 per-tile 比率，比值 1 = 无效应）：

  alloc：各类别（字牌/中张两端/中张）持有率、未见张数梯度（壁）、河牌效应、
         宝牌效应、舍牌邻接效应（早切/晚切分开）。
  wait ：真实有效等待集上的各类别 per-tile 比率、未见张数梯度、筋牌效应
         （早切/晚切、立直/非立直分开）、舍牌邻接效应、宝牌效应、字牌等待
         的巡目分布。

用法：
    python tools/read_model/calibrate_a3.py --root .../202508 --samples 4000
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D                                     # noqa: E402
import rule_baseline as RB                              # noqa: E402
import train as T                                       # noqa: E402

NUM_TILES = 34
HONOR = np.zeros(NUM_TILES, dtype=bool)
HONOR[27:] = True
TERMINAL = np.zeros(NUM_TILES, dtype=bool)
for _t in range(27):
    if _t % 9 in (0, 8):
        TERMINAL[_t] = True
MIDDLE = ~HONOR & ~TERMINAL


def early_discards(tl_tokens: np.ndarray, tl_mask: np.ndarray, k: int = 3) -> np.ndarray:
    """各目标家前 k 次舍牌的牌种集合 [n,3,34]（早切 = 配牌形状的直接信息）。"""
    n = tl_tokens.shape[0]
    dahai = tl_tokens[..., 35] > 0.5
    tile_oh = tl_tokens[..., :NUM_TILES]
    early = np.zeros((n, 3, NUM_TILES), dtype=bool)
    L = tl_mask.shape[-1]
    for s in range(n):
        for j in range(3):
            c = 0
            for p in range(L):
                if tl_mask[s, j, p] and dahai[s, j, p]:
                    early[s, j, int(np.argmax(tile_oh[s, j, p]))] = True
                    c += 1
                    if c >= k:
                        break
    return early


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',
                    default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202508',
                    help='标定月数据根目录（必须不同于评估月 202609）')
    ap.add_argument('--files', type=int, default=12)
    ap.add_argument('--samples', type=int, default=4000)
    args = ap.parse_args()

    files = sorted(glob.glob(f'{args.root}/**/*.json.gz', recursive=True))[:args.files]
    batch = T.build_fixed_eval_set(files, args.samples)
    b = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in batch.items()}

    H = b['alloc']                                        # [n,3,34]
    U = np.round(b['unseen']).astype(int)
    R = b['river']                                        # [n,3,34]
    n = H.shape[0]
    u3 = np.broadcast_to(U[:, None, :], (n, 3, NUM_TILES))
    held = H / np.maximum(u3, 1)                          # 每容量持有率
    dora = RB.indicators_to_dora(b['dora_indicators']) > 0
    dora3 = np.broadcast_to(dora[:, None, :], (n, 3, NUM_TILES))
    early = early_discards(b['tl_tokens'], b['tl_mask'])
    late = R & ~early

    print(f'[calib] root={args.root} samples={n}')
    print('================ alloc 持有率比率（相对中张 base） ================')
    base = held[np.broadcast_to(MIDDLE[None, None, :], (n, 3, NUM_TILES))].mean()
    for name, m in [('honor', HONOR), ('terminal', TERMINAL), ('middle', MIDDLE)]:
        mm = np.broadcast_to(m[None, None, :], (n, 3, NUM_TILES))
        r = held[mm].mean() / base
        print(f'  {name:9s}: ratio {r:.3f}')
    for uu in range(1, 5):
        m = u3 == uu
        print(f'  u={uu}: ratio {held[m].mean() / held[u3 == 4].mean():.3f}')
    m3 = u3 == 3
    print(f'  river(u=3): ratio {held[m3 & R].mean() / held[m3 & ~R].mean():.3f}')
    print(f'  dora      : ratio {held[dora3].mean() / held[~dora3].mean():.3f}')
    adj_e = RB._dilate_suit(early) & ~R
    adj_l = RB._dilate_suit(late) & ~R
    print(f'  adj_early : ratio {held[m3 & adj_e].mean() / held[m3 & ~adj_e & ~R].mean():.3f}')
    print(f'  adj_late  : ratio {held[m3 & adj_l].mean() / held[m3 & ~adj_l & ~R].mean():.3f}')

    print('================ wait 有效等待 per-tile 比率（条件于听牌） ================')
    sup = b['wait'][..., :NUM_TILES] > 0                  # [n,3,34]
    eff = sup.any(-1)
    eff3 = np.broadcast_to(eff[:, :, None], (n, 3, NUM_TILES))
    own = b['own_hand']
    ri = b['riichi'] > 0.5
    expressible = (U > 0) | own
    support = np.broadcast_to(expressible[:, None, :], (n, 3, NUM_TILES)) & ~R
    ri3 = np.broadcast_to(ri[:, :, None], (n, 3, NUM_TILES))
    pool_all = support & eff3          # 全部比率条件于「目标家听牌」（纯放置效应）

    def wratio(mask, pool=pool_all):
        mm = mask & pool
        share = sup[mm].sum() / sup[pool].sum()
        tshare = mm.sum() / pool.sum()
        return share / tshare, share, tshare

    print(f'  [全体] memberships={int(sup.sum())} eff_slots={int(eff.sum())} '
          f'riichi_eff={int((ri & eff).sum())}')
    for name, m in [('honor', HONOR), ('terminal', TERMINAL), ('middle', MIDDLE)]:
        r, sh, ts = wratio(np.broadcast_to(m[None, None, :], (n, 3, NUM_TILES)))
        print(f'  {name:9s}: ratio {r:.3f}  (wait share {sh:.3f} / tile share {ts:.3f})')
    for uu in range(1, 5):
        r, _, _ = wratio(np.broadcast_to((U == uu)[:, None, :], (n, 3, NUM_TILES)))
        print(f'  u={uu}: ratio {r:.3f}')
    suji_e = RB._dilate_suit(early, offsets=(-3, 3)) & ~R
    suji_l = RB._dilate_suit(late, offsets=(-3, 3)) & ~R
    r, _, _ = wratio(suji_e)
    print(f'  suji_early(all): ratio {r:.3f}')
    r, _, _ = wratio(suji_e, pool=pool_all & ri3)
    print(f'  suji_early(riichi): ratio {r:.3f}')
    r, _, _ = wratio(suji_l)
    print(f'  suji_late (all): ratio {r:.3f}')
    adj_e = RB._dilate_suit(early) & ~R
    adj_l = RB._dilate_suit(late) & ~R
    r, _, _ = wratio(adj_e)
    print(f'  adj_early (all): ratio {r:.3f}')
    r, _, _ = wratio(adj_l)
    print(f'  adj_late  (all): ratio {r:.3f}')
    r, _, _ = wratio(adj_e, pool=pool_all & ri3)
    print(f'  adj_early (riichi): ratio {r:.3f}')
    r, _, _ = wratio(dora3)
    print(f'  dora      : ratio {r:.3f}')
    dora_adj = RB._dilate_suit(dora[:, None, :] > 0) & ~dora3
    r, _, _ = wratio(dora_adj)
    print(f'  dora_adj  : ratio {r:.3f}')
    # 分类条件化（排除中张构成的混杂）：只看中张
    mid3 = np.broadcast_to(MIDDLE[None, None, :], (n, 3, NUM_TILES))
    for uu in range(1, 5):
        r, _, _ = wratio(np.broadcast_to((U == uu)[:, None, :], (n, 3, NUM_TILES)),
                         pool=pool_all & mid3)
        print(f'  u={uu} (middle only): ratio {r:.3f}')
    r, _, _ = wratio(suji_e, pool=pool_all & mid3)
    print(f'  suji_early(middle only): ratio {r:.3f}')
    r, _, _ = wratio(suji_l, pool=pool_all & mid3)
    print(f'  suji_late (middle only): ratio {r:.3f}')
    r, _, _ = wratio(adj_e, pool=pool_all & mid3)
    print(f'  adj_early (middle only): ratio {r:.3f}')
    r, _, _ = wratio(adj_l, pool=pool_all & mid3)
    print(f'  adj_late  (middle only): ratio {r:.3f}')
    r, _, _ = wratio(dora3, pool=pool_all & mid3)
    print(f'  dora      (middle only): ratio {r:.3f}')
    # 字牌等待的巡目分布
    turn = b['turn']
    print('  ---- honor wait share by turn ----')
    for lo, hi in [(0, 3), (4, 7), (8, 11), (12, 99)]:
        m = np.broadcast_to(((turn >= lo) & (turn <= hi))[:, None, None], (n, 3, NUM_TILES))
        mm = m & support
        hs = sup[mm & np.broadcast_to(HONOR[None, None, :], (n, 3, NUM_TILES))].sum() / max(sup[mm].sum(), 1)
        print(f'    turn {lo}-{hi}: honor wait share {hs:.4f}  eff rate {eff[m[:, :, 0]].mean():.3f}')
    # 听牌先验逻辑回归的特征-标签关联（方向性核对）
    print('  ---- tenpai prior features (non-riichi) ----')
    feat = RB.timeline_features(b['tl_tokens'], b['tl_mask'])
    nr = ~ri
    tp = b['tenpai'] > 0.5
    for fname in ('n_disc', 'tg_tail', 'last_tg'):
        v = feat[fname][nr]
        y = tp[nr]
        for lo, hi in [(0, 0.5), (0.5, 1.5), (1.5, 99)]:
            m = (v > lo) & (v <= hi)
            if m.sum() > 20:
                print(f'    {fname} in ({lo},{hi}]: tenpai rate {y[m].mean():.3f} (n={int(m.sum())})')
    print(f"    called=1 tenpai rate {tp[nr & (b['called'] > 0.5)].mean():.3f} "
          f"(n={int((nr & (b['called'] > 0.5)).sum())})  "
          f"called=0 {tp[nr & (b['called'] <= 0.5)].mean():.3f}")


if __name__ == '__main__':
    main()
