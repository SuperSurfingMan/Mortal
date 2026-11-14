#!/usr/bin/env python3
"""把 mjai-reviewer 的 4.1c 报告与本仓库模型在同一批决策点上对齐。

为什么需要
----------
`parse_review_reports.py` 抽出了 4.1c 在每个决策点的完整 Q 表，但那只是单边的。
要回答「我们的分歧和它的分歧是不是同一批局面」，必须把两边的 Q 放在同一行上。

做法
----
1. 用 `libriichi.state.PlayerState` 重放 mjai 牌谱，复现 reviewer 的决策点集合
   （`mask.sum() >= 2`，与 `src/review/mortal.rs` 的 `total_reviewed` 同口径）。
2. 用 `(kyoku, 手牌计数, 向听)` 与报告 entry 配对。
3. 对配上的点跑我们的模型，得到 46 维 Q。
4. 输出对齐后的 JSONL，并统计：argmax 一致率、Q 相关性、分歧交集。

⚠️ 口径
--------
- 4.1c 的 Q 是**名次 EV**（约 −2 ~ +0.4），本仓库是 MC 回报尺度；
  比较相关性可以，直接比数值不行。
- 报告把三种「吃」渲染成同一个文本，所以吃只按类别比较；打牌可逐牌对齐。

用法
----
    MORTAL_CFG=config.s6c.toml python align_teacher41c.py \
        --teacher logs/teacher41c/teacher_41c.jsonl \
        --ckpt checkpoints/s6c/mortal.pth \
        --out logs/teacher41c/aligned.jsonl \
        --report logs/teacher41c/ALIGN.md
"""

import argparse
import gzip
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prelude  # noqa: E402,F401
import s6c_features  # noqa: E402
from config import config  # noqa: E402
from libriichi.consts import ACTION_SPACE  # noqa: E402
from libriichi.state import PlayerState  # noqa: E402
from model import Brain, DQN  # noqa: E402

# 动作空间（libriichi/src/consts.rs）
A_DISCARD = range(0, 34)
A_RIICHI = 37
A_CHI = range(38, 41)
A_PON = 41
A_KAN = 42
A_AGARI = 43
A_RYUKYOKU = 44
A_PASS = 45

MJAI_TILES = (
    [f'{n}m' for n in range(1, 10)]
    + [f'{n}p' for n in range(1, 10)]
    + [f'{n}s' for n in range(1, 10)]
    + ['E', 'S', 'W', 'N', 'P', 'F', 'C']
)
IDX2TILE = {i: t for i, t in enumerate(MJAI_TILES)}

# 报告里的牌名 → mjai 牌名
REPORT2MJAI = {'e': 'E', 's': 'S', 'w': 'W', 'n': 'N', 'p': 'P', 'f': 'F', 'c': 'C'}
for _i in range(1, 10):
    for _s in 'mps':
        REPORT2MJAI[f'{_i}{_s}'] = f'{_i}{_s}'
REPORT2MJAI.update({'5mr': '5m', '5pr': '5p', '5sr': '5s'})


def report_tile_to_mjai(t):
    return REPORT2MJAI.get(t, t)


def counts_of(names):
    c = [0] * 34
    for n in names:
        m = report_tile_to_mjai(n)
        if m in MJAI_TILES:
            c[MJAI_TILES.index(m)] += 1
    return tuple(c)


def action_kind(a):
    """把 4.1c 的动作文本归一成类别（吃不细分）。"""
    a = (a or '').strip()
    if a.startswith('打'):
        return 'discard'
    if '碰' in a:
        return 'pon'
    if '吃' in a:
        return 'chi'
    if '杠' in a:
        return 'kan'
    if '立直' in a:
        return 'riichi'
    if '自摸' in a:
        return 'tsumo'
    if a == '和':
        return 'ron'
    if a == '流局':
        return 'ryukyoku'
    if '跳过' in a:
        return 'pass'
    return a or '?'


def action_of_label(label):
    """4.1c Q 表的一行 → (kind, tile_idx or None)。"""
    k = action_kind(label)
    if k == 'discard':
        tiles = [t for t in REPORT2MJAI if '{' + t + '}' in label]
        return 'discard', (MJAI_TILES.index(report_tile_to_mjai(tiles[0]))
                           if tiles and report_tile_to_mjai(tiles[0]) in MJAI_TILES
                           else None)
    return k, None


def canon(s):
    """动作文本 → 可比较的规范 key（打牌精确到牌，和/自摸合并）。

    同时接受两种牌名写法：报告的小写（`打 {n}`）与 mjai 的大写（`打 {N}`）。
    首版只查了 REPORT2MJAI，导致我们自己的 `打 {N}` 全被归一成 `discard:?`，
    打牌一致率被错算成 54%。
    """
    k = action_kind(s)
    if k in ('ron', 'tsumo'):
        k = 'agari'
    if k == 'discard':
        s = s or ''
        for t in REPORT2MJAI:
            if '{' + t + '}' in s:
                return 'discard:' + report_tile_to_mjai(t)
        for t in MJAI_TILES:
            if '{' + t + '}' in s:
                return 'discard:' + t
        return 'discard:?'
    return k


def same_action(human, best):
    """人类动作（文本）与我们的动作索引是否**完全**相同。

    ⚠️ 不能只比类别：「打 X」与「打 Y」类别都是 discard，那样会把
    约 20% 的打牌分歧全部算成一致（首版就踩了这个坑：算出 96.25%）。
    和/自摸统一归到 agari。
    """
    hk = action_kind(human)
    bk = idx_to_kind(best)
    if hk in ('ron', 'tsumo'):
        hk = 'agari'
    if hk != bk:
        return False
    if hk == 'discard':
        tiles = [t for t in REPORT2MJAI if '{' + t + '}' in (human or '')]
        if not tiles:
            return False
        m = report_tile_to_mjai(tiles[0])
        return m in MJAI_TILES and MJAI_TILES.index(m) == best
    return True


def idx_to_kind(a):
    if a in A_DISCARD:
        return 'discard'
    if a == A_RIICHI:
        return 'riichi'
    if a in A_CHI:
        return 'chi'
    if a == A_PON:
        return 'pon'
    if a == A_KAN:
        return 'kan'
    if a == A_AGARI:
        return 'agari'
    if a == A_RYUKYOKU:
        return 'ryukyoku'
    if a == A_PASS:
        return 'pass'
    return 'kan_choice'


CAN_FLAGS = (
    'can_discard', 'can_chi', 'can_pon', 'can_kan', 'can_riichi',
    'can_daiminkan', 'can_ankan', 'can_kakan', 'can_tsumo_agari',
    'can_ron_agari', 'can_ryukyoku', 'can_pass', 'can_agari',
)


def replay(path, seat, version):
    """重放一局，返回 reviewer 口径的决策点（mask.sum() >= 2）。

    局号口径：报告用 `kyoku-<n>-<honba>`，其中 n = 东1..南4 的序号（0..7），
    **连庄不改 n**。所以不能用 `start_kyoku` 计数（连庄会多算），
    必须由 (bakaze, kyoku) 算。
    """
    out = []
    st = PlayerState(seat)
    kyoku = -1
    BAKAZE = {'E': 0, 'S': 1, 'W': 2, 'N': 3}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get('type') == 'start_kyoku':
                kyoku = (BAKAZE.get(ev.get('bakaze'), 0) * 4
                         + int(ev.get('kyoku', 1)) - 1)
            cans = st.update(line)
            if not any(getattr(cans, k) for k in CAN_FLAGS):
                continue
            obs, mask = st.encode_obs(version, False)
            if int(mask.sum()) < 2:
                continue
            out.append({
                'obs': np.asarray(obs, dtype=np.float32),
                'mask': np.asarray(mask, dtype=bool),
                'tehai': tuple(int(x) for x in st.tehai),
                'shanten': int(st.shanten),
                'kyoku': kyoku,
                'n_cand': int(mask.sum()),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='logs/teacher41c/teacher_41c.jsonl')
    ap.add_argument('--results',
                    default='deliver/_extract/ai_similarity_kit/results.csv')
    ap.add_argument('--hands',
                    default='deliver/_extract/ai_similarity_kit/hands.csv')
    ap.add_argument('--ckpt', default='checkpoints/s6c/mortal.pth')
    ap.add_argument('--mjai-root', default='../dataset/mjai/houou4')
    ap.add_argument('--out', default='logs/teacher41c/aligned.jsonl')
    ap.add_argument('--report', default='logs/teacher41c/ALIGN.md')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    # ---- 官方 entry ----
    metas, entries = {}, []
    for line in open(args.teacher, encoding='utf-8'):
        d = json.loads(line)
        if d.get('kind') == 'meta':
            metas[d['report']] = d
        else:
            entries.append(d)

    # report → (log_id, seat)
    import csv
    rep2log = {}
    with open(args.results, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rep2log[r['report_url'].rstrip('/').split('/')[-1].split('.')[0]] = (
                r['log_id'], int(r['seat']))
    by_pair = defaultdict(list)
    for e in entries:
        by_pair[rep2log[e['report']]].append(e)

    # ---- 模型 ----
    st_ck = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    cfg = st_ck['config']
    ver = cfg['control'].get('version', 4)
    spec, extra = s6c_features.resolve(cfg['control'])
    dev = torch.device(args.device)
    brain = Brain(version=ver, extra_channels=extra, **cfg['resnet']).to(dev).eval()
    dqn = DQN(version=ver).to(dev).eval()
    brain.load_state_dict(st_ck['mortal'])
    dqn.load_state_dict(st_ck['current_dqn'])
    print(f'模型 {args.ckpt}  steps={st_ck.get("steps")}  version={ver}  extra={spec!r}')

    # ---- 逐份对齐 ----
    aligned = []
    stat = []
    for (log_id, seat), ents in sorted(by_pair.items()):
        sub = os.path.join(args.mjai_root, log_id[:6], log_id + '.json.gz')
        if not os.path.exists(sub):
            stat.append((log_id, seat, -1, len(ents), 'mjai 缺失'))
            continue
        pts = replay(sub, seat, ver)
        # 对齐：**不能**用 (手牌, 向听) 做键——
        #   报告的「跳过」类 entry 展示的是 14 张（摸牌未打），本地 13 张；
        #   报告的向听也因此差 1。但两边的 (局号, 候选数) 序列是一致的，
        #   用 LCS 对齐既能保持单调性，又能吃掉本地多出的边界点（如杠选择）。
        import difflib
        seq_r = [(e['kyoku'], e['n_cand']) for e in ents]
        seq_l = [(p['kyoku'], p['n_cand']) for p in pts]
        sm = difflib.SequenceMatcher(a=seq_r, b=seq_l, autojunk=False)
        pairs = []
        for blk in sm.get_matching_blocks():
            for d in range(blk.size):
                pairs.append((ents[blk.a + d], pts[blk.b + d]))
        # tehai 校验（只对打牌点有意义）
        check = [(e, p) for e, p in pairs if action_kind(e['human']) == 'discard']
        tok = sum(1 for e, p in check if counts_of(e['hand']) == p['tehai'])
        ok = (len(pairs) == len(ents) == len(pts)
              and (not check or tok == len(check)))
        note = (f'配对 {len(pairs)}/{len(ents)}，打牌手牌一致 '
                f'{tok}/{len(check)}')
        stat.append((log_id, seat, len(pts), len(ents), note + ('' if ok else ' ⚠️')))

        if not pairs:
            continue
        obs = np.stack([p['obs'] for _, p in pairs])
        feat = s6c_features.compute(obs) if spec else None
        obs_in = s6c_features.attach(obs, feat) if spec else obs
        qs = []
        B = 256
        with torch.no_grad():
            for i in range(0, len(obs_in), B):
                x = torch.as_tensor(obs_in[i:i + B], device=dev)
                mk = torch.as_tensor(np.stack([p['mask'] for _, p in pairs[i:i + B]]),
                                     device=dev)
                phi = brain(x, None)
                qs.append(dqn(phi, mk).float().cpu().numpy())
        Q = np.concatenate(qs)

        for k, (e, p) in enumerate(pairs):
            q = Q[k]
            best = int(np.argmax(q))
            aligned.append({
                'report': e['report'], 'log_id': log_id, 'seat': seat,
                'kyoku': e['kyoku'], 'turn': e['turn'], 'shanten': e['shanten'],
                'human': e['human'], 'mortal_41c': e['mortal'],
                's6c': (f'打 {{{IDX2TILE[best]}}}' if best in A_DISCARD
                        else idx_to_kind(best)),
                's6c_kind': idx_to_kind(best),
                'human_kind': action_kind(e['human']),
                'consistent_41c': e['consistent'],
                'consistent_s6c': same_action(e['human'], best),
                'human_rank_41c': e['human_rank'],
                'k41c': action_kind(e['mortal']),
                # 打牌可精确对齐，存两边 Q（4.1c 只在打牌行上）
                'q41c': ({str(action_of_label(c['a'])[1]): c['q']
                          for c in e['cands']
                          if action_of_label(c['a'])[0] == 'discard'
                          and action_of_label(c['a'])[1] is not None}
                         if action_kind(e['human']) == 'discard' else None),
                'q_s6c': ([round(float(v), 5) for v in q[:34]]
                          if action_kind(e['human']) == 'discard' else None),
                'mask_sum': p['n_cand'],
            })

    with open(args.out, 'w', encoding='utf-8') as f:
        for d in aligned:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'对齐写出 {args.out}：{len(aligned)} 点')

    # ---- 统计 ----
    def pct(x, n):
        return f'{x / n * 100:.3f}%' if n else '—'

    L = []
    A = L.append
    n = len(aligned)
    A('# 4.1c 与本仓库 S6c 的同局面对齐（2026-11-14）')
    A('')
    A('## 0. 对齐质量')
    A('')
    A(f'- 报告 entry 总数：**{len(entries)}**；成功配对并前向：**{n}**'
      f'（{n / len(entries) * 100:.2f}%）')
    bad = [s for s in stat if '⚠️' in s[4] or s[2] < 0]
    A(f'- 完全对齐（本地点数 == 报告 reviewed）的报告：'
      f'**{sum(1 for s in stat if "⚠️" not in s[4] and s[2] >= 0)}/{len(stat)}**')
    if bad:
        A('- 未完全对齐：')
        for log_id, seat, a, b, note in bad:
            A(f'  - `{log_id}` seat{seat}：本地 {a} / 报告 {b}（{note}）')
    A('')

    A('## 1. 人类一致率：同一批决策点')
    A('')
    c41 = sum(1 for d in aligned if d['consistent_41c'])
    c6 = sum(1 for d in aligned if d['consistent_s6c'])
    A('| 模型 | 一致 | 一致率 |')
    A('|---|---:|---:|')
    A(f'| 官方 4.1c | {c41} | **{pct(c41, n)}** |')
    A(f'| 本仓库 S6c | {c6} | **{pct(c6, n)}** |')
    A(f'| 差 | {c6 - c41} | {(c6 - c41) / n * 100:+.3f}pp |')
    A('')

    A('## 2. 4.1c 分歧时，S6c 站在哪边')
    A('')
    dis41 = [d for d in aligned if not d['consistent_41c']]
    m = len(dis41)
    same = sum(1 for d in dis41 if d['consistent_s6c'])
    A(f'- 4.1c 分歧点：**{m}** 个')
    A(f'- 其中 S6c **与人类一致**：**{same}**（{pct(same, m)}）')
    A(f'- 其中 S6c **也与人类分歧**：{m - same}（{pct(m - same, m)}）')
    A('')
    A('> 读法：若后一项高，说明这些点是「两个 AI 都反对人类」的区域——'
      '更可能是人类的习惯性选择；若前一项高，说明 4.1c 在这里有独有偏好（或错误）。')
    A('')
    dis6 = [d for d in aligned if not d['consistent_s6c']]
    m6 = len(dis6)
    same6 = sum(1 for d in dis6 if d['consistent_41c'])
    A(f'- 反向看：S6c 分歧点 **{m6}** 个，其中 4.1c 与人类一致 '
      f'**{same6}**（{pct(same6, m6)}）')
    A('')

    A('## 3. argmax 完全一致率')
    A('')
    agree = 0
    for d in aligned:
        if canon(d['mortal_41c']) == canon(d['s6c']):
            agree += 1
    A(f'- 逐动作完全一致（打牌精确到牌，吃按类别）：**{agree}/{n} = {pct(agree, n)}**')
    by_kind = defaultdict(lambda: [0, 0])
    for d in aligned:
        hk = d['human_kind']
        by_kind[hk][1] += 1
        if canon(d['mortal_41c']) == canon(d['s6c']):
            by_kind[hk][0] += 1
    A('')
    A('| 人类动作类型 | 点数 | 两模型 argmax 完全一致 |')
    A('|---|---:|---:|')
    for k, (a, b) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        A(f'| {k} | {b} | {pct(a, b)} |')
    A('')

    A('## 4. 打牌决策上的 Q 相关性（逐决策点 Spearman）')
    A('')
    rhos = []
    for d in aligned:
        if not d['q41c'] or not d['q_s6c']:
            continue
        common = [int(k) for k in d['q41c'] if d['mask_sum'] and 0 <= int(k) < 34]
        if len(common) < 4:
            continue
        a = np.array([d['q41c'][str(i)] for i in common])
        b = np.array([d['q_s6c'][i] for i in common])
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
    if rhos:
        rhos.sort()
        A(f'- n = **{len(rhos)}** 个打牌决策点（两边都有 >=4 个候选）')
        A(f'- Spearman ρ：均值 **{statistics.mean(rhos):.3f}**、'
          f'中位 **{statistics.median(rhos):.3f}**、'
          f'p10 {rhos[int(len(rhos) * .1)]:.3f} / p90 {rhos[int(len(rhos) * .9)]:.3f}')
        A(f'- ρ > 0.5 的占比：**{pct(sum(1 for r in rhos if r > 0.5), len(rhos))}**'
          f'；ρ < 0 的占比：{pct(sum(1 for r in rhos if r < 0), len(rhos))}')
    A('')

    A('## 5. 按 4.1c 的「把握」分层')
    A('')
    A('| 4.1c top1−top2 | 点数 | 4.1c 分歧率 | S6c 一致率 |')
    A('|---|---:|---:|---:|')
    rows = []
    for d in aligned:
        if not d['q41c']:
            continue
        qs = sorted(d['q41c'].values(), reverse=True)
        if len(qs) < 2:
            continue
        rows.append((qs[0] - qs[1], d))
    if rows:
        rows.sort(key=lambda x: x[0])
        k = len(rows) // 4
        for i in range(4):
            chunk = rows[i * k:(i + 1) * k] if i < 3 else rows[3 * k:]
            if not chunk:
                continue
            lo, hi = chunk[0][0], chunk[-1][0]
            A(f'| [{lo:.3f}, {hi:.3f}) | {len(chunk)} | '
              f'{pct(sum(1 for _, d in chunk if not d["consistent_41c"]), len(chunk))} | '
              f'{pct(sum(1 for _, d in chunk if d["consistent_s6c"]), len(chunk))} |')
    A('')

    open(args.report, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'写出 {args.report}')
    print('\n'.join(L[:30]))


if __name__ == '__main__':
    main()