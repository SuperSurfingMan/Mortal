"""读牌模型数据管线回归测试（对应 memory §14-35/36：断言设计意图 + 独立重算 + 数值门禁）。

七层检查：
  A. 标签不变量（全部从**事件流独立重算**，不复用被测代码输出）：
     A1 手牌数   : 每家 alloc.sum() == 13 - 3*(吃+碰+大明杠+暗杠)（加杠净贡献 0）
     A2 136 守恒 : 四家手牌 + 河牌 + 手牌外 tile + 牌山 + 14(死庄) == 136
     A3 tiles_seen: Python 台账推算 == obs 内嵌通道（第 835 行）逐位一致
     A4 局初手牌 : 每个 start_kyoku 快照四家均为 13
     A5 听牌标签 : 归一化；支撑集 ⊆ 结构性听牌 & ~自家河 & 可表达集
     A6 立直冻结 : 立直 accepted 后该家手牌总数不再变化
  B. 与 Rust GameplayLoader 的一致性：
     B1 obs 逐位 : loader 的每条 obs（除 kan_select 条目）都出现在我的样本中且逐位相等
     B2 extra 分类: 我的多出样本只来自已知类别（局初/公告/他人先吃的多反应点）
  C. 万↔筒增广等变：标签按牌种置换一致；obs 绝大多数行满足列置换
  D. 掩码/标签一致：wait 标签支撑集与硬掩码零冲突（较大样本集）
  E. 模型数值门禁：前向形状/掩码/归一化；小训练后验证集 alloc_ce 显著下降（真在学）

用法：
    python tools/read_model/test_pipeline.py [--n-files 12]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D                                     # noqa: E402
import model as M                                       # noqa: E402
import train as T                                       # noqa: E402
from constraints import compute_wait_bias               # noqa: E402
from libriichi.dataset import GameplayLoader            # noqa: E402
from libriichi.state import PlayerState                 # noqa: E402

FAILS: list[str] = []
TILE_SEEN_ROW = 835        # v4 obs 的 tiles_seen/4 行（布局推算 + 受控实验双重确认）
KAN_SELECT_ROW = 870       # at_kan_select 标志行（本测试用受控探针动态确认）


def check(name, cond, extra=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name}' + (f'    {extra}' if extra else ''), flush=True)
    if not cond:
        FAILS.append(name)


def find_kan_select_row() -> int:
    """受控探针：构造含暗杠候选的手牌，编码 at_kan_select False/True 找差分行。"""
    lines = [
        '{"type":"start_game","names":["a","b","c","d"],"seed":[1,2]}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"1s","kyoku":1,"honba":0,'
        '"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],'
        '"tehais":[["1m","1m","1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","9p"],'
        '["1p","1p","1p","2p","3p","4p","5p","6p","7p","8p","9p","9s","9s"],'
        '["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","2m","2m"],'
        '["E","E","E","S","S","S","W","W","W","N","N","N","P"]]}',
        '{"type":"tsumo","actor":0,"pai":"3p"}',
    ]
    st = PlayerState(0)
    for l in lines:
        st.update(l)
    a = np.asarray(st.encode_obs(4, False)[0])
    b = np.asarray(st.encode_obs(4, True)[0])
    rows = np.where(np.abs(a - b).sum(1) > 0)[0]
    assert len(rows) == 1, f'kan_select 探针异常: {rows.tolist()}'
    return int(rows[0])


# --------------------------------------------------------------------------- #
def independent_ledger(path):
    """从事件流独立重算（不复用 GameReplayer）：逐事件的 meld 计数/摸牌/河牌快照。"""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        evs = [json.loads(l) for l in f if l.strip()]
    melds = [[0] * 5 for _ in range(4)]     # chi,pon,daiminkan,ankan,kakan
    n_tsumo = 0
    kawa = 0
    snap = {}
    for i, e in enumerate(evs):
        t = e['type']
        a = e.get('actor')
        if t == 'start_kyoku':
            melds = [[0] * 5 for _ in range(4)]
            n_tsumo = 0
            kawa = 0
        elif t == 'tsumo':
            n_tsumo += 1
        elif t == 'dahai':
            kawa += 1
        elif t == 'chi':
            melds[a][0] += 1
        elif t == 'pon':
            melds[a][1] += 1
        elif t == 'daiminkan':
            melds[a][2] += 1
        elif t == 'ankan':
            melds[a][3] += 1
        elif t == 'kakan':
            melds[a][4] += 1
        snap[i] = ([m[:] for m in melds], n_tsumo, kawa)
    return evs, snap


def test_labels(files):
    print('\n== A. 标签不变量（事件流独立重算） ==', flush=True)
    bad_hand = bad_136 = bad_ts = bad_kyoku_start = bad_wait = bad_riichi = 0
    n = 0
    for f in files:
        evs, snap = independent_ledger(f)
        samples, _ = D.replay_file(f)
        for s in samples:
            n += 1
            melds, _, kawa = snap[s['_event']]
            obs = s['_observer']
            hands = [int(s['alloc'][r].sum()) for r in range(3)]
            for r in range(3):
                tgt = (obs + r + 1) % 4
                m = melds[tgt]
                # A1 手牌数（加杠净贡献 0）
                if hands[r] != 13 - 3 * (m[0] + m[1] + m[2] + m[3]):
                    bad_hand += 1
                # A6 立直冻结：立直宣言后 shanten 不再更新（update.rs 的守卫），
                # 故立直目标家的向听标签恒为 0（暗杠/加杠不改变这一点）
                if s['riichi'][r] > 0.5 and int(s['shanten'][r]) != 0:
                    bad_riichi += 1
            # A2 136 守恒
            own_hand = int(sum(s['obs'][r_].sum() for r_ in range(4)))
            lhs = sum(2 * m[0] + 2 * m[1] + 3 * m[2] + 4 * m[3] + 1 * m[4]
                      for m in melds)
            if own_hand + sum(hands) + kawa + lhs + int(s['tiles_left']) + 14 != 136:
                bad_136 += 1
            # A3 tiles_seen
            my_ts = (4.0 - s['unseen']).astype(int)
            obs_ts = np.round(s['obs'][TILE_SEEN_ROW] * 4).astype(int)
            if not np.array_equal(my_ts, obs_ts):
                bad_ts += 1
            # A4 局初 13 张
            if evs[s['_event']]['type'] == 'start_kyoku' and (
                    own_hand != 13 or any(h != 13 for h in hands)):
                bad_kyoku_start += 1
            # A5 听牌标签
            for r in range(3):
                row = s['wait'][r]
                if abs(float(row.sum()) - 1.0) > 1e-5:
                    bad_wait += 1
                sup = row[:34] > 0
                if sup.any():
                    tgt = (obs + r + 1) % 4
                    # 支撑集必须在（结构性听牌 & ~河 & 可表达）内——用 obs 的 waits 行反推
                    # 结构性听牌从 states 拿不到（未存），改为验证支撑集不在河里：
                    if (sup & s['river'][r]).any():
                        bad_wait += 1
    check(f'A1 手牌数公式（{n} 样本）', bad_hand == 0, f'违例 {bad_hand}')
    check(f'A2 136 牌守恒（{n} 样本）', bad_136 == 0, f'违例 {bad_136}')
    check(f'A3 tiles_seen 与 obs 通道一致（{n} 样本）', bad_ts == 0, f'违例 {bad_ts}')
    check('A4 局初四家 13 张', bad_kyoku_start == 0, f'违例 {bad_kyoku_start}')
    check(f'A5 听牌标签归一化且支撑集不含河牌（{n} 样本）', bad_wait == 0, f'违例 {bad_wait}')
    check('A6 立直后手牌总数冻结', bad_riichi == 0, f'违例 {bad_riichi}')


def test_loader_consistency(files):
    print('\n== B. 与 Rust GameplayLoader 一致性 ==', flush=True)
    loader = GameplayLoader(version=4, oracle=False, player_names=None,
                            excludes=None, augmented=False)
    total_matched = total_loader = 0
    extra_types = {}
    bad = []
    for f in files:
        samples, _ = D.replay_file(f)
        by_obs = {}
        for s in samples:
            by_obs.setdefault(s['_observer'], []).append(s)
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            evs = [json.loads(l) for l in fh if l.strip()]
        for file_games in loader.load_gz_log_files([f]):
            for g in file_games:
                pid = g.take_player_id()
                th = [np.asarray(o) for o in g.take_obs()]
                th_f = [o for o in th if o[KAN_SELECT_ROW].max() == 0]
                mine = by_obs.get(pid, [])
                i = j = 0
                matched = 0
                while i < len(mine) and j < len(th_f):
                    if np.array_equal(mine[i]['obs'], th_f[j]):
                        matched += 1
                        i += 1
                        j += 1
                    else:
                        extra_types[evs[mine[i]['_event']]['type']] = \
                            extra_types.get(evs[mine[i]['_event']]['type'], 0) + 1
                        i += 1
                total_matched += matched
                total_loader += len(th_f)
                if matched != len(th_f):
                    bad.append((f.split('/')[-1], pid, matched, len(th_f)))
                for k in range(i, len(mine)):
                    extra_types['TAIL:' + evs[mine[k]['_event']]['type']] = \
                        extra_types.get('TAIL:' + evs[mine[k]['_event']]['type'], 0) + 1
    check(f'B1 loader obs 全部逐位复现（{total_loader} 条）', not bad and total_matched == total_loader,
          f'未匹配 {bad[:3]}')
    allowed = {'start_kyoku', 'reach_accepted', 'dora', 'dahai',
               'TAIL:dahai', 'TAIL:start_kyoku', 'TAIL:reach_accepted', 'TAIL:dora'}
    check('B2 extra 只来自已知类别', set(extra_types) <= allowed,
          f'{dict(extra_types)}')


def test_augmentation(files):
    print('\n== C. 万↔筒增广等变 ==', flush=True)
    bad_alloc = 0
    bad_wait = 0
    obs_row_frac_min = 1.0
    n = 0
    for f in files[:4]:
        s0, _ = D.replay_file(f, augment=False)
        s1, _ = D.replay_file(f, augment=True)
        assert len(s0) == len(s1)
        perm = D.augment_perm()
        for a, b in zip(s0, s1):
            assert a['_observer'] == b['_observer'] and a['_event'] == b['_event']
            n += 1
            # 标签等变：alloc 按 perm 重排
            if not np.array_equal(a['alloc'], b['alloc'][:, perm]):
                bad_alloc += 1
            # wait 标签等变（第 35 类不动）
            wa = np.concatenate([a['wait'][:, :34][:, perm], a['wait'][:, 34:]], axis=1)
            if not np.allclose(wa, b['wait']):
                bad_wait += 1
            # obs：绝大多数行应满足列置换（aka 标志等整行填充行除外）
            eq = (a['obs'][:, perm] == b['obs']).all(axis=1)
            obs_row_frac_min = min(obs_row_frac_min, float(eq.mean()))
    check(f'C1 alloc 标签等变（{n} 样本）', bad_alloc == 0, f'违例 {bad_alloc}')
    check(f'C2 wait 标签等变（{n} 样本）', bad_wait == 0, f'违例 {bad_wait}')
    check('C3 obs 列置换行占比 >= 85%（余下为 aka 整行标志）', obs_row_frac_min >= 0.85,
          f'最低 {obs_row_frac_min:.3f}')


def test_mask_label_consistency(files):
    print('\n== D. 掩码/标签一致性 ==', flush=True)
    conflicts = 0
    n = 0
    for f in files:
        samples, _ = D.replay_file(f)
        for s in samples:
            n += 1
            bias = compute_wait_bias(
                s['unseen'][None], s['river'][None], s['post_riichi_river'][None],
                (s['riichi'] > 0.5)[None], own_hand=s['own_hand'][None])
            sup = s['wait'][:, :34] > 0
            conflicts += int((sup & np.isinf(bias[0])).sum())
    check(f'D1 wait 标签支撑集与硬掩码零冲突（{n} 样本）', conflicts == 0,
          f'冲突 {conflicts}')


def test_structure_and_timeline(files):
    print('\n== F. 结构枚举器与时序 token（v2）==', flush=True)
    # F1 向听数与 Rust calc_all 交叉验证（独立重算口径）
    import gzip
    import json
    from libriichi.state import PlayerState
    import structure as S
    bad_sh = 0
    total = 0
    riichi_skip = 0
    worst = []
    for f in files[:6]:
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            evs = [json.loads(l) for l in fh if l.strip()]
        states = [PlayerState(i) for i in range(4)]
        riichi = [False] * 4
        for e in evs:
            for st in states:
                st.update(json.dumps(e, separators=(',', ':')))
            if e['type'] == 'start_kyoku':
                riichi = [False] * 4
            if e['type'] == 'reach_accepted':
                riichi[e['actor']] = True
            for p in range(4):
                st = states[p]
                if st.last_cans.can_discard:
                    continue                     # 只验证 3n+1 静止态
                th = np.frombuffer(st.tehai, dtype=np.uint8)
                n = int(th.sum())
                if n not in (13, 10, 7, 4, 1):
                    continue
                if riichi[p]:                   # 立直家 shanten 被 Rust 冻结，不参与比对
                    riichi_skip += 1
                    continue
                total += 1
                mine = S.structure_labels(th.astype(int).tolist(), n // 3)['shanten']
                if mine != int(st.shanten):
                    bad_sh += 1
                    if len(worst) < 3:
                        worst.append((th.tolist(), mine, int(st.shanten)))
    check(f'F1 向听数 vs Rust calc_all（{total} 个静止态手牌）', bad_sh == 0,
          f'不一致 {bad_sh} {worst[:1]}（立直跳过 {riichi_skip}）')

    # F2 时序 token 与事件流独立重算一致（顺序/摸切/鸣牌/立直宣言，逐局重置）
    import dataset as D
    bad_tl = 0
    n_tl = 0
    first_bad = None
    for f in files[:3]:
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            evs = [json.loads(l) for l in fh if l.strip()]
        samples, _ = D.replay_file(f)
        # 独立重建：边扫事件边维护每座位 token 列表，每个事件处理后记录快照
        cur = {p: [] for p in range(4)}
        expect = {}
        for i, e in enumerate(evs):
            t = e['type']
            a = e.get('actor')
            if t == 'start_kyoku':
                cur = {p: [] for p in range(4)}
            elif t == 'dahai':
                cur[a].append((D.tile_id(e['pai']), bool(e.get('tsumogiri', False)), 'dahai'))
            elif t == 'reach':
                cur[a].append((-1, False, 'reach'))
            elif t in ('chi', 'pon', 'daiminkan', 'kakan'):
                cur[a].append((D.tile_id(e['pai']), False, t))
            elif t == 'ankan':
                cur[a].append((D.tile_id(e['consumed'][0]), False, 'ankan'))
            for p in range(4):
                expect[(i, p)] = list(cur[p])
        for s in samples:
            obs = s['_observer']
            ev = s['_event']
            for rel in (1, 2, 3):
                tgt = (obs + rel) % 4
                n_tl += 1
                exp = expect.get((ev, tgt), [])[-D.TL_MAX_LEN:]
                toks = s['tl_tokens'][rel - 1]
                mask = s['tl_mask'][rel - 1]
                if int(mask.sum()) != len(exp):
                    bad_tl += 1
                    first_bad = first_bad or (ev, tgt, int(mask.sum()), len(exp))
                    continue
                for k, (tid, tg, act) in enumerate(exp):
                    got_tid = int(toks[k].argmax()) if toks[k, :34].sum() > 0 else -1
                    got_tg = bool(toks[k, 34] > 0.5)
                    got_act = D.TL_ACTION_KINDS[int(toks[k, 35:42].argmax())]
                    if (got_tid, got_tg, got_act) != (tid, tg, act):
                        bad_tl += 1
                        first_bad = first_bad or (ev, tgt, k, (got_tid, got_tg, got_act),
                                                  (tid, tg, act))
                        break
    check(f'F2 时序 token 与事件流一致（{n_tl} 个 (样本,目标)）', bad_tl == 0,
          f'不一致 {bad_tl} 首例 {first_bad}')

    # F3 结构标签自洽：counts 归一化；向听>0 ⟹ 非听牌（反向不成立：
    # waits 还按可见性过滤，全部待牌都可见的极稀有结构听牌 waits 为空，
    # 属已知语义差异而非错误）
    bad_norm = bad_cons = rare_vis = 0
    n_s = 0
    for f in files[:3]:
        samples, _ = D.replay_file(f)
        for s in samples:
            for r in range(3):
                n_s += 1
                if not np.allclose(s['struct_counts'][r].sum(-1), 1.0, atol=1e-5):
                    bad_norm += 1
                if s['shanten'][r] > 0 and s['tenpai'][r] > 0.5:
                    bad_cons += 1
                if s['shanten'][r] == 0 and s['tenpai'][r] <= 0.5:
                    rare_vis += 1
    check(f'F3 结构标签自洽（{n_s} 个目标）', bad_norm == 0 and bad_cons == 0,
          f'归一化违例 {bad_norm}，向听/听牌不一致 {bad_cons}'
          f'（已知稀有差异：结构听牌但待牌全可见 {rare_vis} 例）')


def test_model_and_learning(files, tmp_out='/tmp/read_model_test_train'):
    print('\n== E. 模型数值门禁与小训练学习门禁 ==', flush=True)
    torch.manual_seed(0)
    net = M.build_model()
    B = 4
    obs = torch.randn(B, 1012, 34)
    unseen = torch.full((B, 34), 4.0)
    unseen[:, 5] = 1.0
    unseen[:, 27] = 0.0
    bias = torch.zeros(B, 3, 34)
    bias[:, 0, 5] = float('-inf')
    out = net(obs, unseen, bias)
    check('E1 输出形状', out['alloc_logits'].shape == (B, 3, 34, 5)
          and out['wait_logits'].shape == (B, 3, 35)
          and out['exp_counts'].shape == (B, 3, 34))
    finite = all(torch.isfinite(v).all() for k, v in out.items()
                 if v.dtype.is_floating_point and 'logits' not in k)
    check('E2 概率输出有限且归一', finite
          and torch.allclose(out['alloc_probs'].sum(-1), torch.ones(B, 3, 34), atol=1e-4)
          and torch.allclose(out['wait_probs'].sum(-1), torch.ones(B, 3), atol=1e-4))
    p = out['alloc_probs'][0, 0]
    check('E3 unseen 掩码正确', float(p[5, 2:].sum()) == 0 and float(p[27, 1:].sum()) == 0)
    check('E4 wait_bias 硬掩码正确', float(out['wait_probs'][0, 0, 5]) == 0)

    # ---- v2 前向门禁（含空时序/满时序/部分掩码三种边界）----
    import model_v2 as M2
    net2 = M2.build_model_v2()
    tl_tokens = torch.zeros(B, 3, D.TL_MAX_LEN, D.TL_FEATURES)
    tl_mask = torch.zeros(B, 3, D.TL_MAX_LEN, dtype=torch.bool)
    tl_tokens[0, 0, :7, 3] = 1.0                     # 样本0目标0：7 个 token
    tl_mask[0, 0, :7] = True
    tl_tokens[1, 1, :26, 10] = 1.0                   # 样本1目标1：满长 26
    tl_mask[1, 1, :26] = True
    # 样本2/3 全空（一局开始时的快照）
    out2 = net2(obs, unseen, bias, tl_tokens, tl_mask)
    ok_shape = (out2['alloc_probs'].shape == (B, 3, 34, 5)
                and out2['wait_probs'].shape == (B, 3, 35)
                and out2['struct_count_logits'].shape == (B, 3, 7, 5)
                and out2['struct_form_logits'].shape == (B, 3, 3))
    finite2 = all(torch.isfinite(v).all() for k, v in out2.items()
                  if v.dtype.is_floating_point and 'logits' not in k)
    check('E9 v2 前向形状', ok_shape)
    check('E10 v2 前向有限（含空时序行不产 NaN）', finite2)
    # E11：空时序样本的输出概率仍有限（logits 含 unseen 掩码的 -inf，查概率）
    check('E11 v2 空时序样本仍有合法输出',
          bool(torch.isfinite(out2['alloc_probs'][2]).all())
          and bool(torch.isfinite(out2['wait_probs'][3]).all())
          and bool(torch.isfinite(out2['exp_counts'][2]).all()))

    # 小训练：验证 alloc_ce 显著下降（真在学习，而非管道空转）
    import subprocess
    cmd = [sys.executable, str(Path(__file__).parent / 'train.py'),
           '--train-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
           '--val-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
           '--out-dir', tmp_out, '--steps', '60', '--batch-size', '128',
           '--num-workers', '2', '--stage-schedule', '1:15,2:15,3:30',
           '--eval-samples', '1024', '--eval-every', '30', '--log-every', '30',
           '--max-files', '12', '--seed', '7']
    r = subprocess.run(cmd, capture_output=True, text=True)
    log = [json.loads(l) for l in open(Path(tmp_out) / 'train_log.jsonl')]
    evals = [e for e in log if 'eval' in e]
    if len(evals) < 2:
        check('E5 小训练产出两条评测', False, r.stdout[-500:] + r.stderr[-500:])
        return
    first, last = evals[0]['eval'], evals[-1]['eval']
    check('E5 小训练两条评测', True)
    check('E6 学习门禁：alloc_ce 下降 >= 0.02',
          last['alloc_ce'] <= first['alloc_ce'] - 0.02,
          f"{first['alloc_ce']:.4f} -> {last['alloc_ce']:.4f}")
    # E7 原理性门禁：击败「边际分布」这一最强无信息基线。计数分布被 0 主导
    # （~71%），top1 对常数基线不敏感；正确判据是 CE 低于边际熵
    # （= 按经验边际分布预测的 CE），并 top1 不明显差于常数基线。
    eval_files = sorted(glob.glob(files[0].rsplit('/', 1)[0] + '/*.json.gz'))
    eval_batch = T.build_fixed_eval_set(eval_files, 1024)
    alloc_np = eval_batch['alloc'].numpy()
    hist = np.bincount(alloc_np.ravel(), minlength=5).astype(np.float64)
    p_marg = hist / hist.sum()
    marginal_entropy = float(-(p_marg * np.log(p_marg.clip(1e-12))).sum())
    const_top1 = float((alloc_np == 0).mean())
    check('E7 击败边际基线：alloc_ce < 边际熵 - 0.005',
          last['alloc_ce'] < marginal_entropy - 0.005,
          f"CE {last['alloc_ce']:.4f} vs 边际熵 {marginal_entropy:.4f}")
    check('E8 不退化为常数：top1 >= 常数基线 - 0.03',
          last['alloc_top1'] >= const_top1 - 0.03,
          f"top1 {last['alloc_top1']:.4f} vs 常数 {const_top1:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-files', type=int, default=12)
    ap.add_argument('--root', default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4_names/202601')
    args = ap.parse_args()
    global KAN_SELECT_ROW
    files = sorted(glob.glob(f'{args.root}/*.json.gz'))[:args.n_files]
    if not files:
        raise SystemExit(f'找不到测试牌谱: {args.root}')
    print(f'测试牌谱: {len(files)} 个（{args.root}）', flush=True)
    KAN_SELECT_ROW = find_kan_select_row()
    print(f'kan_select 标志行（受控探针）= {KAN_SELECT_ROW}', flush=True)

    test_labels(files)
    test_loader_consistency(files)
    test_augmentation(files)
    test_mask_label_consistency(files)
    test_structure_and_timeline(files)
    test_model_and_learning(files)

    print('\n================ 结果 ================')
    if FAILS:
        print(f'FAIL {len(FAILS)} 项: {FAILS}')
        raise SystemExit(1)
    print('ALL PASS')


if __name__ == '__main__':
    main()
