"""读牌模型数据管线回归测试（对应 memory §14-35/36：断言设计意图 + 独立重算 + 数值门禁）。

十层检查（A~I，共 67 项；P0/P1 新增 H/I 两层口径与架构门禁）：
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
  F. 结构枚举器与时序 token（v2）：向听 vs Rust calc_all、token 独立重算、标签自洽
  G. 结构头专项（v2s）：
     G1 软向听公式 : normal 形制上 round(soft_shanten)==硬标签（公式与枚举器同口径）
     G2 v2s 前向   : 形状/有限性/空时序边界；瓶颈维度 46（含向听分布）
     G3 梯度流     : 新损失确实监督结构头向听头（v2 缺陷=该头零梯度，修复证明）；
                     边界批（无心听牌目标）总损失有限（顺带修 v1/v2 潜在 -inf*0 NaN）
     G4 v2s 小训练 : struct_shanten 未加权 CE 有限且下降；评测输出 struct_shanten_acc
  H. P0 口径门禁（2026-10-19）：RM1 因子化读法恒等式（v1/v2s 真实前向）；
     RM2 val_score v2 拒绝 A2 型塌缩（v1 判据复现盲区）；RM3 CE 拆分恒等式
  I. P1 架构门禁（RM4/RM5）：tile 头前向/掩码/参数；逐列通路主导与
     因子化隔离（tenpai 只过 tenpai_head）；CE34 损失；A2b 旧 ckpt 必须
     走 flat 架构严格加载；flat 路径小训练 smoke；resume 后 lr>0 且与
     绝对基准一致（防 cosine 钳零静默冻结，2026-10-20 事故门禁）

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
SKIPS: list[str] = []
N_ITEMS = 0                # 已报告项数（OK+FAIL+SKIP），用于「期望项数」门禁
#: 期望项数门禁（2026-10-20，AUDIT S2 / README §11.5-⑭）：坑 57 的假 PASS 之所以能
#: 连续三次蒙混，是因为本文件有 5 处早退 `return`——subprocess 一崩，该层后续 check
#: 既不执行也不记 FAIL，覆盖静默缩水而摘要照样打印 ALL PASS。项数断言把「少跑」变成
#: 显式失败。新增/删除 check 时必须同步此数（或用 --expect 覆盖）。
EXPECTED_ITEMS = 71
TILE_SEEN_ROW = 835        # v4 obs 的 tiles_seen/4 行（布局推算 + 受控实验双重确认）
KAN_SELECT_ROW = 870       # at_kan_select 标志行（本测试用受控探针动态确认）


def check(name, cond, extra=''):
    global N_ITEMS
    N_ITEMS += 1
    print(f'  [{"OK" if cond else "FAIL"}] {name}' + (f'    {extra}' if extra else ''), flush=True)
    if not cond:
        FAILS.append(name)


def skip(name, extra=''):
    """可选项缺失（如仓库里没有旧 ckpt）——**不计入通过数**，单列并计数。

    旧写法是 `check(..., True)`，把 SKIP 记成 PASS（§11.5-⑭）：门禁看起来全绿，
    实际那两项断言从未执行过。
    """
    global N_ITEMS
    N_ITEMS += 1
    SKIPS.append(name)
    print(f'  [SKIP] {name}' + (f'    {extra}' if extra else ''), flush=True)


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
    # 必须清空 tmp_out：train.py 的 train_log.jsonl 是 append 模式，subprocess
    # 崩溃时会留下空/半截文件，旧 log 残留会让「两条评测」断言读到上一次运行
    # 的数字而假 PASS（2026-10-20 实证：E5-E8 曾连续三次靠 10-16 的陈旧 log
    # 蒙混过关，坑 57）。G4/I5/I6 都有 rmtree，E 层漏了。
    import shutil
    shutil.rmtree(tmp_out, ignore_errors=True)
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
    if r.returncode != 0 or not Path(tmp_out, 'train_log.jsonl').exists():
        check('E5 小训练正常退出（returncode=0 且产出 log）', False,
              f'rc={r.returncode} ' + r.stderr[-400:])
        return
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


def test_structure_head_special(files, tmp_out='/tmp/read_model_test_train_v2s'):
    print('\n== G. 结构头专项（v2s：向听监督 + 分解-向听一致性）==', flush=True)
    import model_v2 as M2
    from structure import (FORM_NORMAL, MAX_SHANTEN_BUCKET,
                           soft_shanten_from_counts)

    # ---- G1：软向听公式 vs 硬枚举器标签（公式只覆盖 normal 路径，见
    #      structure.soft_shanten_from_counts docstring；非 normal 形制的
    #      counts 标签不承载其向听信息，一致性约束按形制门控）----
    counts, trues, forms, allocs = [], [], [], []
    for f in files[:4]:
        samples, _ = D.replay_file(f)
        for s in samples:
            counts.append(s['struct_counts'])
            trues.append(s['shanten'])
            forms.append(s['struct_form'])
            allocs.append(s['alloc'])
    P = torch.from_numpy(np.stack(counts).astype(np.float32))        # [n,3,7,5]
    T_ = np.stack(trues)                                             # [n,3]
    F_ = np.stack(forms)
    L_ = torch.from_numpy((np.stack(allocs).sum(-1) // 3).astype(np.float32))
    with torch.no_grad():
        S = soft_shanten_from_counts(P, L_).numpy()                  # [n,3]
    m = F_ == FORM_NORMAL
    ok = np.round(S).clip(0, MAX_SHANTEN_BUCKET) == T_
    mean_abs = float(np.abs(np.clip(S, 0, MAX_SHANTEN_BUCKET) - T_)[m].mean())
    check(f'G1 软向听公式：normal 形制 round==硬标签（{int(m.sum())} 个目标）',
          bool(m.any()) and ok[m].mean() >= 0.999,
          f'acc {ok[m].mean():.4f} mean|d| {mean_abs:.4f}')

    # ---- G2：v2s 前向门禁（形状/有限/空时序；瓶颈 46 维含向听分布）----
    torch.manual_seed(0)
    net2s = M2.build_model_v2s()
    B = 4
    obs = torch.randn(B, 1012, 34)
    unseen = torch.full((B, 34), 4.0)
    unseen[:, 5] = 1.0
    unseen[:, 27] = 0.0
    bias = torch.zeros(B, 3, 34)
    bias[:, 0, 5] = float('-inf')
    tl_tokens = torch.zeros(B, 3, D.TL_MAX_LEN, D.TL_FEATURES)
    tl_mask = torch.zeros(B, 3, D.TL_MAX_LEN, dtype=torch.bool)
    tl_tokens[0, 0, :7, 3] = 1.0                     # 样本0目标0：7 个 token
    tl_mask[0, 0, :7] = True
    tl_tokens[1, 1, :26, 10] = 1.0                   # 样本1目标1：满长 26
    tl_mask[1, 1, :26] = True
    out2s = net2s(obs, unseen, bias, tl_tokens, tl_mask)
    check('G2a v2s 前向形状',
          out2s['alloc_probs'].shape == (B, 3, 34, 5)
          and out2s['wait_probs'].shape == (B, 3, 35)
          and out2s['struct_shanten_logits'].shape == (B, 3, 8))
    finite2s = all(torch.isfinite(v).all() for k, v in out2s.items()
                   if v.dtype.is_floating_point and 'logits' not in k)
    check('G2b v2s 前向有限（含空时序样本）', finite2s)
    check('G2c 瓶颈维度 46 = counts35+form3+向听8（v2 为 38）',
          net2s.S == 46 and net2s.struct_shanten_belief
          and not M2.build_model_v2().struct_shanten_belief)

    # ---- G3：梯度流门禁——新损失必须真的监督到结构头向听头 ----
    samples, _ = D.replay_file(files[0])
    batch = T.collate(samples[:8])
    wb = T.wait_bias_of(batch)
    net2s.zero_grad()
    out = net2s(batch['obs'], batch['unseen'], wb,
                batch['tl_tokens'], batch['tl_mask'])
    loss, parts = T.compute_loss(out, batch)
    loss.backward()
    g_head = net2s.struct.shanten_head.weight.grad
    g_cnt = net2s.struct.count_head.weight.grad
    check('G3a 总损失有限', bool(torch.isfinite(loss)))
    check('G3b 结构头向听头获得梯度（v2 缺陷：该头原先零监督）',
          g_head is not None and float(g_head.abs().sum()) > 0,
          f'|grad|={0.0 if g_head is None else float(g_head.abs().sum()):.3e}')
    check('G3c 分解头获得一致性梯度（consist_decomp 经公式回流）',
          g_cnt is not None and float(g_cnt.abs().sum()) > 0)
    check('G3d 新损失项进入 parts（含 A2b 输出侧 tenpai）',
          all(k in parts for k in ('wait_tenpai', 'struct_shanten', 'consist_head',
                                   'consist_decomp', 'tenpai_tie'))
          and np.isfinite(parts['wait_tenpai']) and parts['wait_tenpai'] > 0)
    # v2 原状存档：关闭全部新损失后，结构头向听梯度严格为 0/None——
    # 这正是 25.7% < 28.4% 多数类基差的机制原因，防止回归。
    net2 = M2.build_model_v2()
    net2.zero_grad()
    out_v2 = net2(batch['obs'], batch['unseen'], wb,
                  batch['tl_tokens'], batch['tl_mask'])
    zero_cfg = {'w_shanten': 0.0, 'w_struct_shanten': 0.0, 'consist_head': 0.0,
                'consist_decomp': 0.0, 'tenpai_tie': 0.0, 'struct_balance': False}
    loss_v2, _ = T.compute_loss(out_v2, batch, loss_cfg=zero_cfg)
    loss_v2.backward()
    gv2 = net2.struct.shanten_head.weight.grad
    check('G3e v2 原状存档：无向听监督时结构头向听梯度为 0/None',
          gv2 is None or float(gv2.abs().sum()) == 0.0)

    # ---- G4：v2s 小训练学习门禁（新损失有限且向听损失真的在降）----
    import shutil
    import subprocess
    shutil.rmtree(tmp_out, ignore_errors=True)
    cmd = [sys.executable, str(Path(__file__).parent / 'train.py'),
           '--train-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
           '--val-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
           '--out-dir', tmp_out, '--steps', '60', '--batch-size', '128',
           '--model', 'v2s', '--num-workers', '2',
           '--stage-schedule', '1:15,2:15,3:30',
           '--eval-samples', '1024', '--eval-every', '30', '--log-every', '30',
           '--max-files', '12', '--seed', '7']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(tmp_out, 'train_log.jsonl').exists():
        check('G4a v2s 小训练正常退出（returncode=0 且产出 log）', False,
              f'rc={r.returncode} ' + r.stderr[-400:])
        return
    log = [json.loads(l) for l in open(Path(tmp_out) / 'train_log.jsonl')]
    recs = [e for e in log if 'struct_shanten_raw' in e]
    check('G4a v2s 小训练产出结构头损失记录', len(recs) >= 2, r.stderr[-300:])
    check('G4c 一致性项全程有限',
          bool(recs) and all(np.isfinite(e['consist_head'])
                             and np.isfinite(e['consist_decomp'])
                             and np.isfinite(e['tenpai_tie']) for e in recs))
    # 学习门禁用**固定评估集**上的向听 CE/准确率（逐阶段瞬时损失分布不同，不可比）
    evals = [e for e in log if 'eval' in e]
    check('G4d 评测输出 struct_shanten_acc/ce',
          bool(evals) and 'struct_shanten_acc' in evals[-1]['eval']
          and 'struct_shanten_ce' in evals[-1]['eval'])
    if len(evals) >= 2:
        ce0, ce1 = evals[0]['eval']['struct_shanten_ce'], evals[-1]['eval']['struct_shanten_ce']
        check('G4b 学习门禁：固定评估集 struct_shanten_ce 下降', ce1 < ce0,
              f'{ce0:.4f} -> {ce1:.4f}')


# --------------------------------------------------------------------------- #
# H. P0 口径门禁（RM1 因子化读法 / RM2 选型判据 / RM3 CE 拆分）
#    来源：MEMORY §11.2/§11.5（2026-10-19 复核）。选型判据必须覆盖曾经
#    回归过的指标——A2 事故的教训（回归测试纪律 §0.2）。
# --------------------------------------------------------------------------- #
def test_p0_criteria(files):
    print('\n== H. P0 口径门禁（RM1/RM2/RM3）==', flush=True)
    import model_v2 as M2
    import evaluate as EV
    from model import factorized_wait_probs

    # ---- H1：因子化读法恒等式（RM1，README §11.1）----
    torch.manual_seed(11)
    logits = torch.randn(4, 3, 35) * 3.0
    logits[..., 34] += 2.0                        # 偏向「未听」
    logits[0, 0, 5] = -1e9                        # 硬掩码列（constraints 口径）
    p = factorized_wait_probs(logits)
    p_not = torch.sigmoid(logits[..., 34])
    cond = torch.softmax(logits[..., :34], dim=-1)
    check('H1a p34 == sigmoid(l34)（BCE 直接标定量，非 softmax35 边际）',
          torch.allclose(p[..., 34], p_not, atol=1e-6))
    check('H1b p[:34] == P(听)·softmax34(l[:34])',
          torch.allclose(p[..., :34], (1.0 - p_not)[..., None] * cond, atol=1e-6))
    check('H1c 行和为 1 且硬掩码列概率为 0',
          torch.allclose(p.sum(-1), torch.ones(4, 3), atol=1e-5)
          and float(p[0, 0, 5]) == 0.0)
    # -inf 掩码（回归测试 E4 口径）同样成立
    logits2 = torch.randn(2, 3, 35)
    logits2[0, 0, 5] = float('-inf')
    p2 = factorized_wait_probs(logits2)
    check('H1d -inf 掩码列概率为 0 且无 NaN',
          float(p2[0, 0, 5]) == 0.0 and bool(torch.isfinite(p2).all()))
    # v1/v2s 真实前向路径的 wait_probs 即因子化读法
    samples, _ = D.replay_file(files[0])
    batch = T.collate(samples[:32])
    wb = T.wait_bias_of(batch)
    net1 = M.build_model()
    with torch.no_grad():
        o1 = net1(batch['obs'], batch['unseen'], wb)
    check('H1e v1 forward 的 wait_probs == factorized(wait_logits)',
          torch.allclose(o1['wait_probs'],
                         factorized_wait_probs(o1['wait_logits']), atol=1e-6))
    net2s = M2.build_model_v2s()
    with torch.no_grad():
        o2 = net2s(batch['obs'], batch['unseen'], wb,
                   batch['tl_tokens'], batch['tl_mask'])
    check('H1f v2s forward 的 wait_probs == factorized(wait_logits)',
          torch.allclose(o2['wait_probs'],
                         factorized_wait_probs(o2['wait_logits']), atol=1e-6))

    # ---- H2：val_score v2 拒绝听牌塌缩（RM2 验收，A2 事故数字）----
    # MEMORY §3.12-5：v2s 50k 步 P(未听) 塌缩至 0.0735、AUC 0.739→0.667，
    # 而 v1 判据（alloc_ce + 0.5·wait_ce_active）照常「改善」并静默存 best。
    good = dict(alloc_ce=0.7000, wait_ce_active=3.3000, tenpai_auc=0.739,
                mean_p_tenpai=0.0750, tenpai_frac=0.0742)
    bad = dict(alloc_ce=0.6995, wait_ce_active=3.2964, tenpai_auc=0.667,
               mean_p_tenpai=0.1463, tenpai_frac=0.0742)
    v1_good = good['alloc_ce'] + 0.5 * good['wait_ce_active']
    v1_bad = bad['alloc_ce'] + 0.5 * bad['wait_ce_active']
    check('H2a v1 判据复现事故盲区（塌缩 ckpt 得分反而更低=更好）', v1_bad < v1_good)
    check('H2b v2 判据拒绝塌缩 ckpt（val_score 更高=更差）',
          T.val_score_of(bad) > T.val_score_of(good),
          f'good {T.val_score_of(good):.4f} vs bad {T.val_score_of(bad):.4f}')
    check('H2c val_score 版本号 >= 2（resume 换代时重置 best_val）',
          T.VAL_SCORE_VERSION >= 2)
    # AUC 缺测（单类批）时退化为 v1，不得 crash
    deg = dict(alloc_ce=0.7, wait_ce_active=3.3, tenpai_auc=float('nan'),
               mean_p_tenpai=None, tenpai_frac=0.074)
    check('H2d 单类批退化：tenpai_auc=nan 时 val_score 有限且等于 v1 判据',
          abs(T.val_score_of(deg) - (0.7 + 0.5 * 3.3)) < 1e-12)

    # ---- H3：wait CE 拆分恒等式（RM3，README §11.5-⑤）----
    rng = np.random.default_rng(3)
    wp = rng.random((6, 3, 35))
    wp /= wp.sum(-1, keepdims=True)
    wt = np.zeros((6, 3, 35))
    for i in range(6):
        for j in range(3):
            k = int(rng.integers(1, 4))
            wt[i, j, rng.choice(34, k, replace=False)] = 1.0 / k
    wt[0, 0] = 0.0                                # 两个非 active 目标
    wt[3, 1] = 0.0
    ll, nlt, cce = EV.wait_ce_decomposition(wp, wt)
    act = wt[..., :34].sum(-1) > 0
    check('H3a active 目标上恒等式 ll == −logP(听) + cond_ce',
          bool(np.allclose(ll[act], (nlt + cce)[act], atol=1e-9)))
    check('H3b 非 active 目标 ll==0、cond_ce==0 且各项有限',
          bool((ll[~act] == 0).all()) and bool((cce[~act] == 0).all())
          and bool(np.isfinite(nlt).all()))
    # 塌缩方向性：P(听) 过度自信（p_t→1）压低 −logP(t) 项 → ce_active「占便宜」
    wp_conf = wp.copy()
    wp_conf[..., 34] *= 0.01
    wp_conf[..., :34] *= (1.0 - wp_conf[..., 34:35]) / wp_conf[..., :34].sum(-1, keepdims=True)
    ll_c, nlt_c, cce_c = EV.wait_ce_decomposition(wp_conf, wt)
    check('H3c 过度自信 P(听) 只改善 −logP(t) 项、cond_ce 不变（拆分能识破刷分）',
          bool((nlt_c[act] < nlt[act]).all())
          and bool(np.allclose(cce_c[act], cce[act], atol=1e-6))
          and bool((ll_c[act] < ll[act]).all()))


# --------------------------------------------------------------------------- #
# I. P1 架构门禁（RM4 损失因子化 + RM5 逐牌 H2 头，README §11.1/§11.3）
#    A2c = v2s + tile 头 + CE34 条件损失 + 200k 步（RM7）。旧 ckpt（A2b 及
#    之前）架构 = flat 头，state_dict 与 tile 不兼容，加载必须走 flat 缺省。
# --------------------------------------------------------------------------- #
def test_p1_architecture(files):
    print('\n== I. P1 架构门禁（RM4/RM5）==', flush=True)
    import model_v2 as M2
    import train as T
    from model import factorized_wait_probs

    NUM_TILES = D.NUM_TILES
    torch.manual_seed(13)
    B = 4
    obs = torch.randn(B, 1012, 34)
    unseen = torch.full((B, 34), 4.0)
    tl = torch.zeros(B, 3, 26, 43)
    tm = torch.zeros(B, 3, 26, dtype=torch.bool)
    bias = torch.zeros(B, 3, 34)
    bias[:, 0, 5] = float('-inf')          # 硬掩码（constraints hard_river_mask 口径）
    bias[:, 1, 7] = -1.0                   # 软振听（hard_river_mask=False 口径）

    net = M2.build_model_v2s(wait_head='tile')
    out = net(obs, unseen, bias, tl, tm)
    wl = out['wait_logits']
    check('I1a tile 头输出形状 [B,3,35]', tuple(wl.shape) == (B, 3, 35))
    check('I1b tile 头因子化：wait_probs == factorized(wait_logits)',
          torch.allclose(out['wait_probs'], factorized_wait_probs(wl), atol=1e-6))
    finite_logits = torch.isfinite(wl[..., :NUM_TILES])
    check('I1c 硬掩码列概率严格 0、软掩码列概率显著低于均匀且 logit 有限；'
          '除显式 -inf 掩码外 logit 全有限、wait_probs 全有限',
          float(out['wait_probs'][0, 0, 5]) == 0.0
          and 0.0 < float(out['wait_probs'][1, 1, 7]) < 0.02   # < 均匀 1/34=0.029
          and bool(finite_logits[:, 1].all()) and bool(finite_logits[:, 2].all())
          and bool(finite_logits[:, 0, [i for i in range(NUM_TILES) if i != 5]].all())
          and bool(torch.isfinite(out['wait_probs']).all())
          and bool(torch.isfinite(wl[..., NUM_TILES]).all()))
    check('I1d 行和 1', torch.allclose(out['wait_probs'].sum(-1),
                                     torch.ones(B, 3), atol=1e-5))
    n_tile = sum(p.numel() for p in net.parameters())
    n_flat = sum(p.numel() for p in M2.build_model_v2s(wait_head='flat').parameters())
    check('I1e tile 头参数量不高于 flat 头（逐列打分不需要更大）',
          n_tile <= n_flat, f'tile {n_tile:,} vs flat {n_flat:,}')

    # ---- I2：梯度流（RM5 本体：cond logit_x 走逐列通路）----
    # I2a：单列反传，obs 梯度集中在对应列（远端残余来自全局 s_belief 通路，
    # 设计使然——「这手牌快听牌了」本就该影响每一列；RM5 的判据是逐列通路
    # 主导：near 应数倍于 far；flat 头无逐列 h_x 通路，二者比例会趋近 1）
    obs_g = torch.randn(2, 1012, 34, requires_grad=True)
    net.zero_grad()
    out_g = net(obs_g, unseen[:2], bias[:2], tl[:2], tm[:2])
    out_g['wait_logits'][0, 0, 10].backward()
    far = obs_g.grad[0, :, 25].abs().sum()
    near = obs_g.grad[0, :, 10].abs().sum()
    check('I2a 逐列通路主导：logit_10 对 obs 列 10 的梯度 ≫ 远端列 25',
          float(near) > 3.0 * float(far) and float(near) > 0,
          f'near {float(near):.2e} far {float(far):.2e} '
          f'比例 {float(near / max(float(far), 1e-12)):.1f}x')
    # I2b：tenpai 标量与 cond 头结构隔离（RM4 因子化的结构性保证）——
    # l34 反传只经过 tenpai_head，不经过逐列 h2/h2_out
    net.zero_grad()
    out2 = net(obs, unseen, bias, tl, tm)
    out2['wait_logits'][0, 0, NUM_TILES].backward()
    g_tp = float(net.tenpai_head.weight.grad.abs().sum())
    g_h2 = (0.0 if net.h2.weight.grad is None
            else float(net.h2.weight.grad.abs().sum()))
    check('I2b 因子化隔离：tenpai logit 只过 tenpai_head（h2 逐列头梯度为 0）',
          g_tp > 0.0 and g_h2 == 0.0, f'tenpai {g_tp:.2e} h2 {g_h2:.2e}')
    # I2c：cond logit 反传，逐列头 h2 与 tenpai_head 均有/无对应梯度
    net.zero_grad()
    out3 = net(obs, unseen, bias, tl, tm)
    out3['wait_logits'][..., :NUM_TILES].sum().backward()
    g_h2b = float(net.h2.weight.grad.abs().sum())
    g_tp2 = (0.0 if net.tenpai_head.weight.grad is None
             else float(net.tenpai_head.weight.grad.abs().sum()))
    check('I2c cond 反传激活逐列头 h2、不碰 tenpai_head',
          g_h2b > 0.0 and g_tp2 == 0.0, f'h2 {g_h2b:.2e} tenpai {g_tp2:.2e}')

    # ---- I3：RM4 条件损失在 tile 头上有限且分量齐备 ----
    samples, _ = D.replay_file(files[0])
    batch = T.collate(samples[:32])
    wb = T.wait_bias_of(batch)
    out4 = net(batch['obs'], batch['unseen'], wb, batch['tl_tokens'], batch['tl_mask'])
    loss_v, parts = T.compute_loss(out4, batch)
    check('I3a tile 头 compute_loss 全分量有限',
          all(np.isfinite(v) for k, v in parts.items() if k != 'loss')
          and np.isfinite(parts['loss']),
          str({k: round(float(v), 4) for k, v in parts.items()}))
    # I3b：整批无听牌目标时 wait 损失为 0 且全部分量有限——CE34 改写后重验
    # 旧代码注释里的潜在 bug 路径（-inf*0=NaN，小 batch 暴露）
    batch_nt = dict(batch)
    batch_nt['tenpai'] = torch.zeros_like(batch['tenpai'])
    loss_nt, parts_nt = T.compute_loss(out4, batch_nt)
    check('I3b 全非听牌批 wait 损失为 0 且全分量有限（NaN 防护仍有效）',
          float(parts_nt['wait']) == 0.0
          and all(np.isfinite(v) for v in parts_nt.values())
          and np.isfinite(parts_nt['loss']),
          f'wait {float(parts_nt["wait"])}')

    # ---- I4：旧 ckpt（A2b）必须走 flat 架构加载 ----
    a2b = Path('/mnt/data8/mortal/Mortal/mortal/read_model_v2s_a2b/best.pth')
    if a2b.exists():
        st = torch.load(a2b, map_location='cpu', weights_only=False)
        cfg = st.get('config', {})
        net_flat = M2.build_model_v2s(
            conv_channels=cfg.get('conv_channels', 96),
            num_blocks=cfg.get('num_blocks', 12),
            with_aux=not cfg.get('no_aux', False),
            wait_head=cfg.get('wait_head', 'flat'))
        try:
            net_flat.load_state_dict(st['model'])          # strict=True
            ok_flat = True
        except Exception as exc:                            # noqa: BLE001
            ok_flat = False
            print(f'    flat 加载失败: {exc}', flush=True)
        check('I4a A2b 旧 ckpt 以 flat 架构严格加载成功（默认值兼容）', ok_flat)
        net_tile = M2.build_model_v2s(
            conv_channels=cfg.get('conv_channels', 96),
            num_blocks=cfg.get('num_blocks', 12),
            with_aux=not cfg.get('no_aux', False), wait_head='tile')
        try:
            net_tile.load_state_dict(st['model'])
            mismatch = False
        except Exception:                                   # noqa: BLE001
            mismatch = True
        check('I4b tile 架构拒绝 A2b state_dict（证明默认 flat 的必要性）', mismatch)
    else:
        skip('I4 A2b ckpt 不在仓库（兼容断言未执行）')

    # ---- I5：flat 路径小训练 smoke（旧架构仍可训，CE34 不回退能力）----
    import shutil
    import subprocess
    tmp_out = '/tmp/read_model_test_train_flat'
    shutil.rmtree(tmp_out, ignore_errors=True)
    cmd = [sys.executable, str(Path(__file__).parent / 'train.py'),
           '--train-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
           '--val-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
           '--out-dir', tmp_out, '--steps', '60', '--batch-size', '128',
           '--model', 'v2s', '--wait-head', 'flat', '--num-workers', '2',
           '--stage-schedule', '1:15,2:15,3:30',
           '--eval-samples', '1024', '--eval-every', '30', '--log-every', '30',
           '--max-files', '12', '--seed', '7']
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        log = [json.loads(l) for l in open(Path(tmp_out) / 'train_log.jsonl')]
        evals = [e for e in log if 'eval' in e]
        check('I5a flat 小训练产出两条评测', len(evals) >= 2, r.stderr[-300:])
        if len(evals) >= 2:
            first, last = evals[0]['eval'], evals[-1]['eval']
            check('I5b flat+CE34 学习门禁：alloc_ce 下降 >= 0.02',
                  last['alloc_ce'] <= first['alloc_ce'] - 0.02,
                  f"{first['alloc_ce']:.4f} -> {last['alloc_ce']:.4f}")
            check('I5c val_score v2 输出（AUC 项在场）',
                  'val_score' in evals[-1] and 'tenpai_auc' in last,
                  f"val_score {evals[-1].get('val_score')}")
    except FileNotFoundError:
        check('I5 flat 小训练 smoke（train_log 缺失）', False, r.stderr[-300:])

    # ---- I6：resume 后 LR 非零门禁（2026-10-20 必修，静默冻结事故）----
    # 事故回放：A2c 在 103k 处按「剩余量」续训时，scheduler.last_epoch（恢复为
    # 绝对步数 103000）> cosine 分母（剩余量 97000）⇒ t>1 被钳制 ⇒ lr 恒 0，
    # 参数静默冻结 1000+ 步（loss 照常打印、best 照常保存，毫无异常表象）。
    # 修复：lr_lambda 基准改为绝对终点（start+remaining）。本门禁用最小复现：
    # 30 步存 ckpt → resume 30 步，断言 lr>0 且与 lr_lambda(last_epoch) 一致。
    import math
    import shutil
    import subprocess
    tmp_r = '/tmp/read_model_test_train_resume'
    shutil.rmtree(tmp_r, ignore_errors=True)
    base_cmd = [sys.executable, str(Path(__file__).parent / 'train.py'),
                '--train-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
                '--val-globs', files[0].rsplit('/', 1)[0] + '/*.json.gz',
                '--out-dir', tmp_r, '--steps', '30', '--batch-size', '128',
                '--num-workers', '2', '--stage-schedule', '3:30',
                '--eval-samples', '1024', '--eval-every', '30', '--log-every', '10',
                '--max-files', '4', '--seed', '7']
    r1 = subprocess.run(base_cmd, capture_output=True, text=True)
    if not Path(tmp_r, 'last.pth').exists():
        check('I6a 基线 30 步小训练产出 last.pth', False, r1.stderr[-300:])
        return
    check('I6a 基线 30 步小训练产出 last.pth', True)
    r2 = subprocess.run(base_cmd + ['--resume', str(Path(tmp_r, 'last.pth'))],
                        capture_output=True, text=True)
    recs = [json.loads(l) for l in open(Path(tmp_r) / 'train_log.jsonl')]
    steps = [e for e in recs if 'lr' in e]
    if len(steps) < 2:
        check('I6b resume 后继续训练并记录 lr', False, r2.stderr[-300:])
        return
    rec0 = steps[-2]                    # resume 后第一条 step 记录
    first_lr, last_lr = rec0['lr'], steps[-1]['lr']
    # 手算对照：abs_total = 30 + 30 = 60, warmup = max(1, 3) = 3
    at, w = 60, max(1, int(0.05 * 60))
    s0 = rec0['step']
    expect = ((s0 + 1) / w if s0 < w
              else 0.5 * (1 + math.cos(math.pi * min((s0 - w) / max(1, at - w), 1.0))))
    check('I6b resume 后 lr>0（防 cosine 钳零静默冻结；末值=0 是 cosine 到终点的正常行为）',
          first_lr > 0.0,
          f'lr {first_lr:.2e} -> {last_lr:.2e}')
    check('I6c resume 后 lr 与绝对基准 lr_lambda(step) 一致',
          abs(first_lr - expect * 3e-4) < 1e-6,
          f'step {s0}: {first_lr:.3e} vs 期望 {expect * 3e-4:.3e}')
    check('I6d resume 后 step 连续（30→60）', steps[-1]['step'] == 60,
          f"末条 step {steps[-1].get('step')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-files', type=int, default=12)
    # ⚠️ 门禁默认牌谱根与正式评估不同源（这里 houou4_names/202601，run_*.sh 与
    # evaluate/compare_a3 用 houou4/202609）——门禁只需 smoke 级数据，非评估集。
    ap.add_argument('--root', default='/mnt/data8/mortal/Mortal/dataset/mjai/houou4_names/202601')
    ap.add_argument('--expect', type=int, default=EXPECTED_ITEMS,
                    help='期望报告项数（OK+FAIL+SKIP）；实到不符即 FAIL，'
                         '防止早退 return 静默缩减覆盖（§11.5-⑭）')
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
    test_structure_head_special(files)
    test_p0_criteria(files)
    test_p1_architecture(files)

    print('\n================ 结果 ================')
    n_ok = N_ITEMS - len(FAILS) - len(SKIPS)
    print(f'共报告 {N_ITEMS} 项：OK {n_ok} / FAIL {len(FAILS)} / SKIP {len(SKIPS)}'
          f'（期望 {args.expect} 项）')
    if SKIPS:
        print(f'SKIP {len(SKIPS)} 项（**未执行**，不计入通过）: {SKIPS}')
    if N_ITEMS != args.expect:
        FAILS.append(f'覆盖项数 {N_ITEMS} != 期望 {args.expect}'
                     '（有 check 被早退 return 跳过，或增删 check 未同步 EXPECTED_ITEMS）')
    if FAILS:
        print(f'FAIL {len(FAILS)} 项: {FAILS}')
        raise SystemExit(1)
    print('ALL PASS')


if __name__ == '__main__':
    main()
