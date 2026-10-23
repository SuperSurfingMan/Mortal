#!/usr/bin/env python3
"""S6c 特征回归（脚本式，无 pytest）：装进去之前先证明「口径没跑偏」。

覆盖：
  1. 向听表实现 vs libriichi 单测 19 例（含 `len_div3` 副露口径与 −1 和了语义）；
  2. 真实牌谱上 `obs[0:4]` = 自家手牌、`obs[835]` = tiles_seen（`seen >= own`）；
  3. 3n+1 状态的向听与 obs 第 862..868 行 one-hot **逐位相等**；3n+2 失配率落在
     历史区间（obs 在打牌后才更新 → 自摸后状态是摸牌前的值）；
  4. `compute` 形状/范围/确定性/单样本路径；`attach` 不改前 1012 行、新行常量；
  5. `resolve` 的合法/非法组合。

用法（必须在 mortal/ 里跑，libriichi.so 在那儿）：
    cd /mnt/data8/mortal/Mortal/mortal && python ../tools/test_s6c_features.py
    python ../tools/test_s6c_features.py --files 6 --month 201202
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

MORTAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/mortal'
sys.path.insert(0, MORTAL)
os.chdir(MORTAL)

import s6c_features as S  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name:<46s} {detail}')
    if not cond:
        FAILS.append(name)


def hand(s):
    c = np.zeros(34, dtype=np.int64)
    cur = []
    for tok in s.split():
        for ch in tok:
            if ch in 'mpsz':
                base = {'m': 0, 'p': 9, 's': 18, 'z': 27}[ch]
                for d in cur:
                    c[base + int(d) - 1] += 1
                cur = []
            else:
                cur.append(ch)
    return c


# libriichi/src/algo/shanten.rs 的两个单测函数，逐条照抄
RUST_CASES = [
    ('1111m 333p 222s 444z', 4, 1), ('147m 258p 369s 1234z', 4, 6),
    ('468m 33346p 7s', 3, 2), ('147m 258p 3s', 2, 4), ('4455s', 1, 0),
    ('7z', 0, 0), ('15559m 19p 19s 1234z', 4, 3), ('9999m 6677p 88s 355z', 4, 2),
    ('19m 19p 159s 123456z', 4, 1),
    ('2344456m 14p 127s 2z 7p', 4, 3), ('2344456m 14p 127s 2z 5p', 4, 2),
    ('344455667p 1139s 9m', 4, 2), ('344455667p 1139s 9p', 4, 1),
    ('122334m 678p 37s 22z 5s', 4, 0), ('122334m 678p 12s 22z 4s', 4, 0),
    ('12223456m 78889p 2m', 4, -1),
    ('34778p', 1, 0), ('34s', 0, 0), ('55m', 0, -1),
]


def block_index(obs):
    """obs[862:869] 是否为合法 one-hot 块（向听 one-hot）→ 槽号或 None。"""
    seg = obs[862:869]
    one = seg.min(1) == 1.0
    zero = seg.max(1) == 0.0
    if one.sum() == 1 and zero.sum() == 6:
        return int(np.where(one)[0][0])
    return None


def test_tables():
    print('== 1. 向听表 vs libriichi 单测 ==')
    bad = []
    for s, L, want in RUST_CASES:
        got = int(S.shanten_all(hand(s)[None], L)[0])
        if got != want:
            bad.append((s, L, want, got))
    check('Rust 单测 19 例逐位一致', not bad, f'{(len(RUST_CASES) - len(bad))}/{len(RUST_CASES)}')
    for b in bad:
        print('       mismatch:', b)

    # 越界索引走 .unwrap_or_default（与 Rust 一致）：不可能的高张数不应崩
    weird = np.full((1, 34), 4, dtype=np.int64)
    try:
        v = int(S.shanten_all(weird, 4)[0])
        check('超大张数不崩（表越界取默认）', True, f'shanten={v}')
    except Exception as ex:                                    # noqa: BLE001
        check('超大张数不崩（表越界取默认）', False, repr(ex))


def load_obs(files):
    from libriichi.dataset import GameplayLoader
    loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None,
                            augmented=False, always_include_kan_select=True)
    out = []
    for file in loader.load_gz_log_files(files):
        for game in file:
            out.extend(np.asarray(o, np.float32) for o in game.take_obs())
    return np.stack(out) if out else np.zeros((0, 1012, 34), np.float32)


def test_real(obs):
    print(f'== 2~3. 真实牌谱（{len(obs):,} 决策点） ==')
    counts = np.rint(obs[:, S.OWN_HAND_ROWS, :].sum(1)).astype(np.int64)
    n = counts.sum(1)
    seen = np.rint(obs[:, S.TILES_SEEN_ROW, :] * S.NUM_COPIES).astype(np.int64)
    check('obs[0:4] 是整数手牌计数', bool(np.all(np.abs(obs[:, S.OWN_HAND_ROWS, :].sum(1)
                                                     - counts).max() < 1e-6)))
    check('手牌张数属于 3n+1/3n+2', bool(np.isin(n % 3, (1, 2)).all()),
          f'sizes={sorted(set(n.tolist()))}')
    check('tiles_seen 行可信（seen >= 自家手牌）', bool((seen >= counts).all()),
          f'violations={int((seen < counts).any(1).sum())}')

    pairs = [(int(nn), int(S.shanten_all(counts[i:i + 1], int(nn // 3))[0]),
              block_index(obs[i])) for i, nn in enumerate(n)]
    ok1 = [(m, r) for nn, m, r in pairs if nn % 3 == 1 and r is not None]
    two = [(m, r) for nn, m, r in pairs if nn % 3 == 2 and r is not None]
    n_ok1 = sum(max(m, 0) == r for m, r in ok1)
    n_two = sum(max(m, 0) == r for m, r in two)
    check('3n+1 向听 == obs 862 行 one-hot', n_ok1 == len(ok1), f'{n_ok1}/{len(ok1)}')
    rate = n_two / max(len(two), 1)
    check('3n+2 失配率落在历史区间（obs 滞后一抽）', 0.60 <= rate <= 0.90,
          f'{n_two}/{len(two)} = {rate:.4f}（历史 0.757）')
    # 摸牌只能保持或推进向听 → obs（摸牌前）≥ 真值（摸牌后），且一般只差 1
    deltas = [r - max(m, 0) for m, r in two]
    check('3n+2 失配方向恒为 obs ≥ 真值', all(d >= 0 for d in deltas),
          f'max delta={max(deltas) if deltas else 0}')
    check('3n+2 滞后量 ≤ 2', all(d <= 2 for d in deltas))


def test_compute_attach(obs):
    print('== 4. compute / attach ==')
    t0 = time.time()
    feat = S.compute(obs)
    dt = time.time() - t0
    check('compute 形状 (N,2) float32',
          feat.shape == (len(obs), S.NUM_EXTRA_CHANNELS) and feat.dtype == np.float32)
    check('取值范围 [0,1]', bool((feat >= 0).all() and (feat <= 1).all()),
          f'ch0∈[{feat[:, 0].min():.3f},{feat[:, 0].max():.3f}] '
          f'ch1∈[{feat[:, 1].min():.3f},{feat[:, 1].max():.3f}]')
    check('确定性', bool(np.array_equal(feat, S.compute(obs))))
    one = S.compute(obs[0])
    check('单样本路径与批量一致', bool(np.array_equal(one[None], feat[:1])),
          f'shapes {one.shape} / {feat[:1].shape}')

    attached = S.attach(obs, feat)
    check('attach 形状 (N,1014,34)',
          attached.shape == (len(obs), 1012 + S.NUM_EXTRA_CHANNELS, 34)
          and attached.dtype == np.float32)
    check('attach 不改前 1012 行', bool(np.array_equal(attached[:, :1012], obs)))
    rows_ok = all(np.all(attached[i, 1012 + k] == feat[i, k]) for i in range(min(len(obs), 64))
                  for k in range(S.NUM_EXTRA_CHANNELS))
    check('新行 = 该样本特征（34 列同值）', rows_ok)
    check('attach 单样本形状', S.attach(obs[0], feat[:1]).shape == (1014, 34),
          str(S.attach(obs[0], feat[:1]).shape))
    check('有效牌占比中位数合理（1~45 枚）',
          bool(2.0 <= np.median(feat[:, 1] * S._KAWA_DIV) <= 45.0),
          f'median={np.median(feat[:, 1] * S._KAWA_DIV):.1f} 枚, '
          f'mean={feat[:, 1].mean() * S._KAWA_DIV:.1f} 枚')
    print(f'  [info] compute: {dt * 1e3:.1f} ms / {len(obs):,} = '
          f'{dt / max(len(obs), 1) * 1e3:.3f} ms/sample')


def test_resolve():
    print('== 5. resolve（配置守卫） ==')
    check('空配置 → 关闭', S.resolve({}) == ('', 0))
    check('s6c + 2 通道', S.resolve({'extra_features': 's6c', 'extra_channels': 2})
          == (S.SPEC, S.NUM_EXTRA_CHANNELS))
    check('s6c 不写通道数也认', S.resolve({'extra_features': 's6c'})
          == (S.SPEC, S.NUM_EXTRA_CHANNELS))
    check('s6c + 错误通道数报错', _raises(lambda: S.resolve(
        {'extra_features': 's6c', 'extra_channels': 3})))
    check('有通道数但无特征族报错（旧 S6b ckpt）',
          _raises(lambda: S.resolve({'extra_channels': 2})))


def _raises(fn):
    try:
        fn()
    except Exception:                                          # noqa: BLE001
        return True
    return False


def test_missing_tables():
    print('== 6. 向听表缺失时的错误信息 ==')
    saved_env = os.environ.get('MORTAL_S6C_DATA_DIR')
    saved_tables = S._TABLES
    os.environ['MORTAL_S6C_DATA_DIR'] = '/nonexistent-s6c'
    S._TABLES = None
    try:
        S.tables()
        check('缺表时报错', False)
    except FileNotFoundError as ex:
        check('缺表时报错且给出修法', 'MORTAL_S6C_DATA_DIR' in str(ex))
    finally:
        if saved_env is None:
            os.environ.pop('MORTAL_S6C_DATA_DIR', None)
        else:
            os.environ['MORTAL_S6C_DATA_DIR'] = saved_env
        S._TABLES = saved_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default='2012')
    ap.add_argument('--files', type=int, default=4)
    args = ap.parse_args()
    files = sorted(glob.glob(f'/mnt/data8/mortal/Mortal/dataset/mjai/houou4/{args.month}*/*.json.gz'))
    files = files[:args.files]
    if not files:
        print(f'没有牌谱文件：{args.month}', file=sys.stderr)
        return 2

    test_tables()
    obs = load_obs(files)
    if len(obs) == 0:
        print('没有决策点', file=sys.stderr)
        return 2
    test_real(obs)
    test_compute_attach(obs)
    test_resolve()
    test_missing_tables()

    print()
    if FAILS:
        print(f'FAIL: {len(FAILS)} 项未过 → {FAILS}')
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
