#!/usr/bin/env python3
"""S6a 回归测试：version-5「逆转紧迫度」obs 特征 + ckpt 手术。

分层验证（对应 MEMORY §14-35 的教训：断言验设计意图，不复用被测实现的输出）：

  A. 常量：MAX_VERSION>=5；obs_shape(4) 不变 (1012,34)；obs_shape(5)==(1018,34)。
  B. v5 ⊃ v4：同一批牌谱，v5 obs 的前 1012 行与 v4 **逐位一致**（旧通道无回归）；
     新增 6 行值域 [0,1]；全部 obs 形状 (1018,34)。
  C. 独立复核：只从 v4 obs 自带的分数通道（行 7..14，/100k 序）解回 4 家当前分数，
     用纯 Python 重算 6 个特征，与 v5 obs 尾部逐样本比较（跳过被 clamp 的分数）。
  D. 语义不变量：点差剖面 d1<=d2<=d3<=d4、d1<=0<=d4、至少一个 d≈0、
     overtake/lead >=0 且与 d 剖面一致。
  E. ckpt 手术：v4 -> v5 后非首层参数全等、新增通道全零；
     新通道输入为 0 时 v5 前向与 v4 逐位等价（等价于「原模型 + 新特征从零学起」）。
  F. 端到端前向：v5 Brain+DQN 对真实 v5 obs 批前向，输出形状 [B,46] 且全部有限。

用法（任意目录均可；脚本自己 chdir 到 mortal/）：
    python tools/training/test_urgency_features.py
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MORTAL = Path(os.environ.get('S6A_CODE', str(REPO / 'mortal')))
os.chdir(MORTAL)
os.environ.setdefault('MORTAL_CFG', 'config.round4.toml')
sys.path.insert(0, str(MORTAL))

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
from libriichi.consts import MAX_VERSION, obs_shape       # noqa: E402
from libriichi.dataset import GameplayLoader              # noqa: E402
from model import Brain, DQN                              # noqa: E402

V4, V5 = 4, 5
OLD_IN, NEW_IN = 1012, 1018
KL = 6                                                     # 新增特征行数
SCORE_ROW0 = 7                                             # v4 分数通道起始行（4 家 /100k 序）

FILES = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4_names/202601/*.json.gz')))[:3]
if not FILES:
    FILES = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4/202209/*.json.gz')))[:3]
if not FILES:
    raise SystemExit('找不到测试牌谱（houou4_names/202601 或 houou4/202209）')

PARENT = MORTAL / 'checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth'

fails: list[str] = []


def check(name, cond, extra=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name}' + (f'    {extra}' if extra else ''))
    if not cond:
        fails.append(name)


def enc(d: float) -> float:
    """Rust `encode_urgency` 的归一化（与实现同式的独立 Python 版本）。"""
    return min(max(d, -100_000.0), 100_000.0) / 200_000.0 + 0.5


def expected_features(scores_rot: list[float]) -> list[float]:
    """scores_rot[0] = 自家（相对序）。返回 6 个特征值。"""
    me = scores_rot[0]
    allsc = sorted(scores_rot, reverse=True)
    feats = [enc(me - s) for s in allsc]
    above = [s for s in allsc if s > me]
    below = [s for s in allsc if s < me]
    feats.append(enc((min(above) - me) if above else 0.0))
    feats.append(enc(me - (max(below) if below else me)))
    return feats


print('== A) 常量 ==')
check('MAX_VERSION >= 5', MAX_VERSION >= 5, f'MAX_VERSION={MAX_VERSION}')
check('obs_shape(4) 未变', obs_shape(4) == (OLD_IN, 34), f'{obs_shape(4)}')
check('obs_shape(5) == (1018,34)', obs_shape(5) == (NEW_IN, 34), f'{obs_shape(5)}')

print('\n== B) v5 ⊃ v4（逐位一致）+ 值域 ==')
# take_obs() 是消费型接口：每局只取一次，后续所有分层复用
loader4 = GameplayLoader(version=V4, oracle=False, player_names=None)
loader5 = GameplayLoader(version=V5, oracle=False, player_names=None)
data4 = loader4.load_gz_log_files(FILES)
data5 = loader5.load_gz_log_files(FILES)
games4 = [g.take_obs() for f in data4 for g in f]
games5 = [g.take_obs() for f in data5 for g in f]

n_obs = 0
shape_ok = prefix_ok = range_ok = True
for o4s, o5s in zip(games4, games5):
    if len(o4s) != len(o5s):
        shape_ok = False
        continue
    for a, b in zip(o4s, o5s):
        n_obs += 1
        if b.shape != (NEW_IN, 34) or a.shape != (OLD_IN, 34):
            shape_ok = False
        if not np.array_equal(a, b[:OLD_IN]):
            prefix_ok = False
        u = b[OLD_IN:, 0]
        if not ((u >= 0.0) & (u <= 1.0)).all():
            range_ok = False
check('obs 形状 (1012,34) / (1018,34)', shape_ok, f'{n_obs} 条 obs')
check('v5 前 1012 行与 v4 逐位一致（旧通道无回归）', prefix_ok)
check('新增 6 行值域 [0,1]', range_ok)

print('\n== C) 独立复核：从 v4 分数通道重算特征 ==')
n_checked = n_skip = 0
max_err = 0.0
err_examples = []
for o4s, o5s in zip(games4, games5):
    for a, b in zip(o4s, o5s):
        # 行 7/9/11/13 = 4 家 score.clamp(0,1e5)/1e5（相对序，[0]=自家）
        v = a[SCORE_ROW0:SCORE_ROW0 + 8:2, 0].astype(np.float64)
        if (v <= 0.0).any() or (v >= 1.0).any():
            n_skip += 1          # 分数被 clamp（0 或 >=1e5），无法精确反解
            continue
        scores = (v * 100_000.0).tolist()
        exp = np.array(expected_features(scores))
        got = b[OLD_IN:, 0].astype(np.float64)
        err = float(np.abs(exp - got).max())
        max_err = max(max_err, err)
        n_checked += 1
        if err > 1e-5 and len(err_examples) < 3:
            err_examples.append((scores, exp, got, err))
check('独立重算与 v5 obs 一致（tol 1e-5）', n_checked > 0 and max_err <= 1e-5,
      f'复核 {n_checked} / clamp 跳过 {n_skip} / 最大误差 {max_err:.2e}')
for s, e, g, err in err_examples:
    print(f'        scores={s} exp={np.round(e, 5)} got={np.round(g, 5)} err={err:.2e}')

print('\n== D) 语义不变量 ==')
inv_ok = True
for o5s in games5:
    for b in o5s:
        u = b[OLD_IN:, 0].astype(np.float64)
        d = (u[:4] - 0.5) * 200_000.0           # 反归一化回点差
        overtake, lead = (u[4] - 0.5) * 200_000.0, (u[5] - 0.5) * 200_000.0
        if not (d[0] <= d[1] + 1e-2 <= d[2] + 2e-2 and d[1] <= d[2] + 1e-2 <= d[3] + 2e-2):
            inv_ok = False
        if not (d[0] <= 1.0 and d[3] >= -1.0):   # d1<=0<=d4（±1 分）
            inv_ok = False
        if not (np.abs(d).min() <= 1.0):         # 自家分数必在剖面里 → 某个 d≈0
            inv_ok = False
        want_overtake = -max(d[d < -1.0]) if (d < -1.0).any() else 0.0
        want_lead = min(d[d > 1.0]) if (d > 1.0).any() else 0.0
        if abs(overtake - want_overtake) > 1.0 or abs(lead - want_lead) > 1.0:
            inv_ok = False
check('d1<=d2<=d3<=d4、d1<=0<=d4、某 d≈0、overtake/lead 与剖面一致', inv_ok)

print('\n== E) ckpt 手术（v4 -> v5）==')
if not PARENT.exists():
    check('父代 ckpt 存在（跳过）', True, f'缺少 {PARENT}（非失败）')
else:
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / 'mortal_v5.pth'
        r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'expand_ckpt_v5.py'),
                            str(PARENT), str(dst)], capture_output=True, text=True)
        check('expand_ckpt_v5.py 运行成功', r.returncode == 0,
              (r.stdout.strip().splitlines() or [''])[-1] if r.returncode == 0 else r.stderr[-200:])
        if r.returncode == 0:
            st4 = torch.load(PARENT, map_location='cpu', weights_only=False)
            st5 = torch.load(dst, map_location='cpu', weights_only=False)
            check('内嵌 config.version 已改 5', st5['config']['control']['version'] == 5)
            w4, w5 = st4['mortal']['encoder.net.0.weight'], st5['mortal']['encoder.net.0.weight']
            check('首层形状 1012->1018',
                  tuple(w4.shape)[1] == OLD_IN and tuple(w5.shape)[1] == NEW_IN,
                  f'{tuple(w4.shape)} -> {tuple(w5.shape)}')
            check('新增通道全零', bool((w5[:, OLD_IN:, :] == 0).all()))
            check('旧通道权重原样', bool(torch.equal(w4, w5[:, :OLD_IN, :])))
            same_other = all(torch.equal(t, st5['mortal'][k]) for k, t in st4['mortal'].items()
                             if k != 'encoder.net.0.weight')
            check('其余 mortal 参数全等', same_other)
            check('current_dqn 全等',
                  all(torch.equal(t, st5['current_dqn'][k]) for k, t in st4['current_dqn'].items()))

            cfg4 = st4['config']
            b4 = Brain(version=V4, **cfg4['resnet']).eval()
            d4_ = DQN(version=V4).eval()
            b4.load_state_dict(st4['mortal'])
            d4_.load_state_dict(st4['current_dqn'])
            b5 = Brain(version=V5, **cfg4['resnet']).eval()
            d5_ = DQN(version=V5).eval()
            b5.load_state_dict(st5['mortal'])
            d5_.load_state_dict(st5['current_dqn'])

            torch.manual_seed(0)
            x4 = torch.randn(4, OLD_IN, 34)
            x5 = torch.cat([x4, torch.zeros(4, NEW_IN - OLD_IN, 34)], dim=1)
            mask = torch.ones(4, 46, dtype=torch.bool)
            with torch.no_grad():
                q4 = d4_(b4(x4), mask)
                q5 = d5_(b5(x5), mask)
            check('新通道输入为 0 时 v5 前向与 v4 逐位等价（tol 1e-5）',
                  bool(torch.allclose(q4, q5, atol=1e-5, rtol=0)),
                  f'max|dq|={float((q4 - q5).abs().max()):.2e}')

print('\n== F) 端到端前向（真实 v5 obs）==')
obs = [b for o5s in games5[:1] for b in o5s[:4]]
x = torch.as_tensor(np.stack(obs), dtype=torch.float32)
b = Brain(version=V5, conv_channels=192, num_blocks=40).eval()
d_ = DQN(version=V5).eval()
mask = torch.ones(x.shape[0], 46, dtype=torch.bool)
with torch.no_grad():
    q = d_(b(x), mask)
check('前向输出形状 [B,46] 且有限', q.shape == (x.shape[0], 46) and bool(torch.isfinite(q).all()),
      f'{tuple(q.shape)}')

print()
if fails:
    print(f'FAIL: {len(fails)} 项未通过：{fails}')
    sys.exit(1)
print('ALL PASS')
