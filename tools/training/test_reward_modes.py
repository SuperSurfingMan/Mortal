#!/usr/bin/env python3
"""奖励口径（env.reward_mode / reward_scale）验证（2026-10-12，对应分析文档 §3.1）。

检查三层：
  A. **公式独立核对**：用 mjai 事件流独立重算每局真实分数差，与 `calc_delta_points`
     完全一致；并独立复现 `Grp::final_scores` 的推导规则（末局起始分 + 末局 hora/ryukyoku
     deltas + 立直供托 −1000，总和 <10 万时补给第 1 名）→ 必须与 Rust 输出逐位相同。
  B. **默认口径回归**：`rank_pt`（默认）在当前数据上仍等于直接调用 `calc_delta_pt`
     的取值集合与范围（确认没把老路径改坏）。
  C. **接线正确性**：走完整 `FileDatasetsIter`，两种模式样本数相同、奖励集合不同、
     `reward_scale` 精确等比缩放、delta_points 的取值都是 100 的整数倍（天凤分数特征）、
     非法 reward_mode 会被 assert 拦住。

用法（任意目录均可；脚本自己会 chdir 到 mortal/ 并设 MORTAL_CFG）：
    python tools/training/test_reward_modes.py
    MORTAL_CFG=config.round4.toml python tools/training/test_reward_modes.py   # 覆盖配置
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MORTAL = REPO / 'mortal'
os.chdir(MORTAL)
os.environ.setdefault('MORTAL_CFG', 'config.round4.toml')
sys.path.insert(0, str(MORTAL))

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
from config import config                                 # noqa: E402
from model import GRP                                     # noqa: E402
from reward_calculator import RewardCalculator            # noqa: E402
from dataloader import FileDatasetsIter                    # noqa: E402
from libriichi.dataset import GameplayLoader              # noqa: E402

FILES = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4_names/202601/*.json.gz')))[:3]
if not FILES:
    raise SystemExit('找不到测试牌谱（dataset/mjai/houou4_names/202601）')

fails: list[str] = []


def check(name, cond, extra=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name}' + (f'    {extra}' if extra else ''))
    if not cond:
        fails.append(name)


# --- 被验证的实现（与训练同一来源）-------------------------------------------
grp_net = GRP(**config['grp']['network'])
grp_net.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True,
                                   map_location=torch.device('cpu'))['model'])
reward_calc = RewardCalculator(grp_net, config['env']['pts'])
print(f'配置 : MORTAL_CFG={os.environ["MORTAL_CFG"]} | grp={config["grp"]["state_file"]} '
      f'({config["grp"]["network"]}) | pts={config["env"]["pts"]}')
print(f'牌谱 : {len(FILES)} 个（{Path(FILES[0]).parent.name}）')


def expected_from_events(path: str):
    """独立重算：返回 [(start_scores_k, ...), final_scores(4)]，规则见 grp.rs::load_events。"""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        events = [json.loads(l) for l in f if l.strip()]
    starts = [e['scores'] for e in events if e['type'] == 'start_kyoku']
    last_start_idx = max(i for i, e in enumerate(events) if e['type'] == 'start_kyoku')
    final_deltas = [0, 0, 0, 0]
    for e in events[last_start_idx + 1:]:
        if e['type'] in ('hora', 'ryukyoku'):
            for i in range(4):
                final_deltas[i] += e['deltas'][i]
        elif e['type'] == 'reach_accepted':
            final_deltas[e['actor']] -= 1000
    final_scores = [s + d for s, d in zip(starts[-1], final_deltas)]
    if sum(final_scores) < 100_000:                      # grp.rs: 补给第 1 名（平分按座位小者优先）
        top = sorted(range(4), key=lambda i: -final_scores[i])[0]
        final_scores[top] += 100_000 - sum(final_scores)
    return starts, final_scores


print('\n== A) calc_delta_points 与独立重算 ==')
loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)
n_seats = 0
n_kyoku = 0
n_exact_final = 0
max_err = 0.0
for path in FILES:
    starts, final_expected = expected_from_events(path)
    for game in [g for file in loader.load_gz_log_files([path]) for g in file]:
        player_id = game.take_player_id()
        at_kyoku = game.take_at_kyoku()
        grp = game.take_grp()
        feature = grp.take_feature()
        final_scores_rust = list(grp.take_final_scores())
        rank_by_player = grp.take_rank_by_player()

        if final_scores_rust == final_expected:
            n_exact_final += 1
        else:
            check(f'{Path(path).name[:12]} seat{player_id} final_scores 独立重算一致', False,
                  f'rust={final_scores_rust} expected={final_expected}')

        dp = reward_calc.calc_delta_points(player_id, feature, final_scores_rust)
        exp = [starts[k + 1][player_id] - starts[k][player_id] for k in range(len(starts) - 1)]
        exp.append(final_scores_rust[player_id] - starts[-1][player_id])
        err = float(np.max(np.abs(np.asarray(dp) - np.asarray(exp, dtype=np.float64))))
        max_err = max(max_err, err)
        n_kyoku += len(exp)
        n_seats += 1
        if abs(dp.sum() - (final_scores_rust[player_id] - starts[0][player_id])) > 1e-6:
            check('delta_points 总和 = 终局分 − 起始分', False, f'seat{player_id} {path}')
        # 顺带确认 rank_pt 仍可正常计算（默认路径没被破坏）
        rp = reward_calc.calc_delta_pt(player_id, feature, rank_by_player)
        if len(rp) != len(dp):
            check('rank_pt 与 delta_points 局数一致', False, f'seat{player_id}')
check('final_scores 独立重算逐位一致', n_exact_final == n_seats, f'{n_exact_final}/{n_seats} 个座位')
check('calc_delta_points == 独立重算（最大误差 < 1e-6）', max_err < 1e-6, f'max_err={max_err:.2e}')
print(f'      覆盖 {n_seats} 个座位 / {n_kyoku} 局')

print('\n== B) rank_pt 默认口径回归 ==')
loader2 = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)
direct_rp = []
for path in FILES:
    for game in [g for file in loader2.load_gz_log_files([path]) for g in file]:
        grp = game.take_grp()
        direct_rp.extend(np.asarray(
            reward_calc.calc_delta_pt(game.take_player_id(), grp.take_feature(),
                                      grp.take_rank_by_player())).tolist())
print(f'      直接计算 rank_pt 取值 {len(direct_rp)} 个，范围 '
      f'[{min(direct_rp):.3f}, {max(direct_rp):.3f}]')
check('rank_pt 取值落在 pts 极差内（|reward| <= max|pts|）',
      max(abs(min(direct_rp)), abs(max(direct_rp))) <= max(abs(p) for p in config['env']['pts']) + 1e-9)


def collect(mode, scale):
    ds = FileDatasetsIter(
        version=config['control']['version'], file_list=sorted(FILES), pts=config['env']['pts'],
        file_batch_size=4, reserve_ratio=0.0, player_names=None, num_epochs=1,
        enable_augmentation=False, augmented_first=False,
        reward_mode=mode, reward_scale=scale,
    )
    rewards, steps = [], []
    for entry in ds:
        rewards.append(float(entry[4]))
        steps.append(int(entry[3]))
    return rewards, steps


print('\n== C) FileDatasetsIter 接线 ==')
# 注意：load_files 里每轮都会 random.shuffle(file_list)（无固定种子）→ 三次 collect 的
# 样本顺序不同，因此下面一律**按多重集比较**，不做逐元素比较。
rp_r, rp_s = collect('rank_pt', 1.0)
dp_r, dp_s = collect('delta_points', 1.0)
dp_scaled_r, _ = collect('delta_points', 1e-3)
print(f'      rank_pt: {len(rp_r)} 样本，范围 [{min(rp_r):.3f}, {max(rp_r):.3f}]')
print(f'      delta_pt: {len(dp_r)} 样本，范围 [{min(dp_r):.1f}, {max(dp_r):.1f}]（点）')
check('两种模式样本数相同', len(rp_r) == len(dp_r) and len(rp_r) > 0, f'{len(rp_r)} vs {len(dp_r)}')
check('两种模式奖励集合不同', sorted(set(rp_r)) != sorted(set(dp_r)))


def is_multiple_of(v, base, tol=1e-6):
    """浮点近似判定：score/1e4*1e4 会带 ~1e-12 误差，故用四舍五入后回代比较。"""
    return abs(round(v / base) * base - v) < tol


not_mult100 = [v for v in dp_r if not is_multiple_of(v, 100.0)]
check('delta_points 取值都是 100 的整数倍（真实分数特征）', not not_mult100,
      f'非整百值 {len(not_mult100)} 个（例 {not_mult100[:2]}）')
a_sorted = sorted(round(v, 6) for v in dp_scaled_r)
b_sorted = sorted(round(v * 1e-3, 6) for v in dp_r)
check('reward_scale 精确等比缩放（多重集比较）', a_sorted == b_sorted,
      f'期望 max={max(dp_r) * 1e-3:.4f} 实得 max={max(dp_scaled_r):.4f}')
rp_iter_set = set(round(v, 6) for v in rp_r)
rp_direct_set = set(round(v, 6) for v in direct_rp)
check('rank_pt 与直接计算同源（iter 取值 ⊆ 直接计算取值）', rp_iter_set <= rp_direct_set,
      f'iter 取值 {len(rp_iter_set)} 种 / 直接计算 {len(rp_direct_set)} 种'
      f'（直接计算多出的 {len(rp_direct_set - rp_iter_set)} 种来自"该局无动作"的局）')
try:
    FileDatasetsIter(version=4, file_list=FILES[:1], pts=config['env']['pts'], reward_mode='bogus')
    check('非法 reward_mode 被拦住', False, '没有抛异常')
except AssertionError:
    check('非法 reward_mode 被拦住', True)

print('\n== D) 续训目标守卫 env_reward_diff（train.py）==')
import train as train_mod  # noqa: E402  （模块级只有函数/常量定义，导入不会启动训练）

check('同口径 → 无差异', train_mod.env_reward_diff(
    {'gamma': 1, 'pts': [6, 4, 2, 0]}, {'gamma': 1, 'pts': [6, 4, 2, 0]}) == '')
check('旧 ckpt 缺 reward_mode/reward_scale 键 → 按默认处理，不误报', train_mod.env_reward_diff(
    {'gamma': 1, 'pts': [6, 4, 2, 0]},
    {'gamma': 1, 'pts': [6, 4, 2, 0], 'reward_mode': 'rank_pt', 'reward_scale': 1.0}) == '')
_d = train_mod.env_reward_diff({'gamma': 1, 'pts': [6, 4, 2, 0]}, {'gamma': 1, 'pts': [2, 1, 0, -3]})
check('pts 变化 → 报差异', 'pts' in _d, _d)
_d = train_mod.env_reward_diff({'gamma': 1, 'pts': [6, 4, 2, 0]},
                               {'gamma': 1, 'pts': [6, 4, 2, 0], 'reward_mode': 'delta_points'})
check('reward_mode 变化 → 报差异', 'reward_mode' in _d, _d)
_d = train_mod.env_reward_diff({'gamma': 1, 'pts': [6, 4, 2, 0], 'reward_scale': 1},
                               {'gamma': 1, 'pts': [6, 4, 2, 0], 'reward_scale': 1e-3})
check('reward_scale 变化 → 报差异（int/float 归一）', 'reward_scale' in _d, _d)
_d = train_mod.env_reward_diff({'gamma': 0.995}, {'gamma': 1})
check('gamma 变化 → 报差异', 'gamma' in _d, _d)
check('optimizer reset 同令牌 → 不重置',
      train_mod.optimizer_reset_required('reward-v1', 'reward-v1') is False)
check('optimizer reset false → 不重置',
      train_mod.optimizer_reset_required(False, False) is False)
check('optimizer reset 新令牌 → 重置一次',
      train_mod.optimizer_reset_required(False, 'reward-v1') is True)

print()
if fails:
    print(f'FAIL（{len(fails)}）: {fails}')
    sys.exit(1)
print('ALL PASS')
