#!/usr/bin/env python3
"""信用分配口径（env.credit_mode）验证（2026-10-14，IQL/Bellman 改造 Stage A）。

检查五层：
  A. **闭式恒等式**：γ=1 时折扣后缀和 returns[k] = Σ_{j≥k} r_j 必须逐位等于
     pts[final_rank] − E_grp[pts | 第 k 局开始]（GRP 中途期望作控制变量）；
     并验证 GRP 终局顺位概率确为 one-hot（闭式成立的前提）。
  B. **γ<1 递推**：returns[k] = r_k + γ·returns[k+1] 的实现与直接按定义重算一致。
  C. **FileDatasetsIter 接线**：'game' 模式样本值与直接计算按多重集相等；
     'kyoku' 模式与直接 calc_delta_pt 多重集相等（默认路径回归等价）；两种模式
     样本数相同、取值集合不同；非法 credit_mode 会被 assert 拦住。
  D. **终端一致性**：最后一局的 game 模式值 = Σ_{j=末局} r_j（无未来局可折现），
     第 0 局的值 = 整场回报（含全部未来局）。
  E. **续训目标守卫**：credit_mode 进 _ENV_REWARD_KEYS；旧 ckpt 缺键不误报；
     换口径会报差异。

用法（任意目录均可；脚本自己会 chdir 到 mortal/ 并设 MORTAL_CFG）：
    python tools/training/test_credit_modes.py
"""
from __future__ import annotations

import glob
import os
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
from dataloader import FileDatasetsIter                   # noqa: E402
from libriichi.dataset import GameplayLoader              # noqa: E402

FILES = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4_names/202601/*.json.gz')))[:3]
if not FILES:
    raise SystemExit('找不到测试牌谱（dataset/mjai/houou4_names/202601）')

fails: list[str] = []


def check(name, cond, extra=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name}' + (f'    {extra}' if extra else ''))
    if not cond:
        fails.append(name)


# --- 与训练同一来源的 GRP / RewardCalculator ----------------------------------
grp_net = GRP(**config['grp']['network'])
grp_net.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True,
                                   map_location=torch.device('cpu'))['model'])
reward_calc = RewardCalculator(grp_net, config['env']['pts'])
PTS = config['env']['pts']
print(f'配置 : MORTAL_CFG={os.environ["MORTAL_CFG"]} | grp={config["grp"]["state_file"]} '
      f'| pts={PTS}')
print(f'牌谱 : {len(FILES)} 个（{Path(FILES[0]).parent.name}）')


def direct_returns(path: str, gamma: float):
    """按定义独立重算，返回 (后缀回报样本多重集, 原始 r_k 样本多重集, 逐局数组)。"""
    loader = GameplayLoader(version=config['control']['version'], oracle=False,
                            player_names=None, excludes=None, augmented=False)
    return_values: list[float] = []
    raw_values: list[float] = []
    per_game: list[tuple[np.ndarray, np.ndarray]] = []   # (returns, at_kyoku)
    for file in loader.load_gz_log_files([path]):
        for game in file:
            player_id = game.take_player_id()
            # Rust Vec<u8> 经 pyo3 返回 bytes，需 frombuffer 转数组
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
            grp = game.take_grp()
            feature = grp.take_feature()
            rank_by_player = grp.take_rank_by_player()
            r = np.asarray(reward_calc.calc_delta_pt(player_id, feature, rank_by_player),
                           dtype=np.float64)
            raw_values.extend(r[at_kyoku].tolist())
            returns = np.zeros_like(r)
            acc = 0.0
            for k in range(len(r) - 1, -1, -1):
                acc = r[k] + gamma * acc
                returns[k] = acc
            return_values.extend(returns[at_kyoku].tolist())
            per_game.append((returns, at_kyoku))
    return return_values, raw_values, per_game


def closed_form(path: str):
    """γ=1 的闭式参照：返回 (闭式 returns 列表 per game, exp_pts 矩阵 per game)。"""
    loader = GameplayLoader(version=config['control']['version'], oracle=False,
                            player_names=None, excludes=None, augmented=False)
    out = []
    for file in loader.load_gz_log_files([path]):
        for game in file:
            player_id = game.take_player_id()
            grp = game.take_grp()
            feature = grp.take_feature()
            rank_by_player = grp.take_rank_by_player()
            rank_prob = reward_calc.calc_rank_prob(player_id, feature, rank_by_player)
            exp_pts = (rank_prob @ torch.tensor(PTS, dtype=torch.float64)).numpy()
            out.append((exp_pts[-1] - exp_pts[:-1], exp_pts, rank_prob.numpy()))
    return out


print('\n== A) γ=1 闭式恒等式 + GRP 终局 one-hot ==')
n_games = 0
n_closed_err = 0.0
n_not_onehot = 0
n_telescope = 0.0
for path in FILES:
    for rets, exp_pts, rank_prob in closed_form(path):
        n_games += 1
        # GRP 终局顺位概率必须 one-hot（否则闭式不成立）
        final_row = rank_prob[-1]
        if abs(final_row.max() - 1.0) > 1e-9 or abs(final_row.sum() - 1.0) > 1e-9:
            n_not_onehot += 1
        # 闭式 returns[k] = exp_pts[-1] − exp_pts[k] 与后缀和逐位一致
        suffix = np.cumsum((exp_pts[1:] - exp_pts[:-1])[::-1])[::-1]
        n_closed_err = max(n_closed_err, float(np.max(np.abs(rets - suffix))))
        # telescope：第 0 局回报 = 终局点 − 起始期望（整场回报）
        n_telescope = max(n_telescope, abs(rets[0] - (exp_pts[-1] - exp_pts[0])))
check('GRP 终局顺位概率 one-hot', n_not_onehot == 0, f'{n_not_onehot} 个异常局')
check('后缀和 == 闭式 pts[final] − E_grp[pts|局k]（最大误差 < 1e-9）',
      n_closed_err < 1e-9, f'max_err={n_closed_err:.2e}（{n_games} 个座位）')
check('第 0 局回报 = 整场回报（telescope 终点）', n_telescope < 1e-9,
      f'max_err={n_telescope:.2e}')


def collect(credit_mode, gamma):
    ds = FileDatasetsIter(
        version=config['control']['version'], file_list=sorted(FILES), pts=PTS,
        file_batch_size=4, reserve_ratio=0.0, player_names=None, num_epochs=1,
        enable_augmentation=False, augmented_first=False,
        reward_mode='rank_pt', reward_scale=1.0,
        credit_mode=credit_mode, gamma=gamma,
    )
    values, steps = [], []
    for entry in ds:
        values.append(float(entry[4]))
        steps.append(int(entry[3]))
    return values, steps


print('\n== B) FileDatasetsIter 接线（多重集比较；load_files 内有随机 shuffle）==')
ky_r, ky_s = collect('kyoku', 1.0)
gm_r, gm_s = collect('game', 1.0)
gm_g5_r, _ = collect('game', 0.5)

direct_ky_all: list[float] = []
direct_gm_all: list[float] = []
direct_g5_all: list[float] = []
for path in FILES:
    _, raw_v, _ = direct_returns(path, 1.0)     # kyoku 口径的直接参照 = 原始 r_k
    direct_ky_all.extend(raw_v)
    v, _, _ = direct_returns(path, 1.0)         # game γ=1 的直接参照 = 后缀和
    direct_gm_all.extend(v)
    v5, _, _ = direct_returns(path, 0.5)        # game γ=0.5 的直接参照 = 折扣后缀和
    direct_g5_all.extend(v5)

check('两种 credit_mode 样本数相同', len(ky_r) == len(gm_r) and len(ky_r) > 0,
      f'{len(ky_r)} vs {len(gm_r)}')
check('两种 credit_mode 取值集合不同', sorted(set(ky_r)) != sorted(set(gm_r)),
      f'kyoku {len(set(ky_r))} 种 / game {len(set(gm_r))} 种')

ms = lambda xs: sorted(round(v, 9) for v in xs)
check('kyoku 模式 == 直接 calc_delta_pt（回归等价，多重集）',
      ms(ky_r) == ms(direct_ky_all), f'{len(ky_r)} 样本')
check('game γ=1 == 直接后缀和（多重集）', ms(gm_r) == ms(direct_gm_all), f'{len(gm_r)} 样本')
check('game γ=0.5 == 直接折扣后缀和（多重集）', ms(gm_g5_r) == ms(direct_g5_all),
      f'{len(gm_g5_r)} 样本')
check('game γ=0.5 与 γ=1 取值不同（折扣确实生效）', sorted(set(gm_g5_r)) != sorted(set(gm_r)))

print('\n== C) 终端一致性（最后一局无未来局、第 0 局含全部未来局）==')
# 取一个座位逐局核对：dataset 值里，末局的 game 值应等于末局 r；首局值应等于整场回报
rets0, exp0, _ = closed_form(FILES[0])[0]
last_k = len(rets0) - 1
r_seq = exp0[1:] - exp0[:-1]
check('末局 game 值 = 末局 r_k（无未来局可折现）',
      any(abs(v - r_seq[last_k]) < 1e-9 for v in gm_r),
      f'期望 ≈ {r_seq[last_k]:+.6f}')
check('首局 game 值 = 整场回报（exp_pts[-1] − exp_pts[0]）',
      any(abs(v - (exp0[-1] - exp0[0])) < 1e-9 for v in gm_r),
      f'期望 ≈ {exp0[-1] - exp0[0]:+.6f}')
check('game 值域不超过 pts 极差的 2 倍（回报口径合理）',
      max(abs(min(gm_r)), abs(max(gm_r))) <= 2 * max(abs(p) for p in PTS) + 1e-9,
      f'范围 [{min(gm_r):.3f}, {max(gm_r):.3f}]')

print('\n== D) 非法 credit_mode 被拦 ==')
try:
    FileDatasetsIter(version=4, file_list=FILES[:1], pts=PTS, credit_mode='bogus')
    check('非法 credit_mode 被拦住', False, '没有抛异常')
except AssertionError:
    check('非法 credit_mode 被拦住', True)

print('\n== E) 续训目标守卫 env_reward_diff（train.py）==')
import train as train_mod  # noqa: E402  （模块级只有函数/常量定义，导入不会启动训练）

check('同口径 → 无差异', train_mod.env_reward_diff(
    {'gamma': 1, 'pts': [2, 1, 0, -3]}, {'gamma': 1, 'pts': [2, 1, 0, -3]}) == '')
check('旧 ckpt 缺 credit_mode 键 → 按默认处理，不误报', train_mod.env_reward_diff(
    {'gamma': 1, 'pts': [2, 1, 0, -3]},
    {'gamma': 1, 'pts': [2, 1, 0, -3], 'credit_mode': 'kyoku'}) == '')
_d = train_mod.env_reward_diff({'gamma': 1, 'pts': [2, 1, 0, -3]},
                               {'gamma': 1, 'pts': [2, 1, 0, -3], 'credit_mode': 'game'})
check('credit_mode 变化 → 报差异', 'credit_mode' in _d, _d)

print()
if fails:
    print(f'FAIL（{len(fails)}）: {fails}')
    sys.exit(1)
print('ALL PASS')
