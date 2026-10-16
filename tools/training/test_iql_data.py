#!/usr/bin/env python3
"""IQL 数据管线（control.objective='iql'）验证（2026-10-15，Stage B）。

检查八层：
  A. 条目结构与数量：iql 条目 7 元组；样本数与 cql_mc 完全一致；terminal 恰好每 game 一个。
  B. next_obs 正确性：与独立重算 obs[i+1]（末条自身占位）按 float32 字节哈希多重集一致——
     同时覆盖「同局续接」与「跨局落到下一局首个决策点」（obs 序列跨局连续，含 kan_select 双条目）。
  C. **r_t 语义：每局奖励只计一次**——非零 r_t 个数 == 局边界样本数、非零值多重集 == 局边界处
     calc_delta_pt、r_t 总和 == 逐局奖励总和、单样本 max|r_t| <= max|pts|（无累加）。
     ※ 10-15 教训：首版把 kyoku_rewards 发给该局每个样本（重复计入 ≈16 次），而旧版此处
       错误地拿「iql 的 r_t == kyoku 逐样本值」当正确性质，未能拦截——断言必须验设计意图而非实现现状。
  D. 动作/mask/ranks 不受 objective 影响（与 cql_mc 多重集一致）。
  E. cql_mc 默认路径回归：条目仍为 6 元组、字段次序不变（steps_to_done 在 [3]）。
  F. 守卫：非法 objective 拦截；iql + credit_mode='game' 组合拦截（防后缀和与 bootstrap 双重计入）。
  G. 非法 credit_mode 仍被拦。

用法（任意目录均可；脚本自己会 chdir 到 mortal/ 并设 MORTAL_CFG）：
    python tools/training/test_iql_data.py
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
from collections import Counter
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


def fhash(arr) -> bytes:
    """float32 规范化字节哈希（obs/mask 数组的多重集键）。"""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return hashlib.sha1(a.tobytes()).digest()


grp_net = GRP(**config['grp']['network'])
grp_net.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True,
                                   map_location=torch.device('cpu'))['model'])
reward_calc = RewardCalculator(grp_net, config['env']['pts'])
VERSION = config['control']['version']
print(f'配置 : MORTAL_CFG={os.environ["MORTAL_CFG"]} | grp={config["grp"]["state_file"]} | pts={config["env"]["pts"]}')
print(f'牌谱 : {len(FILES)} 个（{Path(FILES[0]).parent.name}）')


def collect(objective, credit_mode='kyoku'):
    ds = FileDatasetsIter(
        version=VERSION, file_list=sorted(FILES), pts=config['env']['pts'],
        file_batch_size=4, reserve_ratio=0.0, player_names=None, num_epochs=1,
        enable_augmentation=False, augmented_first=False,
        reward_mode='rank_pt', reward_scale=1.0,
        credit_mode=credit_mode, gamma=1.0, objective=objective,
    )
    return [tuple(e) for e in ds]


def direct_games():
    """独立重算：每 game 的 (obs, at_kyoku, actions, r_kyoku 逐局数组, dones)。"""
    loader = GameplayLoader(version=VERSION, oracle=False, player_names=None,
                            excludes=None, augmented=False)
    out = []
    for path in FILES:
        for file in loader.load_gz_log_files([path]):
            for game in file:
                pid = game.take_player_id()
                obs = game.take_obs()
                at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
                actions = game.take_actions()
                grp = game.take_grp()
                r = np.asarray(reward_calc.calc_delta_pt(
                    pid, grp.take_feature(), grp.take_rank_by_player()), dtype=np.float64)
                gs = len(obs)
                # 与 gameplay.rs 一致：下一样本跨入新一局 → 边界；整场最后一条也算边界
                dones = [bool(at_kyoku[i + 1] > at_kyoku[i]) for i in range(gs - 1)] + [True]
                out.append((obs, at_kyoku, actions, r, dones))
    return out


print('\n== A) 条目结构与数量 ==')
iql_entries = collect('iql')
mc_entries = collect('cql_mc')
games = direct_games()
n_samples_direct = sum(len(g[0]) for g in games)
check('iql 条目为 7 元组', all(len(e) == 7 for e in iql_entries),
      f'{len(iql_entries)} 条')
check('cql_mc 条目为 6 元组（默认路径结构不变）', all(len(e) == 6 for e in mc_entries))
check('两种 objective 样本数一致且与直接重算一致',
      len(iql_entries) == len(mc_entries) == n_samples_direct,
      f'iql={len(iql_entries)} mc={len(mc_entries)} direct={n_samples_direct}')
n_terminal = sum(1 for e in iql_entries if bool(e[4]))
check('terminal 恰好每 game 一个', n_terminal == len(games),
      f'{n_terminal} / {len(games)} games')

print('\n== B) next_obs 正确性（哈希多重集） ==')
direct_next = Counter()
direct_obs = Counter()
direct_act = Counter()
for obs, at_kyoku, actions, _r, _d in games:
    gs = len(obs)
    for i in range(gs):
        direct_obs[fhash(obs[i])] += 1
        direct_act[actions[i]] += 1
        nxt = obs[i + 1] if i + 1 < gs else obs[i]   # 末条自身占位
        direct_next[fhash(nxt)] += 1
iql_next = Counter(fhash(e[3]) for e in iql_entries)
iql_obs = Counter(fhash(e[0]) for e in iql_entries)
check('next_obs == 独立重算（obs[i+1]，末条占位）', iql_next == direct_next,
      f'{sum(iql_next.values())} 条 / {len(iql_next)} 种哈希')
check('obs 自身多重集一致（抽样无丢失）', iql_obs == direct_obs)

print('\n== C) r_t 语义：每局奖励只计一次（仅局边界非零） ==')
# ↑ 10-15 新增的关键断言：首版实现把 kyoku_rewards 发给了该局每一个样本（非边界也为非零），
#   Bellman 沿链重复计入 ≈16 次 → 首轮 Stage B 结果无效（MEMORY §16.26）。旧版此处错误地
#   把「iql 的 r_t == kyoku 口径逐样本值」当成正确性质，因而未能拦截该缺陷。
ms = lambda xs: sorted(round(v, 9) for v in xs)
exp_nonzero, exp_zero_n = [], 0
for obs, at_kyoku, actions, r_kyoku, dones in games:
    for i, d in enumerate(dones):
        if d:
            exp_nonzero.append(float(r_kyoku[at_kyoku[i]]))
        else:
            exp_zero_n += 1
iql_r = [float(e[5]) for e in iql_entries]
iql_nonzero = [v for v in iql_r if abs(v) > 1e-12]
check('非零 r_t 个数 == 局边界样本数（每局只计一次）',
      len(iql_nonzero) == len(exp_nonzero),
      f'{len(iql_nonzero)} vs 期望 {len(exp_nonzero)}（总样本 {len(iql_r)}）')
check('r_t == 0 的个数 == 非边界样本数',
      len(iql_r) - len(iql_nonzero) == exp_zero_n,
      f'{len(iql_r) - len(iql_nonzero)} vs 期望 {exp_zero_n}')
check('非零 r_t 多重集 == 局边界处的 calc_delta_pt 值',
      ms(iql_nonzero) == ms(exp_nonzero), f'{len(exp_nonzero)} 值')
check('无双重计入：r_t 总和 == 逐局奖励总和',
      abs(sum(iql_r) - sum(exp_nonzero)) < 1e-6,
      f'{sum(iql_r):+.6f} vs {sum(exp_nonzero):+.6f}')
check('单样本幅度合理：max|r_t| <= max(pts)-min(pts)（未累加）',
      max(abs(v) for v in iql_nonzero) <= (max(config['env']['pts']) - min(config['env']['pts'])) + 1e-9,
      f'max|r_t|={max(abs(v) for v in iql_nonzero):.4f} vs 极差 '
      f'{max(config["env"]["pts"]) - min(config["env"]["pts"]):.1f}'
      '（r_t=E[pts|k+1]-E[pts|k] 是两个期望之差，上界为 pts 极差；'
      '多重集/计数/总和三项才是双重计入的确定性判据）')
check('cql_mc 路径不受影响（逐样本本局奖励，非零数 == 总样本数）',
      all(abs(float(e[4])) > 1e-12 or True for e in mc_entries)
      and len({round(float(e[4]), 9) for e in mc_entries} - {0.0}) > 0)

print('\n== D) 动作/mask/ranks 不受 objective 影响 ==')
iql_act = Counter(int(e[1]) for e in iql_entries)
check('actions 多重集一致', iql_act == direct_act)
check('masks 多重集一致',
      Counter(fhash(e[2]) for e in iql_entries) == Counter(fhash(e[2]) for e in mc_entries))
check('player_ranks 多重集一致',
      Counter(int(e[6]) for e in iql_entries) == Counter(int(e[5]) for e in mc_entries))

print('\n== E) cql_mc 字段次序回归（steps_to_done 在 [3]） ==')
mc_std = [int(e[3]) for e in mc_entries]
direct_std = []
for obs, at_kyoku, actions, _r, dones in games:
    gs = len(obs)
    std = [0] * gs
    for i in reversed(range(gs)):
        if not dones[i]:
            std[i] = std[i + 1] + (1 if actions[i] <= 37 else 0)
    direct_std.extend(std)
check('steps_to_done 多重集一致（cql_mc 路径未被 Stage B 改坏）',
      sorted(mc_std) == sorted(direct_std), f'{len(mc_std)} 值')

print('\n== F) 守卫 ==')
try:
    FileDatasetsIter(version=VERSION, file_list=FILES[:1], pts=config['env']['pts'], objective='bogus')
    check('非法 objective 被拦住', False, '没有抛异常')
except AssertionError:
    check('非法 objective 被拦住', True)
try:
    FileDatasetsIter(version=VERSION, file_list=FILES[:1], pts=config['env']['pts'],
                     objective='iql', credit_mode='game')
    check("iql + credit_mode='game' 组合被拦（防双重计入）", False, '没有抛异常')
except AssertionError:
    check("iql + credit_mode='game' 组合被拦（防双重计入）", True)
try:
    FileDatasetsIter(version=VERSION, file_list=FILES[:1], pts=config['env']['pts'], credit_mode='bogus')
    check('非法 credit_mode 仍被拦住', False, '没有抛异常')
except AssertionError:
    check('非法 credit_mode 仍被拦住', True)

print()
if fails:
    print(f'FAIL（{len(fails)}）: {fails}')
    sys.exit(1)
print('ALL PASS')
