#!/usr/bin/env python3
"""诊断：iql 条目的 r_t 是否把每局奖励重复计给了该局所有决策点（应为「仅局边界非零」）。"""
import glob, os, sys
from pathlib import Path
REPO = Path('/mnt/data8/mortal/Mortal'); MORTAL = REPO / 'mortal'
os.chdir(MORTAL); sys.path.insert(0, str(MORTAL))
import numpy as np, torch
from config import config
from model import GRP
from reward_calculator import RewardCalculator
from dataloader import FileDatasetsIter
from libriichi.dataset import GameplayLoader

FILES = sorted(glob.glob(str(REPO / 'dataset/mjai/houou4_names/202601/*.json.gz')))[:3]
grp_net = GRP(**config['grp']['network'])
grp_net.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True,
                                   map_location=torch.device('cpu'))['model'])
rc = RewardCalculator(grp_net, config['env']['pts'])
ds = FileDatasetsIter(version=config['control']['version'], file_list=sorted(FILES),
    pts=config['env']['pts'], file_batch_size=4, reserve_ratio=0.0, player_names=None,
    num_epochs=1, enable_augmentation=False, augmented_first=False,
    reward_mode='rank_pt', reward_scale=1.0, objective='iql')
entries = [tuple(e) for e in ds]
ds_sum = float(np.sum([float(e[5]) for e in entries]))
ds_nonzero = sum(1 for e in entries if abs(float(e[5])) > 1e-12)

loader = GameplayLoader(version=config['control']['version'], oracle=False,
                        player_names=None, excludes=None, augmented=False)
true_sum = 0.0
n_kyoku = 0
n_games = 0
samples = 0
for path in FILES:
    for file in loader.load_gz_log_files([path]):
        for game in file:
            pid = game.take_player_id()
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8)
            grp = game.take_grp()
            r = np.asarray(rc.calc_delta_pt(pid, grp.take_feature(), grp.take_rank_by_player()))
            true_sum += float(r.sum())          # 每局奖励只应被计一次
            n_kyoku += len(r)
            n_games += 1
            samples += len(at_kyoku)

print(f'样本数={samples}  game 数={n_games}  局数={n_kyoku}')
print(f'每局平均决策点数 = {samples / n_games:.2f} / {n_kyoku / n_games:.2f} = {samples / n_kyoku:.2f}')
print(f'dataset r_t 总和  = {ds_sum:+.4f}   非零条数 = {ds_nonzero}')
print(f'应有 r_t 总和     = {true_sum:+.4f}   非零条数 = {n_kyoku}（每局只计一次）')
print(f'放大倍数 = {ds_sum / true_sum if abs(true_sum) > 1e-9 else float("nan"):.2f}×   '
      f'（非零条数比 = {ds_nonzero / n_kyoku:.2f}×）')
