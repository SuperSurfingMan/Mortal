"""第 4 轮启动前冒烟：确认 ① 白名单 v2 生效；② enable_augmentation 确实改变样本（不是空操作）。

做法：同一批文件，分别用 augmented=False / True 的 GameplayLoader（都带 v2 白名单）加载，
比较样本数与首个样本 obs 的差异；再用 config.round4 的 FileDatasetsIter 取一个 batch 确认形状。
"""
import glob
import random
import os
import sys

BASE = '/mnt/data8/mortal/Mortal/mortal'
os.environ.setdefault('MORTAL_CFG', BASE + '/config.round4.toml')
os.chdir(BASE)                      # config 里的 player_names_files/相对路径都以 mortal/ 为基准
sys.path.insert(0, BASE)            # 保证 import libriichi 拿到 mortal/libriichi.so 而非被同名 crate 目录遮蔽

import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import filtered_trimmed_lines  # noqa: E402
from config import config  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402
from dataloader import FileDatasetsIter  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

names = set()
for fn in config['dataset']['player_names_files']:
    with open(fn, encoding='utf-8') as f:
        names.update(filtered_trimmed_lines(f))
print(f'白名单人数 = {len(names)}（config 指到 {config["dataset"]["player_names_files"]}）')

fl = sorted(glob.glob('/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202503/*.json.gz'))
random.Random(7).shuffle(fl)
fl = fl[:6]

for aug in (False, True):
    ld = GameplayLoader(version=4, oracle=False, player_names=list(names), excludes=None, augmented=aug)
    tot = 0
    first_obs = None
    for fp in fl:
        for file in ld.load_gz_log_files([fp]):
            for g in file:
                obs = g.take_obs()
                tot += len(obs)
                if first_obs is None and len(obs) > 0:
                    first_obs = np.asarray(obs[5], dtype=np.float32)   # 取每局第 6 步，避开起始差异过小的位置
    print(f'augmented={aug}: 样本数={tot}  首个观测(第6步) sum={first_obs.sum():.2f} '
          f'nonzero={int((first_obs != 0).sum())}  sha1={hash(first_obs.tobytes()) & 0xffffffff:08x}')

print('--- 用 config.round4 的 FileDatasetsIter 实取一批（应走增强分支）---')
ds = FileDatasetsIter(
    version=config['control']['version'],
    file_list=sorted(set(fl)),
    pts=config['env']['pts'],
    file_batch_size=8,
    reserve_ratio=0.0,
    player_names=list(names),
    num_epochs=config['dataset']['num_epochs'],
    enable_augmentation=config['dataset']['enable_augmentation'],
    augmented_first=config['dataset']['augmented_first'],
)
it = iter(DataLoader(ds, batch_size=config['control']['batch_size'], drop_last=True, num_workers=0))
obs, actions, masks, std, kr, ranks = next(it)
print(f'OK: obs={tuple(obs.shape)} action_space_ok={bool(masks.any())} reward_std={kr.double().std():.3f} '
      f'(config: aug={config["dataset"]["enable_augmentation"]}, aug_first={config["dataset"]["augmented_first"]})')
