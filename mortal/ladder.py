# 两个 checkpoint 之间的 1v3 duplicate-mahjong 对抗（真实强度阶梯评测）
# 用法: MORTAL_CFG=... python ladder.py <challenger.pth> <champion.pth> [seed_start] [n_sets]
import sys
import shutil
import torch
from os import path
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from libriichi.stat import Stat

def load_engine(ckpt, name, device='cuda:0'):
    st = torch.load(ckpt, map_location=torch.device('cpu'), weights_only=True)
    cfg = st['config']
    version = cfg['control'].get('version', 1)
    mortal = Brain(version=version, **cfg['resnet']).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(st['mortal'])
    dqn.load_state_dict(st['current_dqn'])
    return MortalEngine(
        mortal, dqn,
        is_oracle = False,
        version = version,
        device = torch.device(device),
        enable_amp = False,
        enable_rule_based_agari_guard = True,
        name = name,
    )

def main():
    chal_ckpt = sys.argv[1]
    cham_ckpt = sys.argv[2]
    seed_start = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    n_sets = int(sys.argv[4]) if len(sys.argv) > 4 else 500      # 1 set = 4 局同牌山
    key = 0x2000
    log_dir = path.abspath('runs/ladder')
    if path.isdir(log_dir):
        shutil.rmtree(log_dir)
    chal = load_engine(chal_ckpt, 'challenger')
    cham = load_engine(cham_ckpt, 'champion')
    env = OneVsThree(disable_progress_bar = False, log_dir = log_dir)
    rankings = env.py_vs_py(
        challenger = chal,
        champion = cham,
        seed_start = (seed_start, key),
        seed_count = n_sets,
    )
    stat = Stat.from_dir(log_dir, 'challenger')
    print(f'challenger = {chal_ckpt}')
    print(f'champion   = {cham_ckpt}')
    print(f'sets = {n_sets} (hanchans = {n_sets * 4}), seed_start = {seed_start}')
    print(f'raw rankings = {rankings}')
    print(f'avg_rank = {stat.avg_rank:.4f}   avg_pt = {stat.avg_pt([90, 45, 0, -135]):.3f}')
    print(f'1st/2nd/3rd/4th = {stat.rank_1_rate:.3f} {stat.rank_2_rate:.3f} {stat.rank_3_rate:.3f} {stat.rank_4_rate:.3f}')
    print(f'agari={stat.agari_rate:.3f} houjuu={stat.houjuu_rate:.3f} fuuro={stat.fuuro_rate:.3f} riichi={stat.riichi_rate:.3f}')

main()
