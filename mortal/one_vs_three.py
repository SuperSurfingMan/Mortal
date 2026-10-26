import prelude

import numpy as np
import torch
import secrets
import os
from model import Brain, DQN
from engine import MortalEngine
import s6c_features
from libriichi.arena import OneVsThree
from config import config

def main():
    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    use_akochan = cfg['akochan']['enabled']

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        state = torch.load(cfg['champion']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cham_cfg = state['config']
        version = cham_cfg['control'].get('version', 1)
        conv_channels = cham_cfg['resnet']['conv_channels']
        num_blocks = cham_cfg['resnet']['num_blocks']
        cham_extra = s6c_features.resolve(cham_cfg['control'])[1]
        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks,
                       extra_channels=cham_extra).eval()
        dqn = DQN(version=version).eval()
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        if cfg['champion']['enable_compile']:
            mortal.compile()
            dqn.compile()
        engine_cham = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = version,
            device = torch.device(cfg['champion']['device']),
            enable_amp = cfg['champion']['enable_amp'],
            enable_rule_based_agari_guard = cfg['champion']['enable_rule_based_agari_guard'],
            extra_features = s6c_features.resolve(cham_cfg['control'])[0],
            name = cfg['champion']['name'],
        )

    state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    chal_cfg = state['config']
    version = chal_cfg['control'].get('version', 1)
    conv_channels = chal_cfg['resnet']['conv_channels']
    num_blocks = chal_cfg['resnet']['num_blocks']
    chal_extra_features, chal_extra = s6c_features.resolve(chal_cfg['control'])
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks,
                   extra_channels=chal_extra).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    if cfg['challenger']['enable_compile']:
        mortal.compile()
        dqn.compile()
    engine_chal = MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        version = version,
        device = torch.device(cfg['challenger']['device']),
        enable_amp = cfg['challenger']['enable_amp'],
        enable_rule_based_agari_guard = cfg['challenger']['enable_rule_based_agari_guard'],
        extra_features = chal_extra_features,
        name = cfg['challenger']['name'],
    )

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        if use_akochan:
            # `ako_vs_py` = 1 akochan vs 3 Mortal；`py_vs_ako` = 1 Mortal vs 3 akochan。
            # 绝对锚点必须用后者（avg_rank/avg_pt 才代表 Mortal 自己，且与人类场同尺度）。
            if cfg.get('mortal_vs_ako', False):
                rankings = env.py_vs_ako(
                    engine = engine_chal,
                    seed_start = (seed, key),
                    seed_count = seeds_per_iter,
                )
            else:
                rankings = env.ako_vs_py(
                    engine = engine_chal,
                    seed_start = (seed, key),
                    seed_count = seeds_per_iter,
                )
        else:
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = engine_cham,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
