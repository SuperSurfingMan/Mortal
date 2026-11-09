import torch
import numpy as np
import os
import random
import shutil
import secrets
import logging
from os import path
from model import Brain, DQN
from engine import MortalEngine
import s6c_features
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from config import config

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        # 卫冕者自己的 obs 口径（S6c 之前的 ckpt = 0/''）
        base_extra_features, base_extra = s6c_features.resolve(cfg['control'])
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks,
                              extra_channels=base_extra).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = bool(getattr(stable_mortal, 'is_oracle', False)),
            version = version,
            device = device,
            enable_amp = config['test_play'].get('enable_amp', False),
            enable_rule_based_agari_guard = config['test_play'].get('enable_rule_based_agari_guard', True),
            extra_features = base_extra_features,
            name = 'baseline',
        )
        self.chal_version = config['control']['version']
        # 挑战者（训练中的模型）用训练配置自己的特征族
        self.chal_extra_features, _ = s6c_features.resolve(config['control'])
        self.log_dir = path.abspath(config['test_play']['log_dir'])

    def test_play(self, seed_count, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = bool(getattr(mortal, 'is_oracle', False)),
            version = self.chal_version,
            device = device,
            enable_amp = config['test_play'].get('enable_amp', False),
            enable_rule_based_agari_guard = config['test_play'].get('enable_rule_based_agari_guard', True),
            extra_features = self.chal_extra_features,
            name = 'mortal',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

class TrainPlayer:
    def __init__(self):
        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]

        # 纯四 Mortal 自博弈（2026-11-08）：
        #   'baseline' = 旧行为：1 个学习者席位 vs 3 个固定 baseline 席位
        #   'self'     = champion 也用**当前训练中的策略**，且两个 engine 同名 'trainee'
        #                ⇒ mjai 日志的 start_game.names = 4×'trainee'
        #                ⇒ GameplayLoader 按名筛选会给同一局产出 4 份样本（4 倍数据），
        #                  见 libriichi/src/dataset/gameplay.rs::load_events 的 filter。
        #                ⇒ rankings 仍只统计 challenger 那一席（one_vs_three 的 i%4），
        #                  而行为画像 Stat 会统计 4 席（同名 ⇒ 都是 'trainee'）。
        self.champion_mode = cfg.get('champion', 'baseline')
        self.champ_boltzmann_epsilon = float(cfg.get('champ_boltzmann_epsilon', 0.0))
        self.champ_boltzmann_temp = float(cfg.get('champ_boltzmann_temp', 1.0))
        self.champ_top_p = float(cfg.get('champ_top_p', 1.0))
        # 代际对手池（2026-11-09）：champion='pool' 时每轮从这些 ckpt 里抽一个当对手。
        # 动机：4M 的 4 席同权重是严格零和（期望 reward = 0），数据分布无新信息；
        # 跨代对手能提供真实的策略差异，是唯一“改变数据分布”而不只是改损失的手段。
        self.champion_pool = list(cfg.get('champion_pool', []))
        logging.info(f'[train_play] champion = {self.champion_mode}')
        assert self.champion_mode in ('baseline', 'self', 'pool'), self.champion_mode
        if self.champion_mode == 'pool':
            assert self.champion_pool, 'champion=pool 需要 [train_play].champion_pool 列表'

        self.baseline_engine = None
        self.pool_engines = []
        if self.champion_mode == 'baseline':
            baseline_cfg = config['baseline']['train']
            device = torch.device(baseline_cfg['device'])

            state = torch.load(baseline_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
            base_cfg = state['config']
            version = base_cfg['control'].get('version', 1)
            conv_channels = base_cfg['resnet']['conv_channels']
            num_blocks = base_cfg['resnet']['num_blocks']
            # 卫冕者自己的 obs 口径（S6c 之前的 ckpt = 0/''）
            base_extra_features, base_extra = s6c_features.resolve(base_cfg['control'])
            stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks,
                                  extra_channels=base_extra).eval()
            stable_dqn = DQN(version=version).eval()
            stable_mortal.load_state_dict(state['mortal'])
            stable_dqn.load_state_dict(state['current_dqn'])
            if baseline_cfg['enable_compile']:
                stable_mortal.compile()
                stable_dqn.compile()

            self.baseline_engine = MortalEngine(
                stable_mortal,
                stable_dqn,
                is_oracle = bool(getattr(stable_mortal, 'is_oracle', False)),
                version = version,
                device = device,
                enable_amp = True,
                enable_rule_based_agari_guard = True,
                extra_features = base_extra_features,
                name = 'baseline',
            )
        elif self.champion_mode == 'pool':
            # 预加载池里的所有对手（各自一份 Brain/DQN，共享不了权重）
            dev = torch.device(config['baseline']['train']['device'])
            for p in self.champion_pool:
                st = torch.load(p, weights_only=True, map_location=torch.device('cpu'))
                c = st['config']
                ver = c['control'].get('version', 1)
                ef, ex = s6c_features.resolve(c['control'])
                b = Brain(version=ver, conv_channels=c['resnet']['conv_channels'],
                          num_blocks=c['resnet']['num_blocks'], extra_channels=ex).eval()
                d = DQN(version=ver).eval()
                b.load_state_dict(st['mortal'])
                d.load_state_dict(st['current_dqn'])
                self.pool_engines.append(MortalEngine(
                    b, d,
                    is_oracle = bool(getattr(b, 'is_oracle', False)),
                    version = ver,
                    device = dev,
                    enable_amp = True,
                    enable_rule_based_agari_guard = True,
                    extra_features = ef,
                    name = 'baseline',
                ))
                logging.info(f'[train_play] 对手池 += {p} (v{ver}, ef={ef!r})')
            logging.info(f'[train_play] 对手池共 {len(self.pool_engines)} 个模型')
        else:
            logging.info('[train_play] 纯四 Mortal：跳过 baseline 加载，champion 复用当前策略')

        self.chal_version = config['control']['version']
        # 挑战者（训练中的模型）用训练配置自己的特征族
        self.chal_extra_features, _ = s6c_features.resolve(config['control'])
        # 多 worker 隔离（2026-11-07 在线自博弈）：每个 client 启动时会 rmtree(log_dir)，
        # 多个 client 共用同一目录会互相删除对方刚写入的牌谱。用环境变量分流，
        # 不设时行为与原来完全一致。
        wid = os.environ.get('MORTAL_WORKER_ID', '')
        self.log_dir = path.abspath(cfg['log_dir'] + (f'_w{wid}' if wid else ''))
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        self.seed_count = cfg['games'] // 4
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        self.boltzmann_temp = cfg['boltzmann_temp']
        self.top_p = cfg['top_p']
        # top-k 探索（2026-11-09）：0 = 旧行为（在全部合法动作上采样）；>0 = 只在当前模型的
        # top-k 候选里抽。后者把反事实样本集中在"质量相近"的动作上。
        self.boltzmann_top_k = int(cfg.get('boltzmann_top_k', 0))
        if self.boltzmann_top_k > 0:
            logging.info(f'[train_play] top-k 探索：k={self.boltzmann_top_k}、'
                         f'eps={self.boltzmann_epsilon}、temp={self.boltzmann_temp}')

        self.repeats = cfg['repeats']
        self.repeat_counter = 0

    def train_play(self, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = bool(getattr(mortal, 'is_oracle', False)),
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            boltzmann_top_k = self.boltzmann_top_k,
            device = device,
            enable_amp = True,
            extra_features = self.chal_extra_features,
            name = 'trainee',
        )

        if self.champion_mode == 'self':
            # 与 challenger 共享同一份 (mortal, dqn) 的**引用**（不复制权重），
            # 因此权重自动同步；名字也设 'trainee' 才能让 4 席都进训练集。
            engine_champ = MortalEngine(
                mortal,
                dqn,
                is_oracle = bool(getattr(mortal, 'is_oracle', False)),
                version = self.chal_version,
                boltzmann_epsilon = self.champ_boltzmann_epsilon,
                boltzmann_temp = self.champ_boltzmann_temp,
                top_p = self.champ_top_p,
                device = device,
                enable_amp = True,
                extra_features = self.chal_extra_features,
                name = 'trainee',
            )
        else:
            engine_champ = self.baseline_engine

        if self.champion_mode == 'pool':
            engine_champ = random.choice(self.pool_engines)

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = engine_champ,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
