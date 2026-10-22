import random
import torch
import numpy as np
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
        reward_mode = 'rank_pt',
        reward_scale = 1.0,
        credit_mode = 'kyoku',
        gamma = 1.0,
        objective = 'cql_mc',
        always_include_kan_select = True,
    ):
        """reward_mode（2026-10-12 加，对应 NEXT_STEPS_ANALYSIS_20261012.md §3.1）：

          'rank_pt'      —— GRP 预测的顺位概率 × pts 的每局期望点差（默认，等价历史行为）
          'delta_points' —— 每局**真实分数差**（由 grp.feature 各家分数与终局分算得，单位：点），
                            与顺位无关；量级约 ±1e4，通常配 reward_scale（如 1e-3）归一化，
                            以免 CQL 项与 lr 的相对尺度被打乱。

        credit_mode（2026-10-14 加，IQL/Bellman 改造 Stage A，见 memory/IQL_BELLMAN_DESIGN）：

          'kyoku' —— 现状：样本只携带**本局**的奖励 r_k（train.py 里再乘 γ^steps_to_done），
                     未来局的收益被完全丢弃（无跨局信用分配）。
          'game'  —— 跨局折扣回报：样本携带 kyoku_returns[k] = Σ_{j≥k} γ^(j−k) · r_j，
                     γ 取 env.gamma（跨局折扣，γ=1 时闭式为 pts[final_rank] − E_grp[pts|第 k 局开始]，
                     GRP 中途期望自动成为控制变量）。train.py 在此模式下不再乘 γ^steps_to_done。

        always_include_kan_select：对齐 libriichi GameplayLoader 的默认值（True）。部署侧
        quick-eval 只在多候选杠时才请求 kan-select 状态，改成 False 可复现部署侧口径（用于回放对照）。

        objective（2026-10-15 加，IQL Stage B）：

          'cql_mc' —— 现状：MC/CQL 回归（条目 [obs, actions, masks, steps_to_done, r, ranks]），
                      credit_mode 决定 r 是本局奖励还是跨局后缀和。
          'iql'    —— 真 IQL：条目 [obs, actions, masks, next_obs, terminal, r_t, ranks]。
                      next_obs = 下一决策点 obs（同局 obs[i+1]；跨局自然落到下一局首个决策点），
                      最后一条样本 terminal=True（next_obs 用自身占位，训练端丢弃 bootstrap 项）；
                      **r_t 仅局边界样本非零**（= kyoku 口径 r_k，每局只计一次），中间步 r=0，
                      未来价值由 V(s') bootstrap 传递。此模式下 credit_mode 被强制视为 'kyoku'。
        """
        super().__init__()
        assert reward_mode in ('rank_pt', 'delta_points'), f'未知 reward_mode: {reward_mode}'
        assert credit_mode in ('kyoku', 'game'), f'未知 credit_mode: {credit_mode}'
        assert objective in ('cql_mc', 'iql'), f'未知 objective: {objective}'
        if objective == 'iql' and credit_mode != 'kyoku':
            # iql 的未来回报由 bootstrap 接管，后缀和会双重计入（守卫而非静默改写）
            raise AssertionError("objective='iql' 要求 credit_mode='kyoku'（后缀和会与 bootstrap 双重计入）")
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.credit_mode = credit_mode
        self.gamma = float(gamma)
        self.objective = objective
        self.always_include_kan_select = always_include_kan_select
        self.iterator = None

    def reward_desc(self):
        """一行描述当前奖励口径（训练/评测必须一致，故打日志用同一来源）。"""
        if self.reward_mode == 'delta_points':
            desc = f'delta_points(real score diff, scale={self.reward_scale})'
        else:
            desc = f'rank_pt(pts={self.pts}, scale={self.reward_scale})'
        if self.objective == 'iql':
            desc += ', objective=iql(next_obs bootstrap, r_t=kyoku boundary only)'
        elif self.credit_mode == 'game':
            desc += f', credit=game(discounted suffix sum, gamma={self.gamma})'
        return desc

    def build_iter(self):
        # do not put it in __init__, it won't work on Windows
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
            always_include_kan_select = self.always_include_kan_select,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                final_scores = grp.take_final_scores()
                if self.reward_mode == 'delta_points':
                    # 每局真实分数差（点）：seq = [各局起始分…, 终局分]，相邻相减
                    kyoku_rewards = self.reward_calc.calc_delta_points(player_id, grp_feature, final_scores)
                else:
                    kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                if self.reward_scale != 1.0:
                    kyoku_rewards = kyoku_rewards * self.reward_scale

                if self.credit_mode == 'game':
                    # 跨局信用分配（Stage A）：沿局轴做折扣后缀和，把未来所有局的回报折现回
                    # 当前决策点。γ=1 时闭式为 pts[final_rank] − E_grp[pts|第 k 局开始]：逐局 r_j
                    # telescope 求和使 GRP 中间项精确抵消（无双重计入），终局处 GRP 期望收敛为
                    # one-hot，最后一局无需任何终端特判。注意 reward_scale 在此之前生效（线性缩放
                    # 与后缀和可交换）。
                    returns = np.zeros_like(kyoku_rewards)
                    acc = 0.0
                    for k in range(len(kyoku_rewards) - 1, -1, -1):
                        acc = kyoku_rewards[k] + self.gamma * acc
                        returns[k] = acc
                    kyoku_rewards = returns
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                if self.objective == 'iql':
                    # Stage B（真 IQL）：next 决策点 = 整场 obs 序列的 i+1（局内续接、跨局自然落到
                    # 下一局首个决策点，因为 obs/at_kyoku 序列跨局连续）；最后一条样本 terminal=True。
                    #
                    # ⚠ r_t **只在局边界样本非零**（dones[i]=True：下一样本跨入新一局，或整场最后一条）。
                    #   10-15 修正：首版误把 kyoku_rewards 发给该局**每一个**样本，Bellman 沿链把每局
                    #   奖励重复计入 ≈16 次（实测每局平均决策点数 16.00）→ Q 尺度爆到 logsumexp=-176、
                    #   top1 0.81→0.69、ladder δ=-31.04，Stage B 首轮结果无效（详 MEMORY §16.26）。
                    #   中间决策点 r=0，未来价值由 V(s') bootstrap 传递。
                    for i in range(game_size):
                        if i + 1 < game_size:
                            next_i, terminal_i = i + 1, False
                        else:
                            next_i, terminal_i = i, True   # 占位：训练端按 terminal 丢弃 bootstrap 项
                        r_t_i = kyoku_rewards[at_kyoku[i]] if dones[i] else 0.0
                        entry = [
                            obs[i],
                            actions[i],
                            masks[i],
                            obs[next_i],
                            terminal_i,
                            r_t_i,
                            player_ranks[at_kyoku[i] + 1],
                        ]
                        if self.oracle:
                            entry.insert(1, invisible_obs[i])
                        self.buffer.append(entry)
                else:
                    for i in range(game_size):
                        entry = [
                            obs[i],
                            actions[i],
                            masks[i],
                            steps_to_done[i],
                            kyoku_rewards[at_kyoku[i]],
                            player_ranks[at_kyoku[i] + 1],
                        ]
                        if self.oracle:
                            entry.insert(1, invisible_obs[i])
                        self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    # DataLoader 会为每个 worker 分配确定的 torch seed；同步 Python/NumPy 随机源，
    # 使 load_files() 内的 shuffle 在固定训练/holdout seed 下可复现。
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
