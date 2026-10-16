# ---- file_index 的 manifest 机制（2026-10-12）--------------------------------
# 背景：旧版索引只存 file_list，train.py 发现文件存在就直接复用 → 换白名单、扩 globs、
# 修数据后若沿用旧索引名，会**静默**使用过期的文件集合（MEMORY §14 把它列为坑位，但只靠
# 人肉纪律保证）。manifest 记录「这份文件表是在什么条件下构建的」，加载时逐项比对，
# 不一致就自动重建（重建 813k 文件约 1~2 分钟，代价远小于拿错数据集训一整轮）。
#
# 与数据修复的配合：`tools/tenhou_data/repair_names.py` 把 2026 缺 names 的牌谱补回后，
# 把 `dataset.data_version` 改成新值（如 'names-fix-v1'）即可让所有旧索引自动失效。
FILE_INDEX_VERSION = 2          # 1 = 仅 file_list（旧格式）；2 = 带 manifest
_INDEX_MANIFEST_KEYS = (        # 只比对**会改变文件集合**的键；纯记录项不参与比对
    'index_version',
    'globs',
    'player_names_files',
    'excludes_files',
    'data_version',
)


def _sha256_of_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_index_manifest(config, player_names_set, enable_augmentation=False, augmented_first=False):
    """记录 file_index 的构建条件（供下次启动校验）。"""
    from datetime import datetime
    ds = config['dataset']
    names_files = {}
    for fn in ds.get('player_names_files', []):
        try:
            names_files[fn] = _sha256_of_file(fn)
        except OSError as exc:
            names_files[fn] = f'ERROR:{type(exc).__name__}'
    try:
        import libriichi
        libriichi_version = getattr(libriichi, '__version__', '')
    except Exception:
        libriichi_version = ''
    return {
        # --- 参与比对（会改变文件集合）---
        'index_version': FILE_INDEX_VERSION,
        'globs': list(ds['globs']),
        'player_names_files': names_files,
        'excludes_files': {},          # 预留：excludes 尚未接入主训练
        'data_version': ds.get('data_version', ''),
        # --- 仅记录，不参与比对 ---
        'player_names_count': len(player_names_set),
        'enable_augmentation': bool(enable_augmentation),
        'augmented_first': bool(augmented_first),
        'libriichi_version': libriichi_version,
        'built_at': datetime.now().isoformat(timespec='seconds'),
    }


def manifest_matches(saved, current):
    """索引里的 manifest 是否与当前配置一致（None/旧格式一律视为不一致）。"""
    if not isinstance(saved, dict):
        return False
    return all(saved.get(k) == current.get(k) for k in _INDEX_MANIFEST_KEYS)


def manifest_diff(saved, current):
    """给出参与比对键的差异摘要（写日志用）。"""
    if not isinstance(saved, dict):
        return '旧格式索引（无 manifest）'
    diffs = []
    for k in _INDEX_MANIFEST_KEYS:
        a, b = saved.get(k), current.get(k)
        if a == b:
            continue
        if k == 'player_names_files' and isinstance(a, dict) and isinstance(b, dict):
            changed = sorted(fn for fn in set(a) | set(b) if a.get(fn) != b.get(fn))
            diffs.append(f'{k}: {changed}')
        else:
            diffs.append(f'{k}: {a!r} -> {b!r}')
    return '; '.join(diffs) if diffs else '(无差异)'


# ---- 训练目标（奖励口径）一致性守卫（2026-10-12）------------------------------
# 奖励定义写进 checkpoint 的 config 里，但它决定 dqn_loss 的目标值：续训时若只改配置不改
# state_file，会把"两种目标"混进同一条损失曲线（例如 rank_pt → delta_points、改 pts、改 gamma）。
# 这里只告警不中止（对照组实验可能故意如此），让日志里能看见。
_ENV_REWARD_KEYS = ('gamma', 'pts', 'reward_mode', 'reward_scale', 'credit_mode')
_ENV_REWARD_DEFAULTS = {'reward_mode': 'rank_pt', 'reward_scale': 1.0, 'credit_mode': 'kyoku'}


def env_reward_diff(saved_env, current_env):
    """比较 checkpoint 记录的 env 与当前配置的奖励相关键；无差异返回 ''。"""
    diffs = []
    for k in _ENV_REWARD_KEYS:
        default = _ENV_REWARD_DEFAULTS.get(k)
        a = saved_env.get(k, default) if isinstance(saved_env, dict) else default
        b = current_env.get(k, default) if isinstance(current_env, dict) else default
        if k == 'reward_scale':
            try:
                a = float(a) if a is not None else None
                b = float(b) if b is not None else None
            except (TypeError, ValueError):
                pass
        if a != b:
            diffs.append(f'{k}: {a!r} -> {b!r}')
    return '; '.join(diffs)


def optimizer_reset_required(saved_token, current_token):
    """一次性判断是否跳过旧 optimizer/scheduler（用于奖励目标切换的干净暖启）。"""
    return current_token is not False and current_token != saved_token


def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet, VHead
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    # 训练目标族（2026-10-15，IQL Stage B）：'cql_mc' = 现状（MC/CQL 回归，默认，与 env.credit_mode
    # 搭配）；'iql' = 真 IQL（next_obs bootstrap + expectile V + AWR 政策提取，CQL 关闭）。
    # 两键解耦：iql 模式下忽略 credit_mode（未来回报由 V(s') bootstrap 接管）。
    objective = config['control'].get('objective', 'cql_mc')
    assert objective in ('cql_mc', 'iql'), f'未知 objective: {objective}'
    iql_tau = float(config['control'].get('iql_tau', 0.7))      # expectile 分位
    iql_beta = float(config['control'].get('iql_beta', 3.0))    # AWR 优势温度
    ema_decay = float(config['control'].get('ema_decay', 0.995))
    # L_Q bootstrap 的 V 是否改用 EMA target（10-16 加，IQL v2 价值尺度漂移修复的待测臂）：
    # 0.0 = 关（用在线 V，IQL 原文 Algorithm 1 口径）；>0 = VHead 的 Polyak EMA（如 0.99）。
    # 动机：v2 实测 q_target 中位数 105k 步内 +0.6→+61 纯指数发散（超理论界 [−5,+5] ~12 倍），
    # 机制为 expectile 不对称放大 × γ≈1 × 边界稀疏奖励下的 bootstrap 反馈增益 >1（MEMORY 会话 25）。
    iql_v_target_decay = float(config['control'].get('iql_v_target_decay', 0.0))
    assert 0.0 <= iql_v_target_decay < 1.0, f'iql_v_target_decay 应在 [0,1)：{iql_v_target_decay}'
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0
    max_train_steps = config['control'].get('max_train_steps', 0)
    snapshot_every = config['control'].get('snapshot_every', test_every)
    assert snapshot_every % save_every == 0
    assert max_train_steps == 0 or max_train_steps % save_every == 0

    seed = config['control'].get('seed')
    data_generator = None
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        data_generator = torch.Generator().manual_seed(int(seed))
        logging.info(f'training seed: {seed}')

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    gamma = config['env']['gamma']
    # 奖励口径（2026-10-12，对应 NEXT_STEPS_ANALYSIS_20261012.md §3.1）：
    # 默认 'rank_pt' 保持历史行为；'delta_points' 用每局真实分数差（点），配 reward_scale 归一化。
    reward_mode = config['env'].get('reward_mode', 'rank_pt')
    reward_scale = float(config['env'].get('reward_scale', 1.0))
    # 信用分配口径（2026-10-14，IQL/Bellman 改造 Stage A）：'kyoku' = 现状（只学本局奖励，
    # train_batch 里乘 γ^steps_to_done）；'game' = 跨局折扣回报（dataloader 已做后缀和，
    # 此处不再乘 γ^steps_to_done）。γ 在 'game' 模式下语义迁移为跨局折扣。
    credit_mode = config['env'].get('credit_mode', 'kyoku')
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    # 与部署侧 quick-eval 的 kan-select 口径对照用（默认 True = GameplayLoader 默认，等价历史行为）
    always_include_kan_select = bool(config['dataset'].get('always_include_kan_select', True))
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    if objective == 'iql':
        v_net = VHead().to(device)
        all_models = (mortal, dqn, aux_net, v_net)
    else:
        v_net = None
        all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(
        f'env: gamma={gamma} reward_mode={reward_mode} reward_scale={reward_scale} '
        f'credit_mode={credit_mode} objective={objective} '
        f'pts={pts if reward_mode == "rank_pt" else "(unused in delta_points)"} '
        f'kan_select={always_include_kan_select}'
    )
    if objective == 'iql':
        logging.info(
            f'iql: per-step gamma={gamma} tau={iql_tau} beta={iql_beta} ema_decay={ema_decay} '
            f'v_target_decay={iql_v_target_decay} '
            '(Q̄ 用 EMA 网络；CQL 关闭；未来回报由 V(s\') bootstrap 接管)'
        )
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')
    if objective == 'iql':
        logging.info(f'v_net params: {parameter_count(v_net):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    # 哨兵值取最差档（4 人局 avg_rank 4 / avg_pt −135），保证首次评测必定为新纪录
    init_best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }
    best_perf = dict(init_best_perf)

    # `best_perf` 只在「同一个卫冕者」下可比：test_play 的对手（baseline.test.state_file）一旦换代，
    # 历史阈值就失去意义（例如第 1 轮 vs 随机权重刷出的 avg_rank=1.0 / avg_pt=90 饱和值，
    # 换成真实卫冕者后永不满足 better 条件 → best.pth 永久停写）。因此记下换代标识并在变更时重置。
    #
    # 手动开关 `control.reset_best_perf` 是**一次性令牌**：任何非 false 值（true 或字符串）只在
    # 它与 state 里记录的上次令牌不同时生效一次（存盘时会把令牌记进 state）；想再次强制重置就
    # 换一个值（如日期字符串）。这样即使忘了删开关，也不会每次重启都把 best.pth 的竞争重置。
    champion_baseline = config['baseline']['test']['state_file']
    reset_token = config['control'].get('reset_best_perf', False)

    steps = 0
    loaded_state = None
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        loaded_state = state
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        saved_baseline = state.get('best_perf_baseline')
        saved_token = state.get('best_perf_reset_token', False)
        reset_optimizer_token = config['control'].get('reset_optimizer', False)
        reset_optimizer = optimizer_reset_required(
            state.get('optimizer_reset_token', False), reset_optimizer_token)
        if not online or state['config']['control']['online']:
            if reset_optimizer:
                logging.warning(
                    f'跳过 checkpoint optimizer/scheduler（control.reset_optimizer={reset_optimizer_token!r}；'
                    '仅继承模型权重，避免不同奖励目标的动量混用）'
                )
            else:
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
        if reset_optimizer:
            scaler = GradScaler(device.type, enabled=enable_amp)
        else:
            scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']

        if reset_token is not False and reset_token != saved_token:
            reason = (f'config 里 control.reset_best_perf={reset_token!r} 与上次记录 '
                      f'{saved_token!r} 不同（一次性生效）')
        elif saved_baseline is None:
            reason = '旧 state 未记录当时的卫冕者（来源不明，保守重置）'
        elif path.abspath(saved_baseline) != path.abspath(champion_baseline):
            reason = f'卫冕者已更换: {saved_baseline} → {champion_baseline}'
        else:
            reason = ''
        if reason:
            logging.info(f'resetting best_perf {best_perf} → {init_best_perf}, reason: {reason}')
            best_perf = dict(init_best_perf)

        env_diff = env_reward_diff(state.get('config', {}).get('env', {}), config['env'])
        if env_diff:
            if reset_optimizer:
                logging.warning(
                    f'env 奖励口径与 checkpoint 记录不同（本次只继承模型权重，已重置优化器状态）: {env_diff}'
                )
            else:
                logging.warning(
                    f'env 奖励口径与 checkpoint 记录不同（续训会把两种目标混进同一条损失）: {env_diff}'
                    '；若要做净对照，请另建 state_file 或从同一父代重新暖启'
                )

    # EMA target 网络与 VHead 恢复（iql 专属；放在 online 权重加载之后，保证 EMA 初始=online）
    ema_mortal = ema_dqn = None
    v_ema = None
    if objective == 'iql':
        ema_mortal = Brain(version=version, **config['resnet']).to(device)
        ema_dqn = DQN(version=version).to(device)
        if iql_v_target_decay > 0:
            v_ema = VHead().to(device)
        saved = loaded_state if isinstance(loaded_state, dict) else {}
        need_keys = ['v_net', 'ema_mortal', 'ema_dqn']
        if all(k in saved for k in need_keys):
            v_net.load_state_dict(saved['v_net'])
            ema_mortal.load_state_dict(saved['ema_mortal'])
            ema_dqn.load_state_dict(saved['ema_dqn'])
            if v_ema is not None:
                # 旧 iql ckpt（v_target 关闭时存的）没有 v_ema 键 → 从 v_net 复制（EMA 初始=online）
                v_ema.load_state_dict(saved['v_ema'] if 'v_ema' in saved else saved['v_net'])
            logging.info('iql: 已从 checkpoint 恢复 v_net 与 EMA target 网络')
        else:
            ema_mortal.load_state_dict(mortal.state_dict())
            ema_dqn.load_state_dict(dqn.state_dict())
            if v_ema is not None:
                v_ema.load_state_dict(v_net.state_dict())
            logging.warning(
                'iql: 父代 ckpt 缺 v_net/EMA target 键 → VHead 随机初始化、EMA=online 复制'
                '（首次从 cql_mc 暖启时属预期，恰好 1 条）'
            )

    if not online and max_train_steps > 0 and steps >= max_train_steps:
        logging.info('training budget already reached')
        return

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    if objective == 'iql':
        stats = {
            'q_loss': 0,
            'v_loss': 0,
            'awr_loss': 0,
            'next_rank_loss': 0,
        }
    else:
        stats = {
            'dqn_loss': 0,
            'cql_loss': 0,
            'next_rank_loss': 0,
        }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            manifest = build_index_manifest(
                config, player_names_set,
                enable_augmentation = enable_augmentation,
                augmented_first = augmented_first,
            )
            file_list = None
            if path.exists(file_index):
                index = None
                try:
                    index = torch.load(file_index, weights_only=True)
                except Exception as exc:
                    logging.warning(f'file index 不可读（将重建）: {file_index} ({exc})')
                if index is not None:
                    saved = index.get('manifest') if isinstance(index, dict) else None
                    if manifest_matches(saved, manifest):
                        file_list = index['file_list']
                        logging.info(
                            f'file index manifest 校验通过，复用 {file_index}'
                            f'（{len(file_list):,} 文件，建于 {saved.get("built_at", "?")}）'
                        )
                    else:
                        logging.warning(
                            f'file index 与当前配置不一致 → 弃用并重建: {file_index}'
                            f'（{manifest_diff(saved, manifest)}）'
                        )
            if file_list is None:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                names_coverage = None
                if len(player_names_set) > 0:
                    filtered = []
                    n_with_names = n_without_names = 0
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            # 自建 2026 牌谱（未修复 names 的旧树）的 start_game 只有 rules、
                            # 没有 names，用 .get 避免 KeyError；这类文件在白名单模式下会被排除。
                            # 修复版数据见 tools/tenhou_data/repair_names.py（写入 names 后用
                            # tools/tenhou_data/verify_names.py 验收命中率）。
                            names = start.get('names') or []
                            if any(names):
                                n_with_names += 1
                            else:
                                n_without_names += 1
                            if not set(names).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                    # names 覆盖率是"白名单模式下能取到多少数据"的关键诊断量：
                    # 无 names 的文件永远 0 席位（曾让 2026 的 13 万局静默失效）
                    names_coverage = {'with_names': n_with_names, 'without_names': n_without_names}
                file_list.sort(reverse=True)
                manifest['n_files'] = len(file_list)
                manifest['names_coverage'] = names_coverage
                path.dirname(file_index) and os.makedirs(path.dirname(file_index), exist_ok=True)
                torch.save({'file_list': file_list, 'manifest': manifest}, file_index)
                logging.info(
                    f'file index 已写入 {file_index}（n_files={len(file_list):,}'
                    + (f', names_coverage={names_coverage}' if names_coverage else '') + '）'
                )
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
            reward_mode = reward_mode,
            reward_scale = reward_scale,
            credit_mode = credit_mode,
            gamma = gamma,
            objective = objective,
            always_include_kan_select = always_include_kan_select,
        )
        logging.info(f'reward: {file_data.reward_desc()}')
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
            generator = data_generator,
        ))

        remaining = []      # 每元素 = 一个不满批的字段元组（字段数由 objective 决定）
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, *rest):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            if objective == 'iql':
                next_obs, terminal, r_t, player_ranks = rest
            else:
                steps_to_done, kyoku_rewards, player_ranks = rest

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            assert masks[range(batch_size), actions].all()

            if objective == 'iql':
                # Stage B（真 IQL）：next_obs bootstrap + expectile V + AWR 政策提取（CQL 关闭）。
                # 目标端全部 stopgrad：L_Q 的 V(s')、L_V 的 Q̄（EMA 网络）、L_AWR 的优势权重 w。
                # EMA/next 前向都套 no_grad：省激活内存，且避免 EMA/next 侧的反向图。
                # 注：EMA 与 online 均处 train 模式（BN 用批统计，与 online Q 的归一化口径一致；
                # EMA 的 BN buffer 每步都会被 online 覆盖，自身统计无意义）。
                next_obs = next_obs.to(dtype=torch.float32, device=device)
                terminal = terminal.to(dtype=torch.bool, device=device)
                r_t = r_t.to(dtype=torch.float32, device=device)

                with torch.autocast(device.type, enabled=enable_amp):
                    phi = mortal(obs)
                    with torch.no_grad():
                        # L_Q 的 bootstrap 目标：默认用在线 V（IQL 原文口径）；iql_v_target_decay>0
                        # 时改用 V 的 EMA target（打断 V→Q̄→V 的瞬时正反馈，v2 价值漂移的待测修复）
                        v_src = v_ema if v_ema is not None else v_net
                        v_next = v_src(mortal(next_obs)).squeeze(-1)
                    q_target_mc = torch.where(terminal, r_t, r_t + gamma * v_next)

                    q_out = dqn(phi, masks)
                    q = q_out[range(batch_size), actions]
                    q_loss = 0.5 * mse(q, q_target_mc)

                    with torch.no_grad():
                        q_bar_all = ema_dqn(ema_mortal(obs), masks)
                    v = v_net(phi).squeeze(-1)
                    # L_V 严格按 IQL（Kostrikov et al. 2021 式 9）：期望只取**数据里实际执行的动作**
                    # 的 Q̄(s,a)。⚠ 10-15 修正：首版对**全部合法动作**（~30 个从未被监督的 Q 值）
                    # 取 τ-expectile，使 V 被未约束的 Q 外拖走，再经 V(s') 反馈进 Q 目标自激漂移
                    #（r_t 双计修正后 logsumexp_q 仍 −33.6、top1 0.6975，见 MEMORY §16.26）。
                    q_bar = q_bar_all[range(batch_size), actions].detach()
                    diff = q_bar - v
                    tau_w = torch.where(diff < 0, 1.0 - iql_tau, torch.full_like(diff, iql_tau))
                    v_loss = (tau_w * diff ** 2).mean()

                    with torch.no_grad():
                        w = torch.exp((q.detach() - v.detach()) / iql_beta)
                        w = w / w.mean().clamp_min(1e-8)
                    awr_loss = -(w * q_out.log_softmax(dim=-1)[range(batch_size), actions]).mean()

                    next_rank_logits, = aux_net(phi)
                    next_rank_loss = ce(next_rank_logits, player_ranks)

                    loss = sum((
                        q_loss,
                        v_loss,
                        awr_loss,
                        next_rank_loss * next_rank_weight,
                    ))
            else:
                steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
                kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)

                if credit_mode == 'game':
                    # 'game' 模式：dataloader 已把跨局折扣回报（后缀和）放进 kyoku_rewards，
                    # γ^steps_to_done 的局内折扣不再适用（γ 的语义已迁移为跨局折扣）。
                    q_target_mc = kyoku_rewards.to(torch.float32)
                else:
                    q_target_mc = gamma ** steps_to_done * kyoku_rewards
                    q_target_mc = q_target_mc.to(torch.float32)

                with torch.autocast(device.type, enabled=enable_amp):
                    phi = mortal(obs)
                    q_out = dqn(phi, masks)
                    q = q_out[range(batch_size), actions]
                    dqn_loss = 0.5 * mse(q, q_target_mc)
                    cql_loss = 0
                    if not online:
                        cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                    next_rank_logits, = aux_net(phi)
                    next_rank_loss = ce(next_rank_logits, player_ranks)

                    loss = sum((
                        dqn_loss,
                        cql_loss * min_q_weight,
                        next_rank_loss * next_rank_weight,
                    ))

            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                if objective == 'iql':
                    stats['q_loss'] += q_loss
                    stats['v_loss'] += v_loss
                    stats['awr_loss'] += awr_loss
                else:
                    stats['dqn_loss'] += dqn_loss
                    if not online:
                        stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if objective == 'iql':
                    # EMA target：参数 polyak；BN buffer 每步直接跟随 online（自身统计无意义，见上）
                    with torch.no_grad():
                        for pe, p in zip(ema_mortal.parameters(), mortal.parameters()):
                            pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                        for be, b in zip(ema_mortal.buffers(), mortal.buffers()):
                            be.copy_(b)
                        for pe, p in zip(ema_dqn.parameters(), dqn.parameters()):
                            pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                        if v_ema is not None:
                            for pe, p in zip(v_ema.parameters(), v_net.parameters()):
                                pe.mul_(iql_v_target_decay).add_(p, alpha=1 - iql_v_target_decay)
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                if objective == 'iql':
                    writer.add_scalar('loss/q_loss', stats['q_loss'] / save_every, steps)
                    writer.add_scalar('loss/v_loss', stats['v_loss'] / save_every, steps)
                    writer.add_scalar('loss/awr_loss', stats['awr_loss'] / save_every, steps)
                    # 价值尺度监控（10-16 加，MEMORY 会话 25）：v2 曾因子指数漂移打废价值函数而
                    # 2k 冒烟无感——每个存盘点打一行分位数，供链脚本门禁与事后取证。
                    # 理论界：整场折扣回报 telescope 至 pts[final]−E[pts|start] ∈ [−5,+5]（B 口径）。
                    _qt = all_q_target[torch.isfinite(all_q_target)]
                    _qp = all_q[torch.isfinite(all_q)]
                    if _qt.numel():
                        logging.info(
                            f'iql scale: q_target p50={_qt.median():+.3f} p99={_qt.quantile(0.99):+.3f}'
                            f' | q_pred p50={_qp.median():+.3f}'
                            f' | v_ema={"on" if v_ema is not None else "off"}'
                        )
                else:
                    writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                    if not online:
                        writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                # 非有限值会让 TB 的 histogram 抛 ValueError 直接打死训练进程 → 先过滤再记
                for tag, arr in (('q_predicted', all_q_1d), ('q_target', all_q_target_1d)):
                    finite_arr = torch.as_tensor(arr)
                    finite_arr = finite_arr[torch.isfinite(finite_arr)].numpy()
                    if finite_arr.size:
                        writer.add_histogram(tag, finite_arr, steps)
                writer.flush()

                # ---- NaN/Inf 守卫（在清零 stats 与覆盖 checkpoint 之前）----
                # AMP 下 BatchNorm 的 running stats 在前向就会被污染，GradScaler 只能跳过
                # 参数更新、救不了 buffer；继续跑只会把坏状态写进所有产物。宁可停下来让人接手。
                nonfinite = [
                    f'{mn}.{k}'
                    for mn, mod in (
                        ('mortal', mortal), ('dqn', dqn), ('aux', aux_net)
                    ) + (
                        (('v_net', v_net), ('ema_mortal', ema_mortal), ('ema_dqn', ema_dqn))
                        + ((('v_ema', v_ema),) if v_ema is not None else ())
                        if objective == 'iql' else ()
                    )
                    for k, v in mod.state_dict().items()
                    if not torch.isfinite(v).all()
                ]
                nonfinite += [
                    f'loss_accum.{k}'
                    for k, v in stats.items()
                    if torch.is_tensor(v) and not torch.isfinite(v).all()
                ]
                if nonfinite:
                    logging.error(
                        f'non-finite state at step {steps:,}: refusing to overwrite '
                        f'{state_file} ({len(nonfinite)} tensors, e.g. {nonfinite[:6]}); '
                        f'可回滚到 {path.dirname(state_file)}/mortal_step_*.pth 中的干净快照'
                    )
                    writer.flush()
                    raise SystemExit(1)

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    **({
                        'v_net': v_net.state_dict(),
                        'ema_mortal': ema_mortal.state_dict(),
                        'ema_dqn': ema_dqn.state_dict(),
                        **({'v_ema': v_ema.state_dict()} if v_ema is not None else {}),
                    } if objective == 'iql' else {}),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    # 与 best_perf 配套的换代标识：下一次用不同卫冕者重启训练时可自动重置
                    'best_perf_baseline': champion_baseline,
                    # 上次生效的一次性令牌：同值的 reset_best_perf 不会重复触发重置
                    'best_perf_reset_token': reset_token,
                    'optimizer_reset_token': config['control'].get('reset_optimizer', False),
                    'config': config,
                }
                torch.save(state, state_file)

                if steps % snapshot_every == 0:
                    snapshot_file = path.join(
                        path.dirname(state_file),
                        f'mortal_step_{steps:09d}.pth',
                    )
                    shutil.copy2(state_file, snapshot_file)

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, dqn, device)
                    mortal.train()
                    dqn.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135]) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

                if not online and max_train_steps > 0 and steps >= max_train_steps:
                    pb.close()
                    writer.close()
                    logging.info('training budget reached')
                    raise SystemExit(0)

        for fields in data_loader:
            bs = fields[0].shape[0]
            if bs != batch_size:
                remaining.append(fields)
                remaining_bs += bs
                continue
            train_batch(*fields)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            stacked = tuple(torch.cat(parts, dim=0) for parts in zip(*remaining))
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(*(p[start:end] for p in stacked))
                start = end
                end += batch_size
        pb.close()

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass