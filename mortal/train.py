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
_ENV_REWARD_KEYS = ('gamma', 'pts', 'reward_mode', 'reward_scale')
_ENV_REWARD_DEFAULTS = {'reward_mode': 'rank_pt', 'reward_scale': 1.0}


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
    from model import Brain, DQN, AuxNet
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
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(
        f'env: gamma={gamma} reward_mode={reward_mode} reward_scale={reward_scale} '
        f'pts={pts if reward_mode == "rank_pt" else "(unused in delta_points)"} '
        f'kan_select={always_include_kan_select}'
    )
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

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
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
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

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            assert masks[range(batch_size), actions].all()

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
                    for mn, mod in (('mortal', mortal), ('dqn', dqn), ('aux', aux_net))
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

        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                )
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