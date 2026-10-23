"""未见月离线指标：判断「训练有没有在进步」的便宜监控量（秒级，不打牌局）。

对每个 checkpoint 在一批未见过的牌谱（默认 202609，主模型只训到 202608）上算：
  - top1      : masked Q 的 argmax 与人类实际动作的一致率（随机权重≈0.25 为下限）
  - dqn_loss  : 0.5·MSE(Q(s,a_人类), q_target)，与 const_loss（只预测常数）对比
  - R2        : 1 − MSE/var(target)；var_q/var_t 用来判断 Q 幅度是否高估
  - agari     : 和了动作(43) 被排为第一的比例（agari guard 的健全性）
用法：
  python holdout_metrics.py ckpt1.pth ckpt2.pth ...            # 用默认月份与批量
  python holdout_metrics.py --month 202610 --n 48 ckpt.pth     # 换月份/样本量
"""
import argparse
import glob
import hashlib
import random
import torch
from torch.utils.data import DataLoader
from config import config
from model import Brain, DQN
from dataloader import FileDatasetsIter, worker_init_fn
from train import env_reward_diff
import s6c_features

ROOT = '/mnt/data8/mortal/Mortal/dataset/mjai/houou4'
DEVICE = torch.device(config['control']['device'])


def collect(month, n_batch, bs, seed, workers):
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    # credit_mode/gamma/objective 必须与训练同口径，否则 q_target 不可比（2026-10-14/15 加）
    credit_mode = config['env'].get('credit_mode', 'kyoku')
    gamma = config['env']['gamma']
    objective = config['control'].get('objective', 'cql_mc')
    extra_features, _ = s6c_features.resolve(config['control'])
    files = sorted(glob.glob(f'{ROOT}/{month}/*.json.gz'))
    if not files:
        raise SystemExit(f'没有文件: {ROOT}/{month}')
    # --seed 真正参与抽样（否则排序后的切片是确定的，seed 形同虚设）
    random.Random(seed).shuffle(files)
    files = files[: max(n_batch * bs // 600 + 8, 64)]
    # 与训练相同：worker_init_fn 会把 file_list 按 worker 切分；不加它每个 worker
    # 都会各自完整遍历整个 file_list（重复采样 + 8 倍 IO）
    workers = max(1, min(workers, len(files) // 4))
    ds = FileDatasetsIter(
        version = config['control']['version'],
        file_list = files,
        pts = config['env']['pts'],
        file_batch_size = 8,
        reserve_ratio = 0.0,
        player_names = [],
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
        # 必须与训练同口径，否则 q_target 不可比（2026-10-12 加）
        reward_mode = config['env'].get('reward_mode', 'rank_pt'),
        reward_scale = config['env'].get('reward_scale', 1.0),
        credit_mode = credit_mode,
        gamma = gamma,
        objective = objective,
        always_include_kan_select = bool(config['dataset'].get('always_include_kan_select', True)),
        # S6c：holdout 必须与训练同口径（否则 obs 行数与首层不匹配）
        extra_features = extra_features,
    )
    it = iter(DataLoader(ds, batch_size = bs, drop_last = True,
                         num_workers = workers, pin_memory = True,
                         worker_init_fn = worker_init_fn, generator = generator))
    batches = []
    while len(batches) < n_batch:
        fields = next(it)
        if fields[0].shape[0] != bs:
            continue
        if objective == 'iql':
            # iql 目标 = r+γV(s')，非 MC 口径且依赖在线 V 网络 → R²/dqn_loss 不适用（置 None）
            obs, actions, masks = fields[0], fields[1], fields[2]
            q_target = None
        else:
            obs, actions, masks, std, kr, ranks = fields
            if credit_mode == 'game':
                # 'game' 模式：dataloader 已给出跨局折扣回报，不再乘局内 γ^steps_to_done
                q_target = kr.to(torch.float32)
            else:
                q_target = (gamma ** std.to(torch.int64)
                            * kr.to(torch.float64)).to(torch.float32)
        batches.append((obs, actions, masks, q_target))

    # 同一 seed 不仅应选到同一批文件，也应得到同一 worker 迭代顺序；把动作和目标
    # 的多重批次摘要打印出来，跨 checkpoint/进程复用时可直接核对。
    # obs 只摘要 v4 原生的前 obs_shape 行：S6c 等尾部注入行不应让「同一批数据」的
    # 摘要变化（否则手术前后无法用摘要对齐）。
    from libriichi.consts import obs_shape
    n_obs = obs_shape(config['control']['version'])[0]
    sample_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    sample_digest.update('\\n'.join(files).encode('utf-8'))
    for obs, actions, masks, q_target in batches:
        for tensor in (obs[:, :n_obs], actions, masks):
            sample_digest.update(tensor.contiguous().numpy().tobytes())
        if q_target is not None:
            target_digest.update(q_target.contiguous().numpy().tobytes())
    return batches, sample_digest.hexdigest()[:16], target_digest.hexdigest()[:16]


def evaluate(name, ckpt, batches, bs):
    st = torch.load(ckpt, map_location = torch.device('cpu'), weights_only = True)
    cfg = st['config']
    version = cfg['control'].get('version', 1)
    # S6c：额外通道数从 ckpt 自己的 [control] 推（旧 ckpt = 0）
    _, extra_channels = s6c_features.resolve(cfg['control'])
    mortal = Brain(version = version, extra_channels = extra_channels,
                   **cfg['resnet']).to(DEVICE).eval()
    dqn = DQN(version = version).to(DEVICE).eval()
    mortal.load_state_dict(st['mortal'])
    dqn.load_state_dict(st['current_dqn'])
    pq, pt = [], []
    agree = 0
    total = 0
    lse = 0.0
    n_agari = n_agari_hit = 0
    with torch.inference_mode():
        for obs, actions, masks, q_target in batches:
            n = obs.shape[0]
            total += n
            obs = obs.to(dtype = torch.float32, device = DEVICE)
            actions = actions.to(dtype = torch.int64, device = DEVICE)
            masks = masks.to(dtype = torch.bool, device = DEVICE)
            q_out = dqn(mortal(obs), masks)
            q = q_out[torch.arange(n, device = DEVICE), actions]
            if q_target is not None:
                pq.append(q.float().cpu())
                pt.append(q_target)
            argmax = q_out.argmax(-1)
            agree += int((argmax == actions).sum().item())
            lse += float(torch.logsumexp(q_out, dim = -1).mean().item())
            m = actions == 43
            c = int(m.sum().item())
            n_agari += c
            if c > 0:
                n_agari_hit += int((argmax[m] == 43).sum().item())
    if not pt:
        print(f'{name:<40} steps={str(st.get("steps", "-")):<8} '
              f'top1={agree / total:.4f}  agari_top1={n_agari_hit}/{n_agari}  '
              f'logsumexp_q={lse / len(batches):+.3f}  '
              f'(objective=iql：目标非 MC 口径，dqn_loss/R² 不适用)', flush = True)
        return
    q = torch.cat(pq)
    t = torch.cat(pt)
    mse = float(torch.mean((q - t) ** 2))
    var_t = float(t.var())
    var_q = float(q.var())
    const = float((t - t.mean()).pow(2).mean())
    print(f'{name:<40} steps={str(st.get("steps", "-")):<8} '
          f'top1={agree / total:.4f}  dqn_loss={0.5 * mse:.4f} (const={0.5 * const:.4f})  '
          f'R2={1 - mse / var_t:+.4f}  var_q/var_t={var_q:.4f}/{var_t:.4f}  '
          f'agari_top1={n_agari_hit}/{n_agari}  logsumexp_q={lse / len(batches):+.3f}', flush = True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs = '+')
    ap.add_argument('--month', default = '202609')
    ap.add_argument('--n', type = int, default = 24, help = '批数（每批 batch_size 条）')
    ap.add_argument('--seed', type = int, default = 20261009)
    ap.add_argument('--workers', type = int, default = None,
                    help = '数据 loader 进程数；跟训练共机时建议设小（2）以免抢 CPU')
    args = ap.parse_args()
    bs = config['control']['batch_size']
    workers = args.workers if args.workers is not None else config['dataset']['num_workers']
    batches, sample_digest, target_digest = collect(args.month, args.n, bs, args.seed, workers)
    print(f'holdout={args.month} seed={args.seed} batches={len(batches)} x bs={bs} '
          f'= {len(batches) * bs:,} 样本（同一批数据用于所有 checkpoint；换 seed 会换抽样）', flush = True)
    print(f'holdout_sample_digest={sample_digest} holdout_target_digest={target_digest}', flush = True)
    print(f'env: gamma={config["env"]["gamma"]} '
          f'reward_mode={config["env"].get("reward_mode", "rank_pt")} '
          f'reward_scale={config["env"].get("reward_scale", 1.0)} '
          f'credit_mode={config["env"].get("credit_mode", "kyoku")} '
          f'pts={config["env"]["pts"]}'
          '（q_target 由当前配置决定而非 checkpoint，跨配置比较旧 ckpt 时务必固定这里的口径）', flush = True)
    for ckpt in args.ckpts:
        evaluate(ckpt.split('/')[-1], ckpt, batches, bs)


main()
