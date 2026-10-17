"""读牌模型训练脚本（方案 §5 / §6）。

损失（方案 §5，v1 权重）：
    L = L_alloc + 0.5·L_wait + 0.2·L_aux
      L_alloc  逐列计数交叉熵（unseen 掩码后的 logits），按样本权重加权平均
      L_wait   听牌分布交叉熵（多听牌时为听牌集上的均匀分布目标），仅听牌目标激活
      L_aux    0.1·下一打 CE + 0.05·向听 CE + 0.025·立直 BCE + 0.025·鸣牌 BCE

样本权重（方案 §5.1）：立直目标家 ×3；终盘（巡目 ≥ 12）×2；近和牌（听牌）×2；
开局前 3 巡 ×0.5。三者相乘。

课程学习（方案 §6.1，按阶段顺序切换数据集过滤器）：
    阶段 1  仅「有立直家」的样本
    阶段 2  加入中盘（巡目 ≥ 4）任意样本
    阶段 3  全量任意时点任意家

优化（方案 §6.3，遵循项目 AMP 事故后决定）：AdamW lr=3e-4 wd=1e-2，
LinearWarmUpCosineAnnealingLR（warmup 5%），fp32，按验证 log-loss 早停保存 best。

数据切分（方案 §2.4）：按月 holdout——训练 glob 与验证 glob 分开给，
验证集为固定文件 + 固定抽样（每次评测完全可复现）。

用法示例：
    python tools/read_model/train.py \
        --train-globs '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202[2-4]??/**/*.json.gz' \
        --val-globs   '/mnt/data8/mortal/Mortal/dataset/mjai/houou4/202510/**/*.json.gz' \
        --out-dir runs/read_model_v1 --steps 60000 --batch-size 512 \
        --stage-schedule 1:18000,2:18000,3:24000 --augment
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')  # libriichi 扩展（先插，权重最低）
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 本目录优先（避免 model 名冲突）

import dataset as D                                     # noqa: E402
import model as M                                       # noqa: E402
import model_v2 as M2                                   # noqa: E402
from constraints import compute_wait_bias               # noqa: E402

# 损失权重（方案 §5）
LAMBDA_WAIT = 0.5
LAMBDA_AUX = 0.2
W_NEXT_DISCARD, W_SHANTEN, W_RIICHI, W_CALLED = 0.1, 0.05, 0.025, 0.025
LAMBDA_STRUCT = 0.3                                     # v2 结构估计头权重


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class ReadIterDataset(IterableDataset):
    """流式样本：worker 切分文件 -> 回放（课程阶段在 obs 编码前过滤）-> yield。"""

    def __init__(self, file_list, stage: int = 3, augment: str = 'none', seed: int = 0):
        super().__init__()
        assert augment in ('none', 'random', 'all'), augment
        self.file_list = list(file_list)
        self.stage = stage
        self.augment = augment
        self.seed = seed

    def _augment_for(self, f: str, wid: int) -> bool:
        if self.augment == 'all':
            return True
        if self.augment == 'none':
            return False
        # random：按 (seed, worker, 文件) 确定性哈希，约一半文件增广——
        # 万↔筒置换是游戏的对称性，混合口径与「全量+增广遍历两遍」统计等价。
        h = (hash((self.seed, wid, f)) & 0xffffffff)
        return (h % 2) == 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1
        files = D.worker_shard(self.file_list, wid, nw)
        rng = random.Random(self.seed * 100003 + wid)
        for f in D.iter_files(files, rng):
            try:
                samples, _ = D.replay_file(f, augment=self._augment_for(f, wid),
                                           stage=self.stage)
            except Exception as exc:                 # 坏文件不中断训练
                print(f'[dataset] skip {f}: {exc}', flush=True)
                continue
            yield from samples


def collate(samples):
    batch = {}
    for key in ('obs', 'unseen', 'alloc', 'wait', 'next_discard', 'shanten',
                'riichi', 'called', 'turn', 'tiles_left', 'tenpai',
                'river', 'post_riichi_river', 'own_hand'):
        batch[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
    batch['alloc'] = batch['alloc'].long()
    batch['next_discard'] = batch['next_discard'].long()
    batch['shanten'] = batch['shanten'].long()
    if 'tl_tokens' in samples[0]:                        # v2 字段
        for key in ('tl_tokens', 'tl_mask', 'struct_form', 'struct_counts'):
            batch[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
    return batch


def build_fixed_eval_set(files, n_samples: int, seed: int = 20261016):
    """固定验证集：固定文件顺序 + 固定回放（无增广），超过容量则均匀抽稀。"""
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)
    collected = []
    for f in files:
        try:
            samples, _ = D.replay_file(f, augment=False)
        except Exception as exc:
            print(f'[eval] skip {f}: {exc}', flush=True)
            continue
        collected.extend(samples)
        if len(collected) >= n_samples:
            break
    if len(collected) > n_samples:
        idx = np.linspace(0, len(collected) - 1, n_samples).astype(int)
        collected = [collected[i] for i in idx]
    if not collected:
        raise RuntimeError('验证集为空：检查 --val-globs')
    print(f'[eval] 固定验证集 {len(collected)} 样本（来自 {len(files)} 个候选文件）', flush=True)
    return collate(collected)


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def sample_weights(batch) -> torch.Tensor:
    """方案 §5.1：立直 ×3、终盘(巡目>=12) ×2、听牌 ×2、开局(<=3 巡) ×0.5。"""
    turn = batch['turn'].float()                              # [B]
    base = torch.ones_like(turn)
    base = torch.where(turn >= 12, torch.full_like(base, 2.0), base)
    base = torch.where(turn <= 3, torch.full_like(base, 0.5), base)
    w = base[:, None] * (1.0 + 2.0 * batch['riichi']) * (1.0 + 2.0 * batch['tenpai'])
    return w                                                # [B,3]


def compute_loss(out, batch):
    w = sample_weights(batch)                                # [B,3]

    # ---- L_alloc：逐列计数 CE（logits 已按 unseen 掩码）----
    logp = torch.log_softmax(out['alloc_logits'], dim=-1)    # [B,3,34,5]
    ll = logp.gather(-1, batch['alloc'][..., None]).squeeze(-1)   # [B,3,34]
    denom = (w.sum() * 34.0).clamp(min=1e-8)
    l_alloc = -(ll * w[..., None]).sum() / denom

    # ---- L_wait：听牌分布 CE，仅听牌目标 ----
    # 注意：目标分布的零质量类会碰上硬掩码的 -inf logit，0 * -inf = NaN，
    # 必须先把零质量位置清零再求和（只累加目标支撑集上的对数概率）。
    logw = torch.log_softmax(out['wait_logits'], dim=-1)     # [B,3,35]
    ll_wait = -(batch['wait'] * torch.where(
        batch['wait'] > 0, logw, torch.zeros_like(logw))).sum(-1)   # [B,3]
    active = batch['tenpai'] > 0.5
    if active.any():
        wa = w * active.float()
        # ll_wait 已是正值的 CE（见上），直接加权平均，勿再取负
        l_wait = (ll_wait * wa).sum() / wa.sum().clamp(min=1e-8)
    else:
        l_wait = out['wait_logits'].sum() * 0.0

    # ---- L_aux ----
    log_next = torch.log_softmax(out['next_logits'], dim=-1)
    l_next = -log_next.gather(-1, batch['next_discard'][..., None]).squeeze(-1).mean()
    log_sh = torch.log_softmax(out['shanten_logits'], dim=-1)
    l_sh = -log_sh.gather(-1, batch['shanten'][..., None]).squeeze(-1).mean()
    l_riichi = nn.functional.binary_cross_entropy_with_logits(
        out['riichi_logit'], batch['riichi'])
    l_called = nn.functional.binary_cross_entropy_with_logits(
        out['called_logit'], batch['called'])
    l_aux = (W_NEXT_DISCARD * l_next + W_SHANTEN * l_sh
             + W_RIICHI * l_riichi + W_CALLED * l_called)

    total = l_alloc + LAMBDA_WAIT * l_wait + LAMBDA_AUX * l_aux
    parts = {
        'loss': float(total.detach()),
        'alloc': float(l_alloc.detach()),
        'wait': float(l_wait.detach()),
        'aux': float(l_aux.detach()),
        'next': float(l_next.detach()),
        'shanten': float(l_sh.detach()),
    }

    # ---- L_struct（v2）：搭子构造软分布 + 形制 ----
    if 'struct_counts' in batch:
        # 软 CE：标签是最小向听分解上的边际直方图（保留真实歧义），
        # 对 7 个结构计数分别求 -Σ_k p_k log q_k 再平均。
        logc = torch.log_softmax(out['struct_count_logits'], dim=-1)   # [B,3,7,5]
        ll_c = -(batch['struct_counts'] * torch.where(
            batch['struct_counts'] > 0, logc, torch.zeros_like(logc))).sum(-1)  # [B,3,7]
        l_counts = ll_c.mean()
        l_form = nn.functional.cross_entropy(
            out['struct_form_logits'].reshape(-1, out['struct_form_logits'].shape[-1]),
            batch['struct_form'].reshape(-1))
        l_struct = l_counts + l_form
        total = total + LAMBDA_STRUCT * l_struct
        parts.update({'struct': float(l_struct.detach()),
                      'struct_counts': float(l_counts.detach()),
                      'struct_form': float(l_form.detach())})
        parts['loss'] = float(total.detach())
    return total, parts


def wait_bias_of(batch) -> torch.Tensor:
    """从 collate 后的 batch 计算 H2 加性对数偏置（与 constraints 同口径）。"""
    def cpu(x):
        return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    bias = compute_wait_bias(
        cpu(batch['unseen']),
        cpu(batch['river']),
        cpu(batch['post_riichi_river']),
        cpu(batch['riichi']) > 0.5,
        own_hand=cpu(batch['own_hand']),
    )
    return torch.from_numpy(bias)


def forward_batch(net, batch, device):
    """按模型版本分发前向（v1: 三输入；v2: 加时序 token/mask）。"""
    wb = wait_bias_of(batch).to(device)
    if isinstance(net, M2.ReadNetV2):
        return net(batch['obs'].to(device), batch['unseen'].to(device), wb,
                   batch['tl_tokens'].to(device), batch['tl_mask'].to(device))
    return net(batch['obs'].to(device), batch['unseen'].to(device), wb)


@torch.no_grad()
def evaluate(net, eval_batch, device):
    net.eval()
    n = eval_batch['obs'].shape[0]
    bs = 2048
    alloc_ll, wait_ll, alloc_top1, wait_hit1 = [], [], [], []
    for i in range(0, n, bs):
        chunk = {k: v[i:i + bs].to(device) for k, v in eval_batch.items()}
        o = forward_batch(net, chunk, device)
        logp = torch.log_softmax(o['alloc_logits'], dim=-1)
        ll = logp.gather(-1, chunk['alloc'][..., None]).squeeze(-1)
        alloc_ll.append((-ll).mean(-1))                      # [b,3] 逐列 CE 均值
        alloc_top1.append((logp.argmax(-1) == chunk['alloc']).float().mean(-1))
        logw = torch.log_softmax(o['wait_logits'], dim=-1)
        # 与训练损失同口径：只累加目标支撑集（避免 0 * -inf = NaN）
        wait_ll.append(-(chunk['wait'] * torch.where(
            chunk['wait'] > 0, logw, torch.zeros_like(logw))).sum(-1))   # [b,3]
        top1 = logw[..., :D.NUM_TILES].argmax(-1)          # [b,3]
        true_wait = chunk['wait'][..., :D.NUM_TILES] > 0
        hit = (true_wait.gather(-1, top1[..., None]).squeeze(-1) > 0) & (chunk['tenpai'] > 0.5)
        wait_hit1.append(hit.float())
    net.train()
    alloc_ll = torch.cat(alloc_ll)                          # [n,3]
    wait_ll = torch.cat(wait_ll)                            # [n,3]
    alloc_top1 = torch.cat(alloc_top1)
    wait_hit1 = torch.cat(wait_hit1)
    tenpai_mask = eval_batch['tenpai'] > 0.5
    wait_ce_active = float(wait_ll[tenpai_mask].mean()) if tenpai_mask.any() else float('nan')
    wait_hit = float(wait_hit1[tenpai_mask].mean()) if tenpai_mask.any() else float('nan')
    return {
        'alloc_ce': float(alloc_ll.mean()),
        'alloc_top1': float(alloc_top1.mean()),
        'wait_ce_active': wait_ce_active,
        'wait_hit1': wait_hit,
        'tenpai_frac': float(tenpai_mask.float().mean()),
    }


# --------------------------------------------------------------------------- #
# 训练主循环
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-globs', nargs='+', required=True)
    ap.add_argument('--val-globs', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--steps', type=int, default=60000, help='总训练步数（会被阶段表覆盖）')
    ap.add_argument('--stage-schedule', default='1:0.3,2:0.3,3:0.4',
                    help='阶段:步数或比例，如 1:18000,2:18000,3:24000 或 1:0.3,2:0.3,3:0.4')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--warmup-frac', type=float, default=0.05)
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--augment', choices=('none', 'random', 'all'), default='none',
                    help='万/筒增广：all=全部文件；random=约一半文件（对称性混合口径）；none=不增广')
    ap.add_argument('--conv-channels', type=int, default=96)
    ap.add_argument('--num-blocks', type=int, default=12)
    ap.add_argument('--pool', choices=('mean', 'attention'), default='mean',
                    help='全局池化口径（方案 §4.1 两个选项；attention 保留逐列重要性信息）')
    ap.add_argument('--model', choices=('v1', 'v2'), default='v1',
                    help='v1=单帧；v2=空间+手切时序+结构估计三流融合')
    ap.add_argument('--no-aux', action='store_true')
    ap.add_argument('--eval-samples', type=int, default=8192)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--max-files', type=int, default=0, help='>0 时限制训练文件数（冒烟用）')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train_log.jsonl'

    train_files = []
    for g in args.train_globs:
        train_files.extend(D.glob_files(g))
    train_files = sorted(set(train_files))
    rng.shuffle(train_files)
    if args.max_files:
        train_files = train_files[:args.max_files]
    val_files = []
    for g in args.val_globs:
        val_files.extend(D.glob_files(g))
    val_files = sorted(set(val_files))
    print(f'[data] 训练文件 {len(train_files)}，验证文件 {len(val_files)}', flush=True)
    if not train_files:
        raise SystemExit('没有训练文件')

    eval_batch = build_fixed_eval_set(val_files, args.eval_samples)

    # 阶段步数解析（比例或绝对步数；末段吸收 rounding 残差）
    total_steps = args.steps
    phases = []
    for part in args.stage_schedule.split(','):
        stage_s, val_s = part.split(':')
        stage = int(stage_s)
        if '.' in val_s:
            n = int(float(val_s) * total_steps)
        else:
            n = int(val_s)
        phases.append([stage, n])
    residual = total_steps - sum(n for _, n in phases)
    phases[-1][1] += residual
    phases = [(s, n) for s, n in phases]
    print(f'[plan] 阶段表 {phases}（总 {total_steps} 步）', flush=True)

    if args.model == 'v2':
        net = M2.build_model_v2(conv_channels=args.conv_channels,
                               num_blocks=args.num_blocks,
                               with_aux=not args.no_aux).to(device)
    else:
        net = M.build_model(conv_channels=args.conv_channels, num_blocks=args.num_blocks,
                            with_aux=not args.no_aux, pool=args.pool).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'[model] {args.model} | 参数量 {n_params:,} | device={device} | fp32 | '
          f'channels={args.conv_channels} blocks={args.num_blocks} aux={not args.no_aux} '
          f'pool={args.pool if args.model == "v1" else "-"}', flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_step = 0
    best_val = float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        net.load_state_dict(state['model'])
        if 'optimizer' in state:
            opt.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            sched.load_state_dict(state['scheduler'])
        start_step = int(state.get('step', 0))
        best_val = float(state.get('best_val', float('inf')))
        print(f'[resume] {args.resume} @ step {start_step}, best_val {best_val:.6f}', flush=True)

    def save(tag, step, val):
        state = {
            'model': net.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
            'step': step,
            'best_val': val,
            'config': vars(args),
        }
        torch.save(state, out_dir / f'{tag}.pth')

    step = start_step
    end_step = start_step + total_steps
    t0 = time.time()
    log_f = open(log_path, 'a')
    for stage, n_steps in phases:
        if step >= end_step:
            break
        ds = ReadIterDataset(train_files, stage=stage, augment=args.augment,
                             seed=args.seed + stage)
        loader = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=collate, prefetch_factor=2, persistent_workers=True,
        )
        it = iter(loader)
        done_in_phase = 0
        while done_in_phase < n_steps and step < end_step:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)                 # 数据走完一遍，重新洗牌再来
                try:
                    batch = next(it)
                except StopIteration:
                    print('[data] 训练集过小，无法凑满一批', flush=True)
                    break
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = forward_batch(net, batch, device)
            loss, parts = compute_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            done_in_phase += 1

            if step % args.log_every == 0:
                rec = {'step': step, 'stage': stage, 'lr': sched.get_last_lr()[0],
                       'elapsed': round(time.time() - t0, 1), **parts}
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                print(f"step {step} (stage {stage}) loss {parts['loss']:.4f} "
                      f"alloc {parts['alloc']:.4f} wait {parts['wait']:.4f} "
                      f"aux {parts['aux']:.4f} lr {rec['lr']:.2e} "
                      f"[{rec['elapsed']}s]", flush=True)

            if step % args.eval_every == 0 or step == end_step:
                metrics = evaluate(net, eval_batch, device)
                wait_ce = metrics['wait_ce_active']
                val_score = metrics['alloc_ce'] + (0.5 * wait_ce if wait_ce == wait_ce else 0.0)
                rec = {'step': step, 'stage': stage, 'eval': metrics, 'val_score': val_score}
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                print(f"[eval] step {step}: alloc_ce {metrics['alloc_ce']:.4f} "
                      f"top1 {metrics['alloc_top1']:.4f} wait_ce {wait_ce:.4f} "
                      f"wait_hit1 {metrics['wait_hit1']:.4f} "
                      f"tenpai_frac {metrics['tenpai_frac']:.3f}", flush=True)
                if val_score < best_val:
                    best_val = val_score
                    save('best', step, best_val)
                    print(f'[eval] new best val_score {best_val:.6f} -> best.pth', flush=True)
                save('last', step, best_val)
        print(f'[phase] stage {stage} 完成 {done_in_phase} 步', flush=True)

    save('last', step, best_val)
    print(f'[done] 总步数 {step}，用时 {time.time() - t0:.0f}s，best_val {best_val:.6f}',
          flush=True)
    log_f.close()


if __name__ == '__main__':
    main()
