"""读牌模型训练脚本（方案 §5 / §6）。

损失（方案 §5，v1 权重）：
    L = L_alloc + 0.5·L_wait + 0.2·L_aux
      L_alloc  逐列计数交叉熵（unseen 掩码后的 logits），按样本权重加权平均
      L_wait   听牌分布交叉熵（多听牌时为听牌集上的均匀分布目标），仅听牌目标激活
      L_aux    0.1·下一打 CE + 0.15·向听 CE + 0.025·立直 BCE + 0.025·鸣牌 BCE
              （向听权重 v2s 起 0.05→0.15，×3；见下方「结构头专项」）

样本权重（方案 §5.1）：立直目标家 ×3；终盘（巡目 ≥ 12）×2；近和牌（听牌）×2；
开局前 3 巡 ×0.5。三者相乘。

结构头专项（v2s，README §0ter「下一步的主攻方向」）：
    L_struct = l_counts + l_form
               + w_sh·l_struct_shanten        # 结构头向听直接 CE（v2 完全缺失）
               + w_ch·l_consist_head          # 分解-向听一致性：向听头 ↔ 分解暗含向听
               + w_cd·l_consist_decomp        # 分解暗含软向听 ↔ 真标签
               + w_tt·l_tenpai_tie            # shanten==0 ⟺ 结构性听牌（稠密二值）
  背景：v2 里 StructureHead.shanten_head **无任何直接监督**（aux 头权重仅
  0.05×0.2），诊断向听准确率 25.7% 低于 28.4% 多数类基线；form 头 91.9% =
  多数类基线。向听/形制 CE 按批内频率逆平方根类均衡（--no-struct-balance 关）。

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
from structure import (FORM_NORMAL, MAX_SHANTEN_BUCKET, NUM_FORMS,   # noqa: E402
                       NUM_SHANTEN_BUCKETS, soft_shanten_from_counts)

# 损失权重（方案 §5）
LAMBDA_WAIT = 0.5
LAMBDA_AUX = 0.2
W_NEXT_DISCARD = 0.1
W_SHANTEN = 0.15                                        # v2s 起 ×3（0.05→0.15）：向听估计是主攻方向
W_RIICHI, W_CALLED = 0.025, 0.025
LAMBDA_STRUCT = 0.3                                     # v2 结构估计头权重

# ---- 结构头专项（v2s）新增损失权重 ----
W_STRUCT_SHANTEN = 0.5      # 结构头向听直接 CE（v2 缺失的这一项）
LAMBDA_CONSIST_HEAD = 0.3   # 分解-向听一致性：向听头对齐分解暗含的软向听
LAMBDA_CONSIST_DECOMP = 0.2 # 分解-向听一致性：分解暗含软向听拟合真标签
W_TENPAI_TIE = 0.1          # 向听-听牌一致性：P(shanten==0) vs 结构性听牌标签

#: 损失配置默认值（main 里按 CLI 覆盖；compute_loss 的 loss_cfg 入参）
DEFAULT_LOSS_CFG = {
    'w_shanten': W_SHANTEN,               # aux 头向听 CE
    'w_struct_shanten': W_STRUCT_SHANTEN,  # 结构头向听 CE
    'consist_head': LAMBDA_CONSIST_HEAD,
    'consist_decomp': LAMBDA_CONSIST_DECOMP,
    'tenpai_tie': W_TENPAI_TIE,
    'wait_tenpai': 0.2,                    # 输出侧 tenpai 监督（A2b：wait[34] 拟合 1-tenpai）
    'struct_balance': True,                # 向听/形制类均衡
}


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
                'riichi', 'called', 'meld_counts', 'dora_indicators',
                'turn', 'tiles_left', 'tenpai',
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


def _balanced_class_weights(labels: torch.Tensor, n_classes: int,
                            power: float = 0.5, lo: float = 0.5,
                            hi: float = 3.0) -> torch.Tensor:
    """按批内频率的逆平方根加权（detach）：抑制多数类主导。

    解决的系统性偏差（README §0ter 结论 3）：向听预测「严重低估 shanten=2/4、
    对 shanten=0 过预测」，形制预测 91.9% = 多数类基线。幂 0.5（而非 1）是
    温和均衡——拉回少数类召回的同时不摧毁多数类校准。缺失类权重置 0
    （批内无样本，本就不产生梯度）。
    """
    cnt = torch.bincount(labels.reshape(-1), minlength=n_classes).float()
    freq = cnt / cnt.sum().clamp(min=1.0)
    w = torch.where(freq > 0, freq.clamp(min=1e-6).reciprocal().pow(power),
                    torch.zeros_like(freq))
    w = w * (n_classes / w.sum().clamp(min=1e-6))
    return w.clamp(lo, hi).detach()


def compute_loss(out, batch, loss_cfg: dict | None = None):
    """总损失。loss_cfg 缺省用 DEFAULT_LOSS_CFG（见模块头「结构头专项」）。"""
    cfg = {**DEFAULT_LOSS_CFG, **(loss_cfg or {})}
    w = sample_weights(batch)                                # [B,3]

    # ---- L_alloc：逐列计数 CE（logits 已按 unseen 掩码）----
    logp = torch.log_softmax(out['alloc_logits'], dim=-1)    # [B,3,34,5]
    ll = logp.gather(-1, batch['alloc'][..., None]).squeeze(-1)   # [B,3,34]
    denom = (w.sum() * 34.0).clamp(min=1e-8)
    l_alloc = -(ll * w[..., None]).sum() / denom

    # ---- L_wait：条件分布 CE（34 类 softmax34），仅听牌目标 ----
    # RM4（README §11.1 输出层因子化）：旧写法是 35 类 softmax CE——L_wait
    # （仅 14.6% 听牌目标激活）与 L_wait_tenpai BCE（全目标）抢同一个归一化：
    # BCE 把 l34 钉在标定值，其余 34 个 logit 在 85.4% 非听牌目标上完全无约束，
    # 配分函数噪声（sd 1.94）同时污染判别与标定（A2 事故的结构性根源）。
    # 因子化后 tenpai 标量由 L_wait_tenpai 独占监督，本项只监督 34 类条件
    # 分布，与 evaluate 的 cond_ce 同口径（A2b cond_ce 2.8961；RM4 验收
    # ≤2.896 不退）。注意：目标分布的零质量类会碰上硬掩码的 -inf logit，
    # 0 * -inf = NaN，必须先把零质量位置清零再求和（只累加目标支撑集）。
    wait_t = batch['wait'][..., :D.NUM_TILES]                  # [B,3,34]
    logw = torch.log_softmax(out['wait_logits'][..., :D.NUM_TILES], dim=-1)
    ll_wait = -(wait_t * torch.where(wait_t > 0, logw,
                                     torch.zeros_like(logw))).sum(-1)   # [B,3]
    active = batch['tenpai'] > 0.5
    if active.any():
        wa = w * active.float()
        # ll_wait 已是正值的 CE（见上），直接加权平均，勿再取负
        l_wait = (ll_wait * wa).sum() / wa.sum().clamp(min=1e-8)
    else:
        # 整批无听牌目标 → wait 损失为 0。注意 wait_logits 含硬掩码 -inf，
        # 直接 sum()*0 会得 NaN（-inf*0）——先把 -inf 清零再乘 0，保持计算图
        # 连通且数值为 0。
        l_wait = torch.nan_to_num(out['wait_logits'],
                                  posinf=0.0, neginf=0.0).sum() * 0.0

    # ---- 输出侧 tenpai 监督（A2b）：wait[-1]（未听）拟合 1-tenpai，全目标 ----
    # A2 发现的机制缺陷：l_wait 只在结构听牌目标（约 14.5%）上激活，其余 85.5%
    # 的 not-tenpai 目标对 H2 的 wait[-1]（无可荣牌/未听）输出**没有任何直接监督**。
    # 长训练中"听牌目标 P(未听)→0"的梯度单向传递，导致 P(未听) 在全体目标上集体
    # 塌缩（v2s 50k：非听牌目标 P(未听) 中位数 0.0030→0.0013），tenpai 检测 AUC
    # 0.739→0.667 显著退化。该项对全目标直接监督"未听"输出，口径与 AUC 一致
    # （tenpai 标签、score=1-P(未听)）；权重可调，0 即关闭（消融）。
    l_wait_tenpai = nn.functional.binary_cross_entropy_with_logits(
        out['wait_logits'][..., -1], 1.0 - batch['tenpai'])

    # ---- L_aux ----
    log_next = torch.log_softmax(out['next_logits'], dim=-1)
    l_next = -log_next.gather(-1, batch['next_discard'][..., None]).squeeze(-1).mean()
    log_sh = torch.log_softmax(out['shanten_logits'], dim=-1)
    l_sh = -log_sh.gather(-1, batch['shanten'][..., None]).squeeze(-1).mean()
    l_riichi = nn.functional.binary_cross_entropy_with_logits(
        out['riichi_logit'], batch['riichi'])
    l_called = nn.functional.binary_cross_entropy_with_logits(
        out['called_logit'], batch['called'])
    l_aux = (W_NEXT_DISCARD * l_next + cfg['w_shanten'] * l_sh
             + W_RIICHI * l_riichi + W_CALLED * l_called)

    total = (l_alloc + LAMBDA_WAIT * l_wait + LAMBDA_AUX * l_aux
             + cfg['wait_tenpai'] * l_wait_tenpai)
    parts = {
        'loss': float(total.detach()),
        'alloc': float(l_alloc.detach()),
        'wait': float(l_wait.detach()),
        'wait_tenpai': float(l_wait_tenpai.detach()),
        'aux': float(l_aux.detach()),
        'next': float(l_next.detach()),
        'shanten': float(l_sh.detach()),
    }

    # ---- L_struct（v2/v2s）：搭子构造软分布 + 形制 + 向听专项 ----
    if 'struct_counts' in batch:
        # 软 CE：标签是最小向听分解上的边际直方图（保留真实歧义），
        # 对 7 个结构计数分别求 -Σ_k p_k log q_k 再平均。
        logc = torch.log_softmax(out['struct_count_logits'], dim=-1)   # [B,3,7,5]
        ll_c = -(batch['struct_counts'] * torch.where(
            batch['struct_counts'] > 0, logc, torch.zeros_like(logc))).sum(-1)  # [B,3,7]
        l_counts = ll_c.mean()

        # 形制 CE（类均衡：normal ~97%，否则 form 头退化为多数类基线）
        form_logits = out['struct_form_logits']
        l_form = nn.functional.cross_entropy(
            form_logits.reshape(-1, form_logits.shape[-1]),
            batch['struct_form'].reshape(-1),
            weight=(_balanced_class_weights(batch['struct_form'], NUM_FORMS)
                    if cfg['struct_balance'] else None))

        # ---- 结构头向听直接 CE（v2 缺失的一项：该头原本无任何监督）----
        sh_logits = out['struct_shanten_logits']                       # [B,3,8]
        sh_true = batch['shanten']                                     # [B,3]
        log_sh2 = torch.log_softmax(sh_logits, dim=-1)
        ll_sh2 = -log_sh2.gather(-1, sh_true[..., None]).squeeze(-1)   # [B,3]
        if cfg['struct_balance']:
            w_sh = _balanced_class_weights(sh_true, NUM_SHANTEN_BUCKETS)
        else:
            w_sh = torch.ones(NUM_SHANTEN_BUCKETS, device=ll_sh2.device)
        l_struct_shanten = ((ll_sh2 * w_sh[sh_true]).sum()
                            / w_sh[sh_true].sum().clamp(min=1e-8))
        l_struct_shanten_raw = ll_sh2.mean()      # 未加权 CE（类均衡会改变量纲，监控/门禁用）

        # ---- 分解-向听一致性约束：预测的分解直方图必须对应一个自洽向听数 ----
        # 头侧（l_consist_head）：向听头的预测必须与分解暗含的软向听一致
        #   （target 经 detach，梯度只流向向听头——把分解头的判断蒸馏进来）；
        # 分解侧（l_consist_decomp）：分解暗含的软向听必须拟合真标签（梯度
        #   经 shanten 公式回流到分解头，让 7 个计数的整体形状承担向听责任）。
        # 形制门控：软公式与硬枚举器一样只实现 normal 路径（2L-2M-min(T+K,L-M)
        #   -雀首修正）；七对/国士的 counts 标签仍是 normal 分解边际，不承载
        #   其向听信息（实测 normal 上 round(soft)==真值 100%，非 normal 恒偏）。
        #   故约束只作用于 normal 目标；非 normal 的头侧 target 回退为真标签
        #   （等价直接 CE，不引入错误梯度）。
        # L = 4 - 副露数 由公开信息精确推导（alloc.sum()//3，与 dataset 同口径）。
        len_div3 = (batch['alloc'].sum(-1) // 3).float()               # [B,3]
        soft_sh = soft_shanten_from_counts(
            nn.functional.softmax(out['struct_count_logits'], dim=-1), len_div3)  # [B,3]
        consist_target = torch.where(
            batch['struct_form'] == FORM_NORMAL,
            soft_sh.detach().round().long(),
            sh_true).clamp(0, MAX_SHANTEN_BUCKET)                      # [B,3]
        l_consist_head = nn.functional.cross_entropy(
            sh_logits.reshape(-1, sh_logits.shape[-1]),
            consist_target.reshape(-1))
        normal_mask = (batch['struct_form'] == FORM_NORMAL).float()    # [B,3]
        l_consist_decomp = (((soft_sh.clamp(max=MAX_SHANTEN_BUCKET)
                              - sh_true.float()) ** 2) * normal_mask).sum() \
            / normal_mask.sum().clamp(min=1.0)

        # ---- 向听-听牌一致性：shanten==0 ⟺ 结构性听牌 ----
        # tenpai 标签（waits 非空）比 8 类向听稠密得多；P(shanten==0) 直接
        # 对 tenpai 做 BCE，给向听头第二条梯度通路（稀有「结构听牌但待牌
        # 全可见」例外占比极低，权重小，不撼动主监督）。
        p_sh0 = 1.0 - torch.softmax(sh_logits, dim=-1)[..., 1:].sum(-1)  # [B,3]
        l_tenpai_tie = nn.functional.binary_cross_entropy(
            p_sh0.clamp(1e-6, 1 - 1e-6), batch['tenpai'])

        l_struct = (l_counts + l_form
                    + cfg['w_struct_shanten'] * l_struct_shanten
                    + cfg['consist_head'] * l_consist_head
                    + cfg['consist_decomp'] * l_consist_decomp
                    + cfg['tenpai_tie'] * l_tenpai_tie)
        total = total + LAMBDA_STRUCT * l_struct
        parts.update({'struct': float(l_struct.detach()),
                      'struct_counts': float(l_counts.detach()),
                      'struct_form': float(l_form.detach()),
                      'struct_shanten': float(l_struct_shanten.detach()),
                      'struct_shanten_raw': float(l_struct_shanten_raw.detach()),
                      'consist_head': float(l_consist_head.detach()),
                      'consist_decomp': float(l_consist_decomp.detach()),
                      'tenpai_tie': float(l_tenpai_tie.detach())})
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


# 选型判据版本（RM2，README §11.5-③）：v1 = alloc_ce + 0.5·wait_ce_active；
# v2 追加听牌 AUC 与标定项。换代后 resume 会重置 best_val（旧值不可比）。
VAL_SCORE_VERSION = 2


def _rank_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """秩和 AUC（Mann-Whitney U，并列取平均秩）；单类标签返回 nan。"""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def val_score_of(metrics: dict) -> float:
    """选型判据 v2（RM2）：越小越好。

        val_score = alloc_ce + 0.5·wait_ce_active
                    + (1 − tenpai_auc) + |mean P(听) − 实际听牌率|

    v1 判据（前两项）的盲区——A2 事故（2026-10-18，MEMORY §3.12-5）：
    P(未听) 全体塌缩至 0.0735、tenpai AUC 0.739→0.667，而 alloc_ce /
    wait_ce_active 照常「改善」，best.pth 静默存下塌缩权重。后两项使
    听牌判别力与标定必然进入选型（验收：test_pipeline H2）。
    tenpai_auc 为 nan（整批单类）时跳过该项，行为退化为 v1。
    """
    vs = metrics['alloc_ce']
    wc = metrics.get('wait_ce_active', float('nan'))
    if wc == wc:
        vs += 0.5 * wc
    auc = metrics.get('tenpai_auc', float('nan'))
    if auc == auc:
        vs += 1.0 - auc
    mp = metrics.get('mean_p_tenpai')
    frac = metrics.get('tenpai_frac')
    if mp is not None and frac is not None and mp == mp:
        vs += abs(mp - frac)
    return vs


@torch.no_grad()
def evaluate(net, eval_batch, device):
    net.eval()
    n = eval_batch['obs'].shape[0]
    bs = 2048
    alloc_ll, wait_ll, alloc_top1, wait_hit1 = [], [], [], []
    wait_p_not = []                                            # RM2：P(未听) 收集
    struct_sh_hit, struct_sh_ce = [], []
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
        # RM2：P(未听) = sigmoid(l34)——L_wait_tenpai BCE 直接标定的量（RM1 读法）
        wait_p_not.append(torch.sigmoid(o['wait_logits'][..., D.NUM_TILES]))
        if 'struct_shanten_logits' in o:                    # v2/v2s 结构头监控
            log_sh = torch.log_softmax(o['struct_shanten_logits'], dim=-1)
            ll_sh = -log_sh.gather(-1, chunk['shanten'][..., None]).squeeze(-1)
            struct_sh_hit.append(
                (o['struct_shanten_logits'].argmax(-1) == chunk['shanten']).float())
            struct_sh_ce.append(ll_sh)                     # [b,3] 未加权 CE
    net.train()
    alloc_ll = torch.cat(alloc_ll)                          # [n,3]
    wait_ll = torch.cat(wait_ll)                            # [n,3]
    alloc_top1 = torch.cat(alloc_top1)
    wait_hit1 = torch.cat(wait_hit1)
    tenpai_mask = eval_batch['tenpai'] > 0.5
    wait_ce_active = float(wait_ll[tenpai_mask].mean()) if tenpai_mask.any() else float('nan')
    wait_hit = float(wait_hit1[tenpai_mask].mean()) if tenpai_mask.any() else float('nan')
    metrics = {
        'alloc_ce': float(alloc_ll.mean()),
        'alloc_top1': float(alloc_top1.mean()),
        'wait_ce_active': wait_ce_active,
        'wait_hit1': wait_hit,
        'tenpai_frac': float(tenpai_mask.float().mean()),
    }
    # ---- RM2：听牌判别 AUC + 标定（sigmoid 读法，RM1）进选型判据 ----
    p_not = torch.cat(wait_p_not).cpu().numpy().astype(np.float64)   # [n,3] P(未听)
    p_ten = 1.0 - p_not
    metrics['tenpai_auc'] = _rank_auc(tenpai_mask.numpy().ravel(), p_ten.ravel())
    metrics['mean_p_tenpai'] = float(p_ten.mean())
    if struct_sh_hit:                                       # 结构头向听准确率（v2s 主攻指标）
        metrics['struct_shanten_acc'] = float(torch.cat(struct_sh_hit).mean())
        metrics['struct_shanten_ce'] = float(torch.cat(struct_sh_ce).mean())
    return metrics


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
    ap.add_argument('--prefetch-factor', type=int, default=2,
                    help='DataLoader prefetch_factor（降内存/共享内存压力用，崩溃时调小）')
    ap.add_argument('--augment', choices=('none', 'random', 'all'), default='none',
                    help='万/筒增广：all=全部文件；random=约一半文件（对称性混合口径）；none=不增广')
    ap.add_argument('--conv-channels', type=int, default=96)
    ap.add_argument('--num-blocks', type=int, default=12)
    ap.add_argument('--pool', choices=('mean', 'attention'), default='mean',
                    help='全局池化口径（方案 §4.1 两个选项；attention 保留逐列重要性信息）')
    ap.add_argument('--model', choices=('v1', 'v2', 'v2s'), default='v1',
                    help='v1=单帧；v2=空间+手切时序+结构估计三流融合；'
                         'v2s=v2+结构头专项（向听分布注入瓶颈 + 向听监督/一致性新损失）')
    ap.add_argument('--wait-head', choices=('flat', 'tile'), default='tile',
                    help='v2/v2s 的 H2 听牌头参数化（RM5，README §11.3）：flat=旧 '
                         'tile-agnostic 头（A2b 及之前 ckpt 的架构，加载旧 ckpt 必须 '
                         '用此项）；tile=逐牌列打分头（与 H1 同构，A2c 起新训练默认）')
    ap.add_argument('--no-aux', action='store_true')
    ap.add_argument('--w-shanten', type=float, default=DEFAULT_LOSS_CFG['w_shanten'],
                    help='aux 头向听 CE 权重（v2s 起默认 0.15，即 v1 的 0.05 ×3）')
    ap.add_argument('--struct-shanten-weight', type=float,
                    default=DEFAULT_LOSS_CFG['w_struct_shanten'],
                    help='结构头向听直接 CE 权重（0=关闭，复现 v2 行为）')
    ap.add_argument('--consist-head-weight', type=float,
                    default=DEFAULT_LOSS_CFG['consist_head'],
                    help='分解-向听一致性（向听头对齐分解暗含向听）权重')
    ap.add_argument('--consist-decomp-weight', type=float,
                    default=DEFAULT_LOSS_CFG['consist_decomp'],
                    help='分解-向听一致性（分解暗含软向听拟合真标签）权重')
    ap.add_argument('--tenpai-tie-weight', type=float,
                    default=DEFAULT_LOSS_CFG['tenpai_tie'],
                    help='向听-听牌一致性（shanten==0 ⟺ tenpai）权重')
    ap.add_argument('--wait-tenpai-weight', type=float,
                    default=DEFAULT_LOSS_CFG['wait_tenpai'],
                    help='输出侧 tenpai 监督（wait[34] 拟合 1-tenpai，全目标）权重；'
                         'A2b 机制修复，0 关闭（等价 v2s）')
    ap.add_argument('--no-struct-balance', action='store_true',
                    help='关闭向听/形制的类均衡加权（默认开启）')
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
                               with_aux=not args.no_aux,
                               wait_head=args.wait_head).to(device)
    elif args.model == 'v2s':
        net = M2.build_model_v2s(conv_channels=args.conv_channels,
                                 num_blocks=args.num_blocks,
                                 with_aux=not args.no_aux,
                                 wait_head=args.wait_head).to(device)
    else:
        net = M.build_model(conv_channels=args.conv_channels, num_blocks=args.num_blocks,
                            with_aux=not args.no_aux, pool=args.pool).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'[model] {args.model} | 参数量 {n_params:,} | device={device} | fp32 | '
          f'channels={args.conv_channels} blocks={args.num_blocks} aux={not args.no_aux} '
          f'pool={args.pool if args.model == "v1" else "-"} '
          f'wait_head={args.wait_head if args.model != "v1" else "-"}', flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 结构头专项损失配置（CLI 可调；0 权重即关闭对应项，便于消融）
    loss_cfg = {
        'w_shanten': args.w_shanten,
        'w_struct_shanten': args.struct_shanten_weight,
        'consist_head': args.consist_head_weight,
        'consist_decomp': args.consist_decomp_weight,
        'tenpai_tie': args.tenpai_tie_weight,
        'wait_tenpai': args.wait_tenpai_weight,
        'struct_balance': not args.no_struct_balance,
    }
    print(f'[loss] cfg={loss_cfg}', flush=True)
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
        if int(state.get('val_score_version', 1)) != VAL_SCORE_VERSION:
            # RM2：选型判据换代（v1→v2），旧 best_val 与新公式不可比，重置
            print(f'[resume][warn] val_score 公式 v{state.get("val_score_version", 1)}'
                  f' → v{VAL_SCORE_VERSION}，best_val 重置为 inf', flush=True)
            best_val = float('inf')
        print(f'[resume] {args.resume} @ step {start_step}, best_val {best_val:.6f}', flush=True)

    def save(tag, step, val):
        state = {
            'model': net.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
            'step': step,
            'best_val': val,
            'val_score_version': VAL_SCORE_VERSION,
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
            collate_fn=collate, prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
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
            loss, parts = compute_loss(out, batch, loss_cfg=loss_cfg)
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
                extra = (f" struct_sh {parts['struct_shanten']:.4f}"
                         if 'struct_shanten' in parts else '')
                print(f"step {step} (stage {stage}) loss {parts['loss']:.4f} "
                      f"alloc {parts['alloc']:.4f} wait {parts['wait']:.4f} "
                      f"w_ten {parts['wait_tenpai']:.4f} "
                      f"aux {parts['aux']:.4f}{extra} lr {rec['lr']:.2e} "
                      f"[{rec['elapsed']}s]", flush=True)

            if step % args.eval_every == 0 or step == end_step:
                metrics = evaluate(net, eval_batch, device)
                wait_ce = metrics['wait_ce_active']
                val_score = val_score_of(metrics)          # RM2：v2 判据
                rec = {'step': step, 'stage': stage, 'eval': metrics, 'val_score': val_score,
                       'val_score_version': VAL_SCORE_VERSION}
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                print(f"[eval] step {step}: alloc_ce {metrics['alloc_ce']:.4f} "
                      f"top1 {metrics['alloc_top1']:.4f} wait_ce {wait_ce:.4f} "
                      f"wait_hit1 {metrics['wait_hit1']:.4f} "
                      f"tenpai_frac {metrics['tenpai_frac']:.3f} "
                      f"tenpai_auc {metrics['tenpai_auc']:.4f} "
                      f"meanP(ten) {metrics['mean_p_tenpai']:.4f}"
                      + (f" struct_sh_acc {metrics['struct_shanten_acc']:.4f}"
                         f" (ce {metrics['struct_shanten_ce']:.4f})"
                         if 'struct_shanten_acc' in metrics else ''), flush=True)
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
