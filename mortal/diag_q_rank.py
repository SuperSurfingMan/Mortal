"""Q 排序诊断（2026-11-09）：把「argmax 靠不靠谱」变成三个数。

对 holdout 的每个决策算：
  1. rank  = #{a: Q(s,a) > Q(s,a_人类)}  ⇒ 人类动作在 Q 排序中的名次分布
  2. gap   = top1(Q) - top2(Q)           ⇒ argmax 的「把握」
  3. RMSE  = sqrt(mean((Q(s,a) - target)^2))  ⇒ Q 自身误差

用途：判断「top5 里常有最优动作」是否能靠探索学到 —— 需要 gap 明显大于 RMSE；
若训练后 gap 降得比 RMSE 快（信噪比下降），探索只会让 argmax 更不稳。

用法： CUDA_DEVICE_ORDER=PCI_BUS_ID MORTAL_CFG=<cfg> python diag_q_rank.py <ckpt> [month] [seed]
"""
import sys
import torch
from holdout_metrics import collect, DEVICE
from model import Brain, DQN
import s6c_features

ckpt = sys.argv[1]
month = sys.argv[2] if len(sys.argv) > 2 else '202609'
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261009

batches, _digest_s, _digest_t = collect(month, 60, 256, seed, 0)
st = torch.load(ckpt, map_location='cpu', weights_only=True)
cfg = st['config']
ver = cfg['control'].get('version', 1)
_, extra = s6c_features.resolve(cfg['control'])
mortal = Brain(version=ver, extra_channels=extra, **cfg['resnet']).to(DEVICE).eval()
dqn = DQN(version=ver).to(DEVICE).eval()
mortal.load_state_dict(st['mortal'])
dqn.load_state_dict(st['current_dqn'])

hist = torch.zeros(12, dtype=torch.long)
gaps, q_act_all, tgt_all = [], [], []
with torch.inference_mode():
    for obs, actions, masks, q_target in batches:
        obs = obs.float().to(DEVICE)
        a = actions.long().to(DEVICE)
        masks = masks.bool().to(DEVICE)
        q_out = dqn(mortal(obs), masks)
        q_act = q_out[torch.arange(obs.shape[0], device=DEVICE), a]
        rank = (q_out > q_act.unsqueeze(-1)).sum(-1)          # 2 = 恰好是第 3 名
        for r in rank.tolist():
            hist[min(r, 11)] += 1
        top2 = q_out.topk(2, dim=-1).values
        gaps.append((top2[:, 0] - top2[:, 1]).cpu())
        q_act_all.append(q_act.cpu())
        tgt_all.append(q_target.float())

n = int(hist.sum())
print(f'\n=== {ckpt}  month={month} seed={seed}  samples={n} ===')
print('人类动作在 Q 排序中的名次（「第 k 名」含恰好 k-1 个动作更优）:')
cum = 0
for k in range(12):
    c = int(hist[k])
    cum += c
    label = f'{k+1}' if k < 11 else '12+'
    print(f'  第{label:>3}名  {c:7d}  {c/n*100:5.2f}%   累计 {cum/n*100:5.2f}%')
gap = torch.cat(gaps)
qa = torch.cat(q_act_all)
tg = torch.cat(tgt_all)
rmse = float(((qa - tg) ** 2).mean().sqrt())
print(f'\ntop1−top2 的 Q 差距：均值 {gap.mean():.4f}  中位 {gap.median():.4f}  '
      f'p10 {gap.quantile(0.1):.4f}  p90 {gap.quantile(0.9):.4f}')
print(f'Q(s,a_人类) 的 RMSE（= Q 对回报的估计误差）：{rmse:.4f}')
print(f'  ⇒ |误差| / |top1−top2 差距| = {rmse / gap.median():.1f} 倍'
      f'（>1 意味着 top 名次之间的差异淹没在噪声里）')
print(f'var(Q(s,a_人类))={qa.var():.4f}  var(target)={tg.var():.4f}')
