"""AI similarity 诊断：本项目 S6c（v4 + 2 行 s6c）vs 官方血缘 Mortal v3（v3 obs 934 通道）。

背景与口径
----------
社区模型卡（`VoidShine/mortal-298k`）报 "AI similarity 83%~92% on mjai.ekyu.moe"，
那是把自家模型牌谱提交到官方在线服务、读报告里的「与 Mortal 的一致率」；官方不公开权重。
HF 上有官方血缘的 v3 权重（`hhim8826/mortal3-b40c192`：v3 / 192ch / 40blocks / 10.77M）。

本脚本用**更严格的本地口径**：同一批人类 holdout、同一批决策点，两模型各出 argmax：
  - agree   = argmax(S6c) == argmax(v3)              ← 我们在不在官方谱系上
  - top1_x  = argmax(x) == 人类动作                   ← 谁更接近人类
  - 分歧样本中谁更常与人类一致（附 McNemar 检验）      ← 分歧处谁对

实现要点：v3 与 v4 的 obs 编码走不同分支（不是子集关系），且 FileDatasetsIter 的
样本流受 buffer shuffle 影响，跨文件会漂移。因此**逐文件配对**：同一个 .json.gz
分别用 version=3 与 version=4 解码，逐位校验 actions/masks 一致后再比 argmax。

用法：
    MORTAL_CFG=config.s6c.toml python diag_ai_similarity.py [--files 60] [--seed 20261009]
"""
import argparse
import glob
import math
import os
import random
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import config
from model import Brain, DQN
from dataloader import FileDatasetsIter
import s6c_features

ROOT = '/mnt/data8/mortal/Mortal/dataset/mjai/houou4'
DEV = torch.device('cuda:0')
REF_V3 = 'checkpoints/_ref/mortal3_b40c192.pth'
S6C = 'checkpoints/s6c/mortal.pth'
CHUNK = 256


def build_ds(f, version, extra):
    return FileDatasetsIter(
        version=version,
        file_list=[f],
        pts=config['env']['pts'],
        file_batch_size=1,
        reserve_ratio=0.0,
        player_names=[],
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        reward_mode=config['env'].get('reward_mode', 'rank_pt'),
        reward_scale=config['env'].get('reward_scale', 1.0),
        credit_mode=config['env'].get('credit_mode', 'kyoku'),
        gamma=config['env']['gamma'],
        objective=config['control'].get('objective', 'cql_mc'),
        always_include_kan_select=True,
        extra_features=extra,
        seed=0,
    )


def load_models():
    st3 = torch.load(REF_V3, map_location='cpu', weights_only=True)
    cfg3 = st3['config']
    assert cfg3['control'].get('version') == 3, cfg3['control']
    b3 = Brain(version=3, extra_channels=0, **cfg3['resnet']).to(DEV).eval()
    d3 = DQN(version=3).to(DEV).eval()
    b3.load_state_dict(st3['mortal'])
    d3.load_state_dict(st3['current_dqn'])

    st6 = torch.load(S6C, map_location='cpu', weights_only=True)
    cfg6 = st6['config']
    _, ex = s6c_features.resolve(cfg6['control'])
    b4 = Brain(version=cfg6['control']['version'], extra_channels=ex,
               **cfg6['resnet']).to(DEV).eval()
    d4 = DQN(version=cfg6['control']['version']).to(DEV).eval()
    b4.load_state_dict(st6['mortal'])
    d4.load_state_dict(st6['current_dqn'])
    return (b3, d3), (b4, d4), ex, st6.get('steps')


@torch.no_grad()
def predict(brain, dqn, obs, masks):
    """obs/masks: list -> 拼 batch 后分批前向，返回 CPU long tensor"""
    out = []
    for i in range(0, len(obs), CHUNK):
        o = torch.stack(obs[i:i + CHUNK]).to(DEV).float()
        m = torch.stack(masks[i:i + CHUNK]).to(DEV).bool()
        q = dqn(brain(o), m)
        out.append(q.argmax(-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default='202609')
    ap.add_argument('--files', type=int, default=60)
    ap.add_argument('--seed', type=int, default=20261009)
    args = ap.parse_args()

    allf = sorted(glob.glob(f'{ROOT}/{args.month}/*.json.gz'))
    random.Random(args.seed).shuffle(allf)
    files = allf[:args.files]
    (b3, d3), (b4, d4), ex, steps = load_models()
    print(f'[setup] 官方 v3: 192x40 obs=934 | S6c: steps={steps} extra={ex} obs={1012+ex}', flush=True)
    print(f'[setup] month={args.month} seed={args.seed} files={len(files)}', flush=True)

    n = agree = t1_3 = t1_4 = 0
    dis = d4_win = d3_win = d_both_ok = d_both_bad = 0
    stat = {k: dict(n=0, a3=0, a4=0) for k in ('打牌', '立直', '鸣牌/其它', '和了')}
    skipped = []

    for fi, f in enumerate(files):
        it3 = list(build_ds(f, 3, ''))
        it4 = list(build_ds(f, config['control']['version'], 's6c'))
        if len(it3) != len(it4):
            skipped.append((os.path.basename(f), len(it3), len(it4)))
            continue
        h = torch.tensor([int(x[1]) for x in it3])
        h4 = torch.tensor([int(x[1]) for x in it4])
        m3 = torch.stack([torch.as_tensor(x[2]).bool() for x in it3])
        m4 = torch.stack([torch.as_tensor(x[2]).bool() for x in it4])
        if not (torch.equal(h, h4) and torch.equal(m3, m4)):
            skipped.append((os.path.basename(f), 'act/mask 不一致'))
            continue

        obs3 = [torch.as_tensor(x[0]) for x in it3]
        obs4 = [torch.as_tensor(x[0]) for x in it4]
        masks = [torch.as_tensor(x[2]).bool() for x in it3]

        p3 = predict(b3, d3, obs3, masks)
        p4 = predict(b4, d4, obs4, masks)
        cnt = len(h)

        n += cnt
        same = p3 == p4
        agree += int(same.sum())
        c3 = p3 == h
        c4 = p4 == h
        t1_3 += int(c3.sum())
        t1_4 += int(c4.sum())

        d = ~same
        nd = int(d.sum())
        dis += nd
        d4_win += int((d & c4 & ~c3).sum())
        d3_win += int((d & c3 & ~c4).sum())
        d_both_ok += int((d & c3 & c4).sum())
        d_both_bad += int((d & ~c3 & ~c4).sum())

        for name, sel in (('打牌', h < 34), ('立直', h == 37), ('和了', h == 43),
                          ('鸣牌/其它', (h >= 34) & (h != 37) & (h != 43))):
            if bool(sel.any()):
                stat[name]['n'] += int(sel.sum())
                stat[name]['a3'] += int((c3 & sel).sum())
                stat[name]['a4'] += int((c4 & sel).sum())

        if (fi + 1) % 10 == 0:
            print(f'  [{fi+1}/{len(files)}] n={n} agree={agree/n:.4f} '
                  f'top1_v3={t1_3/n:.4f} top1_s6c={t1_4/n:.4f}', flush=True)

    def se(p):
        return math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))

    a = agree / n
    v3 = t1_3 / n
    v4 = t1_4 / n
    print('\n================ AI similarity（本地严格口径） ================')
    print(f'样本数            : {n}（{len(files) - len(skipped)} 个文件）')
    if skipped:
        print(f'跳过的文件        : {len(skipped)} → {skipped[:3]}')
    print(f'argmax 一致率     : {a:.4f} ± {se(a):.4f}   （S6c vs 官方 v3）')
    print(f'top1 官方 v3      : {v3:.4f} ± {se(v3):.4f}')
    print(f'top1 S6c          : {v4:.4f} ± {se(v4):.4f}')
    # 配对差的 SE（同批样本）
    diff = (t1_4 - t1_3) / n
    if n:
        b = d4_win
        c = d3_win
        se_diff = math.sqrt(max(b + c - (b - c) ** 2 / n, 1e-9)) / n
        print(f'Δ(S6c−v3)        : {100*diff:+.2f} pp ± {100*se_diff:.2f}')
        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            # df=1 卡方 → p 的粗略上界（chi2>3.84 ⇔ p<0.05）
            sig = 'p<0.001' if chi2 > 10.8 else ('p<0.01' if chi2 > 6.63 else ('p<0.05' if chi2 > 3.84 else 'n.s.'))
            print(f'McNemar           : b={b}(S6c独对) c={c}(v3独对) chi2={chi2:.1f} → {sig}')
    print(f'\n分歧样本          : {dis} ({dis/n*100:.1f}%)')
    if dis:
        print(f'  仅 S6c 对       : {d4_win:6d} ({d4_win/dis*100:.1f}%)')
        print(f'  仅 v3 对        : {d3_win:6d} ({d3_win/dis*100:.1f}%)')
        print(f'  两个都对        : {d_both_ok:6d} ({d_both_ok/dis*100:.1f}%)')
        print(f'  两个都错        : {d_both_bad:6d} ({d_both_bad/dis*100:.1f}%)')
    print('\n按人类动作类型：')
    for k, v in stat.items():
        if v['n']:
            print(f'  {k:8s} n={v["n"]:6d}  top1_v3={v["a3"]/v["n"]:.4f}  top1_s6c={v["a4"]/v["n"]:.4f}')


if __name__ == '__main__':
    main()
