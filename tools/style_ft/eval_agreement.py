"""风格微调 P3：holdout 动作一致率评测（argmax(masked Q) vs 选手实际动作）。

接受多个 ckpt（风格模型 / 父代 / baseline）在同一 holdout 文件集上逐决策点比对，
输出总体 + 分动作类 + 局内十分位 top-1 一致率与逐局一致率（供局级 bootstrap CI）。

--snapshots 模式：对 checkpoints/style/<slug>/mortal_step_*.pth 批量评测，
输出「一致率-步数」曲线，作为小数据量级下的离线早停选优依据。

用法（repo root 下）:
    python tools/style_ft/eval_agreement.py \
        --ckpts mortal/checkpoints/style/okada_sayaka/mortal.pth \
                mortal/checkpoints/s6c/mortal.pth \
        --stats dataset/index/style/okada_sayaka_stats.json \
        --device cuda:0 --out runs/style/okada_sayaka/agreement.json

    python tools/style_ft/eval_agreement.py \
        --snapshots 'mortal/checkpoints/style/okada_sayaka/mortal_step_*.pth' \
        --stats dataset/index/style/okada_sayaka_stats.json --device cuda:0 \
        --out runs/style/okada_sayaka/agreement_snapshots.json
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import numpy as np
import torch

MORTAL_DIR = str(Path(__file__).resolve().parents[2] / 'mortal')
if MORTAL_DIR not in sys.path:
    sys.path.insert(0, MORTAL_DIR)

from libriichi.dataset import GameplayLoader          # noqa: E402
from model import Brain, DQN                          # noqa: E402
import s6c_features                                    # noqa: E402


def action_class(a: int) -> str:
    """ACTION_SPACE=46（consts.rs:7-15）：37 discard/kan-choice + riichi + 3 chi +
    pon + kan + agari + ryukyoku + pass。"""
    if a <= 36:
        return 'discard'
    if a == 37:
        return 'riichi'
    if 38 <= a <= 42:
        return 'meld'
    if a == 43:
        return 'agari'
    if a == 44:
        return 'ryukyoku'
    return 'pass'


def load_holdout(holdout_files, player, version, extra_features):
    """加载 holdout 牌谱，摊平为 (obs, action, mask, game_id, decile) 样本表。"""
    loader = GameplayLoader(
        version=version,
        oracle=False,
        player_names=[player],
        excludes=None,
        augmented=False,
        always_include_kan_select=True,   # 与训练口径一致（GameplayLoader 默认）
    )
    obs_list, act_list, mask_list, gid_list, dec_list = [], [], [], [], []
    gid = 0
    for batch in (loader.load_gz_log_files(holdout_files[i:i + 16])
                  for i in range(0, len(holdout_files), 16)):
        for file_games in batch:
            for game in file_games:
                obs = np.stack(game.take_obs())
                actions = np.asarray(game.take_actions(), dtype=np.int64)
                masks = np.stack(game.take_masks())
                # 与训练同款过滤：丢弃「动作不在合法 mask」的坏决策点（M-League
                # 转换器缺陷，diag_mleague.py 实测 holdout 含 7 个；这些点模型
                # argmax 永远不可能命中非法位，会无谓压低一致率基数）
                if len(actions) > 0 and not masks[np.arange(len(actions)), actions].all():
                    keep = masks[np.arange(len(actions)), actions]
                    obs = obs[keep]
                    actions = actions[keep]
                    masks = masks[keep]
                # take_at_kyoku() 返回 bytes（每字节一个局编号），须 frombuffer
                # 而非 np.asarray（后者会把 bytes 当字符串解析而炸）
                at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
                at_kyoku = at_kyoku[:len(actions)]
                if extra_features:
                    arr = obs if obs.dtype == np.float32 else obs.astype(np.float32)
                    feats = s6c_features.compute(arr)
                    obs = s6c_features.attach(arr, feats)
                n = len(obs)
                obs_list.append(obs)
                act_list.append(actions)
                mask_list.append(masks)
                gid_list.append(np.full(n, gid, dtype=np.int64))
                # 局内决策序号十分位（巡目段的近似：mjai 无现成巡目 getter）
                seg_start = {}
                seg_len = {}
                for k in at_kyoku:
                    seg_len[int(k)] = seg_len.get(int(k), 0) + 1
                for k in seg_len:
                    seg_start[k] = 0
                dec = np.empty(n, dtype=np.int64)
                for i, k in enumerate(at_kyoku):
                    k = int(k)
                    dec[i] = min(9, seg_start[k] * 10 // max(seg_len[k], 1))
                    seg_start[k] += 1
                dec_list.append(dec)
                gid += 1
    return (np.concatenate(obs_list), np.concatenate(act_list),
            np.concatenate(mask_list), np.concatenate(gid_list), np.concatenate(dec_list))


def load_policy(ckpt, device):
    """仿 ladder.py load_engine：ckpt 自带 config 决定 version/resnet/extra 口径。"""
    st = torch.load(ckpt, map_location=torch.device('cpu'), weights_only=True)
    cfg = st['config']
    version = cfg['control'].get('version', 1)
    extra_features, extra_channels = s6c_features.resolve(cfg['control'])
    mortal = Brain(version=version, extra_channels=extra_channels,
                   **cfg['resnet']).to(device).eval()
    dqn = DQN(version=version).to(device).eval()
    mortal.load_state_dict(st['mortal'])
    dqn.load_state_dict(st['current_dqn'])
    return mortal, dqn, version, extra_features


@torch.inference_mode()
def eval_ckpt(ckpt, obs_np, act_np, mask_np, device, batch=512):
    mortal, dqn, _, _ = load_policy(ckpt, device)
    preds = []
    for i in range(0, len(obs_np), batch):
        obs = torch.from_numpy(obs_np[i:i + batch]).to(device)
        masks = torch.from_numpy(mask_np[i:i + batch]).to(device)
        q = dqn(mortal(obs), masks)
        preds.append(q.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    return preds == act_np, preds


def summarize(hit, act, gid, dec, preds=None):
    per_game = {}
    for g in np.unique(gid):
        m = gid == g
        per_game[int(g)] = float(hit[m].mean())
    out = {
        'n_samples': int(len(hit)),
        'n_games': int(len(per_game)),
        'overall': float(hit.mean()),
        'per_game': per_game,
        'by_action_class': {},
        'by_decile': {},
    }
    for cls in ('discard', 'riichi', 'meld', 'agari', 'ryukyoku', 'pass'):
        m = np.array([action_class(int(a)) == cls for a in act])
        if m.any():
            # pred_rate = 模型在同一人类决策状态下选择该类的比例（风格分布对齐，
            # 与 agreement 互补：小样本低频动作上二者天然互斥，见 NOTES §二.4）
            entry = {'n': int(m.sum()), 'agreement': float(hit[m].mean())}
            if preds is not None:
                pm = np.array([action_class(int(a)) == cls for a in preds])
                entry['pred_n'] = int(pm.sum())
                entry['pred_rate'] = float(pm.mean())
            out['by_action_class'][cls] = entry
    for d in range(10):
        m = dec == d
        if m.any():
            out['by_decile'][str(d)] = {
                'n': int(m.sum()),
                'agreement': float(hit[m].mean()),
            }
    return out


def main():
    ap = argparse.ArgumentParser(description='holdout 动作一致率评测')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--ckpts', nargs='+', help='一个或多个 ckpt 路径')
    g.add_argument('--snapshots', help='glob 模式，如 mortal/checkpoints/style/<slug>/mortal_step_*.pth')
    ap.add_argument('--stats', required=True, help='<slug>_stats.json（读 holdout_files 与玩家名）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.stats, encoding='utf-8') as f:
        stats = json.load(f)
    player = stats['player']
    holdout_files = stats['holdout_files']
    print(f'holdout: {len(holdout_files)} 局（{player}）')

    if args.snapshots:
        ckpts = sorted(globmod.glob(args.snapshots))
        if not ckpts:
            raise SystemExit(f'--snapshots 无匹配: {args.snapshots}')
    else:
        ckpts = args.ckpts

    # holdout 加载口径以第一个 ckpt 的 obs 版本为准（风格模型与父代须同 version/extra）
    _, _, version, extra_features = load_policy(ckpts[0], torch.device('cpu'))
    obs, act, mask, gid, dec = load_holdout(holdout_files, player, version, extra_features)
    print(f'loaded {len(obs):,} 决策点（{len(ckpts)} 个 ckpt 待评测）')

    results = {}
    for ckpt in ckpts:
        hit, preds = eval_ckpt(ckpt, obs, act, mask, args.device, args.batch)
        results[ckpt] = summarize(hit, act, gid, dec, preds)
        print(f'  {Path(ckpt).name}: overall={results[ckpt]["overall"]:.4f}')

    out = {
        'player': player,
        'holdout_files': holdout_files,
        'ckpts': results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'out: {args.out}')


if __name__ == '__main__':
    main()