# 两个 checkpoint 之间的 1v3 duplicate-mahjong 对抗（真实强度阶梯评测）
# 用法: python ladder.py <challenger.pth> <champion.pth> [seed_start] [n_sets] [log_dir] [选项]
#   并发跑多组时第 5 个参数必传（各自独立的日志目录，否则会互相 rmtree 覆盖）
#
# 2026-10-12（对应 NEXT_STEPS_ANALYSIS_20261012.md §3.5）：执行参数不再"隐式写死"——
#   ① 优先级 **CLI 选项 > 配置 [1v3.ladder] > 历史默认（cuda:0 / amp=False / guard=True）**；
#      默认值刻意等于 R2~R4 判定时的口径，避免升级脚本后旧结论不可比（要换必须显式写出来）；
#      注：roundN 配置里的 [1v3.challenger]/[1v3.champion].enable_amp 是 one_vs_three.py 用的，
#      历史上 ladder 不读它们；现在两者不一致会**打印告警**，不再静默（见下 __main__）。
#   ② 双侧强制同参数（否则胜负里混了执行口径差异，不能当强度差）；
#   ③ 把执行参数与两侧 checkpoint 的 path/sha256/steps 打进 stdout（A_*.log 会保留）与
#      <log_dir>/run_meta.json —— 每个判定结果自带出处，事后能回答"这局是用什么配置跑的"。
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from os import path

import torch

from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from libriichi.stat import Stat

try:                      # 没有 MORTAL_CFG 时退化为纯 CLI 用法（engine/model 都不依赖 config）
    from config import config as _config
except FileNotFoundError:
    _config = {}

LADDER_EXEC_DEFAULTS = {
    'device': 'cuda:0',
    'enable_amp': False,
    'enable_rule_based_agari_guard': True,
}


def ckpt_meta(ckpt):
    """checkpoint 身份：路径 + 内容 sha256（前 16 位）+ 步数 + 版本 + mtime。"""
    st = torch.load(ckpt, map_location=torch.device('cpu'), weights_only=True)
    h = hashlib.sha256()
    with open(ckpt, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    saved_cfg = st.get('config', {}) or {}
    return {
        'path': path.abspath(ckpt),
        'sha256': h.hexdigest()[:16],
        'steps': st.get('steps'),
        'version': saved_cfg.get('control', {}).get('version', 1),
        'resnet': saved_cfg.get('resnet'),
        'mtime': datetime.fromtimestamp(path.getmtime(ckpt)).strftime('%Y-%m-%d %H:%M:%S'),
    }


def resolve_exec_cfg(args):
    sec = _config.get('1v3', {}) if isinstance(_config, dict) else {}
    ladder_sec = sec.get('ladder', {}) if isinstance(sec, dict) else {}
    cfg = {k: ladder_sec.get(k, v) for k, v in LADDER_EXEC_DEFAULTS.items()}
    cli_used = False
    if args.device is not None:
        cfg['device'] = args.device
        cli_used = True
    if args.amp is not None:
        cfg['enable_amp'] = args.amp
        cli_used = True
    if args.agari_guard is not None:
        cfg['enable_rule_based_agari_guard'] = args.agari_guard
        cli_used = True
    cfg['source'] = ('cli' if cli_used
                     else ('config:[1v3.ladder]' if ladder_sec else 'default(=R2~R4 判定口径)'))
    return cfg


def warn_mismatched_sections(exec_cfg):
    """[1v3.challenger/champion] 的执行参数与 ladder 生效值不一致时告警（历史上是静默的）。"""
    sec = _config.get('1v3', {}) if isinstance(_config, dict) else {}
    for side in ('challenger', 'champion'):
        sub = sec.get(side, {}) if isinstance(sec, dict) else {}
        for k, flag in (('enable_amp', '--amp/--no-amp'),
                        ('enable_rule_based_agari_guard', '--agari-guard/--no-agari-guard')):
            if k in sub and bool(sub[k]) != bool(exec_cfg[k]):
                print(f'[warn] [1v3.{side}].{k}={sub[k]} 与本入口生效值 {exec_cfg[k]} 不同；'
                      f'ladder 只认 [1v3.ladder] 或 {flag}（如需对齐请显式设置）', flush=True)


def load_engine(ckpt, name, exec_cfg):
    st = torch.load(ckpt, map_location=torch.device('cpu'), weights_only=True)
    cfg = st['config']
    version = cfg['control'].get('version', 1)
    mortal = Brain(version=version, **cfg['resnet']).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(st['mortal'])
    dqn.load_state_dict(st['current_dqn'])
    return MortalEngine(
        mortal, dqn,
        is_oracle = False,
        version = version,
        device = torch.device(exec_cfg['device']),
        enable_amp = exec_cfg['enable_amp'],
        enable_rule_based_agari_guard = exec_cfg['enable_rule_based_agari_guard'],
        name = name,
    )


def main():
    ap = argparse.ArgumentParser(description='1v3 同牌山配对对抗（挑战者 1 席 vs 3× 卫冕者）')
    ap.add_argument('challenger')
    ap.add_argument('champion')
    ap.add_argument('seed_start', nargs='?', type=int, default=30000)
    ap.add_argument('n_sets', nargs='?', type=int, default=500, help='1 set = 4 局同牌山')
    ap.add_argument('log_dir', nargs='?', default='runs/ladder')
    ap.add_argument('--device', default=None, help="如 'cuda:0' / 'cpu'（默认取 [1v3.ladder].device）")
    ap.add_argument('--amp', dest='amp', action='store_true', default=None, help='开启 AMP')
    ap.add_argument('--no-amp', dest='amp', action='store_false', help='关闭 AMP（R2~R4 判定口径）')
    ap.add_argument('--agari-guard', dest='agari_guard', action='store_true', default=None)
    ap.add_argument('--no-agari-guard', dest='agari_guard', action='store_false')
    ap.add_argument('--tag', default='', help='可选标记，写进 run_meta.json 便于检索')
    args = ap.parse_args()

    exec_cfg = resolve_exec_cfg(args)
    warn_mismatched_sections(exec_cfg)

    key = 0x2000
    log_dir = path.abspath(args.log_dir)
    if path.isdir(log_dir):
        shutil.rmtree(log_dir)

    chal_meta = ckpt_meta(args.challenger)
    cham_meta = ckpt_meta(args.champion)
    chal = load_engine(args.challenger, 'challenger', exec_cfg)
    cham = load_engine(args.champion, 'champion', exec_cfg)

    env = OneVsThree(disable_progress_bar = False, log_dir = log_dir)
    rankings = env.py_vs_py(
        challenger = chal,
        champion = cham,
        seed_start = (args.seed_start, key),
        seed_count = args.n_sets,
    )
    stat = Stat.from_dir(log_dir, 'challenger')

    print(f'challenger = {args.challenger}')
    print(f'champion   = {args.champion}')
    print(f'sets = {args.n_sets} (hanchans = {args.n_sets * 4}), seed_start = {args.seed_start}')
    print(f'log_dir = {log_dir}')
    print(f'exec_cfg = {json.dumps(exec_cfg, sort_keys=True)}')
    print(f'challenger_ckpt = {json.dumps(chal_meta, sort_keys=True)}')
    print(f'champion_ckpt = {json.dumps(cham_meta, sort_keys=True)}')
    print(f'raw rankings = {rankings}')
    print(f'avg_rank = {stat.avg_rank:.4f}   avg_pt = {stat.avg_pt([90, 45, 0, -135]):.3f}')
    print(f'1st/2nd/3rd/4th = {stat.rank_1_rate:.3f} {stat.rank_2_rate:.3f} {stat.rank_3_rate:.3f} {stat.rank_4_rate:.3f}')
    print(f'agari={stat.agari_rate:.3f} houjuu={stat.houjuu_rate:.3f} fuuro={stat.fuuro_rate:.3f} riichi={stat.riichi_rate:.3f}')

    meta = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'tag': args.tag,
        'challenger': chal_meta,
        'champion': cham_meta,
        'exec_cfg': exec_cfg,
        'seed_start': args.seed_start,
        'seed_key': key,
        'sets': args.n_sets,
        'hanchans': args.n_sets * 4,
        'results': {
            'raw_rankings': rankings,
            'avg_rank': stat.avg_rank,
            'avg_pt': stat.avg_pt([90, 45, 0, -135]),
            'rank_rates': [stat.rank_1_rate, stat.rank_2_rate, stat.rank_3_rate, stat.rank_4_rate],
            'agari_rate': stat.agari_rate,
            'houjuu_rate': stat.houjuu_rate,
            'fuuro_rate': stat.fuuro_rate,
            'riichi_rate': stat.riichi_rate,
        },
    }
    meta_path = path.join(log_dir, 'run_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f'run_meta = {meta_path}')


main()
