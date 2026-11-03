import prelude

import os
import sys
import json
import torch
from datetime import datetime, timezone
from model import Brain, DQN, GRP
from engine import MortalEngine
import s6c_features
from common import filtered_trimmed_lines
from libriichi.mjai import Bot
from libriichi.dataset import Grp
from config import config

USAGE = '''Usage: python mortal.py <ID>

ARGS:
    <ID>    The player ID, an integer within [0, 3].'''

def main():
    try:
        player_id = int(sys.argv[-1])
        assert player_id in range(4)
    except:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    review_mode = os.environ.get('MORTAL_REVIEW_MODE', '0') == '1'

    device = torch.device('cpu')
    state = torch.load(config['control']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']
    if 'tag' in state:
        tag = state['tag']
    else:
        time = datetime.fromtimestamp(state['timestamp'], tz=timezone.utc).strftime('%y%m%d%H')
        tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{time}'

    # S6c：特征族与额外通道数从 ckpt 自己的 [control] 推（旧 ckpt = 0/''）
    extra_features, extra_channels = s6c_features.resolve(cfg['control'])
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels,
                   extra_channels=extra_channels).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    guard = None
    # 2026-11-03 终局：修正 risk 双乘 P(听) 后，在全新种子 1000000/1020000 上
    # 重扫 λ=4.5/6/7.5（每臂 64k 半庄）全部 NO-CONCLUSION；legacy λ=10 的历史
    # +0.778 也未复现（本轮 −0.184，CI[-0.61,+0.27]）。按预注册规则默认关闭守卫，
    # 不保留 legacy 默认路径。详见 tools/read_model/docs/GUARD_RISK_FIX_20261102.md。
    guard_lambda = float(os.environ.get('MORTAL_GUARD_LAMBDA', '0') or 0)
    if guard_lambda > 0:
        from defense_guard import DefenseGuard
        guard = DefenseGuard(
            lambda_=guard_lambda,
            ckpt=os.environ.get('MORTAL_GUARD_CKPT') or None,
            device=device,
        )

    engine = MortalEngine(
        mortal,
        dqn,
        version = version,
        is_oracle = False,
        device = device,
        enable_amp = False,
        enable_quick_eval = not review_mode,
        enable_rule_based_agari_guard = True,
        defense_guard = guard,
        extra_features = extra_features,
        name = 'mortal',
    )
    bot = Bot(engine, player_id)

    if review_mode:
        logs = []
    for line in filtered_trimmed_lines(sys.stdin):
        if review_mode:
            logs.append(line)

        if reaction := bot.react(line):
            print(reaction, flush=True)
        elif review_mode:
            print('{"type":"none","meta":{"mask_bits":0}}', flush=True)

    if review_mode:
        grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        grp.load_state_dict(grp_state['model'])

        ins = Grp.load_log('\n'.join(logs))
        feature = ins.take_feature()
        seq = list(map(
            lambda idx: torch.as_tensor(feature[:idx+1], device=device),
            range(len(feature)),
        ))

        with torch.inference_mode():
            logits = grp(seq)
        matrix = grp.calc_matrix(logits)
        extra_data = {
            'model_tag': tag,
            'phi_matrix': matrix.tolist(),
        }
        print(json.dumps(extra_data), flush=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
