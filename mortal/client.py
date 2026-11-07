import prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
from os import path
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg
from config import config
import s6c_features

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']
    # S6c：在线训练客户端与训练配置同口径（离线默认 '' = 0）
    _, extra_channels = s6c_features.resolve(config['control'])

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels,
                   extra_channels=extra_channels).to(device).eval()
    dqn = DQN(version=version).to(device)
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    train_player = TrainPlayer()
    param_version = -1        # 我知道的最新版本（随 get_param 响应更新）
    loaded_version = -2       # 已真正装进模型的版本

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        # 权重未变时跳过 load（2026-11-07）：trainer 现在每 N 轮才推一次权重，
        # 而 client 每轮都会 get_param；不跳过的话每 58 秒白传/白 load 一次 41MB。
        # 必须用独立的 loaded_version 判断：param_version 刚被赋成 server 的返回值，
        # 拿它自比永远相等（首版就踩了这个坑，等于永久冻结在初始权重）。
        if param_version == loaded_version:
            logging.info(f'param unchanged (v{param_version}), skip load')
        else:
            mortal.load_state_dict(rsp['mortal'])
            dqn.load_state_dict(rsp['dqn'])
            loaded_version = param_version
            logging.info(f'param has been updated to v{param_version}')

        rankings, file_list = train_player.train_play(mortal, dqn, device)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
