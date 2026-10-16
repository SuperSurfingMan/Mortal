#!/usr/bin/env python3
"""IQL v2 价值尺度漂移的参数级诊断（10-16，会话 24 中途分析）。

对比父代 B/756k（cql_mc，Q 尺度健康）与 iql_b_v2 当前 ckpt（Q 中位数 ~+60）：
  1. 分模块参数范数（Brain / DQN.v_head / DQN.a_head / AuxNet / VHead）
  2. 同一批真实状态上前向：q(s,·) 分布、v_head、VHead 输出
判别目标：膨胀在「共享价值头 v_head + VHead」（价值通胀）还是「优势头 a_head」（策略锐化漂移）。
"""
import os, sys, glob
import numpy as np
import torch

REPO = '/mnt/data8/mortal/Mortal'
os.chdir(f'{REPO}/mortal')
os.environ['MORTAL_CFG'] = 'config.iql_b_v2.toml'
sys.path.insert(0, f'{REPO}/mortal')
from config import config
from model import Brain, DQN, AuxNet, VHead
from libriichi.dataset import GameplayLoader

V = config['control']['version']
CKP = {
    'parent_B_756k': f'{REPO}/mortal/checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth',
    'iql_b_v2_now': f'{REPO}/mortal/checkpoints/iql_b_v2/mortal.pth',
}

# ---- 取一批真实状态（202601 前 3 个文件，与回归测试同源）----
loader = GameplayLoader(version=V, oracle=False, player_names=None,
                        excludes=None, augmented=False)
FILES = sorted(glob.glob(f'{REPO}/dataset/mjai/houou4_names/202601/*.json.gz'))[:3]
obs_list = []
for path in FILES:
    for file in loader.load_gz_log_files([path]):
        for game in file:
            obs_list.append(np.asarray(game.take_obs()))
obs = np.concatenate(obs_list, axis=0)
print(f'states: {obs.shape}')
xs = torch.as_tensor(obs, dtype=torch.float32)

def load_all(path):
    st = torch.load(path, weights_only=False, map_location='cpu')
    brain = Brain(version=V, **config['resnet'])
    dqn = DQN(version=V)
    aux = AuxNet((4,))
    brain.load_state_dict(st['mortal']); dqn.load_state_dict(st['current_dqn']); aux.load_state_dict(st['aux_net'])
    vnet = None
    if 'v_net' in st:
        vnet = VHead(); vnet.load_state_dict(st['v_net'])
    return brain, dqn, aux, vnet, st

for name, path in CKP.items():
    if not os.path.exists(path):
        print(f'{name}: 缺失 {path}'); continue
    brain, dqn, aux, vnet, st = load_all(path)
    brain.eval(); dqn.eval(); aux.eval()
    if vnet is not None: vnet.eval()
    with torch.no_grad():
        phi = brain(xs)
        if dqn.version == 4:
            va = dqn.net(phi)
            v_head_out = va[:, :1].squeeze(-1)
            a_head_out = va[:, 1:]
        else:
            v_head_out = dqn.v_head(phi).squeeze(-1)
            a_head_out = dqn.a_head(phi)
        vnet_out = vnet(phi).squeeze(-1) if vnet is not None else None
    def rms(x): return float(x.pow(2).mean().sqrt())
    print(f'\n== {name}  steps={st.get("steps")}')
    pnames = [n for n, _ in dqn.named_parameters()]
    print(f'  参数范数  brain={rms(torch.cat([p.flatten() for p in brain.parameters()])):.4f}'
          f'  dqn.net={rms(torch.cat([p.flatten() for n,p in dqn.named_parameters()])):.4f}'
          f'  aux={rms(torch.cat([p.flatten() for p in aux.parameters()])):.4f}'
          + (f'  v_net={rms(torch.cat([p.flatten() for p in vnet.parameters()])):.4f}' if vnet else ''))
    print(f'  前向(5018 状态)  v_head p50={v_head_out.median():+.3f} p99={v_head_out.quantile(0.99):+.3f}'
          f'  a_head p50={a_head_out.median():+.3f} p99={a_head_out.quantile(0.99):+.3f}'
          f'  a_head_max={a_head_out.max():+.3f}'
          + (f'  VHead p50={vnet_out.median():+.3f} p99={vnet_out.quantile(0.99):+.3f}' if vnet_out is not None else ''))
    # 每个状态取数据动作不可得（未加载 actions），看 a_head 的跨动作离散度
    print(f'  a_head 跨动作 std（状态均值）={a_head_out.std(dim=-1).mean():.4f}')
