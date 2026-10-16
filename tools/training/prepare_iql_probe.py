#!/usr/bin/env python3
"""生成 IQL v2 价值尺度漂移的 4 臂探针配置（10-16，MEMORY 会话 25）。

背景：iql_b_v2 正式轮实测 q_target 中位数 105k 步内 +0.6→+61 纯指数发散（倍增 ~32k 步，
超理论界 [−5,+5] ~12 倍）；参数级诊断确认膨胀 confined 在价值通道（dqn v_head/VHead），
优势通道 a_head 统计不变。机制：τ=0.7 expectile 不对称放大 × γ≈1 × 边界稀疏奖励（94% 样本
r=0）→ bootstrap 反馈环增益略大于 1。

4 臂全部从 B/756k 暖启、同数据/同步数（30k）/同 batch，只改一个旋钮：
  A 对照  = v2 原配置（γ=0.997, τ=0.7, 在线 V bootstrap）——预期复现漂移（30k 步 p50 ~+15）
  B v-ema = bootstrap 改用 V 的 Polyak EMA target（iql_v_target_decay=0.99）
  C γ=0.99 = 每决策步折扣收紧（直接压反馈环增益）
  D τ=0.5  = expectile 改对称 L2（机制检查：若稳定则证实不对称放大是根源）
判据：q_target p50 全程 ≤ 8（对照臂按 v2 轨迹在 30k 步应到 ~+15，分离度充足）。
"""
from pathlib import Path

MORTAL = Path('/mnt/data8/mortal/Mortal/mortal')
SRC = MORTAL / 'config.iql_b_v2.toml'
text = SRC.read_text(encoding='utf-8')

ARMS = {
    'a': {'desc': '对照 = v2 原配置', 'subs': {}},
    'b': {'desc': 'bootstrap 用 V 的 EMA target（decay 0.99）',
          'subs': {'ema_decay = 0.995': 'ema_decay = 0.995\niql_v_target_decay = 0.99'}},
    'c': {'desc': 'γ=0.99（收紧反馈环增益）',
          'subs': {'gamma = 0.997': 'gamma = 0.99'}},
    'd': {'desc': 'τ=0.5（expectile 改对称 L2，机制检查）',
          'subs': {'iql_tau = 0.7': 'iql_tau = 0.5'}},
}

# 各臂共同的命名空间/预算替换（相对 v2 正式轮配置）
COMMON_SUBS = [
    ('state_file = "checkpoints/iql_b_v2/mortal.pth"',
     'state_file = "checkpoints/iql_probe_{X}/mortal.pth"'),
    ('best_state_file = "checkpoints/iql_b_v2/best.pth"',
     'best_state_file = "checkpoints/iql_probe_{X}/best.pth"'),
    ('tensorboard_dir = "runs/iql_b_v2"', 'tensorboard_dir = "runs/iql_probe_{X}"'),
    ('max_train_steps = 867000', 'max_train_steps = 786000'),   # 756k + 30k 探针
    ('test_every = 50000', 'test_every = 30000'),
    ('snapshot_every = 50000', 'snapshot_every = 30000'),
    ('reset_best_perf = "iql-b-v2"', 'reset_best_perf = "iql-probe-{X}"'),
    ('reset_optimizer = "iql-b-v2"', 'reset_optimizer = "iql-probe-{X}"'),
    ('log_dir = "runs/iql_b_v2/test_play"', 'log_dir = "runs/iql_probe_{X}/test_play"'),
    ('log_dir = "runs/iql_b_v2/ladder"', 'log_dir = "runs/iql_probe_{X}/ladder"'),
]

for x, arm in ARMS.items():
    t = text
    for old, new in COMMON_SUBS:
        assert old in t, f'[{x}] 公共替换未命中: {old}'
        t = t.replace(old, new.replace('{X}', x))
    for old, new in arm['subs'].items():
        assert old in t, f'[{x}] 臂替换未命中: {old}'
        t = t.replace(old, new, 1)
    # 头部注释替换
    header = (f'# IQL 价值尺度漂移探针 臂{x}（{arm["desc"]}，2026-10-16，MEMORY 会话 25）\n'
              f'# 目的：30k 步内对比各候选修复对 Q 尺度轨迹的影响；判据 q_target p50 ≤ 8（对照臂预期 ~+15）。\n'
              f'# 与 v2 正式轮的差异仅：命名空间/预算(30k)/本臂旋钮；数据/名单/增强/pts/GRP/暖启全部锁死。\n')
    lines = t.split('\n')
    # 替换原文件前 6 行注释头
    assert lines[0].startswith('#'), '期望首行是注释'
    out = header + '\n'.join(lines[6:])
    dst = MORTAL / f'config.iql_probe_{x}.toml'
    dst.write_text(out, encoding='utf-8')
    print(f'written: {dst}  ({arm["desc"]})')
