#!/usr/bin/env python3
"""生成 IQL Stage B v3 正式轮配置（10-16，MEMORY 会话 25/26；探针判据出来后调用）。

用法：
    python tools/training/prepare_iql_b_v3.py --fix tau05      # 默认：τ=0.5（探针唯一验证的稳定修复）
    python tools/training/prepare_iql_b_v3.py --fix clip8      # 备选：τ=0.7 + L_Q 目标钳位 ±8（待臂 E 验证）

探针判决（10-16，5 臂各 30k 步）：对照/v-ema/γ=0.99 均漂移（8k 步 p50 +2.2~+3.4），
τ=0.5 基本平坦（+0.20）→ 根因是 expectile 不对称棘轮（V 持续落后上升目标→残差持续为正→
τ=0.7 强上拉/0.3 弱下拉的净向上蠕动，速率∝尺度），与反馈增益（γ）和 target lag 无关。
v3 = v2 + 探针选出的最小反漂移修复；其余（数据/名单/增强/pts/GRP/预算/判定口径）全部锁死 = v2。
判定用新种子块 220000/230000（200000/210000 已被 v2 消费）。
"""
import argparse
from pathlib import Path

MORTAL = Path('/mnt/data8/mortal/Mortal/mortal')
SRC = MORTAL / 'config.iql_b_v2.toml'
text = SRC.read_text(encoding='utf-8')

FIXES = {
    'tau05': ('τ=0.5（expectile 对称化；探针唯一验证的稳定修复，v2 棘轮根因的对症解）',
              {'iql_tau = 0.7': 'iql_tau = 0.5'}),
    'clip8': ('τ=0.7 + L_Q 目标钳位 ±8（保乐观性；臂 E 验证"撞墙止漂"是否伤价值学习）',
              {'ema_decay = 0.995': 'ema_decay = 0.995\niql_value_clip = 8.0'}),
    # 以下两项已被探针证伪（8k 步 p50 仍 +2.2/+2.6，与对照 +3.0 同量级），保留仅作记录：
    'v_ema': ('bootstrap 改用 V 的 EMA target（探针证伪：仅慢 ~25%）',
              {'ema_decay = 0.995': 'ema_decay = 0.995\niql_v_target_decay = 0.99'}),
    'gamma99': ('γ=0.99（探针证伪：与对照同速）', {'gamma = 0.997': 'gamma = 0.99'}),
}

ap = argparse.ArgumentParser()
ap.add_argument('--fix', choices=list(FIXES), default='tau05')
args = ap.parse_args()
desc, subs = FIXES[args.fix]

COMMON_SUBS = [
    ('state_file = "checkpoints/iql_b_v2/mortal.pth"',
     'state_file = "checkpoints/iql_b_v3/mortal.pth"'),
    ('best_state_file = "checkpoints/iql_b_v2/best.pth"',
     'best_state_file = "checkpoints/iql_b_v3/best.pth"'),
    ('tensorboard_dir = "runs/iql_b_v2"', 'tensorboard_dir = "runs/iql_b_v3"'),
    ('reset_best_perf = "iql-b-v2"', 'reset_best_perf = "iql-b-v3"'),
    ('reset_optimizer = "iql-b-v2"', 'reset_optimizer = "iql-b-v3"'),
    ('log_dir = "runs/iql_b_v2/test_play"', 'log_dir = "runs/iql_b_v3/test_play"'),
    ('log_dir = "runs/iql_b_v2/ladder"', 'log_dir = "runs/iql_b_v3/ladder"'),
    ('seed = 20261015', 'seed = 20261016'),
]

t = text
for old, new in COMMON_SUBS:
    assert old in t, f'公共替换未命中: {old}'
    t = t.replace(old, new)
for old, new in subs.items():
    assert old in t, f'修复替换未命中: {old}'
    t = t.replace(old, new, 1)

header = (f'# IQL/Bellman Stage B 正式轮 v3 配置（2026-10-16，MEMORY 会话 25）\n'
          f'# v2 实测价值尺度指数漂移（q_target 中位数 +0.6→+61，倍增 ~32k 步，超理论界 [−5,+5] ~12 倍；\n'
          f'#   根因：τ=0.7 expectile 不对称放大 × γ≈1 × 边界稀疏奖励 → bootstrap 反馈增益 >1，§14-39）。\n'
          f'# v3 = v2 + 探针选出的最小反漂移修复：**{desc}**；其余全锁死 = v2。\n'
          f'# 判定 vs B/756k，新种子块 A=220000-/B=230000-（200000/210000 已被 v2 消费）。\n')
lines = t.split('\n')
assert lines[0].startswith('#')
out = header + '\n'.join(lines[6:])
dst = MORTAL / 'config.iql_b_v3.toml'
dst.write_text(out, encoding='utf-8')
print(f'written: {dst}')

# ---- 冒烟配置（结构/行为/尺度三门禁，2k 步；尺度充分证据由 30k 探针臂提供）----
SMOKE = MORTAL / 'config.iql_b_smoke.toml'
st = SMOKE.read_text(encoding='utf-8')
smoke_subs = [
    ('state_file = "checkpoints/iql_b_smoke_v3/mortal.pth"',
     'state_file = "checkpoints/iql_b_v3_smoke/mortal.pth"'),
    ('best_state_file = "checkpoints/iql_b_smoke_v3/best.pth"',
     'best_state_file = "checkpoints/iql_b_v3_smoke/best.pth"'),
    ('tensorboard_dir = "runs/iql_b_smoke_v3"', 'tensorboard_dir = "runs/iql_b_v3_smoke"'),
    ('reset_best_perf = "iql-b-smoke-v3"', 'reset_best_perf = "iql-b-smoke-v3"'),
    ('reset_optimizer = "iql-b-smoke-v3"', 'reset_optimizer = "iql-b-smoke-v3"'),
    ('log_dir = "runs/iql_b_smoke_v3/test_play"', 'log_dir = "runs/iql_b_v3_smoke/test_play"'),
    ('log_dir = "runs/iql_b_smoke_v3/ladder"', 'log_dir = "runs/iql_b_v3_smoke/ladder"'),
    ('seed = 20261015', 'seed = 20261016'),
]
for old, new in smoke_subs:
    assert old in st, f'冒烟公共替换未命中: {old}'
    st = st.replace(old, new)
for old, new in subs.items():
    assert old in st, f'冒烟修复替换未命中: {old}'
    st = st.replace(old, new, 1)
smoke_header = (f'# IQL Stage B v3 冒烟配置（2026-10-16）：同 v2 冒烟结构 + 修复「{desc}」；\n'
                f'# 门禁含 ⑧ 价值尺度（run_iql_b_smoke.sh）。尺度充分证据 = 30k 探针臂。\n')
slines = st.split('\n')
assert slines[0].startswith('#')
smoke_out = smoke_header + '\n'.join(slines[10:])
sdst = MORTAL / 'config.iql_b_v3_smoke.toml'
sdst.write_text(smoke_out, encoding='utf-8')
print(f'written: {sdst}')
print(f'fix = {args.fix}: {desc}')
