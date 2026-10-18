#!/usr/bin/env python3
"""生成 IQL AWR 温度 β 扫描的探针配置（2026-10-17，用户选定路线：先 β 后稠密化）。

背景：v3（τ=0.5）判定 δ=+0.646、bootstrap CI[−0.35,+1.63] 无统计结论 → IQL 路线按预注册
规则关闭（MEMORY 会话 28）。归因分析第 3 条：AWR 权重≈均匀（模型-专家 top1 已 0.8167，
剩余分歧 ~18% 多为 indifferent 选择）。τ=0.5 下 V=E[Q̄]（行为均值），优势加权
w=exp((Q−V)/β) 是把"价值学习"转成"策略改进"的**唯一机制**，而 β=3.0 从未扫过。

优势尺度（v2 参数级诊断实测）：a_head 跨动作 std≈1.6~1.7、q_pred p50≈+1.2 → Q−V ~ O(1)。
β=3 → 1 单位优势权重比 e^{1/3}≈1.4（软）；β=1 → 2.7×；β=0.5 → 7.4×（锐）；β=10 → 1.11×（≈纯 BC）。

四臂全部 B/756k 暖启、同数据(indexes/iql_b_r4_v2)/同步数(30k)/同 batch/同 seed(20261016)，
只改 iql_beta 一个旋钮：
  a 对照  = β=3.0（v3 原值；提供同预算同种子参照）
  b       = β=1.0（中度锐化）
  c       = β=0.5（锐化）
  d       = β=10.0（软，≈纯 BC 极限；直接检验 H3"权重本就≈均匀"）

假设：
  H1 锐化有帮助（优势含真信号）→ b/c 的 test_play avg_pt 高于对照
  H2 锐化有害（1:10 SNR 下优势被运气污染，放大噪声）→ b/c 劣化、top1 掉
  H3 无效（权重本就≈均匀）→ 四臂无分离，且 d≈a

判据（预注册，详见 run_iql_beta_probe.sh）：
  健康五门（每臂必过）：无 non-finite / q_target p50≤8 / avg_rank≤2.60 / top1≥0.78 / |lse|≤10
  旋钮生效：awr_w ess/p99 随 β 单调（β↓ → ess↓、p99↑）
  探针级方向（低功效，仅筛查）：avg_pt ≥ 对照+2.0 且 top1 ≥ 对照−0.005 → 建议正式轮
    （正式轮种子块 240000/250000，240000 起为下一未消费块）

另生成 config.iql_beta_smoke.toml（= a 臂 + 2k 步预算）：验证 train.py 新增的
awr_w/behavior 日志行与全部门禁，通过后才启动 4 臂（项目冒烟纪律）。
"""
from pathlib import Path

MORTAL = Path('/mnt/data8/mortal/Mortal/mortal')
SRC = MORTAL / 'config.iql_b_v3.toml'
text = SRC.read_text(encoding='utf-8')

# 各臂唯一的实质变量
ARMS = {
    'a': {'desc': '对照 = v3 原配置（β=3.0）', 'subs': {}},
    'b': {'desc': 'β=1.0（中度锐化）', 'subs': {'iql_beta = 3.0': 'iql_beta = 1.0'}},
    'c': {'desc': 'β=0.5（锐化）', 'subs': {'iql_beta = 3.0': 'iql_beta = 0.5'}},
    'd': {'desc': 'β=10.0（软，≈纯 BC 极限）', 'subs': {'iql_beta = 3.0': 'iql_beta = 10.0'}},
}

# 各臂共同的命名空间/预算替换（相对 v3 正式轮配置；seed=20261016 保持不动 → 四臂数据序配对）
COMMON_SUBS = [
    ('state_file = "checkpoints/iql_b_v3/mortal.pth"',
     'state_file = "checkpoints/iql_beta_{X}/mortal.pth"'),
    ('best_state_file = "checkpoints/iql_b_v3/best.pth"',
     'best_state_file = "checkpoints/iql_beta_{X}/best.pth"'),
    ('tensorboard_dir = "runs/iql_b_v3"', 'tensorboard_dir = "runs/iql_beta_{X}"'),
    ('max_train_steps = 867000', 'max_train_steps = 786000'),   # 756k + 30k 探针
    ('max_train_steps = 786000         # 756k + ~111k = top400 增强样本一遍',
     'max_train_steps = 786000         # 756k + 30k 探针预算'),
    ('test_every = 50000', 'test_every = 30000'),
    ('snapshot_every = 50000', 'snapshot_every = 30000'),
    ('reset_best_perf = "iql-b-v3"', 'reset_best_perf = "iql-beta-{X}"'),
    ('reset_optimizer = "iql-b-v3"', 'reset_optimizer = "iql-beta-{X}"'),
    ('log_dir = "runs/iql_b_v3/test_play"', 'log_dir = "runs/iql_beta_{X}/test_play"'),
    ('log_dir = "runs/iql_b_v3/ladder"', 'log_dir = "runs/iql_beta_{X}/ladder"'),
]

HEADER_TMPL = (
    '# IQL AWR 温度 β 扫描探针 臂{x}（{desc}，2026-10-17）\n'
    '# 目的：30k 步同暖启/同数据/同种子对比 AWR 锐度；τ=0.5 下 β 是策略提取唯一旋钮。\n'
    '# 判据：健康五门 + awr_w 集中度随 β 单调 + avg_pt ≥ 对照+2.0 且 top1 ≥ 对照−0.005。\n'
    '# 与 v3 正式轮的差异仅：命名空间/预算(30k)/本臂旋钮；数据/名单/增强/pts/GRP/暖启/seed 全锁死。\n'
)


def gen(arm_key: str, desc: str, subs: dict, namespace: str, budget: int,
        test_every: int, reset_token: str, header: str,
        extra_subs: dict | None = None) -> Path:
    t = text
    for old, new in COMMON_SUBS:
        assert old in t, f'[{arm_key}] 公共替换未命中: {old}'
        t = t.replace(old, new.replace('{X}', namespace))
    for old, new in subs.items():
        assert old in t, f'[{arm_key}] 臂替换未命中: {old}'
        t = t.replace(old, new, 1)
    for old, new in (extra_subs or {}).items():
        assert old in t, f'[{arm_key}] 额外替换未命中: {old}'
        t = t.replace(old, new, 1)
    # 命名空间/预算/令牌的本臂定制（带注释整行替换，避免数值先改导致注释串失配）
    budget_note = '756k + 2k 冒烟预算' if budget == 758000 else '756k + 30k 探针预算'
    t = t.replace('max_train_steps = 786000         # 756k + 30k 探针预算',
                  f'max_train_steps = {budget}         # {budget_note}')
    t = t.replace('test_every = 30000', f'test_every = {test_every}')
    t = t.replace('snapshot_every = 30000', f'snapshot_every = {test_every}')
    t = t.replace(f'reset_best_perf = "iql-beta-{namespace}"',
                  f'reset_best_perf = "{reset_token}"')
    t = t.replace(f'reset_optimizer = "iql-beta-{namespace}"',
                  f'reset_optimizer = "{reset_token}"')
    # 替换源文件头部注释（首个非注释行之前的所有 # 行）
    lines = t.split('\n')
    i = 0
    while i < len(lines) and lines[i].startswith('#'):
        i += 1
    assert i > 0, '期望源配置首行是注释'
    out = header + '\n'.join(lines[i:])
    dst = MORTAL / f'config.iql_beta_{arm_key}.toml'
    dst.write_text(out, encoding='utf-8')
    print(f'written: {dst}  ({desc})')
    return dst


# 冒烟臂 = a 臂配置 + 2k 预算 + test_play 256 局（与 v3 冒烟同口径，10-17 修正：
# 2000 局在并发负载下需 20-30 分钟，曾导致冒烟窗口内被外部干扰打断）
gen('smoke', '冒烟 = a 臂(β=3.0) + 2k 步预算', {}, 'smoke', 758000, 2000,
    'iql-beta-smoke',
    '# IQL β 扫描预检冒烟（2026-10-17）：= 臂a(β=3.0=v3 原配置) + 2k 步预算。\n'
    '# 目的：验证 train.py 新增 awr_w 集中度与 behavior 日志行 + v3 同款健康门禁全过。\n'
    '# test_play 用 256 局（与 v3 冒烟同口径；2000 局在并发负载下过长易被外部干扰打断）。\n'
    '# 通过后才由 run_iql_beta_probe.sh 启动 4 臂 30k 探针。\n',
    extra_subs={'games = 2000': 'games = 256                   # 冒烟：每次 test_play 只打 64 sets'})

for x, arm in ARMS.items():
    gen(x, arm['desc'], arm['subs'], x, 786000, 30000, f'iql-beta-{x}',
        HEADER_TMPL.format(x=x, desc=arm['desc']))
