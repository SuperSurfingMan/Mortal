#!/usr/bin/env python3
"""在真实牌谱上验证 gamma / apply_gamma / dones / steps_to_done 的语义（2026-10-12）。

**为什么需要它**（NEXT_STEPS_ANALYSIS.md §3.2）：
  MEMORY 一直把 `gamma<1` 描述为"跨局信用分配"，但代码事实并非如此：
  `steps_to_done` 在**局边界归零**（train.py/dataloader.py 里 `if not dones[i]` 才累加），
  而奖励取的是 `kyoku_rewards[at_kyoku[i]]`（当前局）→ gamma 只在**当前局内**给较早动作打折。
  此外 `add_entry` 的注释写 "only discard and kan will discount"，实现却是 `label <= 37`
  （= 打牌 0..36 + 立直 37；**不含**杠 42）。

**它检查什么**（任一不变量失败 → 退出码 1）：
  1. `apply_gamma[i] == (label[i] <= 37)`（在真实牌谱上核对 Rust 实现的判据）；
  2. `dones` 只出现在局边界（`at_kyoku` 变化处），且末位为 True；
  3. **折扣视界 = 当前局内**：`steps_to_done[i]` 等于本局内 i 及其后（不含局末那一条）的
     apply_gamma 之和 —— 跨局从不累计；
  4. 给出「动作 id × 是否折扣」的全局分布（看清 37/42/43/44/45 各是什么待遇）。

用法（仓库根目录，CPU 即可，秒级）：
    python tools/training/inspect_gamma_semantics.py --globs \\
        'dataset/mjai/houou4_names/2026*/*.json.gz' --n-files 200 --gamma 0.995
    python tools/training/inspect_gamma_semantics.py --files a.json.gz b.json.gz   # 指定文件
"""
from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'mortal'))          # libriichi.so 在 mortal/ 下

from libriichi.dataset import GameplayLoader  # noqa: E402

ACTION_NAMES = {
    **{i: '打牌' for i in range(37)},
    37: '立直',
    38: '吃(低)', 39: '吃(中)', 40: '吃(高)',
    41: '碰',
    42: '杠',
    43: '自摸和/和了',
    44: '九种九牌',
    45: 'pass',
}


def action_label(a: int) -> str:
    return ACTION_NAMES.get(a, f'未知({a})')


def load_gameplays(files):
    loader = GameplayLoader(version=4, oracle=False, player_names=None, excludes=None, augmented=False)
    out = []
    for file in loader.load_gz_log_files(files):
        for game in file:
            out.append({
                'actions': game.take_actions(),
                'apply_gamma': game.take_apply_gamma(),
                'dones': game.take_dones(),
                'at_kyoku': game.take_at_kyoku(),
                'at_turns': game.take_at_turns(),
            })
    return out


def check_invariants(games):
    """返回 (失败项列表, 统计)。"""
    fails = []
    n_moves = 0
    action_counts = {}
    action_gamma_true = {}
    first_bad_gamma = None
    first_bad_done = None
    first_bad_horizon = None
    examples = []

    for game in games:
        actions = game['actions']
        apply_gamma = game['apply_gamma']
        dones = game['dones']
        at_kyoku = game['at_kyoku']
        at_turns = game['at_turns']
        n = len(actions)
        if n == 0:
            continue
        n_moves += n

        # 1) apply_gamma 判据
        for i, (a, g) in enumerate(zip(actions, apply_gamma)):
            action_counts[a] = action_counts.get(a, 0) + 1
            if g:
                action_gamma_true[a] = action_gamma_true.get(a, 0) + 1
            if bool(g) != (a <= 37) and first_bad_gamma is None:
                first_bad_gamma = (a, bool(g))

        # 2) dones 只在局边界
        for i in range(n):
            expect = (at_kyoku[i + 1] > at_kyoku[i]) if i + 1 < n else True
            if bool(dones[i]) != expect and first_bad_done is None:
                first_bad_done = (i, bool(dones[i]), expect)

        # 3) 折扣视界 = 当前局内（steps_to_done 定义与"不跨局"性质一起验）
        steps_to_done = [0] * n
        for i in reversed(range(n - 1)):
            if not dones[i]:
                steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])
        if len(at_kyoku) > 0:
            kyoku_end = {}
            for i in range(n):
                kyoku_end[at_kyoku[i]] = i          # 每局最后一条动作的下标
            for i in range(n):
                end = kyoku_end[at_kyoku[i]]
                expect = sum(int(apply_gamma[j]) for j in range(i, end))
                if steps_to_done[i] != expect and first_bad_horizon is None:
                    first_bad_horizon = (i, steps_to_done[i], expect)
            if len(examples) < 3:
                examples.append({
                    'at_kyoku': at_kyoku, 'actions': actions, 'at_turns': at_turns,
                    'steps_to_done': steps_to_done, 'dones': dones, 'apply_gamma': apply_gamma,
                })

    if first_bad_gamma is not None:
        fails.append(f'apply_gamma 判据不符: 首个例外 action={first_bad_gamma[0]} '
                     f'apply_gamma={first_bad_gamma[1]}（期望 {first_bad_gamma[0] <= 37}）')
    if first_bad_done is not None:
        fails.append(f'dones 未落在局边界: 首个例外 index={first_bad_done[0]} '
                     f'dones={first_bad_done[1]} 期望={first_bad_done[2]}')
    if first_bad_horizon is not None:
        fails.append(f'折扣视界不是"当前局内": 首个例外 index={first_bad_horizon[0]} '
                     f'steps_to_done={first_bad_horizon[1]} 期望={first_bad_horizon[2]}')
    return fails, {
        'n_moves': n_moves,
        'action_counts': action_counts,
        'action_gamma_true': action_gamma_true,
    }, examples


def print_action_table(stats):
    print('\n== 动作 id × 折扣分布（全局）==')
    print(f'{"label":>5s} {"名称":<14s} {"次数":>9s} {"apply_gamma=True":>16s}  折扣')
    ids = sorted(stats['action_counts'])
    # 0..36 合并成一行打印，其余逐个
    if 0 in stats['action_counts']:
        cnt = sum(stats['action_counts'].get(i, 0) for i in range(37))
        gt = sum(stats['action_gamma_true'].get(i, 0) for i in range(37))
        print(f'{"0-36":>5s} {"打牌":<14s} {cnt:>9,} {gt:>16,}  是')
    for a in ids:
        if a <= 36:
            continue
        cnt = stats['action_counts'][a]
        gt = stats['action_gamma_true'].get(a, 0)
        discount = '是' if gt == cnt and cnt > 0 else ('否' if gt == 0 else f'部分({gt}/{cnt})')
        mark = '  ← 与旧注释"only discard and kan"不符' if a == 42 and gt == 0 else ''
        print(f'{a:>5d} {action_label(a):<14s} {cnt:>9,} {gt:>16,}  {discount}{mark}')


def print_example(examples, gamma):
    if not examples:
        return
    ex = examples[0]
    print(f'\n== 示例：局内折扣（gamma={gamma}，设每局奖励 R=1，展示 q_target 的相对大小）==')
    print(f'{"i":>5s} {"kyoku":>5s} {"turn":>5s} {"action":<14s} {"apply_gamma":>11s} '
          f'{"steps_to_done":>13s} {"gamma**s":>10s} {"dones":>6s}')
    shown = 0
    for i in range(len(ex['actions'])):
        if shown >= 24:
            print('  …（省略）')
            break
        a = ex['actions'][i]
        print(f'{i:>5d} {ex["at_kyoku"][i]:>5d} {ex["at_turns"][i]:>5d} {action_label(a):<14s} '
              f'{str(bool(ex["apply_gamma"][i])):>11s} {ex["steps_to_done"][i]:>13d} '
              f'{gamma ** ex["steps_to_done"][i]:>10.6f} {str(bool(ex["dones"][i])):>6s}')
        shown += 1
    # 用第一个局边界说明"跨局不累计"
    dones = ex['dones']
    at_kyoku = ex['at_kyoku']
    for i in range(len(dones)):
        if dones[i] and i + 1 < len(dones) and at_kyoku[i + 1] > at_kyoku[i]:
            print(f'  ↑ i={i} 是本局最后一条动作：steps_to_done=0（不被折扣）；'
                  f'i={i+1} 起已是第 {at_kyoku[i+1]} 局，计数从 0 重新开始 → '
                  f'后续局的奖励没有回传给前一局（不是跨局信用分配）')
            break


def main() -> None:
    ap = argparse.ArgumentParser(description='验证 gamma / apply_gamma / dones / steps_to_done 语义')
    ap.add_argument('--globs', nargs='*', default=[],
                    help="牌谱 glob（默认 dataset/mjai/houou4_names/2026*/*.json.gz）")
    ap.add_argument('--files', nargs='*', default=[], help='直接指定文件（优先于 --globs）')
    ap.add_argument('--n-files', type=int, default=200, help='最多取多少个文件')
    ap.add_argument('--gamma', type=float, default=0.995, help='示例里用的 gamma（默认 0.995）')
    args = ap.parse_args()

    files = list(args.files)
    if not files:
        globs = args.globs or [str(REPO / 'dataset/mjai/houou4_names/2026*/*.json.gz')]
        for pat in globs:
            files.extend(globmod.glob(pat, recursive=True))
    files = sorted(files)[: args.n_files]
    if not files:
        raise SystemExit('没有可用的牌谱文件')

    print(f'牌谱 {len(files)} 个（如 {Path(files[0]).name} …）')
    games = load_gameplays(files)
    print(f'座位 gameplays {len(games)} 个')
    fails, stats, examples = check_invariants(games)
    print(f'动作样本 {stats["n_moves"]:,}')

    print('\n== 不变量检查 ==')
    for name, key in (('apply_gamma == (label <= 37)', 'g'),
                      ('dones 只在局边界（末位 True）', 'd'),
                      ('折扣视界 = 当前局内（跨局不累计）', 'h')):
        bad = [f for f in fails if key == 'g' and 'apply_gamma' in f
               or key == 'd' and 'dones' in f
               or key == 'h' and '视界' in f]
        print(f'  [{"OK" if not bad else "FAIL"}] {name}' + (f'    {bad[0]}' if bad else ''))
    if not stats['action_counts'].get(42):
        print('  [提示] 本次样本里没有杠(42)动作，建议加大 --n-files 再确认 42 的折扣待遇')

    print_action_table(stats)
    print_example(examples, args.gamma)

    print('\n== 结论 ==')
    print('  1. 参与折扣的只有 打牌(0-36) 与 立直(37)（`label <= 37`）；鸣牌/杠/和了/pass 均不折扣。')
    print('  2. steps_to_done 在局边界归零，且奖励取 kyoku_rewards[at_kyoku[i]] →')
    print('     gamma<1 只在**当前局内**给较早动作打折，不含任何后续局信息。')
    print('  3. 因此"gamma<1 = 跨局信用分配"的说法不成立；真正的跨局 credit assignment')
    print('     需要显式引入下一局状态与未来回报（数据侧 + target network），属单独的框架改动。')

    if fails:
        print('\nFAIL:')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('\nALL PASS')


if __name__ == '__main__':
    main()
