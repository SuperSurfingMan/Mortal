#!/usr/bin/env python3
"""汇总绝对锚点：把各进程的 `challenger rankings` 累加，给出 Mortal 的 avg_rank / avg_pt。

uma 口径 90/45/0/−135 与 dataset/index/player_ratings_*.csv 同尺度；
人类场参考（5356 人）：均值 −5.44 / 中位 −2.61 / p95 +9.29。
"""
import argparse, glob, re, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    a = ap.parse_args()
    tot = [0, 0, 0, 0]; games = 0; files = 0
    for f in sorted(glob.glob(f'{a.dir}/p*.log')):
        txt = open(f, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'challenger rankings:\s*\[(\d+) (\d+) (\d+) (\d+)\]\s*\(\s*([-\d.]+),\s*([-\d.]+)pt\)', txt):
            r = list(map(int, m.groups()[:4])); tot = [x + y for x, y in zip(tot, r)]
            games += sum(r); files += 1
    if not games:
        print('未找到结果行'); return 1
    uma = [90, 45, 0, -135]
    avg_rank = sum((i + 1) * c for i, c in enumerate(tot)) / games
    avg_pt = sum(c * u for c, u in zip(tot, uma)) / games
    print(f'[anchor] {files} 个进程 / {games} 局')
    print(f'[anchor] 名次分布(1..4): {tot}')
    print(f'[anchor] Mortal avg_rank = {avg_rank:.3f}   avg_pt = {avg_pt:+.2f}')
    print(f'[anchor] 人类场标尺（houou 2022-2026H1）：均值 -5.44 / 中位 -2.61 / p95 +9.29')
    se = 82 / (games ** 0.5)
    print(f'[anchor] 近似 SE = {se:.2f}（SD≈82，未计组内相关）→ 95%CI ≈ [{avg_pt-1.96*se:+.1f}, {avg_pt+1.96*se:+.1f}]')
    return 0

if __name__ == '__main__':
    sys.exit(main())
