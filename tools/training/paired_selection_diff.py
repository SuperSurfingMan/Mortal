"""同种子块下的逐局配对比较：消除牌山差异，直接看两个 checkpoint 的相对表现。

用法:
    python tools/training/paired_selection_diff.py \
        --a logs_or_runs/selection_r2 --b runs/selection_r3
两个目录里的文件名形如 <seed>_<key>_<seat>.json.gz，且挑战者在 start_game.names 里固定记作
'mortal'（座位随 a/b/c/d 轮换）。对每个共同文件累计最终得分 → 排名 → 顺位点 [90,45,0,-135]，
再按“同名文件同牌山”配对求差，给出均值与配对 SE（比独立样本 SE 小得多）。
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mjai_score import challenger_pt  # noqa: E402  权威计分口径（与 stat.rs 对齐）


def chal_pt(path):
    """模块级薄包装：供 ProcessPoolExecutor 调用；实现在 tools/mjai_score.py。"""
    return challenger_pt(path, name='mortal')


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, '*.json.gz')):
        v = chal_pt(p)
        if v is not None:
            out[os.path.basename(p)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='较早的 checkpoint 评测目录')
    ap.add_argument('--b', required=True, help='较晚的 checkpoint 评测目录')
    ap.add_argument('--na', default='A(较早)')
    ap.add_argument('--nb', default='B(较晚)')
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit(f'没有共同文件：A={len(A)} B={len(B)}（目录或命名不对）')
    d = [B[k] - A[k] for k in common]
    mA = sum(A[k] for k in common) / len(common)
    mB = sum(B[k] for k in common) / len(common)
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
    se = math.sqrt(var / len(d))
    print(f'{args.na}: avg_pt = {mA:+.3f}   ({len(common)} 局共同牌山)')
    print(f'{args.nb}: avg_pt = {mB:+.3f}')
    print(f'配对差 Δ = {mean:+.3f} pt/局   配对 SE = {se:.3f}   t = {mean / se:+.2f}'
          f'   95%CI = [{mean - 1.96 * se:+.2f}, {mean + 1.96 * se:+.2f}]')
    un = math.sqrt((var_of(A, common) + var_of(B, common)) / len(common))
    print(f'若不配对（把两批当独立样本）SE 会是 ±{un:.2f} → 配对把噪声压到 ±{se:.2f}'
          f'（缩小 {un / max(se, 1e-9):.1f}×）')
    win = sum(1 for x in d if x > 0)
    lose = sum(1 for x in d if x < 0)
    print(f'逐局胜负：{args.nb} 更好 {win} 局，{args.na} 更好 {lose} 局，同分 {len(d) - win - lose} 局')


def var_of(D, keys):
    m = sum(D[k] for k in keys) / len(keys)
    return sum((D[k] - m) ** 2 for k in keys) / (len(keys) - 1)


main()
