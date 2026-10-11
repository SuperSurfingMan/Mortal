"""交叉校验：用 libriichi::Stat 同口径从 selection 目录复算 avg_rank / avg_pt，与 TB 记录对齐。

计分口径统一收敛到 tools/mjai_score.py（权威实现，与 libriichi/src/stat.rs 对齐）。
用途：任何"我们自己算的对局统计"与 Rust Stat 出现分歧时，先用它复算一遍确认口径。
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mjai_score import load_events, seat_of, final_scores, player_by_rank, PTS  # noqa: E402


def one(path):
    """返回该局里 'mortal' 的名次（1..4）；不含该玩家则 None。"""
    events = load_events(path)
    seat = seat_of(events, 'mortal')
    if seat is None:
        return None
    return player_by_rank(final_scores(events)).index(seat) + 1


for d in sys.argv[1:]:
    ranks = []
    for p in glob.glob(f'{d}/*.json.gz'):
        r = one(p)
        if r is None:
            continue
        ranks.append(r)
    n = len(ranks)
    pt = sum(PTS[r - 1] for r in ranks) / n
    dist = [ranks.count(i) / n for i in (1, 2, 3, 4)]
    print(f'{d}: 局数={n}  avg_rank={sum(ranks)/n:.4f}  avg_pt={pt:+.3f}  '
          f'1/2/3/4位率={[round(x, 4) for x in dist]}')
