"""mjai 牌谱计分的权威口径（与 libriichi/src/stat.rs::from_game + rankings.rs 对齐）。

规则（三处工具此前各自复制实现，且都有同一个 bug，故收敛到此处）：
  ① 每一局 start_kyoku 都从事件里的 scores 重置（不是只在首局）；
  ② reach_accepted 时该玩家 −1000（立直宣言的 1000 点供托**不出现在任何 deltas 里**）；
  ③ hora / ryukyoku 的 deltas 直接累加（已含供托的其他支付）；
  ④ 排名 = 按最终分数稳定排序、平分按座位号小者优先；
     排名**不含**终局残供补齐（stat.rs 里补供发生在 Rankings::new 之后，只影响 `point` 不影响名次）。

用法：
    from mjai_score import load_events, final_scores, player_by_rank, challenger_pt
"""
from __future__ import annotations

import gzip
import json

PTS = [90.0, 45.0, 0.0, -135.0]   # 天凤顺位点（仅用于展示/判定，与训练无关）


def load_events(path: str) -> list[dict]:
    """读一个 mjai .json.gz，返回事件字典列表（每行一个 JSON）。"""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def final_scores(events: list[dict]) -> list[float]:
    """按权威口径算四家最终分数（不含终局残供补齐，与 rank 口径一致）。"""
    scores = [25000.0, 25000.0, 25000.0, 25000.0]
    for o in events:
        t = o.get('type')
        if t == 'start_kyoku':
            s = o.get('scores')
            if isinstance(s, list) and len(s) == 4:
                scores = [float(x) for x in s]
        elif t == 'reach_accepted':
            a = o.get('actor')
            if isinstance(a, int) and 0 <= a < 4:
                scores[a] -= 1000.0
        elif t in ('hora', 'ryukyoku'):
            d = o.get('deltas')
            if isinstance(d, list) and len(d) == 4:
                for i in range(4):
                    scores[i] += float(d[i])
    return scores


def player_by_rank(scores: list[float]) -> list[int]:
    """返回按名次从高到低的座位号（稳定排序 → 平分按座位号小者优先）。"""
    return sorted(range(4), key=lambda i: -scores[i])


def seat_of(events: list[dict], name: str) -> int | None:
    """start_game.names 里找指定名字的座位号；找不到返回 None。"""
    for o in events:
        if o.get('type') == 'start_game':
            names = o.get('names') or []
            return names.index(name) if name in names else None
    return None


def challenger_pt(path: str, name: str = 'mortal') -> float | None:
    """单局（半庄）里指定玩家的顺位点；该名不在局中则返回 None。"""
    events = load_events(path)
    seat = seat_of(events, name)
    if seat is None:
        return None
    rank = player_by_rank(final_scores(events)).index(seat)
    return PTS[rank]
