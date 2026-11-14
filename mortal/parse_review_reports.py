#!/usr/bin/env python3
"""解析 mjai-reviewer 的 HTML 报告，抽出结构化决策数据（只读）。

背景
----
`mjai.ekyu.moe` 的报告并未只给 argmax。每个被检讨的决策都渲染成
`<details class="collapse entry">`，内含：

  1. 局面元信息：`<summary>12巡 (余45) 2向听 #3/10</summary>`
     —— 巡目 / 余牌 / 向听数 / **人类动作在该模型 Q 排序中的名次与候选数**
  2. 手牌（`ul.tehai-state`，含摸牌与副露 `ul.consumed`）
  3. 人类实际动作（不一致时外层 span 带 `background:#ffd5d5`）
  4. 模型的 argmax 动作
  5. **完整 Q 表**：`Action / Q̂^π(s,a) / π_τ(a|s)×100`，每行一个候选

⇒ 一次提交（1 局 × 1 席）就能拿到约 160 个状态的完整 Q 向量，
不需要任何额外请求。本脚本把这些抽成 JSONL，供师生分歧画像使用。

口径注意
--------
- 只有当候选数 >= 2 的决策点才会被渲染成 entry（与 reviewer 的
  `total_reviewed` 一致），所以 entry 数 == 报告的 `reviewed`。
- Q 是**名次 EV 尺度**（约 -1.5 ~ +0.3），与本仓库 MC 回报尺度的 Q 不可
  直接回归，需先做仿射对齐。
- `π_τ` 是 `softmax(Q/τ)`（τ 见报告元数据），是**同一个 Q 的单调变换**，
  因此不携带 Q 之外的额外信息；保留它是为了核对解析正确性。

用法
----
    python parse_review_reports.py <reports_dir> -o out.jsonl
    python parse_review_reports.py <reports_dir> -o out.jsonl --summary out.md
"""

import argparse
import glob
import html
import json
import os
import re
import statistics
from collections import Counter, defaultdict

# ---------------------------------------------------------------- 正则

RE_REPORT_TAG = re.compile(r'model tag</dt><dd>([^<]*)</dd>')
RE_TEMP = re.compile(r'softmax temperature \(τ\)</dt><dd>([\d.]+)</dd>')
RE_MATCH = re.compile(r'AI\s*一致\s*率</dt><dd>\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)%')
RE_GAME_JSON = re.compile(r'<textarea[^>]*>(\{.*?\})</textarea>', re.S)
RE_KYOKU = re.compile(
    r'<h1 class="kyoku-heading" id="kyoku-(\d+)-(\d+)"><div><a[^>]*>\s*([^<]+?)\s*</a>')
RE_ENTRY_HEAD = re.compile(r'<details class="collapse entry"([^>]*)>')
RE_SUMMARY = re.compile(r'<summary>(.*?)</summary>', re.S)
RE_TURN = re.compile(r'(\d+)巡\s*\(余(\d+)\)')
RE_SHANTEN = re.compile(r'(\d+)向听')
RE_ORDER = re.compile(r'order-loss">\s*#(\d+)</span>/(\d+)')
RE_PAI = re.compile(r'<use class="face" href="#pai-([0-9a-z]+)"')
RE_TSUMO = re.compile(r'<li[^>]*class="tsumo"[^>]*>(.*?)</li>', re.S)
RE_FUJI = re.compile(r'<ul class="consumed">(.*?)</ul>', re.S)
RE_PLAYER = re.compile(
    r'<span(?:\s+style="background:#([0-9a-f]{6})")?\s*>\s*'
    r'<span class="role">玩家:\s*</span>(.*?)</span>\s*<br>', re.S)
RE_MORTAL = re.compile(
    r'<span class="role">Mortal:\s*</span>(.*?)(?=<details><table|</details>)', re.S)
RE_QTABLE = re.compile(r'<table[^>]*class="data">(.*?)</table>', re.S)
RE_TR = re.compile(r'<tr>(.*?)</tr>', re.S)
RE_TD = re.compile(r'<td>(.*?)</td>', re.S)


def _strip(s):
    """去标签 + 去实体 + 压空格。"""
    s = re.sub(r'<[^>]+>', '', s)
    return re.sub(r'\s+', ' ', html.unescape(s)).strip()


def _action(s):
    """把动作片段转成可比较的字符串：牌面 SVG → 牌名。"""
    s = re.sub(r'<svg class="tile"[^>]*>.*?</svg>',
               lambda m: '{' + (RE_PAI.search(m.group(0)).group(1)
                                if RE_PAI.search(m.group(0)) else '?') + '}',
               s, flags=re.S)
    s = re.sub(r'<svg[^>]*class="tile"[^>]*/>', '{}', s)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip(' >')


def _num(cell):
    t = _strip(cell).replace(',', '')
    try:
        return float(t)
    except ValueError:
        return None


def _pais(fragment):
    return RE_PAI.findall(fragment)


def parse_report(path):
    """解析一份报告，返回 (meta, entries)。"""
    uuid = os.path.basename(path).rsplit('.', 1)[0]
    s = open(path, encoding='utf-8', errors='ignore').read()

    # ---- 报告级元数据 ----
    tag = RE_REPORT_TAG.search(s)
    temp = RE_TEMP.search(s)
    match = RE_MATCH.search(s)
    gj = RE_GAME_JSON.search(s)
    game = None
    if gj:
        try:
            game = json.loads(html.unescape(gj.group(1)))
        except json.JSONDecodeError:
            game = None
    meta = {
        'report': uuid,
        'model_tag': tag.group(1) if tag else None,
        'temperature': float(temp.group(1)) if temp else None,
        'matches': int(match.group(1)) if match else None,
        'reviewed': int(match.group(2)) if match else None,
        'rate_pct': float(match.group(3)) if match else None,
        'dan': game.get('dan') if game else None,
        'tenhou_rate': game.get('rate') if game else None,
        'names': game.get('name') if game else None,
        'rule': (game.get('rule') or {}).get('disp') if game else None,
    }

    # ---- 按出现位置排序的「局标题」与「决策块」 ----
    marks = [(m.start(), 'kyoku', m) for m in RE_KYOKU.finditer(s)]
    marks += [(m.start(), 'entry', m) for m in RE_ENTRY_HEAD.finditer(s)]
    marks.sort(key=lambda x: x[0])

    # entry 块的结束位置：下一个标记（任意类型）之前
    entries = []
    cur_kyoku = None
    for i, (pos, kind, m) in enumerate(marks):
        if kind == 'kyoku':
            cur_kyoku = {'idx': int(m.group(1)), 'honba': int(m.group(2)),
                         'label': m.group(3)}
            continue
        end = marks[i + 1][0] if i + 1 < len(marks) else len(s)
        blk = s[pos:end]
        ent = _parse_entry(blk, meta, cur_kyoku)
        if ent:
            entries.append(ent)
    return meta, entries


def _parse_entry(blk, meta, kyoku):
    attrs = blk[:200]
    red = 'data-mark-red' in attrs

    sm = RE_SUMMARY.search(blk)
    summary = _strip(sm.group(1)) if sm else ''
    turn = RE_TURN.search(summary)
    shanten = RE_SHANTEN.search(summary)
    order = RE_ORDER.search(sm.group(1)) if sm else None

    # 手牌：tehai-state 里的牌，tsumo 那张单独标出。
    # ⚠️ tehai-state 内嵌 `<li class="fuuro"><ul class="consumed">…</ul></li>` 展示副露，
    #    必须先删除，否则手牌会多出 3 张副露（实测 14 vs 11），配对全错。
    blk_nofuuro = re.sub(r'<ul class="consumed">.*?</ul>', '', blk, flags=re.S)
    ul = re.search(r'<ul class="tehai-state">(.*?)</ul>', blk_nofuuro, re.S)
    hand, draw = [], None
    if ul:
        seg = ul.group(1)
        tsm = RE_TSUMO.search(seg)
        if tsm:
            draw_p = _pais(tsm.group(1))
            draw = draw_p[0] if draw_p else None
        hand = _pais(seg)

    # 副露：可能有多组
    fuuro = [RE_PAI.findall(b) for b in RE_FUJI.findall(blk)]

    pm = RE_PLAYER.search(blk)
    human = _action(pm.group(2)) if pm else None
    human_bg = pm.group(1) if pm else None

    mm = RE_MORTAL.search(blk)
    mortal = _action(mm.group(1)) if mm else None

    cands = []
    tb = RE_QTABLE.search(blk)
    if tb:
        for tr in RE_TR.findall(tb.group(1))[1:]:       # 跳过表头
            tds = RE_TD.findall(tr)
            if len(tds) < 3:
                continue
            q = _num(tds[1])
            pi = _num(tds[2])
            if q is None:
                continue
            cands.append({'a': _action(tds[0]), 'q': round(q, 5),
                          'pi': round(pi, 4) if pi is not None else None})

    return {
        'report': meta['report'],
        'model_tag': meta['model_tag'],
        'temperature': meta['temperature'],
        'kyoku': kyoku['idx'] if kyoku else None,
        'honba': kyoku['honba'] if kyoku else None,
        'kyoku_label': kyoku['label'] if kyoku else None,
        'turn': int(turn.group(1)) if turn else None,
        'tiles_left': int(turn.group(2)) if turn else None,
        'shanten': int(shanten.group(1)) if shanten else None,
        'human_rank': int(order.group(1)) if order else 1,
        'n_cand': int(order.group(2)) if order else len(cands),
        'consistent': not red,
        'human_bg': human_bg,
        'human': human,
        'mortal': mortal,
        'hand': hand,
        'draw': draw,
        'fuuro': fuuro,
        'cands': cands,
    }


# ---------------------------------------------------------------- 汇总

def summarize(metas, entries):
    lines = []
    A = lines.append

    A('# 解析汇总')
    A('')
    A(f'- 报告数：**{len(metas)}**')
    A(f'- 决策点数：**{len(entries)}**')
    ok = sum(1 for e in entries if e['consistent'])
    A(f'- 一致 / 不一致：**{ok} / {len(entries) - ok}**'
      f'（{ok / len(entries) * 100:.3f}%）')

    # 与报告自报的分子分母核对
    a = sum(m['matches'] or 0 for m in metas)
    b = sum(m['reviewed'] or 0 for m in metas)
    A(f'- 报告自报合计：**{a} / {b} = {a / b * 100:.3f}%**'
      f'（与本脚本逐条统计{"一致" if ok == a and len(entries) == b else "不一致⚠️"}）')

    mr = [m['reviewed'] for m in metas if m['reviewed']]
    if mr:
        A(f'- 每份 reviewed：中位 {statistics.median(mr):.0f}'
          f'（{min(mr)}~{max(mr)}）')

    # Q 与 π 的自洽性（softmax 检验）
    A('')
    A('## 1. Q 与 π 的自洽性（验证 τ 与解析正确性）')
    errs = []
    for e in entries:
        cs = [c for c in e['cands'] if c['q'] is not None and c['pi'] is not None]
        if len(cs) < 2:
            continue
        t = e['temperature'] or 1.0
        mx = max(c['q'] for c in cs)
        w = [pow(2.718281828, (c['q'] - mx) / t) for c in cs]
        sw = sum(w)
        # π 归一化到 1（报告的 π×100 未必总和 100，因为可能只列部分候选）
        pp = [c['pi'] for c in cs]
        sp = sum(pp)
        if sp <= 0:
            continue
        errs.append(max(abs(w[i] / sw - pp[i] / sp) for i in range(len(cs))))
    if errs:
        errs.sort()
        A(f'- 逐决策点比较 `softmax(Q/τ)` 与 `π` 归一化后的最大绝对偏差：'
          f'中位 **{statistics.median(errs):.5f}**、p90 **{errs[int(len(errs) * .9)]:.5f}**、'
          f'max **{errs[-1]:.5f}**（n={len(errs)}）')
        A('- 结论：' + ('两者一致 ⇒ Q 与 π 同源，π 不携带 Q 之外的额外信息，'
                        'Q 表即完整教师信号。' if statistics.median(errs) < 1e-3
                        else '两者不一致 ⚠️ 需检查 τ 或解析。'))

    # 按动作类型的分布与分歧
    A('')
    A('## 2. 按动作类型：分歧率')
    A('')
    A('| 类型 | 决策点 | 分歧 | 分歧率 | 人类平均名次 |')
    A('|---|---:|---:|---:|---:|')

    def kind(e):
        h = (e['human'] or '').strip()
        if h.startswith('打 '):
            return '打牌'
        if h == '跳过' or h == 'pass':
            return '跳过/不鸣'
        for k in ('碰', '吃', '杠', '立直', '荣和', '自摸', '暗杠', '加杠'):
            if k in h:
                return k
        return h or '?'

    by_kind = defaultdict(list)
    for e in entries:
        by_kind[kind(e)].append(e)
    for k, v in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        bad = sum(1 for e in v if not e['consistent'])
        mr_ = statistics.mean(e['human_rank'] for e in v)
        A(f'| {k} | {len(v)} | {bad} | {bad / len(v) * 100:.2f}% | {mr_:.2f} |')

    # 人类动作名次分布
    A('')
    A('## 3. 人类动作在 4.1c 的 Q 排序中的名次')
    A('')
    dist = Counter(min(e['human_rank'], 6) for e in entries)
    tot = len(entries)
    A('| 名次 | 1 | 2 | 3 | 4 | 5 | 6+ |')
    A('|---|---:|---:|---:|---:|---:|---:|')
    A('| 占比 | ' + ' | '.join(
        f'{dist.get(i, 0) / tot * 100:.2f}%' for i in range(1, 7)) + ' |')

    # Q 尺度
    A('')
    A('## 4. Q 的尺度（4.1c，名次 EV）')
    A('')
    spreads, gaps, topqs = [], [], []
    for e in entries:
        qs = [c['q'] for c in e['cands']]
        if len(qs) < 2:
            continue
        spreads.append(max(qs) - min(qs))
        topqs.append(max(qs))
        s2 = sorted(qs, reverse=True)
        gaps.append(s2[0] - s2[1])
    if spreads:
        A(f'- 全候选跨度 max−min：中位 **{statistics.median(spreads):.3f}**'
          f'（p10 {sorted(spreads)[int(len(spreads) * .1)]:.3f} / '
          f'p90 {sorted(spreads)[int(len(spreads) * .9)]:.3f}）')
        A(f'- top1−top2 差距：中位 **{statistics.median(gaps):.3f}**')
        A(f'- top1 的 Q：中位 **{statistics.median(topqs):.3f}**'
          f'（p10 {sorted(topqs)[int(len(topqs) * .1)]:.3f} / '
          f'p90 {sorted(topqs)[int(len(topqs) * .9)]:.3f}）')

    # 候选数
    A('')
    A('## 5. 候选数分布（= reviewer 的 reviewed 口径）')
    A('')
    nc = Counter(e['n_cand'] for e in entries)
    A('候选数 → 决策点数：' + '、'.join(f'{k}:{v}' for k, v in sorted(nc.items())))

    # 每局/每席
    A('')
    A('## 6. 报告级元数据（段位/rating 来自报告内嵌的天凤牌谱）')
    A('')
    A('| 报告 | model | τ | reviewed | 一致率 | 段位 |')
    A('|---|---|---:|---:|---:|---|')
    for m in sorted(metas, key=lambda x: x['report']):
        A(f"| {m['report']} | {m['model_tag']} | {m['temperature']} | "
          f"{m['reviewed']} | {m['rate_pct']:.3f}% | {'/'.join(m['dan'] or [])} |")

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reports_dir')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--summary', default=None)
    ap.add_argument('--results', default=None,
                    help='results.csv；给了就只解析其中列出的 report（排除重复/探针提交）')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.reports_dir, '*.html')))
    if not files:
        raise SystemExit(f'没找到 html：{args.reports_dir}')
    if args.results:
        import csv
        keep = set()
        with open(args.results, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                keep.add(r['report_url'].rstrip('/').split('/')[-1].split('.')[0])
        before = len(files)
        files = [f for f in files
                 if os.path.basename(f).rsplit('.', 1)[0] in keep]
        print(f'按 results.csv 过滤：{before} → {len(files)} 份')
    print(f'报告数：{len(files)}')

    metas, all_entries = [], []
    for f in files:
        meta, entries = parse_report(f)
        metas.append(meta)
        all_entries.extend(entries)
        print(f'  {meta["report"]}  reviewed={meta["reviewed"]:>4}  '
              f'解析={len(entries):>4}  match={meta["matches"]}  '
              f'{"✓" if len(entries) == meta["reviewed"] else "⚠️"}')

    with open(args.out, 'w', encoding='utf-8') as fh:
        for m in metas:
            fh.write(json.dumps({'kind': 'meta', **m}, ensure_ascii=False) + '\n')
        for e in all_entries:
            fh.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'写出 {args.out}：{len(metas)} meta + {len(all_entries)} entries')

    if args.summary:
        md = summarize(metas, all_entries)
        open(args.summary, 'w', encoding='utf-8').write(md + '\n')
        print(f'写出 {args.summary}')


if __name__ == '__main__':
    main()