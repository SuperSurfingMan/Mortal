#!/usr/bin/env python3
"""修复自建 2026 牌谱缺失的 start_game.names（数据侧 P0，2026-10-12）。

**背景 / 为什么需要这个工具**
  jansou 的 `dump_mjai` 产出的 `start_game` 只有 `rules`，没有 `names`。而
  libriichi 的 `GameplayLoader::load_events` 在白名单模式下是按
  `start_game.names` 过滤座位的（libriichi/src/dataset/gameplay.rs:169），
  Rust 侧 `StartGame.names` 又是 `#[serde(default)]` → 缺字段时静默变成
  4 个空串 → 白名单永远匹配不上 → **2026 自建的 13 万局对白名单训练完全不产出样本**。

  但原始 mjlog 并没有丢信息：`<UN>` 节点保留 `n0~n3`（四座位名称，另有 dan/rate/sx）。
  所以这是转换管线的丢失，本工具负责补回。

  ⚠ **第二个坑（2026-10-12 实测踩到）**：`sc/raw/dat` 下载回来的 mjlog 里，`n0~n3`
  是 **URL 百分号编码**的玩家名（每个日文字符 → `%XX%XX%XX`，值长 45~63，纯 ASCII），
  **必须先 `urllib.parse.unquote` 才能得到真名**。直接写入会得到永远匹配不上白名单的
  垃圾名（且不会报错，静默无用）——首轮实现就犯过这个错，靠"抽样比对白名单命中率"
  才发现。本工具现在自动 unquote，并对解码后仍含 `%XX` 的名字计数告警。

**做法（不动现有数据，保证 R1~R4 可复现）**
  - 只修补「缺 names」的文件，输出到独立的补丁树 `--dst`（默认 `dataset/mjai/houou4_names`）；
  - 除 `names` 外**逐字节保留**原文件其余内容（只重写第一行 JSON，其余原样拼接）；
  - gzip 用 `mtime=0` 确定性压缩，重跑结果可复现；已存在的输出默认跳过（可断点续跑）；
  - 已有 names 的文件**不复制** → dst 是补丁树而非全量树，训练 globs 需同时列出两棵；
  - 统计报告写在 `dataset/index/repair_names_report.json`（**不要放补丁树内**：validate_logs
    的 `**/*.json` 会把它当牌谱解析并报错）。

**用法（在仓库根目录 /mnt/data8/mortal/Mortal 下运行）**
    # 冒烟：只修 50 个，人工检查输出
    python tools/tenhou_data/repair_names.py --months 2026 --limit 50

    # 正式：修复全部 2026 月份（28 进程）
    python tools/tenhou_data/repair_names.py --months 2026 --jobs 28

    # 审计全树（慢，用来确认 HF 部分是否真的都有 names）
    python tools/tenhou_data/repair_names.py --months all --jobs 28

**修好之后**（训练侧接入，见 NEXT_STEPS_ANALYSIS_20261012.md §2.1）
    dataset.globs = [
        '.../dataset/mjai/houou4/202[2-5]??/**/*.json.gz',   # HF 历史（本就有 names）
        '.../dataset/mjai/houou4_names/2026*/**/*.json.gz',  # 本工具补丁树
    ]
  同时把 `dataset.data_version` 设成新值（如 'names-fix-v1'）→ file_index 的 manifest
  会记录它，旧索引自动失效重建，不会静默复用旧文件表。
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import gzip
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / 'dataset/mjai/houou4'
DEFAULT_RAW = REPO / 'dataset/raw/mjlog'
DEFAULT_DST = REPO / 'dataset/mjai/houou4_names'
DEFAULT_REPORT = REPO / 'dataset/index/repair_names_report.json'

SUFFIX_MJAI = '.json.gz'
SUFFIX_RAW = '.xml.gz'

# 解码后仍匹配 %XX 的名字视为可疑（双重编码/解码失败），只报告不阻断
_RE_PCT = re.compile(r'%[0-9A-Fa-f]{2}')


def names_from_raw(raw_path: Path) -> tuple[list[str] | None, int]:
    """从 mjlog XML 的 <UN> 读四座位名称（n0~n3，座位序）。

    实测（2026-10-12）：`sc/raw/dat` 下载回来的 mjlog 里 n0~n3 是 **URL 百分号编码**
    的玩家名（每个日文字符 → %XX%XX%XX，值长 45~63），直接写进 mjai 会得到永远
    匹配不上白名单的垃圾名 → 必须先 urllib.parse.unquote。
    返回 (names, n_suspicious)；names 为 None 表示该文件无法恢复。
    """
    try:
        with gzip.open(raw_path, 'rb') as f:
            root = ET.fromstring(f.read())
    except Exception:  # noqa: BLE001 - 单个坏文件不应中断整轮
        return None, 0
    un = root.find('UN')
    if un is None:
        return None, 0
    names = []
    n_suspicious = 0
    for i in range(4):
        v = urllib.parse.unquote((un.attrib.get(f'n{i}') or '').strip())
        if _RE_PCT.search(v):
            n_suspicious += 1
        names.append(v)
    return (names if all(names) else None), n_suspicious


def repair_one(job: tuple[Path, Path, Path, bool]) -> tuple[str, str, str]:
    """修补单个文件。返回 (status, log_id, detail)。"""
    src, dst, raw_dir, overwrite = job
    log_id = src.name[: -len(SUFFIX_MJAI)]

    try:
        with gzip.open(src, 'rb') as f:
            raw = f.read()
    except Exception as exc:  # noqa: BLE001
        return ('unreadable', log_id, type(exc).__name__)

    first, sep, rest = raw.partition(b'\n')
    if not sep:
        return ('no_lines', log_id, '')
    try:
        obj = json.loads(first)
    except Exception as exc:  # noqa: BLE001
        return ('bad_json', log_id, type(exc).__name__)
    if obj.get('type') != 'start_game':
        return ('bad_start', log_id, str(obj.get('type'))[:40])

    names = obj.get('names')
    if (isinstance(names, list) and len(names) == 4
            and all(isinstance(n, str) and n for n in names)):
        return ('already_named', log_id, '')

    new_names, n_susp = names_from_raw(raw_dir / f'{log_id}{SUFFIX_RAW}')
    if new_names is None:
        return ('no_raw', log_id, '')

    # 只改第一行：插入 names，其余字节原样拼回（键序无所谓，serde 不关心）
    obj['names'] = new_names
    new_first = json.dumps(obj, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

    dst_path = dst / src.parent.name / src.name
    if dst_path.exists() and not overwrite:
        return ('skipped_exists', log_id, str(n_susp))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.name + '.tmp')
    try:
        tmp.write_bytes(gzip.compress(new_first + sep + rest, mtime=0))
        tmp.replace(dst_path)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return ('write_failed', log_id, type(exc).__name__)
    return ('repaired', log_id, str(n_susp))


def main() -> None:
    ap = argparse.ArgumentParser(
        description='从原始 mjlog 恢复 2026 牌谱缺失的 start_game.names（补丁树，不覆盖原数据）')
    ap.add_argument('--src', default=str(DEFAULT_SRC), help='源 mjai 树（含 YYYYMM 子目录）')
    ap.add_argument('--raw', default=str(DEFAULT_RAW), help='原始 mjlog 目录（<log_id>.xml.gz）')
    ap.add_argument('--dst', default=str(DEFAULT_DST), help='补丁输出树（只放修好的文件）')
    ap.add_argument('--months', default='2026',
                    help="月份过滤：'2026' / '202[2-5]' / 'all' / 逗号分隔多个；支持 fnmatch")
    ap.add_argument('--jobs', type=int, default=28)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个文件（冒烟用）')
    ap.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出（默认跳过）')
    ap.add_argument('--report', default=str(DEFAULT_REPORT),
                    help='统计报告输出路径（默认 dataset/index/；不要放补丁树内，'
                         '否则会被 validate_logs 的 **/*.json 误当牌谱）')
    ap.add_argument('--progress-every', type=int, default=20000)
    args = ap.parse_args()

    src, raw_dir, dst = Path(args.src), Path(args.raw), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f'源树不存在: {src}')
    if not raw_dir.is_dir():
        raise SystemExit(f'原始 mjlog 目录不存在: {raw_dir}')

    all_months = sorted(d.name for d in src.iterdir() if d.is_dir() and d.name.isdigit())
    if args.months.strip().lower() == 'all':
        months = all_months
    else:
        pats = [m.strip() for m in args.months.split(',') if m.strip()]
        months = [m for m in all_months
                  if any(m.startswith(p) or fnmatch.fnmatch(m, p) for p in pats)]
    if not months:
        raise SystemExit(f'月份过滤 {args.months!r} 没匹配到任何目录（可选: {all_months[0]}~{all_months[-1]}）')

    files: list[Path] = []
    for m in months:
        files.extend(sorted((src / m).glob(f'*{SUFFIX_MJAI}')))
    if args.limit:
        files = files[: args.limit]

    print(f'源树     {src}', flush=True)
    print(f'原始 mjlog {raw_dir}', flush=True)
    print(f'补丁树   {dst}', flush=True)
    print(f'月份     {months[0]}~{months[-1]}（{len(months)} 个月）| 待扫文件 {len(files):,}', flush=True)

    if not args.limit:
        raw_ids = {p.name[: -len(SUFFIX_RAW)] for p in raw_dir.glob(f'*{SUFFIX_RAW}')}
        src_ids = {p.name[: -len(SUFFIX_MJAI)] for p in files}
        print(f'参考：raw 共 {len(raw_ids):,} 个 mjlog，其中 {len(raw_ids - src_ids):,} 个没有对应 mjai'
              f'（转换失败或被 prune，不在本工具范围）', flush=True)

    t0 = time.time()
    counter: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    n_susp_names = 0   # unquote 后仍含 %XX 的名字（双重编码嫌疑），来自 repaired 的 detail
    jobs = [(f, dst, raw_dir, args.overwrite) for f in files]

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (status, log_id, detail) in enumerate(ex.map(repair_one, jobs, chunksize=64), 1):
            counter[status] += 1
            if status == 'repaired' and detail.isdigit():
                n_susp_names += int(detail)
            if status not in ('repaired', 'already_named', 'skipped_exists') and len(examples[status]) < 5:
                examples[status].append(f'{log_id}({detail})' if detail else log_id)
            if i % args.progress_every == 0:
                rate = i / (time.time() - t0)
                print(f'  {i:,}/{len(files):,} | {rate:.0f} files/s | {dict(counter)}', flush=True)

    el = time.time() - t0
    print(f'\n扫描完成: {len(files):,} 文件 / {el / 60:.2f} min', flush=True)
    for k in sorted(counter):
        line = f'  {k:<16} {counter[k]:>8,}'
        if examples.get(k):
            line += f'  例: {", ".join(examples[k])}'
        print(line, flush=True)

    dst_files = sorted(dst.rglob(f'*{SUFFIX_MJAI}')) if dst.is_dir() else []
    dst_bytes = sum(p.stat().st_size for p in dst_files)
    print(f'\n补丁树现状: {len(dst_files):,} 文件 / {dst_bytes / 2**20:.1f} MiB'
          f'（含历史运行产物；本次新增 {counter["repaired"]:,}）', flush=True)

    report = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'src': str(src), 'raw': str(raw_dir), 'dst': str(dst),
        'months': [months[0], months[-1]], 'n_months': len(months),
        'n_scanned': len(files), 'elapsed_sec': round(el, 1),
        'counts': dict(counter),
        'suspicious_names': n_susp_names,
        'dst_files': len(dst_files), 'dst_bytes': dst_bytes,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'报告已写 {report_path}', flush=True)

    if n_susp_names:
        print(f'\n[警告] 有 {n_susp_names:,} 个座位名 unquote 后仍含 %XX（双重编码嫌疑）；'
              f'投喂训练前请先抽样核对', flush=True)
    bad = counter['unreadable'] + counter['bad_json'] + counter['bad_start'] + counter['write_failed']
    if bad:
        print(f'\n[警告] 有 {bad:,} 个文件异常，请检查上面的示例', flush=True)
    if counter['repaired'] == 0 and counter['skipped_exists'] == 0:
        print('\n[提示] 本次没有修补任何文件（可能都已修好或都缺原始 mjlog）', flush=True)


if __name__ == '__main__':
    main()
