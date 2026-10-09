#!/usr/bin/env python3
"""天凤 scc（鳳凰卓）牌谱采集管线。

流程：抓取 scc 索引 → 过滤生成对局清单 → 串行限速下载 → 转 mjai → 抽样校验。

数据源与红线（详见 TASKLIST.md §3）：
- 索引: https://tenhou.net/sc/raw/list.cgi 与 https://tenhou.net/sc/raw/dat/<file>
- 牌谱: https://tenhou.net/0/log/?<log_id>（响应本身即 gzip 的 XML，原样存盘）
- 官方要求单会话串行；本脚本默认每请求间隔 2 秒，请勿调小到红线以下。

用法示例：
    python tools/tenhou_data/tenhou_pipeline.py fetch-lists       # 近 7 天小时索引
    python tools/tenhou_data/tenhou_pipeline.py fetch-archives    # 滚动归档日文件（当年 1/1 起）
    python tools/tenhou_data/tenhou_pipeline.py build-index
    python tools/tenhou_data/tenhou_pipeline.py stats             # 统计索引中的对局数与预估体积
    python tools/tenhou_data/tenhou_pipeline.py download --delay 2.0
    python tools/tenhou_data/tenhou_pipeline.py convert
    python tools/tenhou_data/tenhou_pipeline.py merge-hf <tar...>  # 合并 HF 年度包（补历史年份）
    python tools/tenhou_data/tenhou_pipeline.py check --sample 30
    python tools/tenhou_data/tenhou_pipeline.py prune --root dataset/mjai/houou4   # 列出/删除非四鳳南喰赤文件
    python tools/tenhou_data/tenhou_pipeline.py all        # 窗口流程一条龙（可反复运行，自动跳过已完成项）

所有步骤均可中断后重跑：索引按文件名跳过，对局清单按 log_id 去重合并，
下载/转换按文件是否存在与 CSV 状态列断点续传。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import io
import json
import re
import shutil
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from tqdm import tqdm

# --- 路径与常量 -------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]  # c:\Maj\Mortal
DATASET_DIR = ROOT / "dataset"
LIST_DIR = DATASET_DIR / "raw" / "lists"
ARCHIVE_DIR = DATASET_DIR / "raw" / "archives"
RAW_DIR = DATASET_DIR / "raw" / "mjlog"
MJAI_DIR = DATASET_DIR / "mjai"
INDEX_CSV = DATASET_DIR / "index" / "houou_games.csv"

LIST_URL = "https://tenhou.net/sc/raw/list.cgi"
DAT_URL = "https://tenhou.net/sc/raw/dat/"
LOG_URL = "https://tenhou.net/0/log/?"

UA = {"User-Agent": "Mozilla/5.0 (personal mahjong research; serial, rate-limited)"}
JST = timezone(timedelta(hours=9))  # 天凤文件名/时间戳均为日本时间

LOGID_RE = re.compile(r"log=([0-9]{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8})")
RULE_RE = re.compile(r"(三|四)(鳳|特|上|般)(東|南)喰赤(速)?")
SCC_NAME_RE = re.compile(r"scc(\d{10})")
LIST_ENTRY_RE = re.compile(r"\{file:'([^']+)',size:(\d+)\}")

FIELDS = ["log_id", "rule", "players", "src", "downloaded", "converted"]

SESSION = requests.Session()
SESSION.headers.update(UA)


def log(msg: str) -> None:
    print(msg, flush=True)


def get(url: str, timeout: int = 60, retries: int = 1) -> bytes:
    """单次 GET（失败时重试 retries 次），返回响应字节。"""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = SESSION.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001 - 网络层异常统一处理
            last = exc
            if attempt < retries:
                time.sleep(3.0)
    raise last  # type: ignore[misc]


def load_rows() -> dict[str, dict]:
    """读取对局清单 CSV，返回 {log_id: row}。"""
    if not INDEX_CSV.exists():
        return {}
    with open(INDEX_CSV, encoding="utf-8", newline="") as fp:
        return {row["log_id"]: row for row in csv.DictReader(fp)}


def save_rows(rows: dict[str, dict]) -> None:
    """按 log_id 排序原子写回对局清单 CSV。

    临时文件+原子替换期间，若有其他进程（如进度看板）正打开目标文件，
    Windows 会拒绝替换（WinError 5/32），故对替换步骤做有限重试。
    """
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    last: PermissionError | None = None
    for _ in range(20):  # 最多重试约 10 秒
        try:
            tmp.replace(INDEX_CSV)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(0.5)
    raise last  # type: ignore[misc]


def ensure_dirs() -> None:
    for path in (LIST_DIR, ARCHIVE_DIR, RAW_DIR, MJAI_DIR, INDEX_CSV.parent):
        path.mkdir(parents=True, exist_ok=True)


# --- 步骤 1: 抓取 scc 索引 --------------------------------------------------


def cmd_fetch_lists(args: argparse.Namespace) -> None:
    ensure_dirs()
    raw = get(LIST_URL).decode("utf-8", "replace")
    files = [name for name, _ in LIST_ENTRY_RE.findall(raw) if name.startswith("scc")]
    files.sort(key=lambda name: SCC_NAME_RE.search(name).group(1), reverse=True)  # 新→旧
    now_jst = datetime.now(JST)
    todo: list[str] = []
    for name in files:
        path = LIST_DIR / name
        match = SCC_NAME_RE.search(name)
        hour = datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=JST)
        # 近 3 小时内的文件可能仍会插入迟到的对局行，需重新抓取
        if path.exists() and not getattr(args, "force", False) and (now_jst - hour) >= timedelta(hours=3):
            continue
        todo.append(name)
    limit = getattr(args, "limit", 0)
    if limit:
        todo = todo[:limit]
    log(f"list.cgi 共 {len(files)} 个 scc 文件，本次需抓取 {len(todo)} 个")
    for name in tqdm(todo, desc="fetch scc lists", unit="file"):
        (LIST_DIR / name).write_bytes(get(DAT_URL + name, retries=2))
        time.sleep(args.delay)
    log(f"索引目录现有 scc 文件: {len(list(LIST_DIR.glob('scc*.gz')))}")


# --- 步骤 1b: 抓取滚动归档（当年日文件；年度 zip 已 404 不可用） --------------


def cmd_fetch_archives(args: argparse.Namespace) -> None:
    """抓取 list.cgi?old 滚动归档中的 scc 日文件（dat/<年>/scc<日期>.html.gz）。

    实测（2026-10-01）：年度 zip（scrawYYYY.zip）全部 404，滚动归档是目前
    唯一可得的日粒度数据源，覆盖当年 1/1 起至约 7 天前（之后由小时窗口接力）。
    """
    ensure_dirs()
    raw_list = get(LIST_URL + "?old").decode("utf-8", "replace")
    entries = [(name, int(size)) for name, size in LIST_ENTRY_RE.findall(raw_list) if "/scc" in name]
    if args.years:
        low, _, high = args.years.partition("-")
        low, high = int(low), int(high or low)
        entries = [(name, size) for name, size in entries if low <= int(name[:4]) <= high]
    log(f"滚动归档中 scc 日文件 {len(entries)} 个（{sum(s for _, s in entries) / 1048576:.1f} MB）")
    todo = [(name, size) for name, size in entries if not (LIST_DIR / name.split("/")[-1]).exists()]
    if args.limit:
        todo = todo[: args.limit]
    log(f"本次需抓取 {len(todo)} 个（{sum(s for _, s in todo) / 1048576:.1f} MB）")
    for name, _ in tqdm(todo, desc="fetch archive lists", unit="file"):
        (LIST_DIR / name.split("/")[-1]).write_bytes(get(DAT_URL + name, retries=2))
        time.sleep(args.delay)


# --- 步骤 2: 解析 + 过滤 → 对局清单 CSV -------------------------------------


def iter_games(path: Path):
    """解析单个 scc 索引文件，产出通过质量过滤的 (log_id, rule, players)。

    兼容两种来源：7 天窗口的小时文件（html-escaped）与归档日文件（原始 HTML）。
    """
    text = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
    for line in text.splitlines():
        line = html.unescape(line)
        log_id_match = LOGID_RE.search(line)
        rule_match = RULE_RE.search(line)
        if not (log_id_match and rule_match):
            continue
        num, table, length, speed = rule_match.groups()
        # 质量过滤（§4）：四人 / 鳳凰卓 / 半庄（南）/ 喰赤，排除速棋
        if num != "四" or table != "鳳" or length != "南" or speed:
            continue
        players = line.split("牌譜")[-1].strip()
        players = re.sub(r"^</a>\s*\|\s*", "", players)
        players = re.sub(r"<br>\s*$", "", players.strip())
        yield log_id_match.group(1), rule_match.group(0), players


def cmd_build_index(args: argparse.Namespace) -> None:
    ensure_dirs()
    rows = load_rows()
    added = 0
    for path in sorted(LIST_DIR.glob("scc*.gz")):
        for log_id, rule, players in iter_games(path):
            if log_id in rows:
                if not rows[log_id].get("src"):
                    rows[log_id]["src"] = path.name
                continue
            rows[log_id] = {
                "log_id": log_id,
                "rule": rule,
                "players": players[:200],
                "src": path.name,
                "downloaded": "",
                "converted": "",
            }
            added += 1
    save_rows(rows)
    log(f"清单新增 {added} 局，总计 {len(rows)} 局 → {INDEX_CSV}")


def cmd_stats(args: argparse.Namespace) -> None:
    """统计清单目录内全部索引（窗口 + 归档）的去重对局数，按月份汇总并估算体积。"""
    games: dict[str, str] = {}
    for path in sorted(LIST_DIR.glob("scc*.gz")):
        for log_id, _rule, _players in iter_games(path):
            games.setdefault(log_id, log_id[:6])
    counts = Counter(games.values())
    log(f"去重后四鳳南对局总数: {len(games)}")
    for month in sorted(counts):
        log(f"  {month}: {counts[month]}")
    raw_mb = len(games) * args.raw_kb / 1024
    mjai_mb = len(games) * args.mjai_kb / 1024
    log(
        f"预估磁盘（raw {args.raw_kb}KB/局 + mjai {args.mjai_kb}KB/局）: "
        f"raw {raw_mb:.1f} MB + mjai {mjai_mb:.1f} MB = {raw_mb + mjai_mb:.1f} MB"
    )


# --- 步骤 3: 串行限速下载牌谱 -----------------------------------------------


def cmd_download(args: argparse.Namespace) -> None:
    ensure_dirs()
    rows = load_rows()
    # 已存在文件的对局先补正状态
    for row in rows.values():
        if (RAW_DIR / f"{row['log_id']}.xml.gz").exists() and row.get("downloaded") != "ok":
            row["downloaded"] = "ok"
    pending = [row for row in sorted(rows.values(), key=lambda r: r["log_id"])
               if not (RAW_DIR / f"{row['log_id']}.xml.gz").exists()]
    if args.limit:
        pending = pending[: args.limit]
    log(f"待下载 {len(pending)} 局（总计 {len(rows)} 局，间隔 {args.delay}s）")
    chunk = getattr(args, "chunk", 0)
    chunk_sleep = getattr(args, "chunk_sleep", 900)
    cooldown = getattr(args, "cooldown", 1800)
    max_cooldowns = getattr(args, "max_cooldowns", 3)
    consecutive_fails = 0
    cooldowns_used = 0
    done_in_chunk = 0
    ok = fails = 0
    progress = tqdm(pending, desc="download mjlog", unit="game")
    try:
        for i, row in enumerate(progress, 1):
            dst = RAW_DIR / f"{row['log_id']}.xml.gz"
            try:
                data = get(LOG_URL + row["log_id"], retries=1)
                if data[:2] == b"\x1f\x8b":
                    payload = data  # 原始 gzip 字节，直接落盘（勿二次压缩）
                elif data[:5] == b"<mjlo" or data[:5] == b"<?xml":
                    # requests 自动解压了 Content-Encoding: gzip，补压一次保持 .xml.gz 格式
                    payload = gzip.compress(data, mtime=0)
                else:
                    raise ValueError(f"unexpected payload ({len(data)} bytes)")
                dst.write_bytes(payload)
                row["downloaded"] = "ok"
                ok += 1
                consecutive_fails = 0
                done_in_chunk += 1
            except Exception as exc:  # noqa: BLE001
                row["downloaded"] = f"fail:{type(exc).__name__}:{str(exc)[:60]}"
                fails += 1
                consecutive_fails += 1
                if consecutive_fails >= args.max_fail:
                    if cooldowns_used < max_cooldowns:
                        cooldowns_used += 1
                        save_rows(rows)
                        log(
                            f"连续 {consecutive_fails} 次失败，疑似被限流；"
                            f"冷却 {cooldown:.0f}s 后继续（{cooldowns_used}/{max_cooldowns}）"
                        )
                        time.sleep(cooldown)
                        consecutive_fails = 0
                    else:
                        log(f"连续失败达 {args.max_fail} 且冷却次数已用尽，提前停止")
                        break
            if i % 50 == 0:
                save_rows(rows)
            time.sleep(args.delay)
            if chunk and done_in_chunk >= chunk:
                # 例行冷却：避免长时间连续请求触发天凤限流（2026-10-02 实测教训）
                save_rows(rows)
                log(f"已完成 {chunk} 局，例行冷却 {chunk_sleep:.0f}s")
                time.sleep(chunk_sleep)
                done_in_chunk = 0
    finally:
        save_rows(rows)
        progress.close()
    log(f"下载完成: 成功 {ok}，失败 {fails}（失败行保留，重跑自动重试）")


def cmd_verify_raw(args: argparse.Namespace) -> None:
    """校验 raw 目录下 gzip/XML 完整性；损坏文件（中断残留）直接删除，将由下载步骤自动重下。"""
    files = sorted(RAW_DIR.glob("*.xml.gz"))
    bad: list[Path] = []
    for path in tqdm(files, desc="verify raw", unit="file"):
        try:
            if gzip.decompress(path.read_bytes())[:8] != b"<mjloggm":
                bad.append(path)
        except Exception:  # noqa: BLE001
            bad.append(path)
    for path in bad:
        path.unlink()
    log(f"raw 校验: 共 {len(files)} 个，损坏 {len(bad)} 个" + ("（已删除，将自动重下）" if bad else ""))
    for path in bad[:10]:
        log(f"  [损坏] {path.name}")


# --- 步骤 4: mjlog XML → mjai（.json.gz） -----------------------------------

# jansou 把里宝牌指示牌写在 hora 的 uradora_markers 键下，
# 而 Mortal/libriichi 读取的键名是 ura_markers（validate_logs 强制要求存在）。
def _fix_hora_ura(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        obj = json.loads(line)
        if obj.get("type") == "hora":
            if "uradora_markers" in obj:
                obj["ura_markers"] = obj.pop("uradora_markers")
            else:
                obj.setdefault("ura_markers", [])
        out_lines.append(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(out_lines) + "\n"


def _convert_one(raw_path: Path) -> bytes:
    """单局转换：读 mjlog → Paifu → mjai JSONL（含键名修正与结构自查）。"""
    from jansou.io.mjai import dump_mjai
    from jansou.io.mjlog import parse_mjlog

    paifu = parse_mjlog(raw_path)
    text = _fix_hora_ura(dump_mjai(paifu))

    # 结构自查（Mortal GameplayLoader 的实际要求）
    objs = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not objs or objs[0]["type"] != "start_game":
        raise ValueError("first line is not start_game")
    if not any(obj["type"] == "start_kyoku" and "scores" in obj for obj in objs):
        raise ValueError("no start_kyoku with scores")
    for obj in objs:
        if obj["type"] in ("hora", "ryukyoku") and "deltas" not in obj:
            raise ValueError(f"{obj['type']} without deltas")
        if obj["type"] == "hora" and "ura_markers" not in obj:
            raise ValueError("hora without ura_markers")
    return gzip.compress(text.encode("utf-8"), mtime=0)


def cmd_convert(args: argparse.Namespace) -> None:
    ensure_dirs()
    rows = load_rows()
    pending = []
    for row in sorted(rows.values(), key=lambda r: r["log_id"]):
        raw_path = RAW_DIR / f"{row['log_id']}.xml.gz"
        if raw_path.exists() and row.get("converted") != "ok":
            pending.append((row, raw_path))
    if args.limit:
        pending = pending[: args.limit]
    log(f"待转换 {len(pending)} 局")
    ok = fails = 0
    progress = tqdm(pending, desc="convert to mjai", unit="game")
    for i, (row, raw_path) in enumerate(progress, 1):
        log_id = row["log_id"]
        dst = MJAI_DIR / "houou4" / log_id[:6] / f"{log_id}.json.gz"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(_convert_one(raw_path))
            row["converted"] = "ok"
            ok += 1
        except Exception as exc:  # noqa: BLE001
            row["converted"] = f"fail:{type(exc).__name__}:{str(exc)[:60]}"
            fails += 1
        if i % 100 == 0:
            save_rows(rows)
    save_rows(rows)
    log(f"转换完成: 成功 {ok}，失败 {fails}")


# --- 步骤 4b: 合并 HF 年度包（.mjai.json.zst → .json.gz） --------------------

# NikkeTryHard/tenhou-to-mjai 数据集：
#   https://huggingface.co/datasets/nikketryhard/tenhou-houou-mjai （国内用 hf-mirror.com）
# 每包 tar 内为 <log_id>.mjai.json.zst（zstd 压缩的 mjai JSONL），已实测与 Mortal 兼容
# （hora 键名就是 ura_markers）。合并时按 log_id 去重：目标已存在（无论来源）则跳过。
HF_MANIFEST = DATASET_DIR / "index" / "hf_merge_manifest.csv"


def _append_manifest(rows: list[dict]) -> None:
    """把新增文件的明细追加到合并清单（便于溯源/回退）。"""
    if not rows:
        return
    write_header = not HF_MANIFEST.exists()
    with open(HF_MANIFEST, "a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["log_id", "src", "ts"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _merge_one_tar(tar_path: Path, dest_root: Path) -> tuple[Counter, list[dict]]:
    """合并单个年度包到 dest_root/houou4/YYYYMM/，按 log_id 去重。返回 (计数, 新增明细)。"""
    import zstandard

    src = tar_path.stem
    totals: Counter = Counter()
    merged_rows: list[dict] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getnames() if m.endswith(".mjai.json.zst")]
        log(f"{tar_path.name}: {len(members)} 局")
        for member in tqdm(members, desc=f"merge {src}", unit="game"):
            log_id = Path(member).name.removesuffix(".mjai.json.zst")
            dst = dest_root / "houou4" / log_id[:6] / f"{log_id}.json.gz"
            if dst.exists():
                totals["skipped"] += 1
                continue
            try:
                data = tf.extractfile(member).read()
                text = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read().decode("utf-8")
                # 轻量结构自查（数据源已抽验 120/120，此处防单文件异常）
                head = text.split("\n", 1)[0]
                if '"start_game"' not in head or '"start_kyoku"' not in text or '"end_game"' not in text:
                    raise ValueError("mjai structure check failed")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
                merged_rows.append({"log_id": log_id, "src": src, "ts": now})
                totals["merged"] += 1
            except Exception as exc:  # noqa: BLE001
                totals["failed"] += 1
                log(f"  [失败] {log_id}: {type(exc).__name__}: {exc}")
    return totals, merged_rows


def cmd_merge_hf(args: argparse.Namespace) -> None:
    ensure_dirs()
    dest_root = Path(args.dest) if getattr(args, "dest", "") else MJAI_DIR
    totals: Counter = Counter()
    all_rows: list[dict] = []
    for tar_path in [Path(p) for p in args.tars]:
        t, rows = _merge_one_tar(tar_path, dest_root)
        totals.update(t)
        all_rows += rows
    _append_manifest(all_rows)
    log(
        f"合并完成: 新增 {totals['merged']}，去重跳过 {totals['skipped']}，失败 {totals['failed']}"
        f"（目标 {dest_root}，新增明细见 {HF_MANIFEST.name}）"
    )


def cmd_merge_hf_batch(args: argparse.Namespace) -> None:
    """按年批量合并（用于全量归档到 D 盘）。

    每年一个批次：合并 → 写完成标记 → 批间停顿（降低 D 盘长时间连续写入风险）。
    续传：年份有完成标记则跳过；无标记的重跑时按 log_id 去重续合并。
    """
    ensure_dirs()
    src_dir = Path(args.src_dir)
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)
    marker_dir = dest_root / ".merged"
    marker_dir.mkdir(parents=True, exist_ok=True)
    years = _expand_years(args.years)
    log(f"批量合并 {len(years)} 年：{src_dir} → {dest_root}（批间停顿 {args.sleep:.0f}s）")
    grand: Counter = Counter()
    for year in years:
        tar_path = src_dir / f"tenhou-houou-{year}.tar"
        marker = marker_dir / year
        if marker.exists():
            log(f"[跳过] {year}: 已有完成标记（{marker.read_text('utf-8').strip()}）")
            continue
        if not tar_path.exists():
            log(f"[跳过] {year}: 缺少 {tar_path.name}")
            continue
        log(f"=== {year} 开始（{datetime.now().strftime('%H:%M:%S')}）===")
        totals, rows = _merge_one_tar(tar_path, dest_root)
        _append_manifest(rows)
        grand.update(totals)
        if totals["failed"]:
            log(f"[警告] {year}: 失败 {totals['failed']} 局，不写完成标记（重跑自动续）")
        else:
            marker.write_text(
                f"merged={totals['merged']} skipped={totals['skipped']} "
                f"ts={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
                encoding="utf-8",
            )
            log(f"[完成] {year}: 新增 {totals['merged']}，跳过 {totals['skipped']}（{datetime.now().strftime('%H:%M:%S')}）")
            if getattr(args, "delete_src", False):
                try:
                    freed = tar_path.stat().st_size
                    tar_path.unlink()
                    log(f"[清理] {year}: 已删除源包 {tar_path.name}（释放 {freed / 1024 / 1024:.1f} MB）")
                except OSError as exc:  # noqa: BLE001
                    log(f"[清理] {year}: 删除源包失败：{exc}")
        log(f"批间停顿 {args.sleep:.0f}s 保护 D 盘")
        time.sleep(args.sleep)
    log(f"批量合并结束: 累计新增 {grand['merged']}，跳过 {grand['skipped']}，失败 {grand['failed']}")


# --- 步骤 4c: 下载 HF 年度包到 D 盘（先 C 临时 → 复制 D → 删 C 原文件） -------

HF_TREE_API = "https://{mirror}/api/datasets/nikketryhard/tenhou-houou-mjai/tree/main"
HF_RESOLVE = "https://{mirror}/datasets/nikketryhard/tenhou-houou-mjai/resolve/main/{name}"


def _expand_years(text: str) -> list[str]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(part))
    return [str(y) for y in sorted(out)]


def _download_resume(url: str, dst: Path, expected: int, timeout: int = 60) -> None:
    """流式下载（支持断点续传），完成后按 expected 校验大小。"""
    pos = dst.stat().st_size if dst.exists() else 0
    headers = dict(UA)
    if pos:
        headers["Range"] = f"bytes={pos}-"
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        if pos and resp.status_code == 206:
            pass
        elif resp.status_code == 200:
            pos = 0
            dst.write_bytes(b"")
        else:
            raise RuntimeError(f"HTTP {resp.status_code}")
        with open(dst, "ab" if pos else "wb") as fp:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fp.write(chunk)
    if dst.stat().st_size != expected:
        raise RuntimeError(f"大小不符 {dst.stat().st_size} != {expected}")


def cmd_fetch_hf(args: argparse.Namespace) -> None:
    """逐包：下载到 C 盘临时区 → 复制到 D 盘 → 校验 → 删除 C 盘原文件。

    全串行、一次只处理一个包，避免长时间连续写 D 盘（用户本机 D 盘长时间写入易卡死）。
    断点续传：D 盘已存在且大小一致 → 跳过；C 盘临时已存在且完整 → 直接复用。
    """
    mirror = args.mirror
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = ARCHIVE_DIR / "hf"
    tmp_root.mkdir(parents=True, exist_ok=True)

    sizes_raw = json.loads(get(HF_TREE_API.format(mirror=mirror)).decode("utf-8"))
    sizes = {
        x["path"]: x["size"]
        for x in sizes_raw
        if x.get("type") == "file" and x["path"].startswith("tenhou-houou-") and x["path"].endswith(".tar")
    }
    log(f"HF 数据集共 {len(sizes)} 个年度包（{sum(sizes.values()) / 1e9:.1f} GB）")

    for year in _expand_years(args.years):
        name = f"tenhou-houou-{year}.tar"
        if name not in sizes:
            log(f"[跳过] {name}: 数据集无此包")
            continue
        expected = sizes[name]
        dst = dest_dir / name
        tmp = tmp_root / name

        # 1) 目标已在 D 盘且完整 → 跳过（顺带清理 C 盘残留）
        if dst.exists() and dst.stat().st_size == expected:
            if tmp.exists():
                tmp.unlink()
            log(f"[跳过] {name}: D 盘已就位（{expected / 1e9:.2f} GB）")
            continue

        # 2) 下载到 C 盘临时区（已有且完整则复用）
        if tmp.exists() and tmp.stat().st_size == expected:
            log(f"[复用] {name}: C 盘已有完整临时文件")
        else:
            url = HF_RESOLVE.format(mirror=mirror, name=name)
            ok = False
            for attempt in range(1, args.retries + 1):
                try:
                    log(f"[下载] {name} {expected / 1e9:.2f} GB → C 盘临时（尝试 {attempt}/{args.retries}）")
                    _download_resume(url, tmp, expected)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001
                    log(f"  下载异常: {exc}")
                    tmp.unlink(missing_ok=True)  # 损坏就删掉重来
            if not ok:
                raise RuntimeError(f"{name} 下载失败，已停止（重试后可续传）")

        # 3) 复制到 D 盘（一次性拷贝，随后校验）
        log(f"[复制] {name} → {dst}")
        shutil.copyfile(tmp, dst)
        if dst.stat().st_size != expected:
            dst.unlink(missing_ok=True)
            raise RuntimeError(f"{name} 复制后大小不符，目标已删除（C 盘临时保留，重跑即可）")

        # 4) 删除 C 盘原文件
        tmp.unlink()
        log(f"[完成] {name}（{expected / 1e9:.2f} GB），C 盘原文件已删除")
        time.sleep(args.delay)

    done = sum(1 for y in _expand_years(args.years) if (dest_dir / f"tenhou-houou-{y}.tar").exists())
    log(f"fetch-hf 结束：D 盘 {dest_dir} 现有 {done} 个完整包")


# --- 步骤 5: 抽样校验 -------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> None:
    import random
    from glob import glob as _glob

    from jansou.io.mjai import parse_mjai
    from jansou.validation.check import check_paifu

    pattern = getattr(args, "glob", "")
    if pattern:
        files = sorted(Path(p) for p in _glob(pattern, recursive=True))
    else:
        files = sorted(MJAI_DIR.rglob("*.json.gz"))
    if not files:
        log("尚无转换产物，跳过校验")
        return
    random.seed(20261001)
    sample = random.sample(files, min(args.sample, len(files)))

    struct_ok = roundtrip_ok = 0
    wins = wins_passed = 0
    problems: list[str] = []
    log(f"抽样校验 {len(sample)} / {len(files)} 个文件")
    for path in tqdm(sample, desc="check samples", unit="file"):
        try:
            text = gzip.decompress(path.read_bytes()).decode("utf-8")
            objs = [json.loads(line) for line in text.splitlines() if line.strip()]
            assert objs[0]["type"] == "start_game"
            assert sum(1 for obj in objs if obj["type"] == "start_kyoku") >= 1
            assert any(obj["type"] == "start_kyoku" and "scores" in obj for obj in objs)
            struct_ok += 1
            paifu = parse_mjai(text)  # 回环解析
            roundtrip_ok += 1
            verdicts = check_paifu(paifu)  # 回放并复核和牌点数
            wins += len(verdicts)
            wins_passed += sum(1 for v in verdicts if v.passed)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path.name}: {type(exc).__name__}: {exc}")

    # 清单层面统计
    rows = load_rows()
    downloaded = sum(1 for r in rows.values() if r.get("downloaded") == "ok")
    converted = sum(1 for r in rows.values() if r.get("converted") == "ok")
    log("")
    log(f"清单: 总计 {len(rows)} 局 | 已下载 {downloaded} | 已转换 {converted}")
    log(f"抽样: 结构自查通过 {struct_ok}/{len(sample)} | 回环解析通过 {roundtrip_ok}/{len(sample)}")
    log(f"回放计分: 和牌 {wins} 次，复核通过 {wins_passed} 次")
    for problem in problems[:10]:
        log(f"  [问题] {problem}")


# --- 步骤 6: 清理非"真实的四鳳南喰赤"文件 -------------------------------------

# log_id 第二段是天凤 GO type 的十六进制，实测与 scc 规则文本一一对应
# （全量 13 万条验证）：00a9=四鳳南喰赤、00e9=四鳳南喰赤速、00b9=三鳳南喰赤、
# 00e1=四鳳東喰赤速、00b1=三鳳東喰赤……故训练所需规则的唯一编码即 00a9。
RULE_CODE_OK = "00a9"
LOGID_NAME_RE = re.compile(
    r"^([0-9]{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8})\.json\.gz$"
)
PLACEHOLDER_NAME_RE = re.compile(r"^EXAMPLE\d*$", re.IGNORECASE)


def _mjai_reject_reason(path: Path, log_id: str) -> str:
    """判定单个 mjai 文件是否为"真实的四鳳南喰赤对局"。合格返回 ""，否则返回原因。"""
    parts = log_id.split("-")
    if len(parts) != 4:
        return "log_id 结构异常"
    if parts[1] != RULE_CODE_OK:
        return f"规则编码 {parts[1]}≠{RULE_CODE_OK}（速棋/三麻/东风/其他）"

    try:
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"解压失败: {type(exc).__name__}"

    objs: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            return "存在非法 JSON 行"
        if not isinstance(obj, dict):
            return "存在非对象 JSON 行"
        objs.append(obj)

    if not objs:
        return "空文件"
    if objs[0].get("type") != "start_game":
        return "首行不是 start_game"
    if objs[-1].get("type") != "end_game":
        return "末行不是 end_game（疑似截断）"

    kyokus = [o for o in objs if o.get("type") == "start_kyoku"]
    if not kyokus:
        return "缺少 start_kyoku"
    if any("scores" not in o for o in kyokus):
        return "start_kyoku 缺 scores"
    if any(len(o["scores"]) != 4 for o in kyokus):
        return "非四人对局（scores 长度≠4）"
    if not any(o.get("type") in ("hora", "ryukyoku") for o in objs):
        return "无和了/流局记录（空局或截断）"

    head = objs[0]
    rules = head.get("rules")
    if isinstance(rules, dict):  # 本地 jansou 格式：规则可完整自证
        if rules.get("player_count") != 4:
            return f'rules.player_count={rules.get("player_count")}≠4'
        if rules.get("game_length") != "S":
            return f'rules.game_length={rules.get("game_length")!r}≠"S"（非半庄）'
        if rules.get("kuitan") is not True:
            return "rules.kuitan≠true（无食断）"
        if rules.get("aka_dora") is not True:
            return "rules.aka_dora≠true（无赤牌）"
    elif head.get("aka_flag") is not True:  # HF 原始 mjai 格式：仅赤牌可判
        return "aka_flag≠true（无赤牌）"

    names = head.get("names")
    if isinstance(names, list) and any(
        PLACEHOLDER_NAME_RE.match(str(n)) for n in names
    ):
        return "占位玩家名（EXAMPLE…，非真实对局）"
    return ""


def _prune_one(
    path: Path, rejects: list[tuple[Path, str]], misplaced: list[Path]
) -> None:
    """校验单个文件，把结果追加到对应桶。"""
    m = LOGID_NAME_RE.match(path.name)
    if not m:
        rejects.append((path, "文件名不是合法 log_id"))
        return
    log_id = m.group(1)
    if path.parent.name != log_id[:6]:
        misplaced.append(path)
    reason = _mjai_reject_reason(path, log_id)
    if reason:
        rejects.append((path, reason))


def cmd_prune(args: argparse.Namespace) -> None:
    """扫描 mjai 树，删除非"真实的四鳳南喰赤"文件（默认 dry-run）。

    按 <root>/YYYYMM/ 分桶逐月扫描：不做一次性 sorted(rglob)，避免百万级文件在
    收集/排序阶段长时间无输出且吃内存；每月一条带总数的进度条。
    """
    root = Path(args.root) if args.root else (MJAI_DIR / "houou4")
    if not root.exists():
        log(f"目录不存在: {root}")
        return

    months = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda p: p.name) or [root]
    mode = "APPLY（会真删）" if args.apply else "DRY-RUN（只看不删）"
    log("=" * 72)
    log(f"prune 启动 | 根目录: {root}")
    log(f"模式: {mode} | 分桶: {len(months)} 个目录 | 心跳间隔: {args.progress_every or '关闭'}")
    log("=" * 72)

    rejects: list[tuple[Path, str]] = []  # 非真实四鳳南喰赤 → 待删除
    misplaced: list[Path] = []  # 目录与 log_id 前缀不符（真实对局，仅告警）
    total = 0
    t_start = time.time()
    stop = False

    try:
        for mi, mdir in enumerate(months, 1):
            files = sorted(mdir.glob("*.json.gz"), key=lambda p: p.name)
            log(f"[{mi}/{len(months)}] {mdir.name}: {len(files)} 个文件，开始校验…")
            t_month = time.time()
            processed = 0
            bar = tqdm(
                files,
                desc=f"prune {mdir.name}",
                unit="file",
                leave=False,
                file=sys.stdout,
                mininterval=1.0,
            )
            for idx, path in enumerate(bar, 1):
                processed = idx
                total += 1
                _prune_one(path, rejects, misplaced)
                if args.progress_every and idx % args.progress_every == 0:
                    bar.set_postfix(不合格=len(rejects), 错位=len(misplaced))
                    tqdm.write(
                        f"    … {mdir.name} 已校验 {idx}/{len(files)} | 累计 {total} | "
                        f"不合格 {len(rejects)} | 错位 {len(misplaced)} | 本目录 {time.time() - t_month:.0f}s",
                        file=sys.stdout,
                    )
                if args.limit and total >= args.limit:
                    stop = True
                    break
            bar.close()
            log(
                f"    {mdir.name} 完成: 本目录 {processed}/{len(files)} | 累计 {total} | "
                f"不合格 {len(rejects)} | 错位 {len(misplaced)} | 耗时 {time.time() - t_month:.1f}s"
            )
            if stop:
                log(f"    触发 --limit {args.limit}，提前结束扫描")
                break
    except KeyboardInterrupt:
        log("\n!! 收到 Ctrl+C，已停止扫描（尚未删除任何文件）")

    log("=" * 72)
    log(f"扫描结束: 共 {total} 个文件 | 总耗时 {time.time() - t_start:.1f}s")
    log(f"不合格文件 {len(rejects)} 个 | 目录错位 {len(misplaced)} 个（不删除，仅告警）")

    cap = args.show
    if rejects:
        log("不合格原因分布:")
        for reason, cnt in Counter(r for _, r in rejects).most_common():
            log(f"  {cnt:>7}  {reason}")

    show = len(rejects) if cap == 0 else min(cap, len(rejects))
    if show:
        log(f"\n待删除清单（显示 {show}/{len(rejects)} 条）：")
        for path, reason in rejects[:show]:
            log(f"  {path}  <- {reason}")
        if show < len(rejects):
            log(f"  ... 其余 {len(rejects) - show} 条未显示（--show 0 可全部打印）")

    if misplaced:
        mis_show = len(misplaced) if cap == 0 else min(cap, len(misplaced))
        log("\n目录错位（真实对局，建议人工确认后再处理，本次不删除）：")
        for path in misplaced[:mis_show]:
            log(f"  {path}")

    if not args.apply:
        log("\n[dry-run] 未删除任何文件。确认无误后加 --apply 执行删除。")
        return

    log(f"\n开始删除 {len(rejects)} 个文件…")
    failed = 0
    for path, _ in tqdm(rejects, desc="删除", unit="file", leave=False, file=sys.stdout):
        try:
            path.unlink()
        except OSError as exc:
            failed += 1
            log(f"  [删除失败] {path}: {exc}")

    log("清理空目录…")
    removed_dirs = 0
    for d in sorted({p.parent for p, _ in rejects}, key=str):
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
                removed_dirs += 1
        except OSError:
            pass
    log(f"已删除 {len(rejects) - failed} 个文件（失败 {failed}），清理空目录 {removed_dirs} 个。")
    log("=" * 72)


# --- 入口 -------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    download_delay = args.delay
    args.delay = 1.0  # 索引文件很小，1s 间隔即可
    cmd_fetch_lists(args)
    args.delay = download_delay
    cmd_build_index(args)
    cmd_download(args)
    cmd_convert(args)
    cmd_check(args)


def _add_download_args(parser: argparse.ArgumentParser) -> None:
    """download / all 共用参数（限速、冷却、失败保护）。"""
    parser.add_argument("--limit", type=int, default=0, help="最多下载 N 局（调试用，0 为不限）")
    parser.add_argument("--delay", type=float, default=2.0, help="每局间隔秒数（默认 2.0）")
    parser.add_argument("--max-fail", type=int, default=20, help="连续失败 N 次触发冷却/停止（默认 20）")
    parser.add_argument("--chunk", type=int, default=0, help="每成功下载 N 局例行冷却一次（0 为不启用）")
    parser.add_argument("--chunk-sleep", type=float, default=900, help="例行冷却秒数（默认 900）")
    parser.add_argument("--cooldown", type=float, default=1800, help="疑似被限流后的冷却秒数（默认 1800）")
    parser.add_argument("--max-cooldowns", type=int, default=3, help="限流冷却重试次数上限（默认 3）")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="天凤 scc 牌谱采集管线（详见 TASKLIST.md）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-lists", help="抓取近 7 天 scc 索引文件（默认跳过已存在）")
    p.add_argument("--force", action="store_true", help="强制重新下载全部索引")
    p.add_argument("--limit", type=int, default=0, help="最多抓取 N 个文件（调试用）")
    p.add_argument("--delay", type=float, default=1.0, help="请求间隔秒数（默认 1.0）")
    p.set_defaults(func=cmd_fetch_lists)

    p = sub.add_parser("fetch-archives", help="抓取滚动归档 scc 日文件（当年 1/1 起，年度 zip 已 404）")
    p.add_argument("--years", default="", help='年份过滤，如 "2026" 或 "2009-2025"（默认全部）')
    p.add_argument("--limit", type=int, default=0, help="最多抓取 N 个文件（调试用）")
    p.add_argument("--delay", type=float, default=0.5, help="请求间隔秒数（默认 0.5）")
    p.set_defaults(func=cmd_fetch_archives)

    p = sub.add_parser("build-index", help="解析 scc 并过滤（四鳳南喰赤）生成对局清单 CSV")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("stats", help="统计索引中的去重对局数（按月份）与预估磁盘体积")
    p.add_argument("--raw-kb", type=float, default=7.38, help="单局 raw 平均 KB（默认实测 7.38）")
    p.add_argument("--mjai-kb", type=float, default=5.03, help="单局 mjai 平均 KB（默认实测 5.03）")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("download", help="串行限速下载牌谱（可断点续传）")
    _add_download_args(p)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("verify-raw", help="校验 raw 牌谱完整性（清理中断残留的损坏文件）")
    p.set_defaults(func=cmd_verify_raw)

    p = sub.add_parser("convert", help="mjlog XML → mjai .json.gz")
    p.add_argument("--limit", type=int, default=0, help="最多转换 N 局（调试用）")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("merge-hf", help="合并 HF 年度包（.mjai.json.zst → .json.gz，按 log_id 去重）")
    p.add_argument("tars", nargs="+", help="tar 文件路径（可多个）")
    p.add_argument("--dest", default="", help="目标根目录（默认 dataset/mjai；输出到 <dest>/houou4/YYYYMM/）")
    p.set_defaults(func=cmd_merge_hf)

    p = sub.add_parser("merge-hf-batch", help="按年批量合并 HF 年度包（每年一批、批间停顿、可续传）")
    p.add_argument("--years", default="2009-2025", help='年份范围，如 "2009-2025"')
    p.add_argument("--src-dir", default=r"D:\tenhou_hf", help="tar 包所在目录（默认 D:\\tenhou_hf）")
    p.add_argument("--dest", default=r"D:\tenhou_mjai", help="目标根目录（默认 D:\\tenhou_mjai）")
    p.add_argument("--sleep", type=float, default=120, help="批间停顿秒数（默认 120，保护 D 盘）")
    p.add_argument("--delete-src", action="store_true", help="每年合并成功后删除该年源包（释放 D 盘空间）")
    p.set_defaults(func=cmd_merge_hf_batch)

    p = sub.add_parser(
        "fetch-hf",
        help="下载 HF 年度包到 D 盘：先下载到 C 盘临时 → 复制到 D 盘 → 校验 → 删除 C 盘原文件",
    )
    p.add_argument("--years", default="2009-2026", help='年份范围，如 "2010-2025" 或 "2010,2015"')
    p.add_argument("--dest", default=r"D:\tenhou_hf", help="D 盘目标目录（默认 D:\\tenhou_hf）")
    p.add_argument("--mirror", default="hf-mirror.com", help="HF 镜像域名（默认 hf-mirror.com）")
    p.add_argument("--retries", type=int, default=3, help="单包下载重试次数（默认 3）")
    p.add_argument("--delay", type=float, default=0.5, help="包间间隔秒数（默认 0.5）")
    p.set_defaults(func=cmd_fetch_hf)

    p = sub.add_parser("check", help="抽样校验转换产物（结构 + 回环解析 + 回放计分）")
    p.add_argument("--sample", type=int, default=30, help="抽样文件数（默认 30）")
    p.add_argument("--glob", default="", help="只校验匹配该 glob 的文件（如 'dataset/mjai/houou4/2009*/*.json.gz'）")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser(
        "prune",
        help="清理非『真实的四鳳南喰赤』的 mjai 文件（默认只 dry-run，加 --apply 才删除）",
    )
    p.add_argument("--root", default="", help="mjai 根目录（默认 dataset/mjai/houou4，内含 YYYYMM/ 子目录）")
    p.add_argument("--apply", action="store_true", help="实际执行删除（不加则仅列出待删清单）")
    p.add_argument("--show", type=int, default=0, help="最多打印多少条待删/错位清单（0=全部，默认 0）")
    p.add_argument("--limit", type=int, default=0, help="最多扫描 N 个文件（调试用）")
    p.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="每校验 N 个文件打印一次进度心跳（0=关闭，默认 10000）",
    )
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("all", help="一条龙: fetch-lists → build-index → download → convert → check")
    _add_download_args(p)
    p.add_argument("--sample", type=int, default=30, help="校验抽样数")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()