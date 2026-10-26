#!/usr/bin/env python3
"""akochan 的 pipe 协议垫片（shim）。

三处协议/构建不匹配，在这里一次性补齐：

1. **字段缺失**：akochan 的 `get_kyoku_first()` / `get_aka_flag()` 要求 `start_game`
   里带 `kyoku_first` 与 `aka_flag`，缺失时 `assert_with_out` 直接 abort；
   libriichi 的 `Event::StartGame` 只发 `names`/`seed` → 这里补 `kyoku_first=0`（半荘）、
   `aka_flag=true`（红五）。
2. **回复时机不同**：akochan 只在自己的 `tsumo`、或别人的 `dahai`/`kakan` 上回话；
   而 libriichi 的 `AkochanAgent::react` 每批事件写完就**必读一行**——3 个 akochan 会被
   同时提问，其中两个的最后事件不是自己的触发点 → 永不回话 → 整局死锁。
   ⇒ **已改在 Rust 侧修**（`libriichi/src/agent/akochan.rs`：非触发点直接返回 Event::None、
   不读子进程）；垫片这里只在触发点读一行，非触发点不写任何东西，避免请求/应答错位。
3. **OpenMP 线程越界**：本构建 `NPROCS=2`（见 tools/akochan_build_notes.md），
   OpenMP 线程数若超过 NPROCS，`selector` 的 per-thread 表会越界 → SIGSEGV
   （实测不设时默认 56 线程必崩）→ 这里强制子进程 `OMP_NUM_THREADS=2`。

调试：`AKO_DEBUG=1` 把收到的每条事件写到 `ako_pipe_debug.log`；
非 JSON 输出（akochan 的报错/调试行）进 `ako_pipe_errors.log`，不污染协议。
"""
import json
import os
import select
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'system.bin')
ERR_LOG = os.path.join(HERE, 'ako_pipe_errors.log')
DBG_LOG = os.path.join(HERE, 'ako_pipe_debug.log')

EXTRA = {'kyoku_first': 0, 'aka_flag': True}   # start_game 补齐
OMP_THREADS = '2'                              # 必须 ≤ 编译期 NPROCS
TIMEOUT = float(os.environ.get('AKO_REPLY_TIMEOUT', '180'))
NONE_REPLY = '[{"type":"none"}]'


def log(path: str, s: str) -> None:
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(s.rstrip('\n') + '\n')
    except OSError:
        pass


def read_json_line(proc, deadline: float):
    """读到一行合法 JSON 为止（跳过 akochan 的报错行）；超时/EOF 返回 None。"""
    while True:
        remain = deadline - time.time()
        if remain <= 0:
            return None
        r, _, _ = select.select([proc.stdout], [], [], remain)
        if not r:
            return None
        line = proc.stdout.readline()
        if not line:
            return None
        s = line.decode('utf-8', 'replace').lstrip()
        if s[:1] in ('[', '{'):
            return s.rstrip('\n')
        if s.strip():
            log(ERR_LOG, s)


def main() -> int:
    if len(sys.argv) < 4:
        sys.stderr.write('usage: system.exe pipe <tactics.json> <player_id>\n')
        return 2
    pid = int(sys.argv[3])
    child = subprocess.Popen([BIN] + sys.argv[1:], cwd=HERE,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             env={**os.environ, 'OMP_NUM_THREADS': OMP_THREADS},
                             bufsize=0)
    dbg = os.environ.get('AKO_DEBUG') == '1'

    for raw in sys.stdin.buffer:
        try:
            ev = json.loads(raw.decode('utf-8', 'replace'))
        except Exception:
            child.stdin.write(raw)
            child.stdin.flush()
            continue
        if not isinstance(ev, dict):
            child.stdin.write(raw)
            child.stdin.flush()
            continue
        if ev.get('type') == 'start_game':
            for k, v in EXTRA.items():
                ev.setdefault(k, v)
            raw = (json.dumps(ev, ensure_ascii=False) + '\n').encode('utf-8')
        if dbg:
            log(DBG_LOG, f"pid={pid} {ev.get('type')} actor={ev.get('actor')} "
                         f"can_act={ev.get('can_act')}")
        child.stdin.write(raw)
        child.stdin.flush()

        if ev.get('can_act') is False:
            continue
        t, a = ev.get('type'), ev.get('actor')
        trigger = (a == pid and t == 'tsumo') or (a != pid and t in ('dahai', 'kakan'))
        if not trigger:
            continue          # Rust 侧已改为「非触发点不读」，这里也不能多写一行
        out = read_json_line(child, time.time() + TIMEOUT)
        if out is None:
            log(ERR_LOG, f'pid={pid} 在 {t} actor={a} 上无输出（超时/EOF）')
            out = NONE_REPLY
        sys.stdout.write(out + '\n')
        sys.stdout.flush()

    try:
        child.stdin.close()
    except OSError:
        pass
    child.wait()
    return 0


if __name__ == '__main__':
    sys.exit(main())
