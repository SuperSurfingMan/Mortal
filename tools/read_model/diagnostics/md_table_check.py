import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _common
from _common import (CKPT, DEV, EPS, OUT_DIR, SUIT, VAL_GLOB, boot_ci, build_eval,
                     col_ce, counts_tier, fmt_ci, forward, load_net, parse_tl,
                     suit_window)

import re, sys
# 默认校验 MEMORY.md；可传路径校验其它文档，如
#   python md_table_check.py ../README.md
p = sys.argv[1] if len(sys.argv) > 1 else str(_common.REPO / "memory" / "MEMORY.md")
lines = open(p, encoding='utf-8').read().split('\n')
sep = re.compile(r'^\|[\s:\-|]+\|$')
bad = 0; i = 0; tables = 0
while i < len(lines):
    if lines[i].strip().startswith('|') and i + 1 < len(lines) and sep.match(lines[i+1].strip()):
        tables += 1
        ncol = lines[i].count('|') - lines[i].count('\\|')
        j = i
        while j < len(lines) and lines[j].strip().startswith('|'):
            c = lines[j].count('|') - lines[j].count('\\|')
            if c != ncol:
                print(f'  L{j+1}: {c} 列 vs 表头 {ncol} 列 -> {lines[j][:90]}')
                bad += 1
            j += 1
        i = j
    else:
        i += 1
print(f'表格数 {tables} | 异常 {bad} | 总行数 {len(lines)}')
# 编号体检
old = [n for n in range(1, 12) if re.search(rf'(?<![A-Za-z0-9])R{n}(?![0-9A-Za-z])', '\n'.join(lines))]
print('文中仍出现的 R<n> 编号:', old, '（R1~R5 属既有训练轮次，属正常）')
print('RM 编号出现次数:', len(re.findall(r'RM\d+', '\n'.join(lines))))
