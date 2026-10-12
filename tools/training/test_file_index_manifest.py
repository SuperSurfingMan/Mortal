#!/usr/bin/env python3
"""file_index manifest 决策逻辑单测（不训练；端到端见 test_file_index_manifest.sh）。

覆盖（2026-10-12 加的机制，防"静默复用过期文件表"）：
  构造      index_version / 记录项 / 名单 sha256 / data_version
  匹配判定  同配置复用；旧格式、globs、名单内容、data_version、index_version、
            excludes_files 变化一律判不匹配（→ 重建）
  不误伤    纯记录项与统计项（built_at / enable_augmentation / n_files / names_coverage…）
            变化不影响复用，避免无谓的 813k 文件重建
  容错      名单文件缺失时记 ERROR 而不是抛异常
"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MORTAL = HERE.parents[1] / 'mortal'
sys.path.insert(0, str(MORTAL))
import train  # noqa: E402  （模块级只定义函数/常量，导入不会启动训练）

fails: list[str] = []


def check(name, cond, extra=''):
    print(f'  [{"OK" if cond else "FAIL"}] {name}' + (f'    {extra}' if extra else ''))
    if not cond:
        fails.append(name)


tmp = Path(tempfile.mkdtemp(prefix='idxman_'))
wl = tmp / 'wl.txt'
wl.write_text('a\nb\nc\n', encoding='utf-8')
cfg = {'dataset': {'globs': ['/x/**/*.json.gz'], 'player_names_files': [str(wl)], 'data_version': 'v1'}}
cur = train.build_index_manifest(cfg, {'a', 'b', 'c'}, enable_augmentation=False, augmented_first=False)


def variant(**kw):
    ds = dict(cfg['dataset'])
    ds.update(kw)
    return train.build_index_manifest({'dataset': ds}, {'a', 'b', 'c'})


print('— 构造与字段 —')
check('index_version 为当前版本', cur['index_version'] == train.FILE_INDEX_VERSION)
check('记录项齐全', all(k in cur for k in (
    'player_names_count', 'enable_augmentation', 'augmented_first',
    'libriichi_version', 'built_at')))
check('player_names_count 正确', cur['player_names_count'] == 3)
check('名单 sha256 与 hashlib 一致',
      cur['player_names_files'][str(wl)] == hashlib.sha256(wl.read_bytes()).hexdigest())
check('data_version 写入', cur['data_version'] == 'v1')
check('缺 data_version 的 config 也能跑', variant(data_version='')['data_version'] == '')

print('— 匹配判定（应复用）—')
check('同配置 → 匹配', train.manifest_matches(cur, dict(cur)) is True)

print('— 触发重建（参与比对的键）—')
check('旧格式 None → 不匹配', train.manifest_matches(None, cur) is False,
      train.manifest_diff(None, cur))
check('旧格式 dict 缺 manifest → 不匹配', train.manifest_matches({}, cur) is False)
check('globs 变化 → 不匹配', train.manifest_matches(cur, variant(globs=['/y/**/*.json.gz'])) is False,
      train.manifest_diff(cur, variant(globs=['/y/**/*.json.gz'])))
check('data_version 变化 → 不匹配', train.manifest_matches(cur, variant(data_version='v2')) is False,
      train.manifest_diff(cur, variant(data_version='v2')))
wl.write_text('a\nb\nc\nd\n', encoding='utf-8')          # 名单内容变了（同一路径）
check('名单内容变化 → 不匹配', train.manifest_matches(cur, variant()) is False,
      train.manifest_diff(cur, variant()))
wl.write_text('a\nb\nc\n', encoding='utf-8')
_v = dict(cur); _v['index_version'] = train.FILE_INDEX_VERSION + 1
check('index_version 变化 → 不匹配', train.manifest_matches(_v, cur) is False)
_v = dict(cur); _v['excludes_files'] = {'/tmp/ex.txt': 'deadbeef'}
check('excludes_files 变化 → 不匹配', train.manifest_matches(_v, cur) is False)
_v = dict(cur); _v['globs'] = ['/x/**/*.json.gz']       # 列表顺序相同即视为一致
check('globs 相同（含顺序）→ 匹配', train.manifest_matches(_v, cur) is True)

print('— 不误伤（仅记录/统计项变化）—')
_v = dict(cur)
_v.update({
    'built_at': '2099-01-01T00:00:00', 'enable_augmentation': True, 'augmented_first': True,
    'player_names_count': 999, 'libriichi_version': 'zzz',
    'n_files': 12345, 'names_coverage': {'with_names': 1, 'without_names': 2},
})
check('记录项/统计项变化 → 仍匹配', train.manifest_matches(_v, cur) is True, train.manifest_diff(_v, cur))

print('— 容错 —')
_m = train.build_index_manifest(
    {'dataset': {'globs': ['/x'], 'player_names_files': [str(tmp / 'nope.txt')]}}, set())
check('名单文件缺失 → 记 ERROR 不抛异常',
      _m['player_names_files'][str(tmp / 'nope.txt')].startswith('ERROR:'))

shutil.rmtree(tmp, ignore_errors=True)
print()
if fails:
    print(f'FAIL（{len(fails)}）: {fails}')
    sys.exit(1)
print('ALL PASS')
