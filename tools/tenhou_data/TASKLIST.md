# 天凤牌谱数据集构建任务清单

> 目标：拉取天凤目前可获取的对局数据，按质量过滤，转为 mjai 格式，落盘到 `C:\Maj\Mortal\dataset\`，供 Mortal 训练使用。
> 本文所有端点与格式均为 2026-10-01 实测验证，非猜测。
> **2026 YTD 全量采集已完成并通过全量逐文件校验，最终状态与结果见 §8（2026-10-03）。**

---

## 0. 任务总览

| # | 任务 | 状态 |
|---|------|------|
| 1 | 本机环境检查（Python / git / cargo） | ✅ 完成（cargo 未装，见 §2） |
| 2 | 天凤公开日志端点与格式调研 | ✅ 完成（结论见 §3） |
| 3 | 数据集目录结构创建 | ✅ 完成（`dataset/{raw,mjai,index}`） |
| 4 | jansou 转换库安装 | ✅ 完成（jansou 0.14.0，装入默认 Anaconda 环境） |
| 5 | 管线脚本：抓取索引 → 过滤 → 下载 → 转换 → 验证 | ✅ 完成；**2026 YTD 全量跑通（130,407 局，下载/转换失败均为 0）** |
| 6 | 历史年度归档（2009~2025）接入 | ⚠️ 年度 zip 已 404；改用滚动归档，**2026 YTD 索引已抓全（130,407 局）**；2009~2025 走 HF 年度包（§5.7） |
| 7 | libriichi 编译与数据校验闭环 | ⬜ 待执行（cargo 仍未安装，§5.6） |
| 8 | 规模化采集与进度管理 | ✅ 完成（2026 YTD 全量；实际结果见 §8） |

---

## 1. 目录布局（已创建）

```
C:\Maj\Mortal\
├── dataset\
│   ├── raw\
│   │   ├── lists\        # scc 索引（滚动日归档 + 近 7 天小时窗口）：486 个 / 8.8 MiB
│   │   ├── archives\     # HF 年度包临时区（hf/，现已清空；年度 zip 已 404 不可用）
│   │   └── mjlog\        # <log_id>.xml.gz 每局原始牌谱（天凤原样字节）：130,407 个 / 937.3 MiB
│   ├── mjai\
│   │   └── houou4\YYYYMM\<log_id>.json.gz   # 转换后训练数据（Mortal 直接可读）：130,407 个 / 595.5 MiB
│   └── index\
│       ├── houou_games.csv                  # 对局清单（log_id, rule, players, src, downloaded, converted）：130,407 行
│       └── hf_merge_manifest.csv            # HF 合并新增明细（历史遗留记录，见 §5.7）
└── tools\
    └── tenhou_data\
        └── TASKLIST.md    # 本文档
```

---

## 2. 环境现状与准备

- Python 3.13.5（Anaconda，`D:\Anaconda3`）；git 2.45.1
- **cargo 未安装**：训练和 `validate_logs` 都需要 libriichi，先装 [rustup](https://rustup.rs/)（MSVC 工具链），参照 `docs/src/user/build.md`
- 依赖已安装（默认环境）：`requests 2.32.3`、`tqdm 4.68.3`、`jansou 0.14.0`（转换库，API 见 §5.5）

---

## 3. 数据源结论（全部实测验证）

### 3.1 文件清单接口

`GET https://tenhou.net/sc/raw/list.cgi` 返回 JS 数组：

```
list([
{file:'sca20260924.log.gz',size:47641},
{file:'scb2026092400.log.gz',size:53798},
{file:'scc2026092400.html.gz',size:3438}, ...
]);
```

官方定义（摘自 sc/raw 页面原文）：

| 前缀 | 内容 | 牌谱链接 | 起始 |
|------|------|----------|------|
| sca | 個室 | 无 | — |
| scb | 段位战（全卓级，**仅结果无牌谱**） | 无 | — |
| **scc** | **凤凰卓** | **有** | 2009-02-01 |
| scd | 雀荘战 | 无 | 2010-10-14 |
| sce | 技能战（已关闭）+ **琥珀卓** | 琥珀卓有 | 2016-06-06 起 |
| scf | html 汇总 | — | 2015-11-04 |

- 文件下载地址：`https://tenhou.net/sc/raw/dat/<file>`
- **在线窗口仅保留最近 7 天**（scb/scc 按小时滚动）
- 官方注意事项：更新周期约 5 分钟；按对局开始时间排序，**新行可能插在文件中间 → 必须按 log_id 去重**

### 3.2 scc 文件格式（凤凰卓 = 天然的高质量数据池）

- gzip 压缩的 HTML，**内容 html-escaped（`&gt; &lt; &amp;`）**，解析前需 `html.unescape`
- 每行：`21:01 | 27 | 四鳳南喰赤－ | |牌譜| | ogatta(+56.0) 悠楽うらめ(+1.5) ...`
- 实测一个小时的 scc 文件含 62 个牌谱 ID（三凤/四凤混合）
- 牌谱 ID 正则（实测可用）：

```python
LOGID_RE = re.compile(r'log=([0-9]{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8})')
# 示例: 2026093021gm-00a9-0000-09a004a9 (四鳳) / 2026093021gm-00b9-0000-3d77264f (三鳳)
# 规则字段建议用正则从行文本提取，不要按列号切:
RULE_RE = re.compile(r'(三|四)(鳳|特|上|般)(東|南)喰赤(速)?')
```

### 3.3 牌谱下载接口

`GET https://tenhou.net/0/log/?<log_id>` → 返回 **gzip 压缩的 XML**（`<mjloggm ver="2.3">...`），实测约 15KB/局。

⚠️ 响应体形态取决于 HTTP 客户端（2026-10-01 修正）：
- 用 `urllib`（不声明 `Accept-Encoding` 或不自动解压）拿到的就是原始 gzip 字节；
- 用 `requests` 时服务端返回 `Content-Encoding: gzip`，requests **会自动解压**，`resp.content` 已是 XML 明文。
→ 落盘前需判断：`1f 8b` 开头则原样存 `.xml.gz`；XML 明文则补压一次（`gzip.compress`），保证磁盘上始终是 gzip 字节。

### 3.4 历史归档（2026-10-01 实测修正：年度 zip 当前不可用）

- ❌ **年度 zip（`scrawYYYY.zip`）当前全部 404**：页面仍显示各年体量（2025=453M、2024=461M…），
  但抽测 2010/2020/2024/2025 均返回 404（curl + 会话 + Referer 交叉验证）
  → **2009~2025 历史牌谱目前无可用下载入口**，若日后恢复再补（约 450~600MB/年）。
- ✅ **滚动日归档可用**（当前唯一日粒度来源）：
  - 清单：`GET https://tenhou.net/sc/raw/list.cgi?old` → `{file:'2026/scc20260101.html.gz',size:38220}`
  - 下载：`GET https://tenhou.net/sc/raw/dat/2026/scc20260101.html.gz`（常规 GET；支持 Range 返回 206）
  - 覆盖：**当年 1/1 至约 7 天前**，与近 7 天小时窗口无缝衔接
  - 2026-10-01 实测：scc 日文件 266 个共 **8.4MB**；内容为**原始 HTML**（非 html-escaped）
- ⚠️ **当年数据必须当年抓**：年底日文件预计并入年度 zip（而 2025 的 zip 已 404）

### 3.5 官方红线（必须遵守）

> 原文摘录：
> - ※天鳳と競合する製品への開発・応用を目的として牌譜を使用していただくことはできません。
> - ※「天鳳での対戦を必要としないサービス」への応用は無償有償ともに行えません。
> - ※不特定多数が天鳳の牌譜をダウンロードするサービスは作成できません。

- 天凤官方（2024-06）：**牌谱下载同时只能 1 个会话**；拒绝单 IP 数百并发
- 结论：**串行下载 + 合理总量**。脚本默认 2 秒/局间隔（保守值）；2026-10-01 曾用 **0.5 秒/局**（实测 1.46 局/秒）。数据仅限本地研究训练，不得二次分发原始牌谱、不得做成公开下载服务
- **全量最终实际参数（2026-10-03，用户拍板；仍为单会话串行）**：`--delay 0.05`（等效约 **4.3 局/秒**，已接近单会话物理上限——每请求往返约 0.19s）
  + `--chunk 20000 --chunk-sleep 900` 例行冷却 + 限流自愈（`--max-fail 50 --cooldown 1800 --max-cooldowns 6`）。
  该档位跑完全部 130,407 局，**下载/转换失败均为 0**（期间被外部终止 2 次，重启后断点续传无损失）

---

## 4. 质量过滤规则

Mortal 仅支持**四人麻将**，且原作即用天凤标准段位规则训练：

| 过滤项 | 规则 | 说明 |
|--------|------|------|
| 数据池 | 只取 **scc（凤凰卓）** | 凤凰卓入门即 R2000+，天然高质量；琥珀卓（sce）可选加入 |
| 人数 | `四` 开头（排除三麻） | Mortal 不支持三麻 |
| 长度 | `南`（半庄） | 排除东风局，与原作训练分布一致 |
| 规则 | `喰赤` | 标准有喰タン有赤牌 |
| 去重 | 按 log_id | 文件重叠/历史行插入导致重复 |
| 完整性 | 转换后校验 | 见 §5.6 |

→ 目标行匹配：`四鳳南喰赤－`（速棋 `四鳳東喰赤速` 默认排除，训练分布不纯）

---

## 5. 执行步骤

### 5.0 一键管线（已实现，推荐）

全部步骤已实现在 `tools/tenhou_data/tenhou_pipeline.py`（单文件，无额外依赖，可反复重跑自动续传）：

```bash
python tools/tenhou_data/tenhou_pipeline.py all              # 窗口一条龙: 索引→清单→下载→转换→校验
# 或分步执行:
python tools/tenhou_data/tenhou_pipeline.py fetch-lists      # 近 7 天小时索引（默认跳过已存在，近 3 小时内自动重抓）
python tools/tenhou_data/tenhou_pipeline.py fetch-archives   # 滚动归档日文件（当年 1/1 起，见 §3.4）
python tools/tenhou_data/tenhou_pipeline.py build-index      # 解析+过滤 → dataset/index/houou_games.csv（可含归档）
python tools/tenhou_data/tenhou_pipeline.py stats            # 统计去重对局数（按月份）与预估体积
python tools/tenhou_data/tenhou_pipeline.py download         # 串行限速下载（默认 2s/局，连续失败 20 次自停）
python tools/tenhou_data/tenhou_pipeline.py convert          # mjlog → mjai（含 ura_markers 键名修正，见 §5.5）
python tools/tenhou_data/tenhou_pipeline.py merge-hf <tar...> # 合并 HF 年度包（补历史年份，见 §5.7）
python tools/tenhou_data/tenhou_pipeline.py check            # 抽样校验（结构 + 回环解析 + 回放计分）
```

以下 §5.1~§5.5 为各步骤的原理与参考代码（脚本即按其实现）；§5.6 仍需装 cargo 后手动执行。

### 5.1 抓取近 7 天 scc 索引

```python
import re, html, gzip, urllib.request, pathlib

UA = {'User-Agent': 'Mozilla/5.0 (personal mahjong research)'}
def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()

LIST_DIR = pathlib.Path(r'C:\Maj\Mortal\dataset\raw\lists')
raw = get('https://tenhou.net/sc/raw/list.cgi').decode('utf-8', 'replace')
files = re.findall(r"\{file:'([^']+)',size:(\d+)\}", raw)
scc_files = [f for f, _ in files if f.startswith('scc')]
for f in scc_files:
    (LIST_DIR / f).write_bytes(get('https://tenhou.net/sc/raw/dat/' + f))
```

### 5.2 解析 + 过滤 → 生成对局清单 CSV

```python
import csv
LOGID_RE = re.compile(r'log=([0-9]{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8})')
RULE_RE  = re.compile(r'(三|四)(鳳|特|上|般)(東|南)喰赤(速)?')

rows = {}
for path in LIST_DIR.glob('scc*.html.gz'):
    text = gzip.decompress(path.read_bytes()).decode('utf-8', 'replace')
    for line in text.splitlines():
        line = html.unescape(line)
        m = LOGID_RE.search(line)
        r = RULE_RE.search(line)
        if not (m and r):
            continue
        num, tbl, leng, spd = r.groups()
        if num == '四' and tbl == '鳳' and leng == '南' and not spd:   # §4 质量过滤
            rows.setdefault(m.group(1), {
                'log_id': m.group(1),
                'rule': r.group(0),
                'players': line.split('牌譜')[-1].strip()[:200],
                'src': path.name, 'downloaded': '', 'converted': ''})
with open(r'C:\Maj\Mortal\dataset\index\houou_games.csv', 'w', newline='', encoding='utf-8') as fp:
    w = csv.DictWriter(fp, fieldnames=['log_id','rule','players','src','downloaded','converted'])
    w.writeheader(); w.writerows(rows.values())
```

> 实测（2026-10-01）：四鳳南约 **450~570 局/天**；7 天窗口实测 4,020 局。

### 5.3 历史归档接入（方案已因 zip 404 改为滚动归档）

- 年度 zip 当前不可用（见 §3.4），改用 `fetch-archives` 抓取滚动日归档：

```bash
python tools/tenhou_data/tenhou_pipeline.py fetch-archives --years 2026
```

- 抓取的日文件与小时窗口文件同放 `dataset/raw/lists/`，`build-index` 自动解析合并、按 log_id 去重
- **2026 年数据已完整索引并采集完成**：130,407 局（四鳳南喰赤，1/1~10/3；最终规模与全量校验见 §8）
- ⚠️ **年底打包风险**：2026-12-31 后本年日文件预计并入 scraw2026.zip（而 2025 的 zip 当前 404）→ **需在年内完成牌谱落盘**（索引文件 8.4MB 已本地留存）
- 若日后 zip 恢复可用，再补 2009~2025：下载 zip → 解压 `scc*` 成员（跳过几百 MB 无关数据）→ 同一份 build-index 解析合并

### 5.4 串行下载牌谱（单会话 + 限速 + 断点续传）

```python
import time
RAW_DIR = pathlib.Path(r'C:\Maj\Mortal\dataset\raw\mjlog')
rows = list(csv.DictReader(open(r'C:\Maj\Mortal\dataset\index\houou_games.csv', encoding='utf-8')))
for row in rows:
    dst = RAW_DIR / f"{row['log_id']}.xml.gz"
    if dst.exists():
        continue
    try:
        data = get('https://tenhou.net/0/log/?' + row['log_id'])
        assert data[:2] == b'\x1f\x8b'
        dst.write_bytes(data)
        row['downloaded'] = 'ok'
    except Exception as e:
        row['downloaded'] = f'fail:{e}'
    time.sleep(2.0)          # 官方红线：单会话，勿删
```

- 建议加：连续 20 次失败即停（疑似被限流）、每日总量上限、`tqdm` 进度
- 失败的行留在 CSV 里，下次运行自动重试
- **全量任务运行方式**：由 Windows 计划任务 `Mortal-Tenhou-Download` 承载（意外退出自动重启、日志追加）；
  查看 `Get-ScheduledTask -TaskName Mortal-Tenhou-Download`，暂停 `Stop-ScheduledTask -TaskName Mortal-Tenhou-Download`，
  日志在 `tools/tenhou_data/logs/run_stdout.log` 与 `run_stderr.log`
- **限流教训（2026-10-02 实测）**：0.5s/局持续约 8 小时（~3.4 万请求）后被天凤掐断连接（连续 ConnectionError/SSLError），数小时后自行恢复
- **实际采用的限速/冷却配置**（计划任务 `Mortal-Tenhou-Download` 启动参数）：
  `--delay 0.05 --max-fail 50 --chunk 20000 --chunk-sleep 900 --cooldown 1800 --max-cooldowns 6`
  → 每 20,000 局例行冷却 15 分钟；连续失败 50 次自动冷却 30 分钟并重试（最多 6 次=3 小时），不再直接停摆
- **已知坑：`save_rows()` 原子替换被文件锁打断**（2026-10-02）。清单写回用「临时文件 + `tmp.replace()`」，
  若此时有其它进程（如进度看板 `Import-Csv`）短暂持有目标 CSV 句柄，Windows 会抛
  `PermissionError: [WinError 5] 拒绝访问`，早期版本会因此**整个进程退出**；现已对替换步骤加 20 次重试（≈10s）兜底
- **完成（2026-10-03）**：全量 130,407 局下载 + 转换结束，计划任务退出码 0，失败 0；全量校验见 §8

### 5.5 mjlog XML → mjai（.json.gz）

jansou 0.14.0 API 已实测确认（2026-10-01）：

```python
from jansou.io.mjlog import parse_mjlog   # mjlog XML(bytes/Path/str) → Paifu（自动识别 gzip）
from jansou.io.mjai import dump_mjai      # Paifu → mjai JSONL 文本
from jansou.validation.check import check_paifu  # 回放每局并复核和牌点数，返回 list[Verdict]

paifu = parse_mjlog(r'dataset\raw\mjlog\<log_id>.xml.gz')
text = dump_mjai(paifu)   # 每行一个 JSON 对象
# 写入 dataset\mjai\houou4\<YYYYMM>\<log_id>.json.gz（log_id 前 6 位即 YYYYMM）
```

⚠️ **键名差异（必须修正）**：jansou 把里宝牌指示牌写在 `hora.uradora_markers`，
而 Mortal/libriichi 读取的键是 `hora.ura_markers`（validate_logs 强制要求该字段存在）。
转换时需：有 `uradora_markers` 则改名，无则补 `"ura_markers": []`（非立直和牌即此情形）。
牌字记号两库一致（赤五均为 `5mr/5pr/5sr`），`StartKyoku.scores`、`hora.deltas` 等必需字段齐全。

- Mortal 侧格式要求：`**/*.json.gz`，每行一个 mjai 事件（`start_game`/`start_kyoku`/`tsumo`/`dahai`/...），见 `libriichi/src/dataset/gameplay.rs`
- **转换后自查**：首行应为 `{"type":"start_game"}`，且存在带 `scores` 的 `start_kyoku`
- 备选转换器（若 jansou 输出与 mjai 规范有出入）：
  - [PaifuLogger](https://github.com/Jim137/Tenhou-Paifu-Logger)（`plog --mjai`，按 URL 逐局）
  - [mjlog2json](https://github.com/tsubakisakura/mjlog2json)（mjlog→tenhou6 JSON，需二次转换）

### 5.6 用 libriichi 闭环校验（数据可用性的最终标准）

```bash
# 1) 装 rustup 后，在仓库根目录：
cargo build -p libriichi --lib --release
copy target\release\riichi.dll mortal\libriichi.pyd
cargo build -p libriichi --bins --no-default-features --release

# 2) 官方校验器扫全目录：
target\release\validate_logs.exe C:\Maj\Mortal\dataset\mjai

# 3) Python 冒烟测试（Mortal 实际吃数据的入口）：
cd mortal
python -c "from libriichi.dataset import GameplayLoader; gl=GameplayLoader(version=4); games=gl.load_gz_log_files([r'C:\Maj\Mortal\dataset\mjai\houou4\<某月>\<某局>.json.gz']); print(len(games), len(games[0]))"
```

`validate_logs` 零报错 + `GameplayLoader` 能取出 obs/actions/masks，数据才算合格。

### 5.7 合并公开数据集补历史年份（HF 年度包）

官方年度 zip 404 后，历史年份改用社区整理集 **NikkeTryHard/tenhou-to-mjai**
（`huggingface.co/datasets/nikketryhard/tenhou-houou-mjai`，国内走 **hf-mirror.com**）：

```bash
# 下载年度包到 D 盘（默认策略：C盘临时下载 → 复制 D 盘 → 校验 → 删 C 盘原文件，逐包串行）
python tools/tenhou_data/tenhou_pipeline.py fetch-hf --years 2010-2025 --dest D:\tenhou_hf
# 合并进统一 mjai 树（解压 zst → 结构自查 → gzip 落盘，按 log_id 去重）
python tools/tenhou_data/tenhou_pipeline.py merge-hf D:\tenhou_hf\tenhou-houou-2010.tar ...
# 定向校验合并结果
python tools/tenhou_data/tenhou_pipeline.py check --glob "dataset/mjai/houou4/2009*/*.json.gz" --sample 100
```

- `fetch-hf` 支持 `--years`（如 `2010-2025`、`2010,2015`）、`--mirror`（默认 hf-mirror.com）、`--dest`（默认 `D:\tenhou_hf`）
- **D 盘写入保护（2026-10-02 用户要求）**：本机 D 盘长时间连续写入易磁盘错误卡死，
  故逐包下载（写 C 盘临时区 `dataset/raw/archives/hf/`）→ 复制到 D 盘 → 校验字节数 → 删 C 盘原文件；
  全串行、一次只处理一个包；断点续传（D 已有则跳过、C 临时完整则复用）
- 包内为 `<log_id>.mjai.json.zst`（zstd 压缩 mjai JSONL），**实测与 Mortal 兼容**（hora 键名即 `ura_markers`，无需修正）
- 合并仅记录【新增】文件的明细到 `dataset/index/hf_merge_manifest.csv`（便于溯源/回退），去重跳过的只计数
- 2026-10-02 验证：2009 包 6,897 局（200902~200912）全部新增并抽验 100/100 通过；
  2026 包 12,188 局与本管线已下载数据 **100% 重合、全部去重跳过**（证明本管线 1 月数据完整）
- 2010–2025（16 包）已由计划任务 `Mortal-Tenhou-FetchHF` **下载完成**：18/18 包字节校验全过（0 失败），
  `D:\tenhou_hf` 共 17.3 GiB；抽验 2015 包（15.2 万局）随机 3 局结构正常；C 盘临时区已自动清空
- 实测合并体积比（2009）：zstd 包 47.6 MiB → gzip 合并后 27.4 MiB（≈0.576×，gzip 反而更小）；全量逻辑体积 ≈ 9.7 GiB
- **⚠️ 2026-10-02 重大发现：D 盘是 exFAT 分区，簇大小 = 256 KiB**（外置 USB 固态盘 Lenovo PS9 PSSD，非内置 NTFS 盘）
  - 每个 ~4 KiB 的 `.json.gz` 至少占 1 个簇 = 256 KiB → **膨胀 64×**
  - 实测：往 D 盘写入 1 字节文件即占 262,144 字节；`D:\tenhou_mjai` 13 万文件（逻辑 521 MiB）实际占 ~33 GiB
  - 全量历史 2009–2025（~243 万局）若散文件落 D 需 **~620 GiB** → 不可行
  - 这也解释了此前"D 盘空间莫名快速流失"：**并非卷影副本/隐藏进程，纯属簇浪费**（11 万文件 × 256 KiB ≈ 28 GiB，与观测完全吻合）
- **最终决策（用户 2026-10-02 拍板）**：只在 D 盘保留 HF 年度原包（`.tar.zst` 为大文件，不受簇浪费影响，共 18 包 16.86 GiB），**不转换、暂不改 Mortal 训练加载器；删除 D 盘上已生成的散装 `.json.gz`**
  - 背景：Mortal 训练加载器只吃散装 `.json.gz`（gzip + glob，见 `libriichi/src/dataset/gameplay.rs` 的 `load_gz_log_files`），**不支持 zstd / tar**
  - 故 `merge-hf` / `merge-hf-batch`（解压 zst → gzip 落盘）**暂不使用**；`D:\tenhou_mjai` 已整体删除，计划任务 `Mortal-Tenhou-MergeHF` 已注销
  - 若将来要训练历史数据：需选簇小（NTFS 4 KiB）的卷承载散文件，或在 D 上放动态 VHDX（内建 NTFS）后挂载为盘符
- 2026 数据不受影响：仍走自建管线，散装 `.json.gz` 落 C 盘 NTFS（簇 4 KiB，无膨胀）

---

## 6. 规模化采集的时间/容量预算

**单局实测体积（2026-10-01，n=118 全样本）**：
raw `.xml.gz` 平均 **7.59KB**（σ=1.04KB）、mjai `.json.gz` 平均 **4.84KB**（σ=0.96KB）
→ **合计 ≈ 12.43KB/局**（σ=2.0KB）

| 范围 | 局数 | 磁盘（raw+mjai） | 状态 |
|------|------|------------------|------|
| 近 7 天窗口 | 实测 4,020 局 | ≈ 50 MiB | ✅ 早期冒烟 |
| **2026 YTD（1/1~10/3）** | **130,407 局** | **1,532.8 MiB**（raw 937.3 + mjai 595.5） | ✅ **已完成（2026-10-03）** |
| 附：索引 + CSV 开销 | — | ≈ 28 MiB（lists 8.8 + index 25.0，含 hf manifest） | — |
| 附：仅保留 mjai（删 raw） | — | ≈ 595.5 MiB | — |
| 历史 2009~2025（HF 年度包，§5.7） | 约 240 万局 | 18 包 16.86 GiB（仅存 tar.zst 于 D 盘） | ⚠️ 未散装落盘（exFAT 簇 256 KiB 问题） |

实测与早期预估（129,475 局、1.53 GiB）几乎一致，偏差 <1%。

后续策略：
- 训练初期几十万局凤凰卓数据即可跑通 GRP + 离线阶段（**2026 YTD 130,407 局已达标**），不必等全部历史
- 历史 2009~2025 若要参与训练：需先装 cargo 跑通 §5.6 闭环，再解决 exFAT 簇浪费（换 NTFS 卷，或在 D 盘挂动态 VHDX）；当前仅在 D 盘保留年度原包
- 2026 年后续新增对局：直接重跑 `all` 即可增量补齐（索引→清单→下载→转换自动跳过已完成项）

---

## 7. 已知坑位备忘

1. 窗口小时文件是 **html-escaped**、归档日文件是**原始 HTML**——统一先 `html.unescape` 再正则（对两者都安全），否则含 `<>&` 的玩家名行会解析错位
2. 排序机制会**把新行插在文件中间**（长时对局），一切以 log_id 去重为准
3. `00a9`=四鳳、`00b9`=三鳳（观测规律），但**过滤逻辑以行内规则文本为准**，别依赖 ID 推断
4. 牌谱 URL `tenhou.net/0/log/?<id>` 的响应内容形态取决于客户端：`urllib` 拿到原始 gzip 字节（勿二次压缩）；`requests` 会自动解压成 XML 明文（需补压一次才能存 `.xml.gz`）。详见 §3.3
5. `list.cgi` 当前 7 天窗口没有 scd/sce 文件属正常（无对局时不出现在列表）
6. **年度 zip 当前全部 404**（2026-10-01 实测），历史数据暂无入口；当年数据靠"滚动日归档 + 7 天小时窗口"拼接，**必须当年抓**（年底打包后可能不可再取）；老牌谱 ID 仍可正常下载（实测 2026-01-01 的 ID 返回 200）
7. jansou 输出的 `hora.uradora_markers` 必须改名为 `ura_markers`（缺省时补 `[]`），否则 libriichi `validate_logs` 报 missing field（详见 §5.5）
8. 索引文件（scc 按小时分桶）在**小时内仍会插入迟到的对局行**，脚本对近 3 小时内的文件会自动重抓；已下载的旧版本不影响——重抓后重跑 `build-index` 即可补齐（按 log_id 去重）
9. `fetch-archives` 覆盖"当年 1/1 ~ 约 7 天前"，与小时窗口（近 7 天）衔接；`list.cgi?old` 是滚动清单，定期重跑该命令即可增量补齐（自动跳过已存在文件）
10. **D 盘是 exFAT（簇 256 KiB）外置 USB 盘**：**绝不能存放海量小文件**（每个 `.json.gz` 白占 256 KiB）。历史牌谱只以年度 `.tar.zst` 原包形式留在 D 盘；散装 mjai 文件只能放 NTFS 卷。排查磁盘空间异常时先确认目标卷的文件系统与簇大小（`Get-Volume`、`Get-CimInstance Win32_Volume -Filter "DriveLetter='D:'"` 看 `BlockSize`）

---

## 8. 最终数据状态与全量校验（2026-10-03）

> 2026 YTD（1/1~10/3）全量采集**已完成**：计划任务 `Mortal-Tenhou-Download` 退出码 0，
> 下载/转换失败均为 0。以下为**逐文件全量校验**结果（非抽样）。

### 8.1 最终规模

| 项 | 数量 | 体积 | 均值 |
|----|------|------|------|
| 对局清单 `dataset/index/houou_games.csv` | 130,407 行 | 19.29 MiB | — |
| raw 牌谱 `dataset/raw/mjlog/*.xml.gz` | 130,407 个 | 937.3 MiB | 7.36 KiB/局 |
| mjai 训练数据 `dataset/mjai/houou4/YYYYMM/*.json.gz` | 130,407 个 | 595.5 MiB | 4.68 KiB/局 |
| scc 索引 `dataset/raw/lists/*.gz` | 486 个 | 8.8 MiB | — |
| **合计（raw + mjai）** | — | **1,532.8 MiB（≈1.50 GiB）** | 12.04 KiB/局 |

- 日期覆盖：**20260101 ~ 20261003**；全部规则为 `四鳳南喰赤`（四麻 / 鳳凰卓 / 半庄 / 喰赤）
- 月份分布（mjai 文件数）：
  202601 16,179｜202602 14,352｜202603 14,294｜202604 13,443｜202605 14,698｜
  202606 13,463｜202607 13,174｜202608 13,827｜202609 15,892｜202610 1,085
- 目录干净：无 `.tmp` / 残留文件；`dataset/raw/archives/`（含 `hf/`）已清空

### 8.2 全量逐文件校验结果

| 校验项 | 范围 | 结果 |
|--------|------|------|
| raw：gzip 可解压 且 以 `<mjloggm` 开头 | 全部 130,407 | ✅ 损坏 **0** |
| mjai：gzip + JSONL 结构（首行 `start_game` / 含 `scores` 的 `start_kyoku` / 末行 `end_game` / `hora`·`ryukyoku` 带 `deltas` / `hora` 带 `ura_markers`） | 全部 130,407 | ✅ 不合格 **0** |
| mjai：归档目录名 == log_id 前 6 位（YYYYMM） | 全部 130,407 | ✅ 错位 **0** |
| CSV：行数 / log_id 唯一性 | 130,407 行 | ✅ 重复 **0** |
| CSV：状态列 | — | ✅ `downloaded=ok` 130,407；`converted=ok` 130,407 |
| CSV ↔ raw ↔ mjai 三方集合一致 | — | ✅ 缺失 0、孤儿 0、差集 0 |
| CSV 字段合法性（rule 全为 `四鳳南喰赤`、log_id 正则、players/src 非空） | 130,407 行 | ✅ 异常 0 |
| jansou 抽样回放计分（`check --sample 1000`） | 1,000 局 | ✅ 结构 1000/1000、回环解析 1000/1000、和牌复核 **8932/8932** |
| libriichi `validate_logs` + `GameplayLoader`（§5.6） | 全量 | ⬜ **未执行**（cargo 仍未安装） |

**结论**：结构层面数据 **100% 可用**且三方一致；唯一未做的是 libriichi 闭环校验（需先装 Rust 工具链）。

### 8.3 复现命令

```bash
python tools/tenhou_data/tenhou_pipeline.py check --sample 1000   # 抽样语义校验（结构+回环+回放计分）
python tools/tenhou_data/tenhou_pipeline.py verify-raw            # raw 完整性（注意：会删除损坏文件）
python tools/tenhou_data/tenhou_pipeline.py all                   # 增量补齐 2026 后续新增对局
```
