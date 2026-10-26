# akochan 外部锚点：构建与现状（2026-10-25）

目标：给主模型一个**非自家的参照对手**（Mortal 自己的 ladder 全是内部对照，`test_play` 的对手
是自家 200k 老模型，所以「我们到底多强」一直无法回答）。

fork：`https://github.com/Equim-chan/akochan`（Mortal `1v3` 模式配套的 mjai 版本）。
源码与产物在 `tools/akochan_src/ako/`（**已 gitignore，不入库**）。

## 构建（可复现）

```bash
git clone --depth 1 https://github.com/Equim-chan/akochan.git tools/akochan_src/ako
cd tools/akochan_src/ako

# 1) 关键坑：默认 NPROCS=核数(56) 会让 selector.o 的 .bss 达到 9.4 GB，
#    链接时 R_X86_64_PC32 溢出（"relocation truncated to fit"）。两种解法：
#    (a) -mcmodel=large（能链上，但慢、且 OpenMP 线程会把机器打满）；
#    (b) ✅ 采用：把 NPROCS 固定成 2（.bss ≈ 0.34 GB，正常小模型编译）。
sed -i 's|^NPROCS = .*|NPROCS = 2|' ai_src/Makefile_Linux

# 2) 不要 libai.so（PC32 溢出风险），把 ai 对象直接静态链进可执行文件
sed -i 's|^LIBS = .*|LIBS = -lboost_system -I./ -no-pie ./ai_src/obj/*.o|' Makefile_Linux

mkdir -p ai_src/obj share/obj obj
(cd ai_src && make -f Makefile_Linux -j16) && make -f Makefile_Linux -j16
# → ./system.exe（22 MB）
cp setup_mjai.json tactics.json        # libriichi 通过 AKOCHAN_TACTICS 指定
```

## 与 libriichi 的接线（已就绪）

- `libriichi/src/agent/akochan.rs`：`AkochanAgent::new` 会 spawn
  `<AKOCHAN_DIR>/system.exe pipe <tactics.json> <player_id>`（cwd = AKOCHAN_DIR）。
- `mortal/one_vs_three.py` 支持 `[1v3.akochan] enabled/dir/tactics`，走
  `OneVsThree.ako_vs_py`（1 个 Mortal + 3 个 akochan）。
- 已建 `mortal/config.anchor_ako.toml`（挑战者 = `B_rank_pt_aligned/mortal.pth`，
  `[1v3.akochan] enabled = true`，dir 指向本机构建目录，tactics = `tactics.json`）。
- 指标：`one_vs_three.py` 输出 `avg_rank` 与 **uma 口径 avg_pt（90/45/0/−135）**，
  与 `dataset/index/player_ratings_2022_2026H1_v3.csv` 的 `avg_pt` **同尺度**，
  因此可直接和人类（houou 场）的 avg_pt 分布比较（全样本均值 −5.44 / 中位 −2.61 / p95 +9.29）。

## 当前状态（重要）

- ✅ 构建成功、`system.exe pipe ...` 能被 libriichi 正常拉起（3 个进程都在），
  比赛能开始（`one_vs_three.rs: seed: [10000, 10001) ... start 1 sets, 4 hanchans`）。
- ❌ **pipe 协议在首次决策处卡死**：实测 8 分钟未完成 4 个半庄，且 akochan 进程
  `CPU time` 30 s 内增量为 **0**、Python 侧 3.9% → 双方都在等对方（不是慢，是死锁）。
  待查方向：`system.exe pipe` 的首个期望消息（可能需要 `start_game`/`player_id` 行）、
  `EventWithCanAct` 的 `can_act` 字段名、或该 fork 的 `pipe` 模式与 libriichi 的协议版本差异；
  另一条可行路是走 TCP：`system.exe mjai_client <port> setup_mjai.json` + 自建 mjai server。
- ⏱ 吞吐未标定（死锁解决后再测）。参考：akochan 每个决策是完整搜索，0.3–3 s/决策，
  一个半庄 ≈ 1–2 min ⇒ **400 半庄的 2σ 锚点需要 3–7 h**（多流并行可压到 1–3 h）。

## 2026-10-25 续：三处不匹配已定位（含复现）

**#1 字段缺失（已修，靠垫片）**
`share/types.cpp:624 get_kyoku_first()` / `get_aka_flag()` 要求 **start_game** 里有
`kyoku_first`（0=半荘、4=東風）与 `aka_flag`；libriichi 的 `Event::StartGame` 只序列化
`names`/`seed` ⇒ `assert_with_out` 打印一行错误后 `assert(false)` 直接 abort。
垫片里对 `start_game` 补 `kyoku_first=0`、`aka_flag=true` 后，单点测试即得到合法回复：
`[{"actor": 0, "pai": "N", "tsumogiri": true, "type": "dahai"}]`。

**#2 OpenMP 线程越界 → SIGSEGV（已修，靠垫片强制 env）**
本构建 `NPROCS=2` 时 `selector` 的 per-thread 表按 2 份分配；若 `OMP_NUM_THREADS` 未设
（默认 = 56 核）则越界 → 子进程 **exit code −11**、无任何输出。垫片强制
`OMP_NUM_THREADS=2` 后崩溃消失（`OMP_NUM_THREADS=1/2` 都能正常回复）。

**#3 回复时机不同 → 死锁（垫片只能部分绕过，仍需改 Rust 侧）**
- akochan 只在「自己的 `tsumo`」或「别人 `dahai`/`kakan`」时回话；
- 但 libriichi 的 `AkochanAgent::react` **每批事件写完必读一行**，而 `ako_vs_py` 会
  **同时**问 3 个 akochan ⇒ 其中两三个的最后事件不是触发点 ⇒ 永不回话 ⇒ 死锁
  （实测 debug 日志末尾：`tsumo actor=0 can_act=False → dahai actor=0 can_act=False →
  tsumo actor=1 can_act=True`，其余两个 seat 就卡在这里）。
- 垫片当前做法：对「`can_act=true` 且不是自己触发点」的批次直接合成 `[{"type":"none"}]`。
  这能让流程继续，但 **libriichi 会把这条 `none` 当成该 agent 的真实动作**消费，
  导致后续状态错位。实测新失败：`RuntimeError: invalid action: Dahai { actor: 0, pai: W,
  tsumogiri: true }`（一个 akochan 的回复被当成了 0 号位的动作），以及
  `system.bin: tehai_ana.cpp:194: Assertion 'num <= 15' failed`（手牌数 15+，状态已乱）。

**下一步（明确、工作量小）**：改 `libriichi/src/agent/akochan.rs` 的 `react`——
在发完事件后先判断「最后一条是否本 seat 的触发点」，不是就直接
`return Ok(EventExt::no_meta(Event::None))`（**不读子进程**），是才读；
同时确认 `ako_vs_py` 传给 `AkochanAgent::new_batched` 的 `player_ids` 只含 1..3、
且引擎占 0 号位（当前现象提示 0 号位也可能起了一个 akochan 实例）。
改完需重编 Rust 扩展（cargo/maturin），再跑 1 set 冒烟。垫片只保留 #1/#2 两个职责，
**不再合成 none**。

## 2026-10-26 01:38：✅ 跑通（RC=0）

根因补充：`sent_new` —— `set_scene` 会**用同一份 log 重复调用**，此时没有新事件发给 akochan，
若还去读就必卡。补上「无新事件 → 直接返回 `Event::None`」后整条链路通：

```
[arena] seeds=4 challenger_ids=[0,1,2,3] champion_ids=[1,2,3, 0,2,3, 0,1,3, 0,1,2]
[arena] agents built: challenger=akochan champion=mortal
[ako p0] ENTER n=2 idx=0 ...（573 次决策无一次卡死、无一次 illegal action）
challenger rankings: [0 1 2 1] (3.0, -22.5pt)   RC=0
```

- **1 set = 4 个并行对局（4 个 split），耗时 14.7 min**（GPU/CPU 都有其他用户争用）
  ⇒ 单进程吞吐 ≈ **16 局/h**。
- 这 4 局的含义：挑战者=akochan 一次都没拿第 1，即 **Mortal 在 4/4 局里都是第 1**，
  akochan 侧 avg_rank 3.0 / −22.5pt（Mortal 侧 +22.5pt）。样本量 n=4，只能算「通路正确」，
  不构成强度结论。
- 锚点规模的现实估算：uma 口径 SD≈82，要 SE≈4 需 **~400 局**；单进程 25 h，
  **6~8 个进程并行 ≈ 3~4 h**（每个进程 3 个 akochan 进程，OMP=2，约 1 核/进程）。

⚠️ 途中发现的另一层卡点：修完 akochan 应答时机后曾出现「akochan 的 react 一次都没被调用」，
那是因为 `set_scene` 的重复调用会让 `event_idx == len`；加上 `sent_new` 判定即解。


## 建议的使用方式

1. 先解决 pipe 死锁（或改用 mjai_client + 本地 server），跑通 1 个半庄；
2. 用 3 条并行流跑 **400 半庄**（SE ≈ 82/√400 ≈ 4.1）作为第一版锚点，
   同时记录 3 个 akochan 的 `avg_rank`（对称性自检：应 ≈2.5）；
3. 锚点结论只用于「判断量级」（例如与人类 p95 +9.3 的对比），不作为 +0.5 级效应的判据。
