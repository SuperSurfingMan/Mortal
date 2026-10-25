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
- ⏱ 吞吐未标定（死锁解决后再测）。参考：akochan 每个决策是完整搜索，预计 0.3–3 s/决策，
  一个半庄 ≈ 1–2 min ⇒ **400 半庄的 2σ 锚点需要 ~3–7 h**（多流并行可压到 1–3 h）。

## 建议的使用方式

1. 先解决 pipe 死锁（或改用 mjai_client + 本地 server），跑通 1 个半庄；
2. 用 3 条并行流跑 **400 半庄**（SE ≈ 82/√400 ≈ 4.1）作为第一版锚点，
   同时记录 3 个 akochan 的 `avg_rank`（对称性自检：应 ≈2.5）；
3. 锚点结论只用于「判断量级」（例如与人类 p95 +9.3 的对比），不作为 +0.5 级效应的判据。
