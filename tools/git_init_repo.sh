# 为本仓库建立本地 git 版本控制（幂等保护 + 防数据泄漏校验 + 必要文件齐全性校验）
# 用法: tmux new -d -s gitinit 'bash /mnt/data8/mortal/Mortal/tools/git_init_repo.sh 2>&1 | tee /mnt/data8/mortal/Mortal/mortal/logs/git_init.log'
set -uo pipefail

ROOT=/mnt/data8/mortal/Mortal
cd "$ROOT"

if [ -d .git ]; then
    echo "!! .git 已存在，为避免破坏既有仓库历史，直接退出。"
    exit 1
fi

echo "== 1. git init =="
git init -q
git symbolic-ref HEAD refs/heads/main          # 本机 git 2.17 不支持 init -b，用 symbolic-ref 命名主分支
git add -A

echo "== 2. 防泄漏校验：以下路径必须【不在】版本控制内 =="
LEAK=0
for p in dataset target backups mortal/checkpoints mortal/indexes mortal/logs mortal/runs \
         mortal/libriichi.so tools/tenhou_data/logs; do
    n=$(git ls-files -- "$p" | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "!! 泄漏 $p 下有 $n 个文件被纳入；示例："
        git ls-files -- "$p" | head -5
        LEAK=1
    else
        echo "   ok 已忽略: $p"
    fi
done
[ "$LEAK" = 1 ] && { echo "!! 存在数据/产物泄漏，已中止，未提交任何东西。"; exit 2; }

echo "== 3. 必要文件齐全性校验：以下必须【在】版本控制内 =="
MISS=0
for f in .gitignore .gitattributes Cargo.toml Cargo.lock Dockerfile environment.yml \
         mortal/train.py mortal/train_grp.py mortal/player.py mortal/dataloader.py mortal/model.py \
         mortal/ladder.py mortal/holdout_metrics.py mortal/bench_pipe.py \
         mortal/run_grp_capacity.sh mortal/run_r2_monitor.sh \
         mortal/config.offline.toml mortal/config.round2.toml mortal/config.grp_h128.toml \
         memory/MEMORY.md tools/training/TRAINING_MANUAL.md tools/training/TRAINING_PLAN.md \
         tools/training/ENV_SETUP_P100.md tools/training/grp_quality_probe.py \
         tools/tenhou_data/TASKLIST.md tools/tenhou_data/tenhou_pipeline.py \
         tools/player_stats/scan_player_ratings.py tools/player_stats/whitelist_coverage.py \
         tools/player_stats/build_whitelist.sh; do
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        echo "   ok tracked: $f"
    else
        echo "!! 缺失: $f 没有进入版本控制"
        MISS=1
    fi
done
[ "$MISS" = 1 ] && { echo "!! 必要文件不齐，已中止。"; exit 3; }

echo "== 4. 入库规模概览 =="
echo "文件数: $(git ls-files | wc -l)"
echo "总大小: $(git ls-files -z | xargs -0 du -ch 2>/dev/null | tail -1 | cut -f1)"
echo "其中 >1MB 的文件（应仅为 Rust 查表数据/文档素材）:"
git ls-files -z | xargs -0 du -m 2>/dev/null | awk '$1>=1' | sort -rn | head -10

echo "== 5. 提交 =="
cat > .git/MSG_TMP <<'MSG'
chore: 建立本地版本控制（第 1 轮完成、第 2 轮在跑的仓库现状快照）

含上游 Mortal(Equim-chan) 源码 + 本项目新增的训练/评测/数据工具链与文档。

已打补丁（与上游不同处，均可从行为反推，待有网时 git diff 精确核对）：
- mortal/train.py      : control.max_train_steps（预算到点自停）、control.snapshot_every
                       （另存 mortal_step_%09d.pth）、构建文件索引时对无 names 的牌谱
                       改用 start.get('names', []) 以避开 KeyError
- mortal/train_grp.py  : grp.control.max_train_steps、grp_best.pth 按 val loss 取优、
                       best_val_loss 入库、验证抽样固定种子、日志加 est. %、每次启动强制 lr
- mortal/player.py     : TestPlayer 的 enable_amp / enable_rule_based_agari_guard 改从
                       [test_play] 配置读；评测改为固定种子块 seed_start=(10000, 0x2000)

配置三份并存：config.offline.toml（第 1 轮实跑配方）、config.round2.toml（第 2 轮：
吞吐调优 + 玩家白名单 + 卫冕者换代 + grp_h128 奖励模型）、config.grp_h128.toml（GRP 容量实验派生件）

新增工具：mortal/ladder.py（checkpoint 间同牌山 1v3 配对对抗）、
mortal/holdout_metrics.py（未见月离线指标：top1 一致率/Q 校准/R2）、
mortal/bench_pipe.py（数据管道吞吐基准）、tools/training/grp_quality_probe.py（GRP 分位置质量探针）、
tools/player_stats/*（自建玩家强度榜与白名单覆盖度分析）

文档：memory/MEMORY.md 新增 §16 服务器实跑现状（含两处自我修正）；
tools/training/TRAINING_MANUAL.md 新增 §14 实跑记录与修正、§15 第 2 轮就绪状态。

刻意不入库（见 .gitignore）：
- dataset/  2,630,640 局天凤凤凰卓牌谱，受天凤 ToS 约束，禁止二次分发与公开下载
- *.pth / *.so / *.dll / *.exe / target/   权重、扩展模块与编译产物
- mortal/runs/ mortal/logs/ backups/ tools/tenhou_data/logs/   运行产物与本地备份

新环境需要自行生成两个前置件（都不在 git 内）：
1. mortal/libriichi.so  <- cargo build -p libriichi --lib --release（步骤见 ENV_SETUP_P100.md）
2. dataset/index/top_players_400.txt <- bash tools/player_stats/build_whitelist.sh（77 秒，需牌谱）
MSG

# 环境里没有任何 git 身份配置；这里只用 -c 临时传入，不改动你的 git 配置。
# 想换成自己的身份：git -c user.name='你的名字' -c user.email='you@example.com' commit --amend --reset-author
GITID=(-c user.name='mortal-trainer' -c user.email='mortal@localhost')
git "${GITID[@]}" commit -q -F .git/MSG_TMP
rm -f .git/MSG_TMP

echo "== 6. 打一个基线 tag，便于以后 baseline 换代与回退 =="
# 注意：annotated tag 自身也需要身份（只给 commit 加 -c 会在tag环节报unable to auto-detect email address）
git "${GITID[@]}" tag -a round1-complete -m "第 1 轮离线 CQL 完成态（GRP 30k + 主模型 200k；评测指标当时不可用）"

echo "== 7. 结果 =="
git log --oneline --stat -1 | head -8
echo "分支: $(git rev-parse --abbrev-ref HEAD)"
echo "工作区状态（应干净）:"
git status --short --branch | head -10
echo "仓库元数据大小: $(du -sh .git | cut -f1)"
echo "被忽略的顶层条目:"
git status --ignored --short | grep '^!!' | head -12
echo "ALL DONE"
