# 清理残留 tmux 会话、重启看板、启动第 2 轮判定长跑
set -u
PY=/mnt/data8/miniconda3/envs/mortal/bin
ROOT=/mnt/data8/mortal/Mortal

echo '=== 1) 清理遗留会话（保留本次要新建的）==='
for s in agtest est cmdcheck env_probe env_map env_e2 grp_probe grp_cap whitelist toplist cfg_check smoke_r2 grp_v2 train_grp train_main tb; do
    tmux kill-session -t $s 2>/dev/null && echo "  已杀: $s"
done

echo '=== 2) 确认无残留 python 进程 ==='
ps -eo pid,args | grep -E '[p]ython' | head -5 || true

echo '=== 3) 重启 TensorBoard（原会话随第 1 轮结束已死）==='
tmux new -d -s tb "cd $ROOT/mortal && export LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib && $PY/tensorboard --logdir runs --host 127.0.0.1 --port 6006 > logs/tensorboard.log 2>&1"
sleep 10
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:6006/ || echo 000)
echo "  tensorboard HTTP $code"

echo '=== 4) 启动判定长跑（5000 sets，5 卡并行两波）==='
tmux new -d -s ladder_r2 "bash $ROOT/mortal/run_r2_ladder.sh > $ROOT/mortal/logs/ladder_r2_run.log 2>&1"
sleep 25
tmux ls
echo '--- 起跑日志 ---'
tail -6 $ROOT/mortal/logs/ladder_r2_run.log 2>/dev/null
echo '--- 每卡占用 ---'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | head -8
echo DONE
