# 由 config.round2.toml 派生 config.round3.toml（只改步数预算/epoch 数/落盘路径，配方不变）
# 用法: bash /mnt/data8/mortal/Mortal/tools/training/make_round3_cfg.sh
set -euo pipefail
ROOT=/mnt/data8/mortal/Mortal
cd "$ROOT/mortal"

sed -e "s|^# Mortal 第 2 轮离线训练配置.*|# Mortal 第 3 轮离线训练配置（= 第 2 轮配方 + 第 2 个 epoch，2026-10-10 由 ladder 判定第 2 轮有效后开启）|" \
    -e "s|^state_file = 'checkpoints/round2/mortal.pth'.*|state_file = 'checkpoints/round3/mortal.pth'      # 从 round2/mortal.pth(424k) 暖启|" \
    -e "s|^best_state_file = 'checkpoints/round2/best.pth'|best_state_file = 'checkpoints/round3/best.pth'|" \
    -e "s|^tensorboard_dir = 'runs/main_r2'|tensorboard_dir = 'runs/main_r3'|" \
    -e "s|^max_train_steps = 450000.*|max_train_steps = 650000        # 累计值：暖启 424k → 再训 226k 步 ≈ 白名单池的第 2 个 epoch|" \
    -e "s|^log_dir = 'runs/selection_r2'|log_dir = 'runs/selection_r3'|" \
    -e "s|^num_epochs = 1|num_epochs = 2                 # ★ 本轮唯一实质变量：同一份白名单池再过一遍|" \
    -e "s|^state_file = 'checkpoints/round2/best.pth'|state_file = 'checkpoints/round3/best.pth'|" \
    config.round2.toml > config.round3.toml

echo '--- 关键项核对 ---'
grep -nE "^(state_file|best_state_file|tensorboard_dir|max_train_steps|num_epochs|num_workers|file_batch_size|player_names_files|file_index|log_dir) *=|^\[test_play\]|^hidden_size|^num_layers|^state_file = 'checkpoints/grp" config.round3.toml
echo '--- 与 round2 的差异行数 ---'
diff config.round2.toml config.round3.toml | grep -c '^[<>]'
