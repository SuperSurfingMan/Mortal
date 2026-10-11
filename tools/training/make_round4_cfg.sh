# 由 config.round2.toml 派生 config.round4.toml
# 第 4 轮：唯一实验变量 = enable_augmentation（样本 ×2 且不引入重复），
#          同时把白名单换成修正计分后的 v2（属"修测量仪"，不算实验变量）。
set -euo pipefail
ROOT=/mnt/data8/mortal/Mortal
cd "$ROOT/mortal"

sed -e "s|^# Mortal 第 2 轮离线训练配置.*|# Mortal 第 4 轮离线训练配置（10-11：白名单 v2 + enable_augmentation=true；暖启自 round2/424k）|" \
    -e "s|^state_file = 'checkpoints/round2/mortal.pth'.*|state_file = 'checkpoints/round4/mortal.pth'      # 启动前从 round2/mortal.pth(424k) 暖启|" \
    -e "s|^best_state_file = 'checkpoints/round2/best.pth'|best_state_file = 'checkpoints/round4/best.pth'|" \
    -e "s|^tensorboard_dir = 'runs/main_r2'|tensorboard_dir = 'runs/main_r4'|" \
    -e "s|^log_dir = 'runs/selection_r2'|log_dir = 'runs/selection_r4'|" \
    -e "s|^max_train_steps = 450000.*|max_train_steps = 645000        # 累计值：暖启 424k → 再训 221k 步 ≈ 一整遍“增强样本”（augmented_first=true）|" \
    -e "s|^player_names_files = .*|player_names_files = ['../dataset/index/top_players_400_v2.txt']   # v2：修正供托计分后的强度榜|" \
    -e "s|^file_index = 'indexes/main_r2_whitelist_v1.pth'|file_index = 'indexes/main_r4_whitelist_v2.pth'  # 名单变了必须换新名，否则沿用旧文件表|" \
    -e "s|^enable_augmentation = false.*|enable_augmentation = true      # ★ 本轮唯一实验变量：席位视角增强（样本×2、不重复原数据）|" \
    -e "s|^augmented_first = false|augmented_first = true           # 先吃增强样本，避免重蹈 R3\"先重复原样本\"的覆辙|" \
    -e "s|^state_file = 'checkpoints/round2/best.pth'|state_file = 'checkpoints/round4/best.pth'|" \
    config.round2.toml > config.round4.toml

echo '--- 关键项核对 ---'
grep -nE "^(state_file|best_state_file|tensorboard_dir|max_train_steps|num_epochs|enable_augmentation|augmented_first|player_names_files|file_index|num_workers|file_batch_size|log_dir) *=" config.round4.toml
echo '--- 与 round2 的差异行数（应≈22 行含注释）---'
diff config.round2.toml config.round4.toml | grep -c '^[<>]'
