# 定量判别：第 2 轮停在 424,847 步 ≈ 白名单池的一个 epoch 吗？
# 口径与训练完全一致：GameplayLoader(player_names=白名单) 只产出名单内座位的决策样本
set -u
PY=/mnt/data8/miniconda3/envs/mortal/bin/python
cd /mnt/data8/mortal/Mortal/mortal
export MORTAL_CFG=$PWD/config.round2.toml LD_LIBRARY_PATH=/mnt/data8/miniconda3/envs/mortal/lib
$PY - <<'PYP'
import glob, random, numpy as np, torch
from common import filtered_trimmed_lines
from config import config
from libriichi.dataset import GameplayLoader

names = set()
for fn in config['dataset']['player_names_files']:
    with open(fn, encoding='utf-8') as f:
        names.update(filtered_trimmed_lines(f))

fl = torch.load(config['dataset']['file_index'], weights_only=True)['file_list']
print(f'索引内文件数 = {len(fl):,}  |  名单人数 = {len(names):,}')

random.seed(11)
sample = random.sample(fl, 120)
ld = GameplayLoader(version=config['control']['version'], oracle=False,
                    player_names=list(names), excludes=None, augmented=False)
per = []
for f in sample:
    data = ld.load_gz_log_files([f])
    n = sum(len(g.take_obs()) for file in data for g in file)
    per.append(n)
mean = float(np.mean(per)); sd = float(np.std(per))
print(f'抽样 120 文件（仅名单座位）: 平均 {mean:.1f} 样本/文件  std={sd:.1f}  区间 {min(per)}~{max(per)}')

steps_1_epoch = len(fl) * mean / config['control']['batch_size']
print(f'→ 1 个 epoch ≈ {steps_1_epoch:,.0f} 步（±{len(fl)*sd/np.sqrt(len(sample))/512:,.0f} 步 95%CI 半宽约）')
obs = 424847 - 200000
print(f'→ 本轮实际消耗 = 424,847 - 200,000(暖启) = {obs:,} 步 = 预测值的 {obs/steps_1_epoch:.1%}')
print(f'→ 到达 max_train_steps=450,000 还需 {450000-424847:,} 步，'
      f'相当于 {((450000-424847)*512)/ (len(fl)*mean):.2f} 个 epoch → 单靠 num_epochs=1 不可能到达')
PYP
echo DONE
