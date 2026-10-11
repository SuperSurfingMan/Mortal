"""拉平三轮的 test_play / loss 轨迹，判断「第 3 轮（epoch 2）」有没有继续进步。"""
import glob
import os
from tensorboard.backend.event_processing import event_accumulator as em

RUNS = ['runs/main', 'runs/main_r2', 'runs/main_r3']
SINGLES = ['loss/dqn_loss', 'loss/cql_loss', 'loss/next_rank_loss',
           'test_play/avg_pt', 'test_play/avg_ranking', 'test_play/point_per_round',
           'test_play/riichi_point', 'test_play/fuuro_point']
GROUPS = ['test_play/ranking', 'test_play/behavior']


def fmt(evs, take=10):
    if not evs:
        return '(无)'
    idx = sorted(set([0] + [round(i * (len(evs) - 1) / (take - 1)) for i in range(take)]))
    return '  '.join(f'{evs[i].step // 1000}k:{evs[i].value:.3f}' for i in idx)


for run in RUNS:
    if not os.path.isdir(run):
        print(f'== {run} 不存在')
        continue
    e = em.EventAccumulator(run)
    e.Reload()
    print(f'== {run}   单值标签 {len(e.Tags()["scalars"])} 个')
    for t in SINGLES:
        if t in e.Tags()['scalars']:
            print(f'   {t:<28s}', fmt(e.Scalars(t)))
    for d in sorted(glob.glob(run + '/test_play_*')):
        ee = em.EventAccumulator(d)
        ee.Reload()
        for t in ee.Tags()['scalars']:
            print(f'   {os.path.basename(d):<34s}', fmt(ee.Scalars(t), 6))
    print()
