import os, sys, glob, time, random
nw = int(sys.argv[1]); ray = sys.argv[2]
os.environ['RAYON_NUM_THREADS'] = ray
import torch
from torch.utils.data import DataLoader
from config import config
from dataloader import FileDatasetsIter, worker_init_fn

BS = config['control']['batch_size']
files = sorted(glob.glob('/mnt/data8/mortal/Mortal/dataset/mjai/houou4/2025??/*.json.gz'))
random.seed(1); random.shuffle(files); files = files[:6000]
ds = FileDatasetsIter(version=config['control']['version'], file_list=files, pts=config['env']['pts'],
                      file_batch_size=8, reserve_ratio=0.0, player_names=[], num_epochs=1,
                      enable_augmentation=False, augmented_first=False)
it = iter(DataLoader(ds, batch_size=BS, drop_last=True, num_workers=nw, pin_memory=True, worker_init_fn=worker_init_fn))
NB = 120
t0 = time.time(); n = 0; warm = 20
for obs, a, m, s, k, r in it:
    n += obs.shape[0]
    if n % (BS*10) == 0:
        rate = n/(time.time()-t0)
        print(f'nw={nw} rayon={ray}: {n} samples, {rate:.0f} samples/s ({rate/BS:.2f} it/s)', flush=True)
    if n >= NB*BS: break
dt = time.time()-t0
print(f'RESULT nw={nw} rayon={ray} {n} samples in {dt:.1f}s -> {n/dt:.0f} samples/s = {n/dt/BS:.2f} it/s', flush=True)
