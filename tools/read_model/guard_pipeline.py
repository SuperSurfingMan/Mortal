#!/usr/bin/env python3
"""推理期守卫的无人值守编排（对话断开后仍推进到初步结论）。

**目标**：北京时间 2026-10-23 12:00 前产出一份可判读的初步结论。

**阶段**（全部幂等：``run_meta.json`` 已存在即跳过，可安全重启）：

- **S0** 等第二轮：λ=50/τ=0.03，双向 2000 sets（16:13 起）。
- **S1** 汇总并决策（`agg_guard_ladder.py`）。
- **S2** 条件分支：
  - ``GO``（avg_pt>+0.2 且 agari 降幅<3%）→ 扩 10000 sets × 双向；
  - ``WEAK``（−0.2≤avg_pt≤+0.2 或 agari 降幅≥3%）→ τ 扫描
    （λ=50，τ=0.02 / 0.05，各 2000×2，串行两批），选优后扩 8000×2；
  - ``NO``（avg_pt<−0.2）→ 低剂量 λ=10 线性 2000×2；若为正再扩 8000×2，
    否则直接判 FAIL。
- **S3** 交叉验证：剩余时间跑第二个参数臂 8000×2（若 S2 已扩到 10000 则跳过）。
- **S4** 生成 `tools/read_model/docs/GUARD_VERDICT_20261023.md`。

**并发**：最多 2 个 ladder（每方向一个，固定 cuda:1 / cuda:3，避免与主任务
抢 cuda:0）。**日志**：`mortal/logs/guard_pipeline/`。

启动::

    tmux new -d -s guard_pipeline "cd /mnt/data8/mortal/Mortal && \
        /mnt/data8/miniconda3/envs/mortal/bin/python \
        tools/read_model/guard_pipeline.py > mortal/logs/guard_pipeline.log 2>&1"

查看::

    tail -f mortal/logs/guard_pipeline.log
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/data8/mortal/Mortal')
MORTAL = ROOT / 'mortal'
PY = '/mnt/data8/miniconda3/envs/mortal/bin/python'
B = MORTAL / 'checkpoints/reward_ablation/B_rank_pt_aligned/mortal.pth'
AGG = ROOT / 'tools/read_model/agg_guard_ladder.py'
RUNS = MORTAL / 'runs'
LOG = MORTAL / 'logs/guard_pipeline'
OUT = ROOT / 'tools/read_model/diagnostics/results/guard_pipeline'
SEEDS = (260000, 270000)
GPU = {260000: 'cuda:1', 270000: 'cuda:3'}
PTS = np.array([90.0, 45.0, 0.0, -135.0])
AGARI_BASE = 0.217          # 100 sets 自对弈（B/756k）


def now() -> str:
    return datetime.now().strftime('%H:%M:%S')


def log(msg: str):
    print(f'[{now()}] {msg}', flush=True)


def meta_path(name: str, seed: int) -> Path:
    return RUNS / f'{name}_{seed}' / 'run_meta.json'


def is_done(name: str, seed: int) -> bool:
    return meta_path(name, seed).exists()


def launch(name: str, lam: float, tau: float, seed: int, sets: int):
    """启动一个 ladder；已完成的跳过。返回 Popen 或 None。"""
    if is_done(name, seed):
        log(f'  skip {name} seed={seed}（已完成）')
        return None
    LOG.mkdir(parents=True, exist_ok=True)
    logf = LOG / f'{name}_{seed}.log'
    cmd = [PY, 'ladder.py', str(B), str(B), str(seed), str(sets),
           str(RUNS / f'{name}_{seed}'),
           '--guard-lambda', str(lam), '--guard-risk-floor', str(tau),
           '--device', GPU[seed], '--no-amp', '--tag', name]
    log(f'  launch {name} seed={seed} sets={sets} gpu={GPU[seed]}')
    return subprocess.Popen(cmd, cwd=MORTAL, stdout=open(logf, 'w'),
                            stderr=subprocess.STDOUT)


def run_batch(name: str, lam: float, tau: float, sets: int,
              retries: int = 1):
    """同一参数臂的双向并行；失败重试。"""
    while True:
        procs = []
        for seed in SEEDS:
            p = launch(name, lam, tau, seed, sets)
            if p is not None:
                procs.append((seed, p))
        if not procs:
            return True
        ok = True
        for seed, p in procs:
            p.wait()
            if p.returncode != 0:
                log(f'  !! {name} seed={seed} 退出码 {p.returncode}')
                ok = False
        if ok:
            return True
        if retries <= 0:
            log(f'  !! {name} 重试耗尽，跳过')
            return False
        retries -= 1
        log(f'  {name} 重试一次')


def wait_for(name: str, seeds=SEEDS, timeout_h: float = 8.0):
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        if all(is_done(name, s) for s in seeds):
            return True
        time.sleep(60)
    log(f'  !! 等待 {name} 超时（{timeout_h}h）')
    return False


def load_arm(name: str):
    rows = []
    for seed in SEEDS:
        p = meta_path(name, seed)
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        r = m['results']
        c = np.array(r['raw_rankings'], float)
        n = c.sum()
        mu = float(r['avg_pt'])
        se = float(np.sqrt(((c / n) * (PTS - mu) ** 2).sum() / n))
        rows.append({'seed': seed, 'n': int(n), 'avg_pt': mu, 'se': se,
                     'avg_rank': float(r['avg_rank']),
                     'agari': float(r['agari_rate']),
                     'houjuu': float(r['houjuu_rate']),
                     'riichi': float(r['riichi_rate']),
                     'lam': m.get('guard_lambda'),
                     'tau': m.get('guard_risk_floor')})
    return rows


def summarize(name: str):
    rows = load_arm(name)
    if not rows:
        return None
    mean = float(np.mean([r['avg_pt'] for r in rows]))
    se = float(np.sqrt(sum(r['se'] ** 2 for r in rows))) / len(rows)
    return {
        'name': name, 'rows': rows, 'n': sum(r['n'] for r in rows),
        'avg_pt': mean, 'se': se, 'lo': mean - 1.96 * se, 'hi': mean + 1.96 * se,
        'agari': float(np.mean([r['agari'] for r in rows])),
        'houjuu': float(np.mean([r['houjuu'] for r in rows])),
        'riichi': float(np.mean([r['riichi'] for r in rows])),
        'lam': rows[0]['lam'], 'tau': rows[0]['tau'],
    }


def dump(name: str, s: dict):
    log(f'  [{name}] n={s["n"]} avg_pt={s["avg_pt"]:+.3f} '
        f'CI[{s["lo"]:+.2f},{s["hi"]:+.2f}] agari={s["agari"]:.3f} '
        f'houjuu={s["houjuu"]:.3f}')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{name}.json').write_text(json.dumps(s, ensure_ascii=False, indent=1))


def decide(s: dict) -> str:
    if s is None:
        return 'NO'
    agari_drop = s['agari'] - AGARI_BASE
    if s['avg_pt'] > 0.2 and agari_drop > -0.03:
        return 'GO'
    if s['avg_pt'] < -0.2:
        return 'NO'
    return 'WEAK'


def write_verdict(head: str, arms: list[dict], decision: str,
                  notes: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        '# 推理期防守守卫：初步结论（自动生成）', '',
        f'- 生成时间：{datetime.now().isoformat(timespec="seconds")}',
        f'- 阶段结论：**{head}**', '',
        '## 各臂（双向合并，n = 半庄数）', '',
        '| 臂 | λ | τ | n | avg_pt | 95%CI | agari | houjuu | riichi |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for s in arms:
        if s is None:
            continue
        lines.append(
            f"| {s['name']} | {s['lam']} | {s['tau']} | {s['n']} | "
            f"{s['avg_pt']:+.3f} | [{s['lo']:+.2f}, {s['hi']:+.2f}] | "
            f"{s['agari']:.3f} | {s['houjuu']:.3f} | {s['riichi']:.3f} |")
    lines += ['', '## 判读', '']
    lines += [f'- {n}' for n in notes]
    lines += [
        '', '## 口径与限制', '',
        '- 对照：B/756k 同 ckpt 自对弈（challenger 带守卫，champion 不带），1v3。',
        '- pt = [90,45,0,−135]；CI 用多项分布近似（未修正同 set 组内相关，偏窄）。',
        '- baseline agari 0.217 / houjuu 0.130 来自 100 sets 自对弈，仅作量级参考。',
        '- 2000 sets 的 SE≈0.66；8000~10000 sets 才能把 SE 压到 0.3~0.35。',
    ]
    p = ROOT / 'tools/read_model/docs/GUARD_VERDICT_20261023.md'
    p.write_text('\n'.join(lines) + '\n')
    log(f'  verdict -> {p}')
    # 同时把原始汇总落一份到 results/
    (OUT / 'verdict.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    log('=== guard pipeline start ===')
    arms: list[dict] = []
    notes: list[str] = []

    # ---------- S0: 等第二轮 ----------
    log('S0 等第二轮 λ=50/τ=0.03（2000×2）...')
    wait_for('guard_l50t03')
    s2 = summarize('guard_l50t03')
    if s2:
        dump('s2_l50t03_2000', s2)
        arms.append(s2)
    dec = decide(s2)
    log(f'S1 决策 = {dec}（avg_pt={s2["avg_pt"]:+.3f}, agari={s2["agari"]:.3f}）')

    # ---------- S2: 条件分支 ----------
    final_name = None
    if dec == 'GO':
        notes.append('第二轮 λ=50/τ=0.03 方向为正且 agari 未塌（>−3pp），'
                     '按 GO 扩到 10000 sets 双向。')
        final_name = 'guard_l50t03_x10k'
        run_batch(final_name, 50, 0.03, 10000)
        wait_for(final_name, timeout_h=8)
    elif dec == 'WEAK':
        notes.append('第二轮 λ=50/τ=0.03 落在弱区（|avg_pt|≤0.2 或 agari 降≥3pp），'
                     '先做 τ 扫描（λ=50，τ=0.02/0.05，各 2000×2）选优。')
        best = None
        for tag, tau in (('t02', 0.02), ('t05', 0.05)):
            name = f'guard_l50t{tag}_2000'
            run_batch(name, 50, tau, 2000)
            s = summarize(name)
            if s:
                dump(f's2_{name}', s)
                arms.append(s)
                if best is None or s['avg_pt'] > best[1]['avg_pt']:
                    best = (name, s)
        if best is not None:
            final_name = f'{best[0]}_x8k'
            log(f'S2b 选优 {best[0]}（avg_pt {best[1]["avg_pt"]:+.3f}）→ 扩 8000×2')
            best_tau = best[1]['tau']
            run_batch(final_name, 50, best_tau, 8000)
            wait_for(final_name, timeout_h=8)
        else:
            notes.append('τ 扫描无有效结果。')
    else:  # NO
        notes.append('第二轮 λ=50/τ=0.03 点估计 < −0.2，按 NO 转低剂量 λ=10 线性。')
        name = 'guard_l10_2000'
        run_batch(name, 10, 0.0, 2000)
        s = summarize(name)
        if s:
            dump(f's2_{name}', s)
            arms.append(s)
        if s and s['avg_pt'] > 0:
            final_name = 'guard_l10_x8k'
            log(f'S2b λ=10 为正（{s["avg_pt"]:+.3f}）→ 扩 8000×2')
            run_batch(final_name, 10, 0.0, 8000)
            wait_for(final_name, timeout_h=8)
        else:
            notes.append('λ=10 线性也未转正，判 FAIL（不再扩样本）。')

    # ---------- S3: 交叉验证（剩余时间） ----------
    if final_name:
        fs = summarize(final_name)
        if fs:
            dump(f's3_{final_name}', fs)
            arms.append(fs)
        # 剩余时间跑「另一个剂量」的 8000×2：若最终臂是 gating，补线性 λ=20；
        # 若最终臂是 λ=10，补 λ=20。避免和已跑过的臂重复。
        if final_name.startswith('guard_l50t03'):
            alt = ('guard_l20_x8k', 20, 0.0)
        elif final_name.startswith('guard_l10'):
            alt = ('guard_l20_x8k', 20, 0.0)
        else:
            alt = ('guard_l20_x8k', 20, 0.0)
        if not is_done(alt[0], SEEDS[0]):
            log(f'S3 交叉验证：{alt[0]}（线性 λ={alt[1]}）8000×2')
            run_batch(alt[0], alt[1], alt[2], 8000)
            ss = summarize(alt[0])
            if ss:
                dump(f's3_{alt[0]}', ss)
                arms.append(ss)

    # ---------- S4: 结论 ----------
    finals = [a for a in arms if a and 'x' in a['name']]
    if finals:
        best = max(finals, key=lambda a: a['avg_pt'])
        if best['lo'] > 0:
            head = f"GO：{best['name']} 显著为正（{best['avg_pt']:+.3f}，CI 下界 {best['lo']:+.2f}）"
            notes.append('大样本臂 CI 下界 > 0 ⇒ 守卫有可判读的正收益，建议作为默认推理配置。')
        elif best['hi'] < 0:
            head = f"NO-GO：{best['name']} 显著为负（{best['avg_pt']:+.3f}）"
            notes.append('大样本臂 CI 上界 < 0 ⇒ 守卫在 1v3 中稳定变弱，关闭。')
        else:
            head = (f"INCONCLUSIVE：最好臂 {best['name']} avg_pt {best['avg_pt']:+.3f}，"
                    f"CI [{best['lo']:+.2f},{best['hi']:+.2f}] 仍跨 0")
            notes.append('扩样本后仍跨 0 ⇒ 效应量小于可测阈值（|δ|<0.3），'
                         '按「无显著收益」处置；若追求定论需再扩到 20000+ sets。')
    else:
        head = 'INCONCLUSIVE：没有完成的大样本臂'
    write_verdict(head, arms, dec, notes)
    log(f'=== done: {head} ===')


if __name__ == '__main__':
    main()
