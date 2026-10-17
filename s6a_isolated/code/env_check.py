"""启动前自检：一条命令验证「目录 + 环境变量 + 依赖 + 选卡」是否全对。

用法（**必须在 mortal/ 目录下运行**，否则会先报目录错误并给出修法）：
    cd /mnt/data8/mortal/Mortal/mortal && python env_check.py
可选：
    python env_check.py --cfg config.round2.toml      # 顺带校验配置文件
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # 本文件所在目录 = mortal/
FAILS = []


def ok(name, detail):
    print(f'  [OK]   {name:<26s} {detail}')


def bad(name, detail, fix):
    FAILS.append((name, fix))
    print(f'  [FAIL] {name:<26s} {detail}')
    print(f'         → 修法: {fix}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=os.environ.get('MORTAL_CFG', ''), help='顺带校验的配置文件路径')
    args = ap.parse_args()

    print('== 1. 工作目录与导入路径 ==')
    cwd = os.getcwd()
    if os.path.realpath(cwd) == os.path.realpath(HERE):
        ok('cwd', cwd)
    else:
        bad('cwd', f'当前在 {cwd}，而 libriichi.so 在 {HERE}',
            f'cd {HERE}   （仓库根的 libriichi/ 是 Rust crate 目录，会遮蔽真扩展）')

    try:
        import libriichi
    except Exception as e:  # noqa: BLE000
        bad('import libriichi', f'{type(e).__name__}: {e}', '确认已 cargo build 并 cp 到 mortal/libriichi.so')
        sys.exit(1)
    ver = getattr(libriichi, '__version__', None)
    if not ver:
        bad('libriichi 是否为真扩展', f'__file__={getattr(libriichi, "__file__", None)} '
            f'__path__={getattr(libriichi, "__path__", None)}', f'cd {HERE} 后重跑（现在是命名空间包）')
    else:
        ok('libriichi', f'{libriichi.__file__} | v{ver} {libriichi.__profile__}')

    print('== 2. torch / CUDA ==')
    import torch
    ok('torch', f'{torch.__version__} | arch={",".join(torch.cuda.get_arch_list()[-4:])}')
    if not torch.cuda.is_available():
        bad('cuda available', 'False', '检查驱动 / CUDA_VISIBLE_DEVICES 是否指向存在的卡')
    else:
        name = torch.cuda.get_device_name(0)
        ok('cuda available', f'{torch.cuda.device_count()} 张可见，cuda:0 = {name}')
        want = os.environ.get('CHECK_GPU_NAME')
        if want and want.split()[0].lower() not in name.lower():
            bad('选卡是否符合预期', f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} → {name}，期望含 "{want}"',
                '记住 CUDA 序号默认≠nvidia-smi：先 export CUDA_DEVICE_ORDER=PCI_BUS_ID 再按 nvidia-smi 序号选卡')
        else:
            ok('选卡', f'{name}（CUDA_DEVICE_ORDER={os.environ.get("CUDA_DEVICE_ORDER", "<未设>")}, '
                       f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<未设>")}）')

    print('== 3. 关键环境变量 ==')
    cfg = args.cfg
    if not cfg:
        bad('MORTAL_CFG', '未设置（默认会去找相对路径 config.toml，本仓库没有这个文件）',
            f'export MORTAL_CFG={HERE}/config.round2.toml')
    else:
        p = cfg if os.path.isabs(cfg) else os.path.join(cwd, cfg)
        if os.path.exists(p):
            ok('MORTAL_CFG', p)
        else:
            bad('MORTAL_CFG', f'文件不存在: {p}', '改成实际存在的配置文件绝对路径')
    for k, need in [('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'), ('RAYON_NUM_THREADS', '8')]:
        v = os.environ.get(k)
        if v is None:
            bad(k, '未设置', f'export {k}={need}   （必须在启动前设，rayon 池进程内只建一次）')
        elif v != need:
            ok(k, f'{v}（文档建议 {need}）')
        else:
            ok(k, v)
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    ok('LD_LIBRARY_PATH', (ld[:60] + '…') if ld else '未设置（训练可省；跑 target/release/validate_logs|stat 时必需）')

    print('== 4. 版本常量与数据/目录 ==')
    from libriichi.consts import ACTION_SPACE, obs_shape
    shp = obs_shape(4)
    if shp[0] == 1012 and ACTION_SPACE == 46:
        ok('obs_shape(4)/ACTION_SPACE', f'{shp} / {ACTION_SPACE}')
    else:
        bad('obs_shape(4)/ACTION_SPACE', f'{shp} / {ACTION_SPACE}', 'libriichi 编译产物与 version=4 不匹配，重新 cargo build')
    # v5 = v4 + 6 行紧迫度特征（S6a）。旧 .so 无此 arm → obs_shape(5) 会 panic，
    # 因此在训练前显式验证，避免带着陈旧编译产物开跑。
    try:
        shp5 = obs_shape(5)
        if shp5[0] == 1018:
            ok('obs_shape(5)', f'{shp5}（v4+6 紧迫度特征，S6a）')
        else:
            bad('obs_shape(5)', f'{shp5}', '与 (1018, 34) 不符，检查 libriichi 编译产物')
    except Exception as e:  # noqa: BLE001
        bad('obs_shape(5)', f'{type(e).__name__}: {e}', 'libriichi 无 version 5：重新 cargo build --release 并 cp 到 mortal/')
    if cfg:
        try:
            from config import config          # 依赖 cwd 或 MORTAL_CFG 可加载
            d = config['dataset']['globs']
            st = config['control']['state_file']
            ok('config 可解析', f'[control].state_file={st} | batch={config["control"]["batch_size"]} '
                f'| nw={config["dataset"]["num_workers"]} | grp={config["grp"]["state_file"]} '
                f'| version={config["control"].get("version", 1)}')
            from glob import glob
            n = sum(len(glob(p, recursive=True)) for p in d)
            if n:
                ok('dataset.globs 命中文件数', f'{n:,}')
            else:
                bad('dataset.globs 命中文件数', '0', '检查 globs 里的绝对路径与目录结构')
            for sub in ('checkpoints', 'runs', 'indexes', 'logs'):
                if not os.path.isdir(sub):
                    bad(f'目录 {sub}/', '不存在', f'mkdir -p {sub}')
            ok('其它目录', 'checkpoints/runs/indexes/logs 齐全')
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE000
            bad('config 解析', f'{type(e).__name__}: {e}', '确认 MORTAL_CFG 指向合法 toml 且 cwd=mortal/')
    gp = os.path.join(HERE, '..', 'target', 'release', 'validate_logs')
    ok('Rust CLI', (gp if os.path.exists(gp) else '缺失') + '  （跑它时需要 LD_LIBRARY_PATH 指向 conda 的 lib 目录）')

    print()
    if FAILS:
        print(f'结论：{len(FAILS)} 项不通过 → ' + '; '.join(n for n, _ in FAILS))
        sys.exit(1)
    print('结论：全部通过，可以启动训练/评测')


main()
