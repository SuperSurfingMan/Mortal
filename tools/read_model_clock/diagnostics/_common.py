"""诊断脚本共享层：路径解析 + 模型/数据装载 + 通用数值工具。

所有 `tools/read_model/diagnostics/*.py` 都从这里取路径与工具函数，避免硬编码。
可用环境变量覆盖：

    RM_CKPT      A2b checkpoint（默认 <repo>/mortal/read_model_v2s_a2b/best.pth）
    RM_VAL_GLOB  holdout 牌谱 glob（默认 <repo>/dataset/mjai/houou4/202609/**）
    RM_OUT_DIR   产物目录（默认本目录下的 results/）

本模块**只读**，不写任何仓库文件、不改 `mortal/` 与 `libriichi/`。
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # tools/read_model_clock/diagnostics → repo root
MORTAL_DIR = REPO / 'mortal'
# 本目录是**独立时钟探针**：本地 dataset/model/train 优先；共享 model/constraints/
# structure/rule_baseline 回退到主 read_model。两条路线互不覆盖。
TOOLS_DIR = REPO / 'tools' / 'read_model_clock'
BASELINE_TOOLS = REPO / 'tools' / 'read_model'

OUT_DIR = Path(os.environ.get('RM_OUT_DIR', str(HERE / 'results')))
OUT_DIR.mkdir(parents=True, exist_ok=True)
# 大体积中间缓存（§3.15.8 的 cache.npz ≈ 120 MB）放仓库外的 scratch，不入库；
# 独立目录避免与主任务的 cache.npz 互相覆盖。
CACHE_DIR = Path(os.environ.get('RM_CACHE_DIR', '/tmp/read_model_diag_cache_clock'))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE = CACHE_DIR / 'cache.npz'
CKPT = Path(os.environ.get(
    'RM_CKPT', str(MORTAL_DIR / 'read_model_v2s_a2b' / 'best.pth')))
VAL_GLOB = os.environ.get(
    'RM_VAL_GLOB', str(REPO / 'dataset' / 'mjai' / 'houou4' / '202609' / '**' / '*.json.gz'))
# 探针训练月（§3.15.3）：必须与评估月 202609 不重叠，否则 P0~P3 的对照失效
TRAIN_GLOB = os.environ.get(
    'RM_TRAIN_GLOB',
    str(REPO / 'dataset' / 'mjai' / 'houou4_names' / '202608' / '**' / '*.json.gz'))

EPS = 1e-12
SUIT = '万筒索'
# 官方固定评估集协议（train.build_fixed_eval_set）：202609 holdout + 该 seed
EVAL_SEED = 20261016


def bootstrap_paths():
    """把 local clock probe、baseline read_model、mortal/ 加进 sys.path（幂等）。

    顺序必须是 **local → baseline → mortal**：否则 `import dataset/model_v2/train`
    会命中主任务脚本，时钟探针就变成改主路线。
    """
    paths = [str(HERE), str(TOOLS_DIR), str(BASELINE_TOOLS), str(MORTAL_DIR)]
    for p in paths:
        while p in sys.path:
            sys.path.remove(p)
    sys.path[:0] = paths


bootstrap_paths()

import torch  # noqa: E402  （必须在 bootstrap_paths 之后）

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------ 装载
def load_net(ckpt=None):
    """按 checkpoint 里存的 config 重建 A2b（v2s）并 eval()。"""
    import model_v2 as M2
    ckpt = Path(ckpt or CKPT)
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = st['config']
    net = M2.build_model_v2s(conv_channels=cfg['conv_channels'],
                             num_blocks=cfg['num_blocks'],
                             with_aux=not cfg['no_aux'],
                             wait_head=cfg.get('wait_head', 'flat'),
                             tl_mode=cfg.get('tl_mode', 'pooled'),
                             next_tedashi=cfg.get('next_tedashi_weight', 0.0) > 0,
                             tile_order=cfg.get('tile_order', False)).to(DEV).eval()
    net.load_state_dict(st['model'])
    return net


def forward(net, batch, keys=('obs', 'unseen', 'tl_tokens', 'tl_mask'),
            tl_override=None, bs=1024):
    """跑读牌模型，返回 (alloc_probs[n,3,34,5], exp_counts[n,3,34])，均 float64。

    `tl_override` 用于配对反事实：只替换 tl_tokens（tensor）或同时替换
    token 及其侧车字段（RM12：dict[tl_tokens/tl_turn/tl_age/tl_consumed/
    tl_flags/tl_pos/tl_age_ev]）；其余输入（尤其 obs）不变。pooled 模式忽略
    侧车字段，A2b/A2c 数值不受影响。
    """
    import train as T
    m = batch['obs'].shape[0]
    AP = np.zeros((m, 3, 34, 5), np.float32)
    EC = np.zeros((m, 3, 34), np.float32)
    extra_keys = ('tl_turn', 'tl_age', 'tl_consumed', 'tl_flags',
                  'tl_pos', 'tl_age_ev')
    with torch.no_grad():
        for i in range(0, m, bs):
            sl = slice(i, i + min(bs, m - i))
            sub = {k: batch[k][sl] for k in
                   ('unseen', 'river', 'post_riichi_river', 'riichi', 'own_hand')}
            wb = T.wait_bias_of(sub).to(DEV)
            ch = {k: batch[k][sl].to(DEV) for k in keys}
            if getattr(net, 'tl_mode', 'pooled') == 'tile':
                for k in extra_keys:
                    ch[k] = batch[k][sl].to(DEV)
            if tl_override is not None:
                if isinstance(tl_override, dict):
                    for k, v in tl_override.items():
                        ch[k] = v[sl].to(DEV)
                else:
                    ch['tl_tokens'] = tl_override[sl].to(DEV)
            o = net(ch['obs'], ch['unseen'], wb, ch['tl_tokens'], ch['tl_mask'],
                    **net.tl_extra(ch))
            AP[sl] = o['alloc_probs'].cpu().numpy()
            EC[sl] = o['exp_counts'].cpu().numpy()
    return AP.astype(np.float64), EC.astype(np.float64)


def build_eval(n_eval=40000, seed=EVAL_SEED):
    """官方固定评估集（202609 holdout）。返回 (batch, batch_as_numpy, 文件数)。"""
    import glob as _glob

    import train as T
    files = sorted(_glob.glob(VAL_GLOB, recursive=True))
    batch = T.build_fixed_eval_set(files, n_eval, seed=seed)
    bn = {k: v.numpy() for k, v in batch.items()}
    return batch, bn, len(files)


# ------------------------------------------------------- counts 精确档
def counts_tier(unseen, hand):
    """零行为假设的精确后验（洗牌 ⇒ 超几何）。

    入参 `unseen[n,34]` = 4 − 自家手牌 − 全部可见；`hand[n,3]` = 13 − 3×副露。
    返回 (E_count[n,3,34], PMF[n,3,34,5])。

    池子 N = Σunseen = 三家暗牌 + 活牌山 + 王牌区（−宝牌指示）。目标家的 h_j 张
    暗牌是这 N 张的一个均匀随机子集（发牌+摸牌等价于随机指派），故
    ``P(count_i = k) = C(u_i,k)·C(N−u_i, h_j−k) / C(N, h_j)`` **精确**成立，
    不含任何行为假设 —— 因此它是「顺序盲 / 行为盲」的天然对照地板。
    """
    n = unseen.shape[0]
    N = unseen.sum(-1)[:, None]                                   # [n,1]
    E = unseen[:, None, :] * (hand / np.maximum(N, 1))[:, :, None]
    PMF = np.zeros((n, 3, 34, 5))
    for j in range(3):
        h = hand[:, j][:, None].astype(np.float64)                # [n,1]
        PMF[:, j, :, 0] = 1.0
        for k in range(1, 5):
            ratio = ((unseen - (k - 1)) / k) * \
                    ((h - (k - 1)) / np.maximum(N - unseen - h + k, EPS))
            ok = (h >= k) & (N - unseen >= h - k)
            PMF[:, j, :, k] = np.where(ok, PMF[:, j, :, k - 1] * np.maximum(ratio, 0), 0.0)
        PMF[:, j] /= np.maximum(PMF[:, j].sum(-1, keepdims=True), EPS)
    return E, PMF


# ------------------------------------------------------- tl_tokens 解析
def parse_tl(tl_tokens, tl_mask):
    """把时序 token 还原成「每家手切/摸切了哪些牌、在第几个 token」。

    token 布局（`dataset.py`，43 维）：
      ``[0:34]``  牌 one-hot（reach 事件 tid=−1 ⇒ 全 0）
      ``[34]``    tsumogiri（摸切）标记
      ``[35:42]`` 动作 one-hot：dahai/reach/chi/pon/daiminkan/ankan/kakan
      ``[42]``    归一化巡目 —— ⚠️ 已知缺陷：写入的是快照时目标家的 at_turn
                  常数，全部 26 个 token 相同（见 MEMORY.md §3.15.5-①）

    返回 ``hc[n,3,34]`` / ``tg[n,3,34]``：该家手切/摸切该牌的**最早** token
    位置，未出现则为 99。
    """
    n, J, L, _ = tl_tokens.shape
    tok = np.asarray(tl_tokens)
    tsum = tok[..., :34].sum(-1)
    tile = np.where(tsum > 0.5, tok[..., :34].argmax(-1), -1)
    is_tg = tok[..., 34] > 0.5
    act = tok[..., 35:42].argmax(-1)
    dahai = np.asarray(tl_mask) & (act == 0) & (tile >= 0)
    pos = np.broadcast_to(np.arange(L, dtype=np.int32), (n, J, L))
    BIG = np.int16(99)
    hc = np.full((n, J, 34), BIG, np.int16)
    tg = np.full((n, J, 34), BIG, np.int16)
    for t in range(34):
        m = tile == t
        hc[..., t] = np.where((dahai & ~is_tg & m).any(-1),
                              np.where(dahai & ~is_tg & m, pos, 99).min(-1),
                              BIG).astype(np.int16)
        tg[..., t] = np.where((dahai & is_tg & m).any(-1),
                              np.where(dahai & is_tg & m, pos, 99).min(-1),
                              BIG).astype(np.int16)
    return hc, tg


# ------------------------------------------------------------- 小工具
def col_ce(P, alloc, eps=EPS):
    """逐列交叉熵：P[n,3,34,5] × alloc[n,3,34] → [n,3,34] nats。"""
    return -np.take_along_axis(np.log(np.clip(P, eps, None)),
                               np.asarray(alloc)[..., None].astype(np.int64), -1)[..., 0]


def suit_window(src, half=2):
    """同色 ±half 窗口求和（跨花色不越界；字牌不参与）。src[n,3,34] → 同形。"""
    tidx = np.arange(34)
    w = np.zeros_like(src)
    for d in range(-half, half + 1):
        s = tidx - d
        ok = (s >= 0) & (s < 27) & (tidx < 27) & (s // 9 == tidx // 9)
        w[..., ok] += src[..., s[ok]]
    return w


def boot_ci(vals, lo=2.5, hi=97.5):
    """返回 (mean, ci_lo, ci_hi, P(>0), P(<0))。"""
    a = np.asarray(vals, float)
    p = np.percentile(a, [lo, hi])
    return float(a.mean()), float(p[0]), float(p[1]), float((a > 0).mean()), float((a < 0).mean())


def fmt_ci(a, lo=2.5, hi=97.5):
    m, l, h, pg, pl = boot_ci(a, lo, hi)
    return f'{m:+.4f}  CI95[{l:+.4f},{h:+.4f}]  P(>0)={pg:.3f}'
