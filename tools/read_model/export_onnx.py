"""读牌模型 ONNX 导出（方案 §9 路径 B：ONNX-in-Rust）。

导出的图只包含可微部分（骨干 + H1/H2/H3），硬约束层留在调用方
（Python ``constraints.py`` / 未来的 Rust 侧）做后处理。

ONNX 签名（动态 batch）：
    输入：
        obs        float32 [B, 1012, 34]  观察者 v4 obs
        unseen     float32 [B, 34]        4 - tiles_seen（H1 容量掩码）
        wait_bias  float32 [B, 3, 34]     H2 加性对数偏置（调用方按
                                          constraints.compute_wait_bias 口径预计算：
                                          硬掩码 -inf / 舍牌振听软惩罚 / 0）
    输出：
        alloc_probs  float32 [B, 3, 34, 5]  逐列计数分布（已按 unseen 掩码）
        exp_counts   float32 [B, 3, 34]     期望张数
        wait_probs   float32 [B, 3, 35]     听牌分布（已加 bias；RM1 因子化读法：
                                            p35 = sigmoid(l34) = P(未听牌)，为 BCE
                                            直接标定概率、注入侧可直接使用；
                                            p_x = P(听)·softmax34(l_x)）
        next_probs   float32 [B, 3, 35]     辅助：下一打分布
        shanten_probs float32 [B, 3, 8]     辅助：向听分布
        riichi_prob  float32 [B, 3]         辅助：立直概率
        called_prob  float32 [B, 3]         辅助：鸣牌概率

用法：
    python tools/read_model/export_onnx.py --state runs/read_model_v1/best.pth \
        --out runs/read_model_v1/read_model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/mnt/data8/mortal/Mortal/mortal')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model as M                                       # noqa: E402
import model_v2 as M2                                   # noqa: E402                                     # noqa: E402
from constraints import compute_wait_bias               # noqa: E402

INPUT_NAMES = ['obs', 'unseen', 'wait_bias']
INPUT_NAMES_V2 = ['obs', 'unseen', 'wait_bias', 'tl_tokens', 'tl_mask']
OUTPUT_NAMES = ['alloc_probs', 'exp_counts', 'wait_probs',
                'next_probs', 'shanten_probs', 'riichi_prob', 'called_prob']


class _OnnxWrapper(torch.nn.Module):
    """固定输出顺序与名字的薄封装（dict 输出在 ONNX 里不携带键名）。"""

    def __init__(self, net, v2: bool = False):
        super().__init__()
        self.net = net
        self.v2 = v2

    def forward(self, obs, unseen, wait_bias, tl_tokens=None, tl_mask=None):
        if self.v2:
            out = self.net(obs, unseen, wait_bias, tl_tokens, tl_mask)
        else:
            out = self.net(obs, unseen, wait_bias)
        return (out['alloc_probs'], out['exp_counts'], out['wait_probs'],
                torch.softmax(out['next_logits'], dim=-1),
                torch.softmax(out['shanten_logits'], dim=-1),
                torch.sigmoid(out['riichi_logit']),
                torch.sigmoid(out['called_logit']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    state = torch.load(args.state, map_location='cpu', weights_only=False)
    cfg = state.get('config', {})
    kind = cfg.get('model', 'v1')
    is_v2 = kind in ('v2', 'v2s')
    if is_v2:
        builder = M2.build_model_v2s if kind == 'v2s' else M2.build_model_v2
        net = builder(conv_channels=cfg.get('conv_channels', 96),
                      num_blocks=cfg.get('num_blocks', 12),
                      with_aux=not cfg.get('no_aux', False))
    else:
        net = M.build_model(conv_channels=cfg.get('conv_channels', 96),
                            num_blocks=cfg.get('num_blocks', 12),
                            with_aux=not cfg.get('no_aux', False),
                            pool=cfg.get('pool', 'mean'))
    net.load_state_dict(state['model'])
    net.eval()
    wrapper = _OnnxWrapper(net, v2=is_v2).eval()

    B = 2
    dummy_obs = torch.randn(B, 1012, 34)
    dummy_unseen = torch.full((B, 34), 4.0)
    dummy_unseen[:, 5] = 1.0
    dummy_river = torch.zeros(B, 3, 34, dtype=torch.bool)
    dummy_river[:, 0, 7] = True
    dummy_bias = torch.from_numpy(compute_wait_bias(
        dummy_unseen.numpy(), dummy_river.numpy(),
        np.zeros((B, 3, 34), bool), np.zeros((B, 3), bool),
        own_hand=np.ones((B, 34), bool)))
    inputs = [dummy_obs, dummy_unseen, dummy_bias]
    if is_v2:
        dummy_tokens = torch.zeros(B, 3, 26, 43)
        dummy_tokens[0, 0, :7, 3] = 1.0                # 7 个动作 token
        dummy_mask = torch.zeros(B, 3, 26, dtype=torch.bool)
        dummy_mask[0, 0, :7] = True
        dummy_mask[1, :, :26] = True                   # 满长序列
        inputs += [dummy_tokens, dummy_mask]
    feed = dict(zip(INPUT_NAMES_V2 if is_v2 else INPUT_NAMES, inputs))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 动态轴：所有输入/输出的 batch 维；v2 额外两个输入的 L 维也固定（tl_len）
    dyn = {'obs': {0: 'B'}, 'unseen': {0: 'B'}, 'wait_bias': {0: 'B'},
           **{name: {0: 'B'} for name in OUTPUT_NAMES}}
    if is_v2:
        dyn.update({'tl_tokens': {0: 'B'}, 'tl_mask': {0: 'B'}})
    with torch.no_grad():
        torch.onnx.export(
            wrapper, tuple(inputs), str(out_path),
            input_names=INPUT_NAMES_V2 if is_v2 else INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=args.opset,
            dynamic_axes=dyn,
        )
    print(f'[export] {out_path}', flush=True)

    # ---- 数值校验：PyTorch vs onnxruntime（若可用）----
    try:
        import onnxruntime as ort
    except ImportError:
        print('[verify] onnxruntime 不可用，跳过数值校验（图已导出）', flush=True)
        return
    sess = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        ref = wrapper(*inputs)
    feed_np = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in feed.items()}
    got = sess.run(None, feed_np)
    max_err = 0.0
    for name, r, g in zip(OUTPUT_NAMES, ref, got):
        r_np = r.numpy()
        err = float(np.abs(r_np - g).max())
        max_err = max(max_err, err)
        print(f'  {name:15s} shape {tuple(g.shape)} max|diff| {err:.2e}')
    assert max_err < 2e-5, f'ONNX 数值偏差过大: {max_err}'
    print(f'[verify] PyTorch vs onnxruntime 最大偏差 {max_err:.2e} < 2e-5 通过', flush=True)


if __name__ == '__main__':
    main()
