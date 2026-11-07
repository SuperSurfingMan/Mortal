import torch
import socket
import struct
import time
from typing import *
from io import BytesIO
from functools import partial
from tqdm.auto import tqdm as orig_tqdm
from config import config

tqdm = partial(orig_tqdm, unit='batch', dynamic_ncols=True, ascii=True)

def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def filtered_trimmed_lines(lines):
    return filter(lambda l: l, map(lambda l: l.strip(), lines))

def iter_grads(parameters, take=False):
    for p in parameters:
        if p.grad is not None:
            if take:
                # Set to zero instead of None to preserve the layout and make it
                # easier to assign back later
                yield p.grad.clone()
                p.grad.zero_()
            else:
                yield p.grad

def drain():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    while True:
        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {'type': 'drain'})
            msg = recv_msg(conn)
        if msg['count'] == 0:
            # 2026-11-07：5 s → 1 s。trainer 比 4 个 client 快，一轮 30 步就训完，
            # 原 5 s 轮询让每轮白等 5 s（实测占总墙钟 23%：1.85 步/s vs 可达 2.4）。
            time.sleep(1)
            continue
        return msg['drain_dir']

def submit_param(mortal, dqn, is_idle=False):
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    with socket.socket() as conn:
        conn.connect(remote)
        send_msg(conn, {
            'type': 'submit_param',
            'mortal': mortal.state_dict(),
            'dqn': dqn.state_dict(),
            'is_idle': is_idle,
        })

def send_msg(conn: socket.socket, msg, packed=False):
    if packed:
        tx = msg
    else:
        buf = BytesIO()
        torch.save(msg, buf)
        tx = buf.getbuffer()
    conn.sendall(struct.pack('<Q', len(tx)))
    conn.sendall(tx)

def recv_msg(conn: socket.socket, map_location='cpu'):
    rx = recv_binary(conn, 8)
    (size,) = struct.unpack('<Q', rx)
    # 协议错位诊断（2026-11-07，在线自博弈 smoke）：size 超出合理上限时直接报清楚，
    # 否则会在 bytearray(size) 里抛 MemoryError（堆栈看不到 header 值）。
    if size > 4 * 1024 ** 3:
        try:
            peer = conn.getpeername()
        except OSError:
            peer = '?'
        raise ValueError(
            f'protocol desync: header={rx.hex()} size={size} peer={peer}'
        ) from None
    rx = recv_binary(conn, size)
    return torch.load(BytesIO(rx), weights_only=True, map_location=map_location)

def recv_binary(conn: socket.socket, size):
    assert size > 0
    ret = bytearray(size)
    buf = memoryview(ret)

    while len(buf) > 0:
        n = conn.recv_into(buf)
        if n == 0:
            raise UnexpectedEOF()
        buf = buf[n:]
    return bytes(ret)

class UnexpectedEOF(Exception):
    def __init__(self):
        super().__init__('unexpected EOF')
