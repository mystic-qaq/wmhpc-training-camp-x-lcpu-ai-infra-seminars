#!/usr/bin/env python3
"""Scan the compiled K2 mappings over a small shape matrix.

The script is intentionally independent of the reference implementation.  It
times the public FlashKDA call, changes the dispatch environment between
variants, and prints one compact CSV row per (shape, state, mapping).  This is
useful for checking that a launch heuristic generalizes beyond the one target
shape used during kernel development.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

import flash_kda


def make_inputs(seq_lens: list[int], heads: int):
    total = sum(seq_lens)
    device = torch.device("cuda")
    dim = 128
    torch.manual_seed(1234)
    q = F.normalize(torch.randn(1, total, heads, dim, device=device), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, total, heads, dim, device=device), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, total, heads, dim, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, total, heads, dim, dtype=torch.bfloat16, device=device)
    beta = torch.randn(1, total, heads, dtype=torch.bfloat16, device=device)
    a_log = torch.rand(heads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(heads, dim, dtype=torch.float32, device=device)
    out = torch.empty_like(q)
    n = len(seq_lens)
    cu_seqlens = None
    if n > 1:
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)),
                                  dtype=torch.long, device=device)
    h0_bf16 = torch.randn(n, heads, dim, dim, dtype=torch.bfloat16, device=device)
    return q, k, v, g, beta, a_log, dt_bias, out, cu_seqlens, h0_bf16


def timed(fn, warmup: int, iters: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / iters)
    samples.sort()
    return sum(samples) / len(samples), samples[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixed", "varlen"), default="fixed")
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 4, 8, 12, 16, 32, 64, 96])
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 1024, 8192])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--variants", nargs="+", default=["auto", "128:2", "64:1", "32:1", "16:1"])
    parser.add_argument("--state", choices=("none", "bf16", "fp32", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    states = ("none", "bf16", "fp32") if args.state == "all" else (args.state,)
    if args.mode == "fixed":
        print("H,T,state,variant,mean_ms,min_ms")
        shapes = ((heads, length, [length])
                  for heads in args.heads for length in args.lengths)
    else:
        print("H,N,avg_len,state,variant,mean_ms,min_ms")
        shapes = ((heads, length * batch_size, [length] * batch_size)
                  for heads in args.heads for batch_size in args.batch_sizes
                  for length in args.lengths)

    for heads, total_length, seq_lens in shapes:
            length = seq_lens[0]
            q, k, v, g, beta, a_log, dt_bias, out, cu_seqlens, h0_bf16 = make_inputs(seq_lens, heads)
            h0_fp32 = h0_bf16.float()
            common = dict(A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
                          cu_seqlens=cu_seqlens)
            iters = max(args.iters, 100) if length <= 128 else args.iters
            for variant in args.variants:
                if variant == "auto":
                    os.environ.pop("FLASH_KDA_V_TILE", None)
                    os.environ.pop("FLASH_KDA_BLOCKS_PER_WARP", None)
                else:
                    v_tile, bpw = (int(x) for x in variant.split(":", 1))
                    os.environ["FLASH_KDA_V_TILE"] = str(v_tile)
                    os.environ["FLASH_KDA_BLOCKS_PER_WARP"] = str(bpw)
                for state in states:
                    if state == "none":
                        initial = final = None
                    elif state == "bf16":
                        initial = h0_bf16
                        final = torch.empty_like(h0_bf16)
                    else:
                        initial = h0_fp32
                        final = torch.empty_like(h0_fp32)

                    def fn() -> None:
                        flash_kda.fwd(q, k, v, g, beta, 1.0 / math.sqrt(128), out,
                                      initial_state=initial, final_state=final, **common)

                    mean, minimum = timed(fn, args.warmup, iters, args.repeats)
                    if args.mode == "fixed":
                        print(f"{heads},{length},{state},{variant},{mean:.6f},{minimum:.6f}", flush=True)
                    else:
                        print(f"{heads},{len(seq_lens)},{length},{state},{variant},"
                              f"{mean:.6f},{minimum:.6f}", flush=True)


if __name__ == "__main__":
    main()
