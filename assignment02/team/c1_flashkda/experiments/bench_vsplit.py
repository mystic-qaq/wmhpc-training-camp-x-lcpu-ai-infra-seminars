#!/usr/bin/env python3
"""Small, reproducible benchmark for the K2 V-split experiment.

The kernel variant is selected by ``FLASH_KDA_V_TILE``.  This script keeps
the timing scope to FlashKDA itself so that running several reference kernels
in one process cannot affect the result.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

import flash_kda


def benchmark(fn, warmup: int, iters: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        # Keep the same event scope as the official benchmark: launches are
        # submitted back-to-back and one synchronization is used per repeat.
        # Synchronizing after every call can change GPU clock behavior and is
        # not representative of a normal inference loop.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, stop in zip(starts, stops):
            start.record()
            fn()
            stop.record()
        torch.cuda.synchronize()
        samples.extend(float(start.elapsed_time(stop))
                       for start, stop in zip(starts, stops))

    samples.sort()
    return sum(samples) / len(samples), samples[0], samples[-1]


def make_case(seq_lens: list[int], heads: int) -> tuple[torch.Tensor, ...]:
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
    return q, k, v, g, beta, a_log, dt_bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--seq-lens", type=int, nargs="+")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--state", choices=("none", "bf16", "fp32", "all"), default="all")
    args = parser.parse_args()

    seq_lens = args.seq_lens or [args.T]
    if args.seq_lens is not None and args.T != 8192:
        parser.error("use either --T or --seq-lens, not both")
    varlen = len(seq_lens) > 1
    q, k, v, g, beta, a_log, dt_bias = make_case(seq_lens, args.H)
    total = q.shape[1]
    n = len(seq_lens)
    cu_seqlens = None
    if varlen:
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)),
                                  dtype=torch.long, device="cuda")

    out = torch.empty_like(q)
    h0_bf16 = torch.randn(n, args.H, 128, 128, dtype=torch.bfloat16, device="cuda")
    h0_fp32 = h0_bf16.float()
    fwd_common = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
    )

    def run(state_dtype: str) -> None:
        if state_dtype == "none":
            initial = final = None
        elif state_dtype == "bf16":
            initial = h0_bf16
            final = torch.empty_like(h0_bf16)
        else:
            initial = h0_fp32
            final = torch.empty_like(h0_fp32)

        def fn() -> None:
            flash_kda.fwd(q, k, v, g, beta, 1.0 / math.sqrt(128), out,
                          initial_state=initial, final_state=final, **fwd_common)

        mean, minimum, maximum = benchmark(fn, args.warmup, args.iters, args.repeats)
        print(f"V_TILE={os.environ.get('FLASH_KDA_V_TILE', '128')} "
              f"state={state_dtype} H={args.H} seq_lens={seq_lens} "
              f"mean={mean:.5f} min={minimum:.5f} max={maximum:.5f} ms")

    states = ("none", "bf16", "fp32") if args.state == "all" else (args.state,)
    for state in states:
        run(state)


if __name__ == "__main__":
    main()
