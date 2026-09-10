#!/usr/bin/env python3
"""Compare several compiled K2 mappings on one fixed input.

The extension reads the two ``FLASH_KDA_*`` variables at dispatch time.  This
driver changes them between measurements, so all variants see the same input
and GPU allocation.  The timer deliberately includes the public
``flash_kda.fwd`` call (including workspace allocation and beta transpose),
which is the useful end-to-end latency for this project.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

import flash_kda


def make_case(seq_lens: list[int], heads: int) -> tuple[torch.Tensor, ...]:
    total = sum(seq_lens)
    dim = 128
    device = torch.device("cuda")
    torch.manual_seed(1234)
    q = F.normalize(torch.randn(1, total, heads, dim, device=device), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, total, heads, dim, device=device), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, total, heads, dim, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, total, heads, dim, dtype=torch.bfloat16, device=device)
    beta = torch.randn(1, total, heads, dtype=torch.bfloat16, device=device)
    a_log = torch.rand(heads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(heads, dim, dtype=torch.float32, device=device)
    return q, k, v, g, beta, a_log, dt_bias


def time_call(fn, warmup: int, iters: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        values.append(float(start.elapsed_time(stop)) / iters)
    values.sort()
    return sum(values) / len(values), values[0], values[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--seq-lens", type=int, nargs="+")
    parser.add_argument("--state", choices=("none", "bf16", "fp32", "all"), default="all")
    parser.add_argument("--variants", nargs="+", default=["128:2", "64:1", "32:1"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    seq_lens = args.seq_lens or [args.T]
    if args.seq_lens is not None and args.T != 8192:
        parser.error("use either --T or --seq-lens, not both")
    q, k, v, g, beta, a_log, dt_bias = make_case(seq_lens, args.H)
    n = len(seq_lens)
    cu_seqlens = None
    if n > 1:
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)),
                                  dtype=torch.long, device="cuda")

    out = torch.empty_like(q)
    h0_bf16 = torch.randn(n, args.H, 128, 128, dtype=torch.bfloat16, device="cuda")
    h0_fp32 = h0_bf16.float()
    common = dict(A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
                  cu_seqlens=cu_seqlens)
    states = ("none", "bf16", "fp32") if args.state == "all" else (args.state,)

    for variant in args.variants:
        if variant == "auto":
            os.environ.pop("FLASH_KDA_V_TILE", None)
            os.environ.pop("FLASH_KDA_BLOCKS_PER_WARP", None)
        else:
            try:
                v_tile, blocks_per_warp = (int(x) for x in variant.split(":", 1))
            except ValueError as exc:
                raise SystemExit(
                    f"bad variant {variant!r}; expected auto or V_TILE:BLOCKS_PER_WARP"
                ) from exc
            os.environ["FLASH_KDA_V_TILE"] = str(v_tile)
            os.environ["FLASH_KDA_BLOCKS_PER_WARP"] = str(blocks_per_warp)

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

            mean, minimum, maximum = time_call(fn, args.warmup, args.iters, args.repeats)
            print(f"variant={variant} state={state} H={args.H} seq_lens={seq_lens} "
                  f"mean={mean:.6f} min={minimum:.6f} max={maximum:.6f} ms")


if __name__ == "__main__":
    main()
