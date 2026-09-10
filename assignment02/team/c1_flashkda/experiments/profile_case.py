#!/usr/bin/env python3
"""Issue one FlashKDA invocation for Nsight Compute."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import flash_kda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--state", choices=("none", "bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    device = torch.device("cuda")
    dim = 128
    torch.manual_seed(1234)
    q = F.normalize(torch.randn(1, args.T, args.H, dim, device=device), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, args.T, args.H, dim, device=device), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, args.T, args.H, dim, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, args.T, args.H, dim, dtype=torch.bfloat16, device=device)
    beta = torch.randn(1, args.T, args.H, dtype=torch.bfloat16, device=device)
    a_log = torch.rand(args.H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(args.H, dim, dtype=torch.float32, device=device)
    out = torch.empty_like(q)

    if args.state == "none":
        initial = final = None
    elif args.state == "bf16":
        initial = torch.randn(1, args.H, dim, dim, dtype=torch.bfloat16, device=device)
        final = torch.empty_like(initial)
    else:
        initial = torch.randn(1, args.H, dim, dim, dtype=torch.float32, device=device)
        final = torch.empty_like(initial)

    flash_kda.fwd(q, k, v, g, beta, 1.0 / math.sqrt(dim), out,
                  A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
                  initial_state=initial, final_state=final)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
