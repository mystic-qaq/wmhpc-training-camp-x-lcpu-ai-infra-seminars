#!/usr/bin/env python3
"""Compare FlashKDA's BF16 state path with an FP64 recurrent reference.

The reference is FLA's fused recurrent implementation with all gate
activation done before the call.  It is intentionally independent of the
FlashKDA chunked torch reference: the latter also models BF16 state storage.
Run this on a CUDA device after installing FlashKDA and flash-linear-attention.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import flash_kda
from fla.ops.kda import fused_recurrent_kda


LOWER_BOUND = -5.0


def error_stats(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (actual.float() - reference.float()).abs()
    ref_rms = reference.float().square().mean().sqrt().item()
    rms = diff.square().mean().sqrt().item()
    return rms / (ref_rms + 1e-12), diff.max().item(), diff.mean().item(), ref_rms


def make_case(T: int, gate_case: str, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    H, D = 1, 128
    q = F.normalize(torch.randn(1, T, H, D, device="cuda"), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, T, H, D, device="cuda"), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(1, T, H, dtype=torch.bfloat16, device="cuda")
    if gate_case == "near_lower":
        # sigmoid(exp(A)*(g+bias)) is close to 1, so the activated gate is
        # close to lower_bound and the accumulated exponent decays fastest.
        g = torch.full((1, T, H, D), 8.0, dtype=torch.bfloat16, device="cuda")
    elif gate_case == "near_zero":
        # The opposite stress: very little decay, so rounding can accumulate
        # over many chunks instead of being hidden by a rapidly vanishing state.
        g = torch.full((1, T, H, D), -8.0, dtype=torch.bfloat16, device="cuda")
    elif gate_case == "random":
        g = torch.randn(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    else:
        raise ValueError(f"unknown gate case: {gate_case}")
    A_log = torch.zeros(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(H, D, dtype=torch.float32, device="cuda")
    h0 = (0.1 * torch.randn(1, H, D, D, device="cuda")).to(torch.bfloat16)
    return q, k, v, g, beta, A_log, dt_bias, h0


@torch.inference_mode()
def run_case(T: int, gate_case: str, seed: int) -> None:
    q, k, v, g, beta, A_log, dt_bias, h0_bf16 = make_case(T, gate_case, seed)
    scale = 1.0 / math.sqrt(q.shape[-1])

    # Independent high-precision path.  All inputs are the same BF16 values
    # that FlashKDA sees; only the recurrent state and arithmetic are wider.
    g64 = g.double() + dt_bias.double().view(1, 1, 1, -1)
    g64 = LOWER_BOUND * torch.sigmoid(torch.exp(A_log.double()).view(1, 1, 1, 1) * g64)
    beta64 = torch.sigmoid(beta.double())
    gold_out, gold_state = fused_recurrent_kda(
        q=q.double(), k=k.double(), v=v.double(), g=g64, beta=beta64,
        A_log=None, dt_bias=None, scale=scale,
        initial_state=h0_bf16.double(), output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
        lower_bound=None, transpose_state_layout=True,
    )

    out = torch.empty_like(q)
    final_bf16 = torch.empty_like(h0_bf16)
    flash_kda.fwd(
        q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias,
        lower_bound=LOWER_BOUND, initial_state=h0_bf16.clone(),
        final_state=final_bf16,
    )
    torch.cuda.synchronize()

    o_rel, o_max, o_mean, o_ref_rms = error_stats(gold_out, out)
    s_rel, s_max, s_mean, s_ref_rms = error_stats(gold_state, final_bf16)
    print(
        f"case={gate_case:>10s} T={T:5d} "
        f"output_rel_rms={o_rel:.6e} output_max={o_max:.6e} "
        f"state_rel_rms={s_rel:.6e} state_max={s_max:.6e}"
    )
    print(
        f"  reference_rms(output,state)=({o_ref_rms:.6e},{s_ref_rms:.6e}) "
        f"mean_abs(output,state)=({o_mean:.6e},{s_mean:.6e})"
    )

    # The external FP32 state API should not change the result when the
    # initial values are exactly representable in BF16: the production kernel
    # converts both forms to the same on-chip BF16 state.
    if T == 1024:
        out32 = torch.empty_like(q)
        final_fp32 = torch.empty(1, 1, 128, 128, dtype=torch.float32, device="cuda")
        flash_kda.fwd(
            q, k, v, g, beta, scale, out32, A_log=A_log, dt_bias=dt_bias,
            lower_bound=LOWER_BOUND, initial_state=h0_bf16.float(),
            final_state=final_fp32,
        )
        torch.cuda.synchronize()
        print(
            "  BF16-vs-FP32-IO: output_exact=%s state_exact=%s state_max_abs=%.6e"
            % (
                bool(torch.equal(out, out32)),
                bool(torch.equal(final_bf16.float(), final_fp32)),
                (final_bf16.float() - final_fp32).abs().max().item(),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[256, 1024, 8192])
    parser.add_argument("--cases", nargs="+", default=["random", "near_lower", "near_zero"])
    args = parser.parse_args()
    for case_index, gate_case in enumerate(args.cases):
        for t_index, T in enumerate(args.T):
            run_case(T, gate_case, seed=1234 + case_index * 100 + t_index)


if __name__ == "__main__":
    main()
