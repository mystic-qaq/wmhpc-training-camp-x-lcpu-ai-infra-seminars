#!/usr/bin/env python3
"""Compare the production kernel with the supplied FLA KDA references.

The official FlashKDA tests use a deliberately kernel-matched reference.  This
script adds an independent check against the assignment's ``fla_kda_ref``
implementations: ``naive_recurrent_kda`` and ``naive_chunk_kda``.  The
production kernel receives raw q/k/g/beta, so the script prepares the
mathematical references with the same normalized BF16 q/k and activated gate
values before calling them.

Run from a CUDA environment in which FlashKDA and flash-linear-attention are
installed, for example::

    python experiments/reference_compare.py --T 16 64 256
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch


# The script lives below the directory containing the copied reference
# package.  Adding that directory explicitly also makes the command work when
# it is launched from FlashKDA/ as shown in the docstring.
TEAM_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(TEAM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEAM_ROOT))

import flash_kda
from fla_kda_ref.naive import naive_chunk_kda, naive_recurrent_kda


LOWER_BOUND = -5.0
H = 1
D = 128


def kernel_match_l2(x: torch.Tensor) -> torch.Tensor:
    """Approximate the production kernel's BF16 input normalization.

    FlashKDA loads BF16 q/k and reduces each 128-element row as 16 groups of
    eight elements followed by a warp XOR tree.  Reproducing that grouping is
    more faithful than normalizing the original FP32 random tensor.
    """

    x_f32 = x.float()
    groups = x_f32.reshape(*x_f32.shape[:-1], 16, 8)
    partials = torch.zeros(
        *x_f32.shape[:-1], 16, dtype=torch.float32, device=x.device
    )
    for i in range(8):
        partials = partials + groups[..., i] * groups[..., i]
    indices = torch.arange(16, device=x.device)
    for offset in (8, 4, 2, 1):
        partials = partials + partials[..., indices ^ offset]
    inv_norm = torch.rsqrt(partials[..., :1] + 1e-6)
    return (x_f32 * inv_norm).to(torch.bfloat16)


def relative_rms(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - reference.float()).abs()
    denom = reference.float().square().mean().sqrt().item()
    rel = diff.square().mean().sqrt().item() / (denom + 1e-12)
    return rel, diff.max().item()


def make_inputs(T: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    shape = (1, T, H, D)
    q = torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    g_raw = torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    beta_raw = torch.randn((1, T, H), device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(H, D, device="cuda", dtype=torch.float32)
    h0 = (0.1 * torch.randn(1, H, D, D, device="cuda")).to(torch.bfloat16)
    return q, k, v, g_raw, beta_raw, A_log, dt_bias, h0


@torch.inference_mode()
def compare_one(T: int, seed: int) -> None:
    q, k, v, g_raw, beta_raw, A_log, dt_bias, h0 = make_inputs(T, seed)
    scale = 1.0 / math.sqrt(D)

    # These are the inputs expected by the assignment references.  The
    # production path computes the same transformations internally, but its
    # sigmoid/tensor-core/state arithmetic is intentionally lower precision.
    q_ref = kernel_match_l2(q)
    k_ref = kernel_match_l2(k)
    g_ref = LOWER_BOUND * torch.sigmoid(
        torch.exp(A_log).view(1, 1, H, 1) * (g_raw.float() + dt_bias.view(1, 1, H, D))
    )
    beta_ref = torch.sigmoid(beta_raw.float())

    out = torch.empty_like(v)
    final_state = torch.empty_like(h0)
    flash_kda.fwd(
        q,
        k,
        v,
        g_raw,
        beta_raw,
        scale,
        out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=h0.clone(),
        final_state=final_state,
    )
    torch.cuda.synchronize()

    # FlashKDA's public [D, D] state is physically V-first.  The simple FLA
    # references represent their recurrent matrix as [K, V], so transpose the
    # initial state on entry and transpose the returned matrix back for the
    # comparison.  (The square shape makes an omitted transpose particularly
    # easy to miss.)
    h0_kv = h0.transpose(-1, -2).contiguous()
    recurrent_out, recurrent_state_kv = naive_recurrent_kda(
        q_ref,
        k_ref,
        v,
        g_ref,
        beta_ref,
        scale=scale,
        initial_state=h0_kv,
        output_final_state=True,
    )
    chunk_out, chunk_state_kv = naive_chunk_kda(
        q_ref,
        k_ref,
        v,
        g_ref,
        beta_ref,
        scale=scale,
        initial_state=h0_kv,
        output_final_state=True,
        chunk_size=16,
    )
    torch.cuda.synchronize()

    recurrent_state = recurrent_state_kv.transpose(-1, -2)
    chunk_state = chunk_state_kv.transpose(-1, -2)

    rec_out_rel, rec_out_max = relative_rms(out, recurrent_out)
    rec_state_rel, rec_state_max = relative_rms(final_state, recurrent_state)
    chunk_out_rel, chunk_out_max = relative_rms(out, chunk_out)
    chunk_state_rel, chunk_state_max = relative_rms(final_state, chunk_state)
    print(
        f"T={T:4d} "
        f"naive_recurrent output_rel_rms={rec_out_rel:.6e} "
        f"state_rel_rms={rec_state_rel:.6e} "
        f"output_max={rec_out_max:.6e} state_max={rec_state_max:.6e}"
    )
    print(
        f"       naive_chunk16   output_rel_rms={chunk_out_rel:.6e} "
        f"state_rel_rms={chunk_state_rel:.6e} "
        f"output_max={chunk_out_max:.6e} state_max={chunk_state_max:.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[16, 64, 256])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    for index, T in enumerate(args.T):
        if T < 1 or T % 16:
            raise SystemExit(f"T must be a positive multiple of 16, got {T}")
        compare_one(T, seed=20260827 + index)


if __name__ == "__main__":
    main()
