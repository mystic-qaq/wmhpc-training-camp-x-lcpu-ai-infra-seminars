# FlashKDA

FlashKDA: Flash Kimi Delta Attention — high-performance KDA kernels built on CUTLASS

## News

- **2026-04-22** — Deep-Dive Blog: the design decisions behind FlashKDA v1, read it [here](docs/20260420-flashkda-v1-deep-dive.md).

## Requirements
- SM90 and above
- CUDA 12.9 and above
- PyTorch 2.4 and above

## Installation
```bash
git clone https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
git submodule update --init --recursive
pip install -v --no-build-isolation .
```

For the assignment snapshot, CUTLASS is checked out next to the snapshot
rather than inside it.  Point the build at that checkout explicitly:

```bash
export FLASH_KDA_CUTLASS_DIR=/path/to/assignment02/FlashKDA/cutlass
pip install -v --no-build-isolation .
```

By default, the build detects the current CUDA device and compiles for that architecture. For wheel or CI builds, compile all supported architectures explicitly:

```bash
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .
```

Supported values are `auto` (default), `all`, or a comma-separated arch list such as `90a,100a`.

## Using FlashKDA as an FLA backend

Once installed, FlashKDA is auto-dispatched from `flash-linear-attention`'s `chunk_kda`. See [fla-org/flash-linear-attention#852](https://github.com/fla-org/flash-linear-attention/pull/852) for integration details.

**Requirements**

1. Install `flash-linear-attention >= 0.5.0`:
   ```bash
   pip install -U flash-linear-attention
   ```
2. Call `chunk_kda` under `torch.inference_mode()`
   ```python
   import torch
   from fla.ops.kda import chunk_kda

   with torch.inference_mode():
       out, final_state = chunk_kda(
           q=q, k=k, v=v, g=g, beta=beta,
           scale=scale,
           initial_state=h0,
           output_final_state=True,
           use_gate_in_kernel=True,
           use_qk_l2norm_in_kernel=True,
           use_beta_sigmoid_in_kernel=True,
           safe_gate=True,
           A_log=A_log, dt_bias=dt_bias,
           lower_bound=lower_bound,
           transpose_state_layout=True,
           cu_seqlens=cu_seqlens,
       )
   ```

**Opt out:** set `FLA_FLASH_KDA=0` to fall back to the Triton path.

On SM10.x (Blackwell), the recurrence uses a conservative shape heuristic to
split the independent value rows when the number of heads is small.  The
heuristic is disabled on other architectures.  Its current policy is:

| shape | automatic mapping |
|---|---|
| one sequence, `H<=16` | `V_TILE=16`, one block per warp |
| one sequence, `17<=H<=32`, `T<=64` | `V_TILE=16`, one block per warp |
| one sequence, `17<=H<=32`, `T>64` | `V_TILE=32`, one block per warp |
| one sequence, otherwise `H<=64` | `V_TILE=64`, one block per warp |
| one sequence, `H=65..96`, BF16/no state | `V_TILE=64`, one block per warp |
| one sequence, FP32 state outside the cases above | original `V_TILE=128` mapping |
| packed/varlen, `H<=4` | `V_TILE=16` for small `N`/`H`, otherwise `V_TILE=32` |
| packed/varlen, `H=5..16`, `N<=2` | `V_TILE=32`, one block per warp |
| packed/varlen, `H=5..16`, long FP32 or large-N case | original `V_TILE=128` mapping |
| other packed/varlen, measured small-chain case | `V_TILE=64`, one block per warp |
| other packed/varlen | original `V_TILE=128` mapping |

This policy is deliberately B300-measured rather than universally optimal; the
original mapping remains available as a fallback.  For experiments or fixed
deployments, the mapping can be overridden at dispatch time:

```bash
# Recommended mapping for the B300 TP8/K3 shape (H=12, long fixed sequence)
export FLASH_KDA_V_TILE=16
export FLASH_KDA_BLOCKS_PER_WARP=1

# Force the original one-CTA-per-head mapping
export FLASH_KDA_V_TILE=128
export FLASH_KDA_BLOCKS_PER_WARP=2
```

The supported `FLASH_KDA_V_TILE` values are `128`, `64`, `32`, and `16`.  For a
split tile, one block per warp is the default; `FLASH_KDA_BLOCKS_PER_WARP=2`
selects the original denser warp mapping.  The variables are optional: when
unset, the SM10.x heuristic chooses a mapping from `H`, sequence packing, and
state dtype.  Builds enable the B-layout publication optimization by default;
set `FLASH_KDA_NVCC_DEFINES=` only for an A/B build without it.  The automatic
state-update warp count is 8 for `V_TILE=16/64`, 4 for `V_TILE=32`, and 0 for
the original `V_TILE=128` mapping.  It can be overridden for experiments with
`FLASH_KDA_STATE_UPDATE_WARPS=0`, `4`, `8`, `16`, or `auto`.  The legacy
`FLASH_KDA_PARALLEL_STATE_UPDATE=0/1` variable remains an alias for disabling
or enabling the parallel path.

**Debug dispatch:** add `logging.basicConfig(level=logging.INFO)` to see `[FLA Backend] kda.chunk_kda -> flashkda` on hit, or `... rejected: <reason>` on miss.

## Performance

See [BENCHMARK_H20.md](BENCHMARK_H20.md).

The B300/SM103 optimization measurements, including mapping tables and the
correctness/NCU/SASS evidence, are recorded in
[`../results/final_optimization_b300.md`](../results/final_optimization_b300.md).

## Tests

```bash
bash tests/test.sh
```

- `tests/test_fwd.py` — correctness tests (exact match against the torch reference; compared with `flash-linear-attention`)
- The assignment validation matrix is `tests/test_fwd_full.py`; the final
  SM103 build passed all 580 cases.


## Kernel API

### `flash_kda.fwd`

```python
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None)
```

**Parameters:**

| Parameter | Dtype | Shape | Description |
|---|---|---|---|
| `q` | bf16 | `[B, T, H, K]` | Query |
| `k` | bf16 | `[B, T, H, K]` | Key |
| `v` | bf16 | `[B, T, H, V]` | Value |
| `g` | bf16 | `[B, T, H, K]` | Gate before activation |
| `beta` | bf16 | `[B, T, H]` | Beta logits (pre-activation; sigmoid applied internally) |
| `scale` | float | scalar | scaling factor |
| `out` | bf16 | `[B, T, H, V]` | Output tensor |
| `A_log` | fp32 | `[H]` | Log-gate parameter |
| `dt_bias` | fp32 | `[H, K]` | Gate bias |
| `lower_bound` | float | scalar | Gate lower bound (range from -5.0 to 0) |
| `initial_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional) Initial recurrent state |
| `final_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional, output) Final recurrent state |
| `cu_seqlens` | int64 | `[N+1]` | (optional) Cumulative sequence lengths for variable-length batching |

- Currently requires `K = V = 128`.
- `initial_state` / `final_state` accept `None` (stateless), bf16, or fp32 tensors. When both are provided, their dtypes must match.
- When `cu_seqlens` is provided, `B` must be 1, `T` is the total length across all sequences, and `initial_state` / `final_state` have shape `[N, H, V, K]`.
- When `cu_seqlens` is `None`, each batch element is treated as an independent sequence, and the state shape is `[B, H, V, K]`.

## Development

To set up IntelliSense (clangd) for the CUDA/C++ sources, run:

```bash
bash setup_clangd.sh
```

This generates a `.clangd` file with the correct repository paths and installs the global clangd `config.yaml` to `~/.config/clangd/`.

## Citation

```bibtex
@misc{flashkda2026,
      title={FlashKDA: Flash Kimi Delta Attention},
      author={Yutian Chen, Zhiyuan Li, Yucheng Wang, Ming Wei},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/FlashKDA}},
}
```
