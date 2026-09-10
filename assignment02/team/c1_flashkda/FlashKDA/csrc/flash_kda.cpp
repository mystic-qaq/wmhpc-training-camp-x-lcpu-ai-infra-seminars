#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cstring>
#include "fwd.h"

int64_t get_workspace_size(
    int64_t T_total,
    int64_t H,
    int64_t N = 1
) {
    constexpr int CHUNK = 16;
    constexpr int D = 128;

    // Upper bound: each of N sequences adds at most 1 extra tile vs floor division
    int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;

    static_assert(CHUNK * D * 2 % 128 == 0, "k_decayed/q_decayed/k_restored size must be 128-byte aligned");
    static_assert(D * 4 % 128 == 0, "g_total size must be 128-byte aligned");
    static_assert(CHUNK * CHUNK * 2 % 128 == 0, "INV/Mqk size must be 128-byte aligned");

    int64_t per_tile_bytes = 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2;

    // Trailing buffer for the tile prefix-sum (N+1 int32), 128-byte aligned.
    int64_t tile_prefix_bytes = ((N + 1) * 4 + 127) / 128 * 128;

    return H * total_tiles * per_tile_bytes + tile_prefix_bytes;
}

void fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    float scale,
    torch::Tensor out,
    torch::Tensor workspace,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    double lower_bound,
    std::optional<torch::Tensor> initial_state = std::nullopt,
    std::optional<torch::Tensor> final_state = std::nullopt,
    std::optional<torch::Tensor> cu_seqlens = std::nullopt
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda() && out.is_cuda() && workspace.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() && beta.is_contiguous() && out.is_contiguous() && workspace.is_contiguous(),
                "all tensors must be contiguous");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(g.dtype() == torch::kBFloat16, "g must be bfloat16");
    TORCH_CHECK(beta.dtype() == torch::kBFloat16, "beta must be bfloat16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");

    // Validate state tensors if present
    bool has_state_in = initial_state.has_value();
    bool has_state_out = final_state.has_value();
    bool state_fp32 = false;

    if (has_state_in) {
        auto& is = initial_state.value();
        TORCH_CHECK(is.is_cuda() && is.is_contiguous(), "initial_state must be contiguous CUDA tensor");
        TORCH_CHECK(is.dtype() == torch::kBFloat16 || is.dtype() == torch::kFloat32,
                     "initial_state must be bfloat16 or float32");
        if (is.dtype() == torch::kFloat32) state_fp32 = true;
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        TORCH_CHECK(fs.is_cuda() && fs.is_contiguous(), "final_state must be contiguous CUDA tensor");
        TORCH_CHECK(fs.dtype() == torch::kBFloat16 || fs.dtype() == torch::kFloat32,
                     "final_state must be bfloat16 or float32");
        if (fs.dtype() == torch::kFloat32) state_fp32 = true;
    }
    // If both present, dtypes must match
    if (has_state_in && has_state_out) {
        TORCH_CHECK(initial_state->dtype() == final_state->dtype(),
                     "initial_state and final_state must have the same dtype");
    }

    TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous(), "A_log must be contiguous CUDA tensor");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.is_contiguous(), "dt_bias must be contiguous CUDA tensor");
    TORCH_CHECK(dt_bias.dtype() == torch::kFloat32, "dt_bias must be float32");

    // Accept 4D input [B, T, H, D]
    TORCH_CHECK(q.dim() == 4, "q must be [B, T, H, D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, T, H, D]");
    TORCH_CHECK(g.dim() == 4, "g must be [B, T, H, D]");
    TORCH_CHECK(beta.dim() == 3, "beta must be [B, T, H]");
    TORCH_CHECK(out.dim() == 4, "out must be [B, T, H, D]");

    int64_t B = q.size(0);
    int64_t T_seq = q.size(1);
    int64_t H = q.size(2);
    int64_t D = q.size(3);
    int64_t T_total = B * T_seq;

    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
    TORCH_CHECK(g.sizes() == q.sizes(), "g must match q shape");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(beta.size(0) == B && beta.size(1) == T_seq && beta.size(2) == H,
                "beta must be [B, T, H] matching q");

    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == D, "dt_bias must be [H, D]");

    TORCH_CHECK(D == 128, "currently only supports D == 128");

    // Flatten [B, T, H, D] -> [B*T, H, D] (contiguous, same data pointer)
    auto q_3d = q.reshape({T_total, H, D});
    auto k_3d = k.reshape({T_total, H, D});
    auto v_3d = v.reshape({T_total, H, D});
    auto g_3d = g.reshape({T_total, H, D});
    auto beta_2d = beta.reshape({T_total, H});
    auto out_3d = out.reshape({T_total, H, D});

    auto q_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(q_3d.data_ptr<at::BFloat16>());
    auto k_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(k_3d.data_ptr<at::BFloat16>());
    auto v_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(v_3d.data_ptr<at::BFloat16>());
    auto g_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(g_3d.data_ptr<at::BFloat16>());
    // Kernel 1/2 consume beta in the [H, T_total] layout.  Keeping this
    // transpose also makes beta a contiguous 1-D TMA source; the attempted
    // direct [T_total, H] loads were substantially slower on B300.
    auto beta_t = beta_2d.t().contiguous();
    auto beta_t_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(beta_t.data_ptr<at::BFloat16>());
    float scale_f = scale;
    auto out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out_3d.data_ptr<at::BFloat16>());
    auto A_log_ptr = A_log.data_ptr<float>();
    auto dt_bias_ptr = dt_bias.data_ptr<float>();
    float gate_scale = float(lower_bound * 1.4426950408889634);

    auto workspace_ptr = workspace.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int CHUNK = 16;

    // Get state pointers (nullptr if not present)
    void const* initial_state_raw = has_state_in ? initial_state->data_ptr() : nullptr;
    void* final_state_raw = has_state_out ? final_state->data_ptr() : nullptr;

    // Determine cu_seqlens and N
    bool is_varlen = cu_seqlens.has_value();
    int64_t N_val;
    int64_t const* cu_seqlens_dev = nullptr;

    if (is_varlen) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        auto& cu_seqlens_t = cu_seqlens.value();
        TORCH_CHECK(cu_seqlens_t.is_cuda(), "cu_seqlens must be on CUDA");
        TORCH_CHECK(cu_seqlens_t.dtype() == torch::kLong, "cu_seqlens must be int64");
        TORCH_CHECK(cu_seqlens_t.dim() == 1, "cu_seqlens must be 1D");
        N_val = cu_seqlens_t.numel() - 1;
        TORCH_CHECK(N_val > 0, "cu_seqlens must have at least 2 elements");
        cu_seqlens_dev = cu_seqlens_t.data_ptr<int64_t>();
    } else {
        N_val = B;
    }

    // Validate state shapes: always [N, H, D, D]
    if (has_state_in) {
        auto& is = initial_state.value();
        TORCH_CHECK(is.dim() == 4, "initial_state must be [N, H, D, D]");
        TORCH_CHECK(is.size(0) == N_val && is.size(1) == H && is.size(2) == D && is.size(3) == D,
                     "initial_state must be [N, H, D, D]");
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        TORCH_CHECK(fs.dim() == 4, "final_state must be [N, H, D, D]");
        TORCH_CHECK(fs.size(0) == N_val && fs.size(1) == H && fs.size(2) == D && fs.size(3) == D,
                     "final_state must be [N, H, D, D]");
    }

    int total_tiles;
    if (is_varlen) {
        total_tiles = int((T_total + CHUNK - 1) / CHUNK + N_val);  // upper bound for varlen
    } else {
        total_tiles = int(N_val * ((T_seq + CHUNK - 1) / CHUNK));   // exact for batched
    }

    // Dispatch based on state configuration, varlen, V-split mapping, and
    // the optional parallel state-update implementation.
    // On Blackwell, the small-head-count cases benefit from exposing the
    // independent value rows as multiple CTAs.  The heuristic is deliberately
    // conservative: it is enabled only for SM10.x and only for shapes that
    // were measured to improve on B300.  An explicit V_TILE always wins, so
    // FLASH_KDA_V_TILE=128 is also the compatibility/performance escape hatch.
    int v_tile = -1;
    bool v_tile_override = false;
    if (const char* v_tile_env = std::getenv("FLASH_KDA_V_TILE")) {
        v_tile = std::atoi(v_tile_env);
        v_tile_override = true;
    }

    cudaDeviceProp device_prop{};
    bool is_blackwell =
        cudaGetDeviceProperties(&device_prop, q.get_device()) == cudaSuccess &&
        device_prop.major == 10;

    // ``avg_len`` is useful in packed mode because the same number of
    // sequence/head CTAs can represent either a short-request workload or a
    // long-context workload.  The mapping choice is a launch-time policy;
    // it does not change the public packed layout.
    int64_t avg_len = (T_total + N_val - 1) / N_val;

    if (!v_tile_override && is_blackwell) {
        if (N_val == 1) {
            // The smallest legal value tile is the best B300 mapping for a
            // single sequence with at most 16 heads.  This remains true from
            // one short tile through long-context runs: the extra CTA-level
            // parallelism and the smaller state/TMA slice outweigh the
            // repeated full-D workspace loads.  The policy is independent of
            // state I/O dtype; all three state modes were scanned.
            if (H <= 16) {
                v_tile = 16;
            } else if (H <= 32 && T_total <= 64) {
                v_tile = 16;
            } else if (H <= 32) {
                v_tile = 32;
            } else if (H <= 64) {
                v_tile = 64;
            } else if (H <= 96 && !state_fp32) {
                v_tile = 64;
            } else {
                v_tile = 128;
            }
        } else {
            // Packed mode has a second useful axis: N changes the number of
            // independent state chains.  V=32 is worthwhile for one or two
            // chains, V=64 is the middle ground for a small/medium batch,
            // and V=128 avoids repeating the full-D workspace traffic once
            // there are already many heads in flight.  FP32 state conversion
            // moves the long-context crossover to V=128 earlier.
            if (H <= 4) {
                // V16 is still best for the common small packed batches
                // (N<=4).  At N>=8, H=3/4 has enough independent chains
                // that V32 avoids the extra CTA count; H<=2 remains small
                // enough for V16 to win in the measured N=8 case.
                v_tile = (N_val <= 4 || H <= 2) ? 16 : 32;
            } else if (H <= 16) {
                if (N_val <= 2) {
                    v_tile = 32;
                } else if (state_fp32 && N_val >= 8 && avg_len > 256) {
                    v_tile = 128;
                } else if (N_val >= 16 && avg_len > 256) {
                    v_tile = 128;
                } else {
                    v_tile = 64;
                }
            } else {
                // For larger head counts, only a very small total number of
                // chains benefits from duplicating q/k/workspace traffic.
                // Keep FP32 state conservative except for the measured
                // H<=32, N<=2 corner.
                int64_t chain_heads = N_val * H;
                if ((!state_fp32 && chain_heads <= 128) ||
                    (state_fp32 && H <= 32 && N_val <= 2)) {
                    v_tile = 64;
                } else {
                    v_tile = 128;
                }
            }
        }
    } else if (!v_tile_override) {
        v_tile = 128;
    }
    TORCH_CHECK(v_tile == 128 || v_tile == 64 || v_tile == 32 || v_tile == 16,
                "FLASH_KDA_V_TILE must be 128, 64, 32, or 16");

    // Parallel state update is a separate compiled kernel variant because it
    // changes the CTA's warp layout.  The default policy is intentionally
    // narrow: on B300 it is useful when V-split leaves only one or two MMA
    // warps per CTA, but additional update warps hurt dense V64/V128 mappings.
    // The new STATE_UPDATE_WARPS knob allows an occupancy-vs-ILP experiment.
    // With persistent state fragments, V32/BPW1 is fastest with four update
    // warps for both state modes; the FP32 variant uses its lower-register
    // C-layout U publication path.  V16 still needs eight because each update
    // warp owns one row tile.  The
    // same eight-warp update path is useful for V64/BPW1: its four independent
    // state fragments per warp are enough to amortize the extra warp group,
    // and this is the useful mapping for medium/large head counts.  V128
    // retains the original single-warp-per-output mapping.  The old
    // PARALLEL_STATE_UPDATE=0/1 switch remains an alias for forcing 0/8.
    int state_update_warps =
        is_blackwell && v_tile <= 64 ?
            (v_tile == 32 ? 4 : 8) : 0;
    bool state_update_warps_override = false;
    if (const char* update_warps_env = std::getenv("FLASH_KDA_STATE_UPDATE_WARPS")) {
        if (std::strcmp(update_warps_env, "auto") != 0 &&
            std::strcmp(update_warps_env, "AUTO") != 0) {
            state_update_warps = std::atoi(update_warps_env);
            state_update_warps_override = true;
        }
    }
    TORCH_CHECK(state_update_warps == 0 || state_update_warps == 4 ||
                    state_update_warps == 8 || state_update_warps == 16,
                "FLASH_KDA_STATE_UPDATE_WARPS must be auto, 0, 4, 8, or 16");

    if (const char* parallel_env = std::getenv("FLASH_KDA_PARALLEL_STATE_UPDATE")) {
        if (std::strcmp(parallel_env, "auto") != 0 &&
            std::strcmp(parallel_env, "AUTO") != 0) {
            int parallel_value = std::atoi(parallel_env);
            TORCH_CHECK(parallel_value == 0 || parallel_value == 1,
                        "FLASH_KDA_PARALLEL_STATE_UPDATE must be auto, 0, or 1");
            if (!state_update_warps_override) {
                state_update_warps = parallel_value != 0 ? 8 : 0;
            }
        }
    }

    int blocks_per_warp = (v_tile == 128) ? 2 : 1;
    if (const char* blocks_env = std::getenv("FLASH_KDA_BLOCKS_PER_WARP")) {
        blocks_per_warp = std::atoi(blocks_env);
    }
    TORCH_CHECK(blocks_per_warp == 1 || blocks_per_warp == 2,
                "FLASH_KDA_BLOCKS_PER_WARP must be 1 or 2");
    #define LAUNCH(VT, BPW, HI, HO, FP32, VL, UPDATE_WARPS) \
        launch_fwd<128, HI, HO, FP32, VL, VT, BPW, UPDATE_WARPS>( \
            q_ptr, k_ptr, v_ptr, g_ptr, beta_t_ptr, \
            initial_state_raw, scale_f, final_state_raw, out_ptr, \
            workspace_ptr, total_tiles, \
            int(T_total), int(H), int(N_val), cu_seqlens_dev, \
            A_log_ptr, dt_bias_ptr, gate_scale, stream)

    #define DISPATCH_STATE(VT, BPW, VL, UPDATE_WARPS) \
        if (!has_state_in && !has_state_out) { \
            LAUNCH(VT, BPW, false, false, false, VL, UPDATE_WARPS); \
        } else if (has_state_in && has_state_out && state_fp32) { \
            LAUNCH(VT, BPW, true, true, true, VL, UPDATE_WARPS); \
        } else if (has_state_in && has_state_out && !state_fp32) { \
            LAUNCH(VT, BPW, true, true, false, VL, UPDATE_WARPS); \
        } else if (!has_state_in && has_state_out && state_fp32) { \
            LAUNCH(VT, BPW, false, true, true, VL, UPDATE_WARPS); \
        } else if (!has_state_in && has_state_out && !state_fp32) { \
            LAUNCH(VT, BPW, false, true, false, VL, UPDATE_WARPS); \
        } else if (has_state_in && !has_state_out && state_fp32) { \
            LAUNCH(VT, BPW, true, false, true, VL, UPDATE_WARPS); \
        } else { \
            LAUNCH(VT, BPW, true, false, false, VL, UPDATE_WARPS); \
        }

    #define DISPATCH_MAPPING(UPDATE_WARPS) \
        if (blocks_per_warp == 1) { \
            if (v_tile == 128) { \
                if (is_varlen) { DISPATCH_STATE(128, 1, true, UPDATE_WARPS); } \
                else { DISPATCH_STATE(128, 1, false, UPDATE_WARPS); } \
            } else if (v_tile == 64) { \
                if (is_varlen) { DISPATCH_STATE(64, 1, true, UPDATE_WARPS); } \
                else { DISPATCH_STATE(64, 1, false, UPDATE_WARPS); } \
            } else if (v_tile == 32) { \
                if (is_varlen) { DISPATCH_STATE(32, 1, true, UPDATE_WARPS); } \
                else { DISPATCH_STATE(32, 1, false, UPDATE_WARPS); } \
            } else { \
                TORCH_CHECK(v_tile == 16, \
                            "one block per warp is only instantiated for V_TILE=128, 64, 32, or 16"); \
                if (is_varlen) { DISPATCH_STATE(16, 1, true, UPDATE_WARPS); } \
                else { DISPATCH_STATE(16, 1, false, UPDATE_WARPS); } \
            } \
        } else if (v_tile == 64) { \
            if (is_varlen) { DISPATCH_STATE(64, 2, true, UPDATE_WARPS); } \
            else { DISPATCH_STATE(64, 2, false, UPDATE_WARPS); } \
        } else if (v_tile == 32) { \
            if (is_varlen) { DISPATCH_STATE(32, 2, true, UPDATE_WARPS); } \
            else { DISPATCH_STATE(32, 2, false, UPDATE_WARPS); } \
        } else { \
            TORCH_CHECK(v_tile == 128, \
                        "two blocks per warp are only instantiated for V_TILE=128, 64, or 32"); \
            if (is_varlen) { DISPATCH_STATE(128, 2, true, UPDATE_WARPS); } \
            else { DISPATCH_STATE(128, 2, false, UPDATE_WARPS); } \
        }

    if (state_update_warps == 16) {
        TORCH_CHECK(blocks_per_warp == 1 && (v_tile == 32 || v_tile == 64),
                    "16 state-update warps require V_TILE=32 or 64 and one block per warp");
        if (v_tile == 32) {
            if (is_varlen) { DISPATCH_STATE(32, 1, true, 16); }
            else { DISPATCH_STATE(32, 1, false, 16); }
        } else {
            TORCH_CHECK(v_tile == 64,
                        "16 state-update warps require V_TILE=32 or 64");
            if (is_varlen) { DISPATCH_STATE(64, 1, true, 16); }
            else { DISPATCH_STATE(64, 1, false, 16); }
        }
    } else if (state_update_warps == 8) {
        DISPATCH_MAPPING(8);
    } else if (state_update_warps == 4) {
        DISPATCH_MAPPING(4);
    } else {
        DISPATCH_MAPPING(0);
    }

    #undef DISPATCH_MAPPING
    #undef DISPATCH_STATE
    #undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "FlashKDA Forward (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
        py::arg("scale"), py::arg("out"),
        py::arg("workspace"),
        py::arg("A_log"), py::arg("dt_bias"), py::arg("lower_bound"),
        py::arg("initial_state") = py::none(), py::arg("final_state") = py::none(),
        py::arg("cu_seqlens") = py::none());
    m.def("get_workspace_size",
        static_cast<int64_t(*)(int64_t, int64_t, int64_t)>(&get_workspace_size),
        "Get workspace size in bytes",
        py::arg("T_total"), py::arg("H"),
        py::arg("N") = 1);
}
