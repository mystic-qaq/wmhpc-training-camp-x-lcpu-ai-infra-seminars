// A state-update-shaped tcgen05 experiment for FlashKDA.
//
// The production recurrence updates a 128 x V state with
//
//   S[m, v] = bf16(decay[m] * S[m, v] + K_restored[m, :] @ U[:, v]).
//
// For one V block, the natural UMMA decomposition is two 64 x V x 16
// operations: tcgen05 requires M >= 64 for this BF16 mode, while FlashKDA's
// existing HMMA path naturally uses eight 16 x 16 products along M.  This
// program performs the actual 2 x (64 x V x 16) product, TMEM -> SMEM
// epilogue, BF16 decay/add, and recurrent state writeback.  It is therefore a
// stronger test than a bare opcode benchmark, while remaining isolated from
// the production kernel until its layout/protocol cost is understood.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

constexpr int kD = 128;
constexpr int kM = 64;
constexpr int kK = 16;
constexpr int kThreads = 128;
constexpr int kHmmaThreads = 256;

#ifndef TEST_V
#define TEST_V 16
#endif

static void check_cuda(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

#define CUDA_CHECK(x) check_cuda((x), #x)

template <int V>
struct TcUpdateShared {
  static_assert(V % 8 == 0 && V <= 256);
  using TiledMMA = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<BF16, BF16, float, kM, V,
                           UMMA::Major::K, UMMA::Major::K>{}));
  using ShapeA = decltype(partition_shape_A(
      TiledMMA{}, make_shape(Int<kM>{}, Int<kK>{})));
  using ShapeB = decltype(partition_shape_B(
      TiledMMA{}, make_shape(Int<V>{}, Int<kK>{})));
  using LayoutA = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, ShapeA{}));
  using LayoutB = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, ShapeB{}));
  using StateLayout = Layout<Shape<Int<kD>, Int<V>>,
                             Stride<Int<V>, Int<1>>>;

  // Two M=64 A tiles cover the production D=128 state rows.  B is shared by
  // both operations, exactly as K_restored is shared by all value blocks.
  alignas(128) ArrayEngine<BF16, cosize_v<LayoutA>> a0;
  alignas(128) ArrayEngine<BF16, cosize_v<LayoutA>> a1;
  alignas(128) ArrayEngine<BF16, cosize_v<LayoutB>> b;

  // These are deliberately ordinary row-major buffers.  The tcgen05 result
  // must leave TMEM before the BF16 recurrent epilogue can update state.
  alignas(128) BF16 state[kD * V];
  alignas(128) float acc[kM * V];
  alignas(128) float decay[kD];

  alignas(16) uint64_t mma_barrier;
  alignas(16) uint32_t tmem_base;

  CUTE_DEVICE auto tensor_a0() {
    return make_tensor(make_smem_ptr(a0.begin()), LayoutA{});
  }
  CUTE_DEVICE auto tensor_a1() {
    return make_tensor(make_smem_ptr(a1.begin()), LayoutA{});
  }
  CUTE_DEVICE auto tensor_b() {
    return make_tensor(make_smem_ptr(b.begin()), LayoutB{});
  }
};

template <int V>
__global__ void tcgen05_state_update_kernel(BF16 const* g_a,
                                            BF16 const* g_b,
                                            BF16 const* g_initial_state,
                                            float const* g_decay,
                                            BF16* g_final_state,
                                            int repeats) {
  using Shared = TcUpdateShared<V>;
  using TiledMMA = typename Shared::TiledMMA;
  using StateLayout = typename Shared::StateLayout;

  extern __shared__ char raw[];
  Shared& smem = *reinterpret_cast<Shared*>(raw);

  auto g_a_all = make_tensor(
      make_gmem_ptr(g_a),
      Layout<Shape<Int<kD>, Int<kK>>,
             Stride<Int<kK>, Int<1>>>{});
  auto g_b_tensor = make_tensor(
      make_gmem_ptr(g_b),
      Layout<Shape<Int<V>, Int<kK>>,
             Stride<Int<kK>, Int<1>>>{});
  auto g_initial = make_tensor(make_gmem_ptr(g_initial_state), StateLayout{});
  auto g_final = make_tensor(make_gmem_ptr(g_final_state), StateLayout{});
  auto s_state = make_tensor(make_smem_ptr(smem.state), StateLayout{});
  using StateTileLayout = Layout<Shape<Int<kM>, Int<V>>,
                                Stride<Int<V>, Int<1>>>;
  auto s_state0 = make_tensor(make_smem_ptr(smem.state), StateTileLayout{});
  auto s_state1 = make_tensor(make_smem_ptr(smem.state + kM * V),
                              StateTileLayout{});
  // Keep an ordinary 2-D shared-memory tensor as the backing storage, then
  // let ThrMMA produce the post-partitioned C tensor.  TMEM copy partitioning
  // is defined in terms of that ((M,N),1,1) post-partitioned form; passing a
  // hand-written nested layout here loses the MMA-specific C layout.
  auto s_acc_base = make_tensor(
      make_smem_ptr(smem.acc),
      Layout<Shape<Int<kM>, Int<V>>,
             Stride<Int<V>, Int<1>>>{});

  TiledMMA tiled_mma{};
  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  auto g_a0 = make_tensor(g_a_all.data(),
                          Layout<Shape<Int<kM>, Int<kK>>,
                                 Stride<Int<kK>, Int<1>>>{});
  auto g_a1 = make_tensor(g_a_all.data() + kM * kK,
                          Layout<Shape<Int<kM>, Int<kK>>,
                                 Stride<Int<kK>, Int<1>>>{});

  auto s_a0 = smem.tensor_a0();
  auto s_a1 = smem.tensor_a1();
  auto s_b = smem.tensor_b();

  // Stage A/B once per kernel launch.  This is outside the repeated
  // state-update loop, matching the way the production recurrence keeps the
  // current workspace tile in shared memory.
  cooperative_copy<kThreads>(threadIdx.x,
                              cta_mma.partition_A(g_a0)(_, _, 0), s_a0);
  cooperative_copy<kThreads>(threadIdx.x,
                              cta_mma.partition_A(g_a1)(_, _, 0), s_a1);
  cooperative_copy<kThreads>(threadIdx.x,
                              cta_mma.partition_B(g_b_tensor)(_, _, 0), s_b);
  cooperative_copy<kThreads>(threadIdx.x, g_initial, s_state);
  for (int i = threadIdx.x; i < kD; i += kThreads) {
    smem.decay[i] = g_decay[i];
  }
  __syncthreads();

  auto t_cr_a0 = cta_mma.make_fragment_A(s_a0);
  auto t_cr_a1 = cta_mma.make_fragment_A(s_a1);
  auto t_cr_b = cta_mma.make_fragment_B(s_b);
  auto t_cg_acc = cta_mma.partition_C(s_acc_base);
  auto t_c_acc = cta_mma.make_fragment_C(t_cg_acc);

  using Allocator = TMEM::Allocator1Sm;
  Allocator allocator{};
  bool one_warp = (threadIdx.x / 32 == 0);
  uint32_t one_thread = elect_one_sync();
  if (one_warp) {
    allocator.allocate(Allocator::Sm100TmemCapacityColumns,
                       &smem.tmem_base);
  }
  __syncthreads();
  t_c_acc.data() = smem.tmem_base;

  if (one_warp && one_thread) {
    initialize_barrier(smem.mma_barrier, 1);
  }
  __syncthreads();

  // The 64xV accumulator has a 16dp32b1x-compatible TMEM load for V=16.
  // V=32/64/128 uses the corresponding wider copy atom below; keeping the
  // choice compile-time lets CUTE validate the TMEM layout.
  auto t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b1x{}, t_c_acc);
  auto thr_t2r = t2r.get_slice(threadIdx.x);
  auto t_src = thr_t2r.partition_S(t_c_acc);
  auto t_cg_state0 = cta_mma.partition_C(s_state0);
  auto t_cg_state1 = cta_mma.partition_C(s_state1);
  auto t_dst0 = thr_t2r.partition_D(t_cg_state0);
  auto t_dst1 = thr_t2r.partition_D(t_cg_state1);
  auto t_reg = make_fragment_like<float>(t_dst0);
  auto t_state_reg0 = make_fragment_like<BF16>(t_dst0);
  auto t_state_reg1 = make_fragment_like<BF16>(t_dst1);

  int phase = 0;
  for (int r = 0; r < repeats; ++r) {
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (one_warp) {
      gemm(tiled_mma, t_cr_a0(_, _, 0), t_cr_b(_, _, 0), t_c_acc);
      cutlass::arch::umma_arrive(&smem.mma_barrier);
    }
    wait_barrier(smem.mma_barrier, phase);
    phase ^= 1;
    copy(t2r, t_src, t_reg);
    for (int i = 0; i < size(t_reg); ++i) {
      int state_offset = int(reinterpret_cast<char const*>(&t_dst0(i)) -
                             reinterpret_cast<char const*>(smem.state)) /
                         int(sizeof(BF16));
      int row = state_offset / V;
      t_state_reg0(i) = BF16(float(t_dst0(i)) * smem.decay[row] + t_reg(i));
    }
    copy(t_state_reg0, t_dst0);

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (one_warp) {
      gemm(tiled_mma, t_cr_a1(_, _, 0), t_cr_b(_, _, 0), t_c_acc);
      cutlass::arch::umma_arrive(&smem.mma_barrier);
    }
    wait_barrier(smem.mma_barrier, phase);
    phase ^= 1;
    copy(t2r, t_src, t_reg);
    for (int i = 0; i < size(t_reg); ++i) {
      int state_offset = int(reinterpret_cast<char const*>(&t_dst1(i)) -
                             reinterpret_cast<char const*>(smem.state)) /
                         int(sizeof(BF16));
      int row = state_offset / V;
      t_state_reg1(i) = BF16(float(t_dst1(i)) * smem.decay[row] + t_reg(i));
    }
    copy(t_state_reg1, t_dst1);
  }

  cooperative_copy<kThreads>(threadIdx.x, s_state, g_final);
  __syncthreads();
  if (one_warp) {
    allocator.release_allocation_lock();
    allocator.free(smem.tmem_base, Allocator::Sm100TmemCapacityColumns);
  }
}

// HMMA reference for the same 128xV state update.  This intentionally uses
// the production-style eight-warp decomposition for V=16: each warp owns one
// 16x16 state tile, performs the corresponding 16x16x16 HMMA product, and
// applies the BF16 recurrent epilogue.  Unlike the tcgen05 path, the state
// tile is loaded/stored through the normal C-layout shared-memory path on
// every repeat, matching the narrow V16 production mapping.
template <int V>
struct HmmaUpdateShared {
  using KLayout = decltype(tile_to_shape(
      GMMA::Layout_MN_INTER_Atom<BF16>{},
      make_shape(Int<kD>{}, Int<kK>{}), LayoutRight{}));
  using ULayout = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{},
      make_shape(Int<kK>{}, Int<V>{}), LayoutLeft{}));
  using StateLayout = decltype(tile_to_shape(
      GMMA::Layout_MN_INTER_Atom<BF16>{},
      make_shape(Int<kD>{}, Int<V>{}), LayoutRight{}));

  alignas(128) ArrayEngine<BF16, cosize_v<KLayout>> k;
  alignas(128) ArrayEngine<BF16, cosize_v<ULayout>> u;
  alignas(128) ArrayEngine<BF16, cosize_v<StateLayout>> state;
  alignas(128) float decay[kD];

  CUTE_DEVICE auto tensor_k() {
    return make_tensor(make_smem_ptr(k.begin()), KLayout{});
  }
  CUTE_DEVICE auto tensor_u() {
    return make_tensor(make_smem_ptr(u.begin()), ULayout{});
  }
  CUTE_DEVICE auto tensor_state() {
    return make_tensor(make_smem_ptr(state.begin()), StateLayout{});
  }
};

template <int V>
__global__ void hmma_state_update_kernel(BF16 const* g_a,
                                          BF16 const* g_b,
                                          BF16 const* g_initial_state,
                                          float const* g_decay,
                                          BF16* g_final_state,
                                          int repeats) {
  using Shared = HmmaUpdateShared<V>;
  using KLayout = typename Shared::KLayout;
  using ULayout = typename Shared::ULayout;
  using StateLayout = typename Shared::StateLayout;

  extern __shared__ char raw[];
  Shared& smem = *reinterpret_cast<Shared*>(raw);

  auto g_a_tensor = make_tensor(
      make_gmem_ptr(g_a),
      Layout<Shape<Int<kD>, Int<kK>>,
             Stride<Int<kK>, Int<1>>>{});
  auto g_b_tensor = make_tensor(
      make_gmem_ptr(g_b),
      Layout<Shape<Int<V>, Int<kK>>,
             Stride<Int<kK>, Int<1>>>{});
  auto g_initial = make_tensor(
      make_gmem_ptr(g_initial_state),
      Layout<Shape<Int<kD>, Int<V>>,
             Stride<Int<V>, Int<1>>>{});
  auto g_final = make_tensor(
      make_gmem_ptr(g_final_state),
      Layout<Shape<Int<kD>, Int<V>>,
             Stride<Int<V>, Int<1>>>{});

  auto s_k = smem.tensor_k();
  auto s_u = smem.tensor_u();
  auto s_state = smem.tensor_state();
  cooperative_copy<kHmmaThreads>(threadIdx.x, g_a_tensor, s_k);
  cooperative_copy<kHmmaThreads>(threadIdx.x, g_b_tensor, s_u);
  cooperative_copy<kHmmaThreads>(threadIdx.x, g_initial, s_state);
  for (int i = threadIdx.x; i < kD; i += kHmmaThreads) {
    smem.decay[i] = g_decay[i];
  }
  __syncthreads();

  auto mma = make_tiled_mma(
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
      Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  if (warp_id < kD / 16) {
    for (int vi = 0; vi < V / 16; ++vi) {
    auto thr_mma = mma.get_slice(lane_id);
    Tensor k_block = local_tile(s_k, make_shape(Int<16>{}, Int<16>{}),
                                make_coord(warp_id, 0));
    Tensor u_block = local_tile(s_u, make_shape(Int<16>{}, Int<16>{}),
                                make_coord(0, vi));
    Tensor state_block = local_tile(
        s_state, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id, vi));

    auto copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto load_state = make_tiled_copy_C(
        Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto store_state = make_tiled_copy_C(
        Copy_Atom<SM90_U16x8_STSM_T, BF16>{}, mma);
    auto thr_copy_a = copy_a.get_thread_slice(lane_id);
    auto thr_copy_b = copy_b.get_thread_slice(lane_id);
    auto thr_load_state = load_state.get_slice(lane_id);
    auto thr_store_state = store_state.get_slice(lane_id);

    auto a_frag = thr_mma.partition_fragment_A(k_block);
    auto b_frag = thr_mma.partition_fragment_B(u_block);
    auto a_copy_frag = make_fragment_like<BF16>(a_frag);
    auto b_copy_frag = make_fragment_like<BF16>(b_frag);
    auto a_copy_view = thr_copy_a.retile_D(a_copy_frag);
    auto b_copy_view = thr_copy_b.retile_D(b_copy_frag);
    copy(copy_a, thr_copy_a.partition_S(k_block), a_copy_view);
    copy(copy_b, thr_copy_b.partition_S(u_block), b_copy_view);
    transform(a_copy_frag, a_frag, identity{});
    transform(b_copy_frag, b_frag, identity{});

    auto c_partition = thr_mma.partition_C(state_block);
    auto acc = thr_mma.make_fragment_C(c_partition);
    auto state_frag = make_fragment_like<BF16>(acc);
    clear(acc);

    for (int r = 0; r < repeats; ++r) {
      copy(load_state, thr_load_state.partition_S(state_block),
           thr_load_state.retile_D(state_frag));
      clear(acc);
      gemm(thr_mma, a_frag(_, _, Int<0>{}), b_frag(_, _, Int<0>{}), acc);
      int group_id = (lane_id / 4) % 8;
      float g0 = smem.decay[warp_id * 16 + group_id];
      float g1 = smem.decay[warp_id * 16 + group_id + 8];
      for (int a = 0; a < 2; ++a) {
        for (int d = 0; d < 2; ++d) {
          auto c0 = make_coord(make_coord(a, 0), 0, d);
          auto c1 = make_coord(make_coord(a, 1), 0, d);
          state_frag(c0) = BF16(float(state_frag(c0)) * g0 + acc(c0));
          state_frag(c1) = BF16(float(state_frag(c1)) * g1 + acc(c1));
        }
      }
      copy(store_state, thr_store_state.retile_S(state_frag),
           thr_store_state.partition_D(state_block));
    }
    }
  }
  __syncthreads();
  cooperative_copy<kHmmaThreads>(threadIdx.x, s_state, g_final);
}

template <int V>
static float time_tcgen(BF16* a, BF16* b, BF16* initial, float* decay,
                        BF16* final, int repeats) {
  auto kernel = tcgen05_state_update_kernel<V>;
  size_t smem = sizeof(TcUpdateShared<V>);
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  constexpr int launches = 20;
  for (int i = 0; i < 3; ++i) {
    kernel<<<1, kThreads, smem>>>(a, b, initial, decay, final, repeats);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i) {
    kernel<<<1, kThreads, smem>>>(a, b, initial, decay, final, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1.e6f / (launches * repeats * 2.f);
}

template <int V>
static float time_hmma(BF16* a, BF16* b, BF16* initial, float* decay,
                       BF16* final, int repeats) {
  auto kernel = hmma_state_update_kernel<V>;
  size_t smem = sizeof(HmmaUpdateShared<V>);
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  constexpr int launches = 20;
  for (int i = 0; i < 3; ++i) {
    kernel<<<1, kHmmaThreads, smem>>>(a, b, initial, decay, final, repeats);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i) {
    kernel<<<1, kHmmaThreads, smem>>>(a, b, initial, decay, final, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1.e6f / (launches * repeats);
}

template <int V>
static int run_case(int repeats) {
  std::vector<BF16> h_a(kD * kK), h_b(V * kK), h_init(kD * V);
  std::vector<float> h_decay(kD);
  for (int row = 0; row < kD; ++row) {
    h_decay[row] = 0.75f + 0.001f * float(row % 17);
    for (int k = 0; k < kK; ++k) {
      h_a[row * kK + k] = BF16(float((3 * row + 5 * k) % 13 - 6) * 0.125f);
    }
    for (int col = 0; col < V; ++col) {
      h_init[row * V + col] = BF16(float((row + 2 * col) % 9 - 4) * 0.03125f);
    }
  }
  for (int col = 0; col < V; ++col) {
    for (int k = 0; k < kK; ++k) {
      h_b[col * kK + k] = BF16(float((7 * col + 3 * k) % 11 - 5) * 0.125f);
    }
  }

  BF16 *d_a = nullptr, *d_b = nullptr, *d_init = nullptr, *d_final = nullptr;
  float* d_decay = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_init, h_init.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_final, h_init.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_decay, h_decay.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(BF16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(BF16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_init, h_init.data(), h_init.size() * sizeof(BF16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_decay, h_decay.data(), h_decay.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto kernel = tcgen05_state_update_kernel<V>;
  size_t smem = sizeof(TcUpdateShared<V>);
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kernel<<<1, kThreads, smem>>>(d_a, d_b, d_init, d_decay, d_final, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<BF16> got(h_init.size());
  CUDA_CHECK(cudaMemcpy(got.data(), d_final, got.size() * sizeof(BF16),
                        cudaMemcpyDeviceToHost));

  std::vector<BF16> ref = h_init;
  for (int group = 0; group < 2; ++group) {
    for (int row = 0; row < kM; ++row) {
      int global_row = group * kM + row;
      for (int col = 0; col < V; ++col) {
        float dot = 0.f;
        for (int k = 0; k < kK; ++k) {
          dot += float(h_a[global_row * kK + k]) *
                 float(h_b[col * kK + k]);
        }
        ref[global_row * V + col] = BF16(
            float(ref[global_row * V + col]) * h_decay[global_row] + dot);
      }
    }
  }
  int bad = 0;
  float max_abs = 0.f;
  for (size_t i = 0; i < got.size(); ++i) {
    float diff = std::fabs(float(got[i]) - float(ref[i]));
    max_abs = std::max(max_abs, diff);
    if (diff > 1.e-2f) ++bad;
  }

  auto hmma_kernel = hmma_state_update_kernel<V>;
  smem = sizeof(HmmaUpdateShared<V>);
  CUDA_CHECK(cudaFuncSetAttribute(
      hmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  hmma_kernel<<<1, kHmmaThreads, smem>>>(d_a, d_b, d_init, d_decay, d_final, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<BF16> got_hmma(h_init.size());
  CUDA_CHECK(cudaMemcpy(got_hmma.data(), d_final,
                        got_hmma.size() * sizeof(BF16),
                        cudaMemcpyDeviceToHost));
  int bad_hmma = 0;
  float max_abs_hmma = 0.f;
  for (size_t i = 0; i < got_hmma.size(); ++i) {
    float diff = std::fabs(float(got_hmma[i]) - float(ref[i]));
    max_abs_hmma = std::max(max_abs_hmma, diff);
    if (diff > 1.e-2f) ++bad_hmma;
  }
  if (bad_hmma && repeats == 1) {
    for (int row : {0, 1, 15, 16, 17, 31, 32, 63, 64, 95, 96, 111, 112, 127}) {
      int i = row * V;
      std::printf("hmma row %d: %.6g %.6g %.6g %.6g | ref %.6g %.6g %.6g %.6g\n",
                  row, float(got_hmma[i]), float(got_hmma[i + 1]),
                  float(got_hmma[i + 2]), float(got_hmma[i + 3]),
                  float(ref[i]), float(ref[i + 1]), float(ref[i + 2]),
                  float(ref[i + 3]));
    }
  }

  float ns = time_tcgen<V>(d_a, d_b, d_init, d_decay, d_final, repeats);
  float ns_hmma = time_hmma<V>(d_a, d_b, d_init, d_decay, d_final, repeats);
  std::printf("tcgen05_state V=%d shape=64x%dx16 ns_per_64xV=%.3f "
              "ns_per_full_state_update=%.3f bad=%d max_abs=%.6g\n",
              V, V, ns, ns * 2.f, bad, max_abs);
  std::printf("hmma_state   V=%d shape=128x%dx16 ns_per_full_state_update=%.3f "
              "bad=%d max_abs=%.6g tcgen05/hmma=%.3fx\n",
              V, V, ns_hmma, bad_hmma, max_abs_hmma, (ns * 2.f) / ns_hmma);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_init));
  CUDA_CHECK(cudaFree(d_final));
  CUDA_CHECK(cudaFree(d_decay));
  return bad || bad_hmma;
}

int main(int argc, char** argv) {
  int repeats = argc > 1 ? std::atoi(argv[1]) : 128;
  if (repeats < 1) repeats = 1;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d repeats=%d\n", prop.name, prop.major,
              prop.minor, repeats);
  if (prop.major != 10) {
    std::fprintf(stderr, "This experiment requires SM100/SM103.\n");
    return 3;
  }
  return run_case<TEST_V>(repeats) != 0;
}
