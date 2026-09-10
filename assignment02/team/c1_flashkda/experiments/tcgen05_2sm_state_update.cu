// Two-CTA tcgen05 experiment for the 128 x V recurrent state update.
//
// A 2x1SM BF16 UMMA can cover M=128, N=16, K=16 in one instruction.  The
// experiment keeps the two peer CTAs explicit: each CTA owns 64 state rows
// and 8 value columns, while the leader CTA issues the CTA-group MMA.  This
// measures the complete product, TMEM load, BF16 recurrent epilogue, and
// state writeback rather than only counting the tensor-core instruction.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

constexpr int kD = 128;
constexpr int kMPerCta = 64;
constexpr int kK = 16;
#ifndef TWO_SM_V
#define TWO_SM_V 16
#endif
constexpr int kV = TWO_SM_V;
#ifndef TWO_SM_THREADS
#define TWO_SM_THREADS 128
#endif
constexpr int kThreads = TWO_SM_THREADS;

#ifndef TWO_SM_LOAD_MODE
#define TWO_SM_LOAD_MODE 256
#endif

#ifndef TWO_SM_ACC_MODE
#define TWO_SM_ACC_MODE 0
#endif

#if TWO_SM_ACC_MODE == 1
using TwoSmAccType = BF16;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp256b1x_16b;
using TwoSmTmemValue = BF16;
#elif TWO_SM_LOAD_MODE == 256
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp256b1x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 128
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp128b1x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 1282
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp128b2x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 1284
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp128b4x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 64
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp64b1x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 644
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp64b4x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 328
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp32b8x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 512
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp256b2x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 1024
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp256b4x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 3201
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_32dp32b1x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 3202
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_32dp32b2x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 3204
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_32dp32b4x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 3208
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_32dp32b8x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 3216
using TwoSmAccType = float;
using TwoSmTmemLoad = SM100_TMEM_LOAD_32dp32b16x;
using TwoSmTmemValue = float;
#elif TWO_SM_LOAD_MODE == 25616
using TwoSmTmemLoad = SM100_TMEM_LOAD_16dp256b1x_16b;
using TwoSmTmemValue = BF16;
#else
#error "unsupported TWO_SM_LOAD_MODE"
#endif

static void check_cuda(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

#define CUDA_CHECK(x) check_cuda((x), #x)

struct TwoSmShared {
  using TiledMMA = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<BF16, BF16, TwoSmAccType, 128, kV,
                                   UMMA::Major::K, UMMA::Major::K>{}));
  using ShapeA = decltype(partition_shape_A(
      TiledMMA{}, make_shape(Int<128>{}, Int<kK>{})));
  using ShapeB = decltype(partition_shape_B(
      TiledMMA{}, make_shape(Int<kV>{}, Int<kK>{})));
  using LayoutA = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, ShapeA{}));
  using LayoutB = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, ShapeB{}));
  using FullLayout = Layout<Shape<Int<kD>, Int<kV>>,
                            Stride<Int<kV>, Int<1>>>;

  alignas(128) ArrayEngine<BF16, cosize_v<LayoutA>> a;
  alignas(128) ArrayEngine<BF16, cosize_v<LayoutB>> b;
  // Both peer CTAs keep a full state-sized backing buffer.  partition_C()
  // selects rows [0,64) or [64,128) according to the CTA rank, which avoids
  // relying on a pointer one object-size before the local allocation.
  alignas(128) BF16 state[kD * kV];
  alignas(128) float decay[kD];

  alignas(16) uint64_t mma_barrier;
  alignas(16) uint32_t tmem_base;

  CUTE_DEVICE auto tensor_a() {
    return make_tensor(make_smem_ptr(a.begin()), LayoutA{});
  }
  CUTE_DEVICE auto tensor_b() {
    return make_tensor(make_smem_ptr(b.begin()), LayoutB{});
  }
  CUTE_DEVICE auto tensor_state() {
    return make_tensor(make_smem_ptr(state), FullLayout{});
  }
};

__global__ void two_sm_state_update_kernel(BF16 const* g_a,
                                           BF16 const* g_b,
                                           BF16 const* g_initial_state,
                                           float const* g_decay,
                                           BF16* g_final_state,
                                           int repeats) {
  using Shared = TwoSmShared;
  using TiledMMA = typename Shared::TiledMMA;

  extern __shared__ char raw[];
  Shared& smem = *reinterpret_cast<Shared*>(raw);

  auto g_a_tensor = make_tensor(
      make_gmem_ptr(g_a),
      Layout<Shape<Int<kD>, Int<kK>>,
             Stride<Int<kK>, Int<1>>>{});
  auto g_b_tensor = make_tensor(
      make_gmem_ptr(g_b),
      Layout<Shape<Int<kV>, Int<kK>>,
             Stride<Int<kK>, Int<1>>>{});
  auto g_initial = make_tensor(
      make_gmem_ptr(g_initial_state), typename Shared::FullLayout{});

  TiledMMA tiled_mma{};
  int cta_rank = int(block_rank_in_cluster());
  auto cta_mma = tiled_mma.get_slice(cta_rank);
  auto s_a = smem.tensor_a();
  auto s_b = smem.tensor_b();
  auto s_state = smem.tensor_state();

  // Each peer CTA fills its own half of the A/B shared-memory descriptors;
  // the 2SM instruction consumes the pair of descriptors as one tile.
  cooperative_copy<kThreads>(threadIdx.x,
                              cta_mma.partition_A(g_a_tensor)(_, _, 0), s_a);
  cooperative_copy<kThreads>(threadIdx.x,
                              cta_mma.partition_B(g_b_tensor)(_, _, 0), s_b);
  cooperative_copy<kThreads>(threadIdx.x, g_initial, s_state);
  for (int i = threadIdx.x; i < kD; i += kThreads) {
    smem.decay[i] = g_decay[i];
  }
  __syncthreads();

  auto t_cr_a = cta_mma.make_fragment_A(s_a);
  auto t_cr_b = cta_mma.make_fragment_B(s_b);
  auto t_cg_acc = cta_mma.partition_C(s_state);
  auto t_c_acc = cta_mma.make_fragment_C(t_cg_acc);

  using Allocator = TMEM::Allocator2Sm;
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
    // The leader's multicast arrival completes the barrier in both peer
    // CTAs.  This matches the 2x1SM CUTLASS tutorial protocol.
    // The peer dimension is part of the CTA-group instruction itself.  The
    // barrier count follows the CUTLASS 2SM protocol: for a 2x1 cluster the
    // non-peer multicast participant count is 1, not 2.
    initialize_barrier(smem.mma_barrier, 1);
  }
  cluster_sync();

  auto t2r = make_tmem_copy(TwoSmTmemLoad{}, t_c_acc);
  auto thr_t2r = t2r.get_slice(threadIdx.x);
  auto t_src = thr_t2r.partition_S(t_c_acc);
  auto t_dst = thr_t2r.partition_D(t_cg_acc);
  auto t_reg = make_fragment_like<TwoSmTmemValue>(t_dst);
  auto t_state_reg = make_fragment_like<BF16>(t_dst);

  int phase = 0;
  for (int r = 0; r < repeats; ++r) {
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if (cta_rank == 0 && one_warp) {
      gemm(tiled_mma, t_cr_a(_, _, 0), t_cr_b(_, _, 0), t_c_acc);
      cutlass::arch::umma_arrive_multicast_2x1SM(&smem.mma_barrier, 0x3);
    }
    wait_barrier(smem.mma_barrier, phase);
    phase ^= 1;

    copy(t2r, t_src, t_reg);
    for (int i = 0; i < size(t_reg); ++i) {
      int state_offset = int(reinterpret_cast<char const*>(&t_dst(i)) -
                             reinterpret_cast<char const*>(smem.state)) /
                         int(sizeof(BF16));
      int row = state_offset / kV;
      t_state_reg(i) = BF16(float(t_dst(i)) * smem.decay[row] + t_reg(i));
    }
    copy(t_state_reg, t_dst);
    __syncthreads();
  }

  __syncthreads();
  // The C partition is row-major in the backing state tensor.  Each CTA
  // writes only its non-overlapping 64-row half to the global output.
  for (int i = threadIdx.x; i < kMPerCta * kV; i += kThreads) {
    int offset = cta_rank * kMPerCta * kV + i;
    g_final_state[offset] = smem.state[offset];
  }
  __syncthreads();

  if (one_warp) {
    allocator.release_allocation_lock();
    allocator.free(smem.tmem_base, Allocator::Sm100TmemCapacityColumns);
  }
}

static void launch_two_sm(BF16* a, BF16* b, BF16* initial, float* decay,
                          BF16* final, int repeats) {
  size_t smem = sizeof(TwoSmShared);
  auto kernel = two_sm_state_update_kernel;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(2, 1, 1);
  dim3 block(128, 1, 1);
  dim3 cluster(2, 1, 1);
  cutlass::ClusterLaunchParams params{grid, block, cluster, int(smem)};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*)kernel, a, b, initial, decay, final, repeats);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "cluster launch failed: %d\n", int(status));
    std::exit(2);
  }
  CUDA_CHECK(cudaGetLastError());
}

static float time_two_sm(BF16* a, BF16* b, BF16* initial, float* decay,
                         BF16* final, int repeats) {
  constexpr int launches = 20;
  for (int i = 0; i < 3; ++i) {
    launch_two_sm(a, b, initial, decay, final, repeats);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i) {
    launch_two_sm(a, b, initial, decay, final, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1.e6f / (launches * repeats);
}

int main(int argc, char** argv) {
  int repeats = argc > 1 ? std::atoi(argv[1]) : 128;
  if (repeats < 1) repeats = 1;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d repeats=%d\n", prop.name, prop.major,
              prop.minor, repeats);
  if (prop.major != 10) return 3;

  std::vector<BF16> h_a(kD * kK), h_b(kV * kK), h_init(kD * kV);
  std::vector<float> h_decay(kD);
  for (int row = 0; row < kD; ++row) {
    h_decay[row] = 0.75f + 0.001f * float(row % 17);
    for (int k = 0; k < kK; ++k) {
      h_a[row * kK + k] = BF16(float((3 * row + 5 * k) % 13 - 6) * 0.125f);
    }
    for (int col = 0; col < kV; ++col) {
      h_init[row * kV + col] = BF16(float((row + 2 * col) % 9 - 4) * 0.03125f);
    }
  }
  for (int col = 0; col < kV; ++col) {
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
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_init, h_init.data(), h_init.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_decay, h_decay.data(), h_decay.size() * sizeof(float), cudaMemcpyHostToDevice));

  launch_two_sm(d_a, d_b, d_init, d_decay, d_final, 1);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<BF16> got(h_init.size());
  CUDA_CHECK(cudaMemcpy(got.data(), d_final, got.size() * sizeof(BF16), cudaMemcpyDeviceToHost));

  std::vector<BF16> ref = h_init;
  for (int row = 0; row < kD; ++row) {
    for (int col = 0; col < kV; ++col) {
      float dot = 0.f;
      for (int k = 0; k < kK; ++k) {
        dot += float(h_a[row * kK + k]) * float(h_b[col * kK + k]);
      }
      ref[row * kV + col] = BF16(float(ref[row * kV + col]) * h_decay[row] + dot);
    }
  }
  int bad = 0;
  float max_abs = 0.f;
  for (size_t i = 0; i < got.size(); ++i) {
    float diff = std::fabs(float(got[i]) - float(ref[i]));
    max_abs = std::max(max_abs, diff);
    if (diff > 1.e-2f) ++bad;
  }
  std::printf("two_sm_state shape=128x%dx16 ns_per_full_state_update=%.3f "
              "bad=%d max_abs=%.6g\n",
              kV, time_two_sm(d_a, d_b, d_init, d_decay, d_final, repeats),
              bad, max_abs);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_init));
  CUDA_CHECK(cudaFree(d_final));
  CUDA_CHECK(cudaFree(d_decay));
  return bad != 0;
}
