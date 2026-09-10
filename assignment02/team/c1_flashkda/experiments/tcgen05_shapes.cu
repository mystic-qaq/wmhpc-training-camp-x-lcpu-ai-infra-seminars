// Compare one SM100/103 tcgen05 BF16 tile with the equivalent collection of
// SM80 16x16 HMMA tiles.  This is a focused experiment for the K2 state
// update: U[V,16] @ K[16,128], where V can be 64 or 128.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

static void check(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}
#define CUDA_CHECK(x) check((x), #x)

template <int M, int N, int K>
struct TcShared {
  using TiledMMA = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<BF16, BF16, float, M, N,
                           UMMA::Major::K, UMMA::Major::K>{}));
  using ShapeA = decltype(partition_shape_A(
      TiledMMA{}, make_shape(Int<M>{}, Int<K>{})));
  using ShapeB = decltype(partition_shape_B(
      TiledMMA{}, make_shape(Int<N>{}, Int<K>{})));
  using LayoutA = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, ShapeA{}));
  using LayoutB = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, ShapeB{}));

  alignas(128) ArrayEngine<BF16, cosize_v<LayoutA>> a;
  alignas(128) ArrayEngine<BF16, cosize_v<LayoutB>> b;
  alignas(16) uint64_t barrier;
  alignas(16) uint32_t tmem_base;

  CUTE_DEVICE auto tensor_a() { return make_tensor(make_smem_ptr(a.begin()), LayoutA{}); }
  CUTE_DEVICE auto tensor_b() { return make_tensor(make_smem_ptr(b.begin()), LayoutB{}); }
};

template <int M, int N, int K>
__global__ void tcgen_kernel(BF16 const* g_a, BF16 const* g_b, float* g_d,
                             int repeats) {
  using Shared = TcShared<M, N, K>;
  using TiledMMA = typename Shared::TiledMMA;
  extern __shared__ char raw[];
  Shared& shared = *reinterpret_cast<Shared*>(raw);

  auto ga = make_tensor(make_gmem_ptr(g_a),
                        Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  auto gb = make_tensor(make_gmem_ptr(g_b),
                        Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  auto mma = TiledMMA{};
  auto cta_mma = mma.get_slice(Int<0>{});
  auto ta = shared.tensor_a();
  auto tb = shared.tensor_b();
  cooperative_copy<128>(threadIdx.x, ga, ta);
  cooperative_copy<128>(threadIdx.x, gb, tb);
  __syncthreads();

  auto frag_a = cta_mma.make_fragment_A(ta);
  auto frag_b = cta_mma.make_fragment_B(tb);
  // Use a real output pointer while constructing the C partition.  The
  // epilogue is outside this experiment's timed region, but a valid GMEM
  // tensor keeps the descriptor/layout path identical to a real K2 update.
  auto gd = make_tensor(make_gmem_ptr(g_d),
                        Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>{});
  auto t_cg_d = cta_mma.partition_C(gd);
  auto frag_c = cta_mma.make_fragment_C(t_cg_d);

  using Allocator = TMEM::Allocator1Sm;
  Allocator allocator{};
  uint32_t one_warp = (threadIdx.x / 32 == 0);
  uint32_t one_thread = elect_one_sync();
  if (one_warp) {
    allocator.allocate(Allocator::Sm100TmemCapacityColumns, &shared.tmem_base);
  }
  __syncthreads();
  frag_c.data() = shared.tmem_base;
  if (one_warp && one_thread) {
    initialize_barrier(shared.barrier, 1);
  }
  __syncthreads();

  mma.accumulate_ = UMMA::ScaleOut::Zero;
  int phase = 0;
  for (int r = 0; r < repeats; ++r) {
    if (one_warp) {
      gemm(mma, frag_a, frag_b, frag_c);
      mma.accumulate_ = UMMA::ScaleOut::One;
      cutlass::arch::umma_arrive(&shared.barrier);
    }
    wait_barrier(shared.barrier, phase);
    phase ^= 1;
  }
  __syncthreads();
  if (one_warp) {
    allocator.release_allocation_lock();
    allocator.free(shared.tmem_base, Allocator::Sm100TmemCapacityColumns);
  }
}

template <int M, int N, int K>
__global__ void hmma_kernel(BF16 const* g_a, BF16 const* g_b, float* g_d,
                            int repeats) {
  constexpr int TilesM = M / 16;
  constexpr int TilesN = N / 16;
  constexpr int kElems = 16 * 16;
  int tile = blockIdx.x;
  int tm = tile / TilesN;
  int tn = tile % TilesN;
  using ALayout = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<16>{}, Int<16>{}), LayoutLeft{}));
  using CLayout = ALayout;
  __shared__ alignas(128) BF16 raw[3 * 16 * 16];
  auto a = make_tensor(make_smem_ptr(raw), ALayout{});
  auto b = make_tensor(make_smem_ptr(raw + 16 * 16), ALayout{});
  auto c = make_tensor(make_smem_ptr(raw + 2 * 16 * 16), CLayout{});
  auto ga = make_tensor(make_gmem_ptr(g_a + tm * 16 * K),
                        Layout<Shape<Int<16>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  auto gb = make_tensor(make_gmem_ptr(g_b + tn * 16 * K),
                        Layout<Shape<Int<16>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  copy(AutoVectorizingCopy{}, ga, a);
  copy(AutoVectorizingCopy{}, gb, b);
  __syncthreads();

  auto mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                            Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
  int tid = threadIdx.x;
  if (tid < 32) {
    auto thr = mma.get_slice(tid);
    auto ar = thr.partition_fragment_A(a);
    auto br = thr.partition_fragment_B(b);
    auto cr = thr.make_fragment_C(thr.partition_C(c));
    auto ca = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto cb = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto ta = ca.get_thread_slice(tid);
    auto tb = cb.get_thread_slice(tid);
    auto ar_copy = make_fragment_like<BF16>(ar);
    auto br_copy = make_fragment_like<BF16>(br);
    copy(ca, ta.partition_S(a), ta.retile_D(ar_copy));
    copy(cb, tb.partition_S(b), tb.retile_D(br_copy));
    transform(ar_copy, ar, identity{});
    transform(br_copy, br, identity{});
    clear(cr);
    for (int r = 0; r < repeats; ++r) {
      gemm(thr, ar(_, _, Int<0>{}), br(_, _, Int<0>{}), cr);
    }
    auto sc = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto st = sc.get_slice(tid);
    auto out = make_fragment_like<BF16>(cr);
    transform(cr, out, [] __device__(float x) { return BF16(x); });
    copy(sc, st.retile_S(out), st.partition_D(c));
  }
  __syncthreads();

  // Keep the HMMA loop observable.  Without a global-memory side effect,
  // ptxas is allowed to remove the shared-memory-only accumulator and the
  // measured time collapses to a launch/barrier artifact.
  for (int i = threadIdx.x; i < kElems; i += blockDim.x) {
    g_d[tile * kElems + i] = float(raw[2 * kElems + i]);
  }
}

// Fair one-CTA reference for the large-tile question.  The earlier hmma
// kernel launches one CTA per 16x16 output tile, which is useful for a
// throughput sanity check but gives HMMA much more grid-level parallelism
// than a single tcgen05 CTA.  Here eight HMMA warps each execute eight
// independent 16x16 products, i.e. the same 64 logical products as one
// 128x128x16 UMMA operation.  Inputs are staged once and are deliberately
// outside the timed loop, matching time_tc().
template <int TilesPerWarp>
__global__ void hmma_one_cta_kernel(BF16 const* g_a, BF16 const* g_b,
                                    float* g_d, int repeats) {
  constexpr int kWarps = 8;
  constexpr int kElems = 16 * 16;
  extern __shared__ BF16 raw_one[];
  BF16* s_a = raw_one + (threadIdx.x / 32) * 3 * kElems;
  BF16* s_b = s_a + kElems;
  BF16* s_c = s_b + kElems;

  if (threadIdx.x < 32) {
    for (int i = threadIdx.x; i < kElems; i += 32) {
      s_a[i] = g_a[i];
      s_b[i] = g_b[i];
    }
  }
  __syncthreads();

  using ALayout = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<16>{}, Int<16>{}), LayoutLeft{}));
  auto a = make_tensor(make_smem_ptr(s_a), ALayout{});
  auto b = make_tensor(make_smem_ptr(s_b), ALayout{});
  auto c = make_tensor(make_smem_ptr(s_c), ALayout{});
  auto mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                            Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
  int tid = threadIdx.x % 32;
  auto thr = mma.get_slice(tid);
  auto ar = thr.partition_fragment_A(a);
  auto br = thr.partition_fragment_B(b);
  auto cr = thr.make_fragment_C(thr.partition_C(c));
  auto ca = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
  auto cb = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
  auto ta = ca.get_thread_slice(tid);
  auto tb = cb.get_thread_slice(tid);
  auto ar_copy = make_fragment_like<BF16>(ar);
  auto br_copy = make_fragment_like<BF16>(br);
  copy(ca, ta.partition_S(a), ta.retile_D(ar_copy));
  copy(cb, tb.partition_S(b), tb.retile_D(br_copy));
  transform(ar_copy, ar, cute::identity{});
  transform(br_copy, br, cute::identity{});
  clear(cr);

  for (int r = 0; r < repeats; ++r) {
    #pragma unroll
    for (int tile = 0; tile < TilesPerWarp; ++tile) {
      gemm(thr, ar(_, _, Int<0>{}), br(_, _, Int<0>{}), cr);
    }
  }
  __syncthreads();

  auto sc = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
  auto st = sc.get_slice(tid);
  auto out = make_fragment_like<BF16>(cr);
  transform(cr, out, [] __device__(float x) { return BF16(x); });
  copy(sc, st.retile_S(out), st.partition_D(c));
  __syncthreads();

  constexpr int kWarpCOffset = 2 * kElems;
  for (int i = threadIdx.x; i < kWarps * kElems; i += blockDim.x) {
    int warp = i / kElems;
    int elem = i % kElems;
    g_d[i] = float(raw_one[warp * 3 * kElems + kWarpCOffset + elem]);
  }
}

template <int M, int N, int K>
float time_tc(BF16* a, BF16* b, float* d, int repeats) {
  using Shared = TcShared<M, N, K>;
  auto kernel = tcgen_kernel<M, N, K>;
  size_t smem = sizeof(Shared);
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  for (int i = 0; i < 5; ++i) kernel<<<1, 128, smem>>>(a, b, d, repeats);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < 20; ++i) kernel<<<1, 128, smem>>>(a, b, d, repeats);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1.e6f / (20.f * repeats);
}

template <int M, int N, int K>
float time_hmma(BF16* a, BF16* b, float* d, int repeats) {
  auto kernel = hmma_kernel<M, N, K>;
  constexpr int tiles = (M / 16) * (N / 16);
  for (int i = 0; i < 5; ++i) kernel<<<tiles, 128, 0>>>(a, b, d, repeats);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < 20; ++i) kernel<<<tiles, 128, 0>>>(a, b, d, repeats);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1.e6f / (20.f * repeats * tiles);
}

float time_hmma_one_cta(BF16* a, BF16* b, float* d, int repeats) {
  constexpr int kWarps = 8;
  constexpr int kElems = 16 * 16;
  size_t smem = size_t(kWarps) * 3 * kElems * sizeof(BF16);
  auto kernel = hmma_one_cta_kernel<8>;
  for (int i = 0; i < 5; ++i) kernel<<<1, kWarps * 32, smem>>>(a, b, d, repeats);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < 20; ++i)
    kernel<<<1, kWarps * 32, smem>>>(a, b, d, repeats);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1.e6f / (20.f * repeats);
}

template <int M, int N, int K>
void run_shape(int repeats) {
  std::vector<BF16> ha(M * K), hb(N * K);
  for (int i = 0; i < M * K; ++i) ha[i] = BF16(float((i * 3) % 11 - 5));
  for (int i = 0; i < N * K; ++i) hb[i] = BF16(float((i * 5) % 7 - 3));
  BF16 *a = nullptr, *b = nullptr;
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&a, ha.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&b, hb.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d, size_t(M) * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(a, ha.data(), ha.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, hb.data(), hb.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  float tc = time_tc<M, N, K>(a, b, d, repeats);
  float hm = time_hmma<M, N, K>(a, b, d, repeats);
  int useful = (M / 16) * (N / 16);
  std::printf("shape=%dx%dx%d tcgen=%.3f ns/instruction useful=%.3f ns/16x16 hmma=%.3f ns/16x16 ratio=%.3f\n",
              M, N, K, tc, tc / useful, hm, (tc / useful) / hm);
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(d));
}

int main(int argc, char** argv) {
  int repeats = argc > 1 ? std::atoi(argv[1]) : 200;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d repeats=%d\n", prop.name, prop.major, prop.minor, repeats);
  if (prop.major != 10) return 3;
  run_shape<64, 16, 16>(repeats);
  run_shape<64, 32, 16>(repeats);
  run_shape<64, 64, 16>(repeats);
  run_shape<64, 128, 16>(repeats);
  run_shape<128, 16, 16>(repeats);
  run_shape<128, 32, 16>(repeats);
  run_shape<128, 64, 16>(repeats);
  run_shape<128, 128, 16>(repeats);

  // Compare one 128x128 tcgen05 tile with one CTA containing eight HMMA
  // warps.  This is the closest instruction-level proxy for K2's full-D
  // state-update product.
  std::vector<BF16> h_small(128 * 16, BF16(1));
  BF16* d_small_a = nullptr;
  BF16* d_small_b = nullptr;
  float* d_small_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_small_a, h_small.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_small_b, h_small.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_small_c, 128 * 128 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_small_a, h_small.data(), h_small.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_small_b, h_small.data(), h_small.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  float tc128 = time_tc<128, 128, 16>(d_small_a, d_small_b, d_small_c, repeats);
  float hm128 = time_hmma_one_cta(d_small_a, d_small_b, d_small_c, repeats);
  std::printf("fair_one_cta_128x128x16 tcgen=%.3f ns/64_products hmma=%.3f ns/64_products ratio=%.3f\n",
              tc128, hm128, tc128 / hm128);
  CUDA_CHECK(cudaFree(d_small_a));
  CUDA_CHECK(cudaFree(d_small_b));
  CUDA_CHECK(cudaFree(d_small_c));
  return 0;
}
