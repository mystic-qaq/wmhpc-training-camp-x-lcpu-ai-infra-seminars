// FlashKDA C1 experiment: compare the instruction that naturally fits the
// 16x16 Neumann products with the smallest 1-CTA SM100 tcgen05 tile.
//
// This is intentionally a microbenchmark, rather than a replacement FlashKDA
// implementation.  FlashKDA's inverse is a collection of 16x16 products.  On
// SM100/103, the smallest BF16 UMMA has M=64, so this program pads the M axis
// from 16 to 64 and reports both the raw time and the time per *useful* 16x16
// product.  The padding is the point of the experiment: changing only the
// instruction changes the amount of work and the execution protocol too.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

constexpr int kLogicalM = 16;
constexpr int kLogicalN = 16;
constexpr int kK = 16;
constexpr int kTcgenM = 64;
constexpr int kThreads = 128;

static void check_cuda(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

#define CUDA_CHECK(x) check_cuda((x), #x)

template <class TA, class TB, class LA, class LB>
struct TcShared {
  alignas(128) cute::ArrayEngine<TA, cute::cosize_v<LA>> a;
  alignas(128) cute::ArrayEngine<TB, cute::cosize_v<LB>> b;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base;

  CUTE_DEVICE auto tensor_a() { return make_tensor(make_smem_ptr(a.begin()), LA{}); }
  CUTE_DEVICE auto tensor_b() { return make_tensor(make_smem_ptr(b.begin()), LB{}); }
};

// Smallest useful 1-CTA tcgen05 BF16 operation on SM100/SM103.  The result is
// kTcgenM x kLogicalN; the first kLogicalM rows are the logical FlashKDA tile.
template <class Shared, class TiledMMA, class LayoutA, class LayoutB,
          class LayoutD>
__global__ void tcgen05_kernel(BF16 const* g_a, BF16 const* g_b, float* g_d,
                               int repeats) {
  extern __shared__ char smem_raw[];
  Shared& smem = *reinterpret_cast<Shared*>(smem_raw);

  auto g_a_tensor = make_tensor(make_gmem_ptr(g_a), LayoutA{});
  auto g_b_tensor = make_tensor(make_gmem_ptr(g_b), LayoutB{});
  auto g_d_tensor = make_tensor(make_gmem_ptr(g_d), LayoutD{});

  auto mma_tiler = make_shape(Int<kTcgenM>{}, Int<kLogicalN>{}, Int<kK>{});
  auto cta_mma = TiledMMA{}.get_slice(Int<0>{});
  auto t_cg_a = cta_mma.partition_A(g_a_tensor);
  auto t_cg_b = cta_mma.partition_B(g_b_tensor);
  auto t_cg_d = cta_mma.partition_C(g_d_tensor);
  (void)mma_tiler;
  (void)t_cg_a;
  (void)t_cg_b;

  auto t_cs_a = smem.tensor_a();
  auto t_cs_b = smem.tensor_b();
  auto t_cr_a = cta_mma.make_fragment_A(t_cs_a);
  auto t_cr_b = cta_mma.make_fragment_B(t_cs_b);
  auto t_c_acc = cta_mma.make_fragment_C(t_cg_d);

  // The global inputs are already padded to the UMMA shape.  A cooperative
  // copy is deliberately outside the timed loop, just like a KDA tile's
  // staging operation is outside the repeated matrix products.
  cooperative_copy<kThreads>(threadIdx.x, t_cg_a(_, _, 0), t_cs_a);
  cooperative_copy<kThreads>(threadIdx.x, t_cg_b(_, _, 0), t_cs_b);
  __syncthreads();

  uint32_t one_warp = (threadIdx.x / 32 == 0);
  uint32_t one_thread = cute::elect_one_sync();
  using Allocator = cute::TMEM::Allocator1Sm;
  Allocator allocator{};

  if (one_warp) {
    allocator.allocate(Allocator::Sm100TmemCapacityColumns,
                       &smem.tmem_base);
  }
  __syncthreads();
  t_c_acc.data() = smem.tmem_base;

  if (one_warp && one_thread) {
    cute::initialize_barrier(smem.mma_barrier, 1);
  }
  __syncthreads();

  int phase = 0;
  TiledMMA mma = TiledMMA{};
  mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int r = 0; r < repeats; ++r) {
    if (one_warp) {
      // K=16 gives one descriptor in the K mode.
      gemm(mma, t_cr_a(_, _, 0), t_cr_b(_, _, 0), t_c_acc);
      mma.accumulate_ = UMMA::ScaleOut::One;
      cutlass::arch::umma_arrive(&smem.mma_barrier);
    }
    cute::wait_barrier(smem.mma_barrier, phase);
    phase ^= 1;
  }

  // One epilogue is enough for correctness.  It is outside the timed work
  // when repeats is large and also gives us a direct result to compare.
  TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp32b1x{}, t_c_acc);
  ThrCopy thr_t2r = t2r.get_slice(threadIdx.x);
  Tensor t_src = thr_t2r.partition_S(t_c_acc);
  Tensor t_dst = thr_t2r.partition_D(t_cg_d);
  Tensor t_reg = make_fragment_like(t_dst);
  copy(t2r, t_src, t_reg);
  copy(t_reg, t_dst);
  __syncthreads();

  if (one_warp) {
    allocator.release_allocation_lock();
    allocator.free(smem.tmem_base, Allocator::Sm100TmemCapacityColumns);
  }
}

// SM80 reference instruction: two m16n8k16 instructions make one 16x16
// product.  The helper is written explicitly here so the SASS can be checked
// for HMMA.16816.F32.BF16 independently of FlashKDA's large kernel.
template <class LayoutA, class LayoutB, class LayoutC>
__global__ void mma_sync_kernel(BF16 const* g_a, BF16 const* g_b, BF16* g_c,
                                 int repeats) {
  extern __shared__ char smem_raw[];
  BF16* s_a = reinterpret_cast<BF16*>(smem_raw);
  BF16* s_b = s_a + kLogicalM * kK;
  BF16* s_c = s_b + kLogicalN * kK;

  using GLayout = Layout<Shape<Int<kLogicalM>, Int<kK>>,
                         Stride<Int<kK>, Int<1>>>;
  auto g_a_tensor = make_tensor(make_gmem_ptr(g_a), GLayout{});
  auto g_b_tensor = make_tensor(make_gmem_ptr(g_b), GLayout{});
  auto s_a_tensor = make_tensor(make_smem_ptr(s_a), LayoutA{});
  auto s_b_tensor = make_tensor(make_smem_ptr(s_b), LayoutB{});
  cooperative_copy<kThreads>(threadIdx.x, g_a_tensor, s_a_tensor);
  cooperative_copy<kThreads>(threadIdx.x, g_b_tensor, s_b_tensor);
  __syncthreads();

  // The fragment layouts and the shared-memory load path are the same ones
  // used by FlashKDA.  Load A/B once, then time only register MMA calls; this
  // avoids charging the SM80 comparison for a repeated LDSM/stmatrix that is
  // not part of the instruction-level question.
  auto a = make_tensor(make_smem_ptr(s_a), LayoutA{});
  auto b = make_tensor(make_smem_ptr(s_b), LayoutB{});
  auto c = make_tensor(make_smem_ptr(s_c), LayoutC{});
  if (threadIdx.x < 32) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_1, _1>>{},
        Tile<_16, _16, _16>{});
    int tid = threadIdx.x;
    auto thr_mma = mma.get_slice(tid);
    auto t_cr_a = thr_mma.partition_fragment_A(a);
    auto t_cr_b = thr_mma.partition_fragment_B(b);
    auto t_cs_c = thr_mma.partition_C(c);
    auto t_cr_c = thr_mma.make_fragment_C(t_cs_c);

    auto copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_copy_a = copy_a.get_thread_slice(tid);
    auto thr_copy_b = copy_b.get_thread_slice(tid);
    auto t_cs_a = thr_copy_a.partition_S(a);
    auto t_cs_b = thr_copy_b.partition_S(b);
    auto t_cr_a_copy = make_fragment_like<BF16>(t_cr_a);
    auto t_cr_b_copy = make_fragment_like<BF16>(t_cr_b);
    auto t_cr_a_view = thr_copy_a.retile_D(t_cr_a_copy);
    auto t_cr_b_view = thr_copy_b.retile_D(t_cr_b_copy);
    copy(copy_a, t_cs_a(_, _, Int<0>{}), t_cr_a_view(_, _, Int<0>{}));
    copy(copy_b, t_cs_b(_, _, Int<0>{}), t_cr_b_view(_, _, Int<0>{}));
    transform(t_cr_a_copy, t_cr_a, cute::identity{});
    transform(t_cr_b_copy, t_cr_b, cute::identity{});
    clear(t_cr_c);

    for (int r = 0; r < repeats; ++r) {
      gemm(thr_mma, t_cr_a(_, _, Int<0>{}), t_cr_b(_, _, Int<0>{}), t_cr_c);
    }

    auto store_c = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto thr_store_c = store_c.get_thread_slice(tid);
    auto t_cs_c_dst = thr_store_c.partition_D(c);
    auto t_cr_c_bf16 = make_fragment_like<BF16>(t_cr_c);
    auto store_bf16 = [] __device__(float x) { return BF16(x); };
    transform(t_cr_c, t_cr_c_bf16, store_bf16);
    auto t_cr_c_view = thr_store_c.retile_S(t_cr_c_bf16);
    copy(store_c, t_cr_c_view, t_cs_c_dst);
  }
  __syncthreads();
  auto s_c_tensor = make_tensor(make_smem_ptr(s_c), LayoutC{});
  for (int i = threadIdx.x; i < kLogicalM * kLogicalN; i += blockDim.x) {
    int m = i / kLogicalN;
    int n = i % kLogicalN;
    g_c[i] = s_c_tensor(m, n);
  }
}

template <class Kernel, class... Args>
static float time_kernel(Kernel kernel, dim3 grid, dim3 block, size_t smem,
                         int repeats, Args... args) {
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // A launch contains `repeats` MMAs, so launch enough work to amortize event
  // resolution while keeping this experiment short.
  int launches = 20;
  kernel<<<grid, block, smem>>>(args..., repeats);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i)
    kernel<<<grid, block, smem>>>(args..., repeats);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / launches / repeats * 1.0e6f;
}

int main(int argc, char** argv) {
  int repeats = argc > 1 ? std::atoi(argv[1]) : 200;
  if (repeats < 1) repeats = 1;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d repeats=%d\n", prop.name, prop.major,
              prop.minor, repeats);
  if (prop.major != 10) {
    std::fprintf(stderr, "This experiment must run on SM100/SM103.\n");
    return 3;
  }

  std::vector<BF16> h_a(kTcgenM * kK), h_b(kLogicalN * kK);
  for (int m = 0; m < kTcgenM; ++m)
    for (int k = 0; k < kK; ++k)
      h_a[m * kK + k] = BF16(float((m + 2 * k) % 7 - 3));
  for (int n = 0; n < kLogicalN; ++n)
    for (int k = 0; k < kK; ++k)
      h_b[n * kK + k] = BF16(float((n + 3 * k) % 5 - 2));

  BF16 *d_a = nullptr, *d_b = nullptr, *d_c_mma = nullptr;
  float* d_c_tc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_c_mma, kLogicalM * kLogicalN * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&d_c_tc, kTcgenM * kLogicalN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(BF16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(BF16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c_mma, 0, kLogicalM * kLogicalN * sizeof(BF16)));
  CUDA_CHECK(cudaMemset(d_c_tc, 0, kTcgenM * kLogicalN * sizeof(float)));

  using ALayoutTc = Layout<Shape<Int<kTcgenM>, Int<kK>>,
                           Stride<Int<kK>, Int<1>>>;
  using BLayoutTc = Layout<Shape<Int<kLogicalN>, Int<kK>>,
                           Stride<Int<kK>, Int<1>>>;
  using DLayoutTc = Layout<Shape<Int<kTcgenM>, Int<kLogicalN>>,
                           Stride<Int<kLogicalN>, Int<1>>>;
  auto tiled_mma_obj = make_tiled_mma(
      SM100_MMA_F16BF16_SS<BF16, BF16, float, kTcgenM, kLogicalN,
                           UMMA::Major::K, UMMA::Major::K>{});
  auto b_m = tile_size<0>(tiled_mma_obj);
  auto b_n = tile_size<1>(tiled_mma_obj);
  auto b_k = tile_size<2>(tiled_mma_obj);
  auto mma_tiler = make_shape(b_m, b_n, b_k);
  auto mma_shape_a = partition_shape_A(
      tiled_mma_obj, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_b = partition_shape_B(
      tiled_mma_obj, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  // B32 is sufficient for K=16 and avoids adding a fake K=64 loop.  The
  // unavoidable mismatch for this experiment is therefore only M=16 -> 64.
  auto s_layout_a = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, mma_shape_a);
  auto s_layout_b = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW32_Atom<BF16>{}, mma_shape_b);
  using TiledMMA = decltype(tiled_mma_obj);
  using SLayoutA = decltype(s_layout_a);
  using SLayoutB = decltype(s_layout_b);
  using SharedTc = TcShared<BF16, BF16, SLayoutA, SLayoutB>;

  auto tc_kernel = tcgen05_kernel<SharedTc, TiledMMA, ALayoutTc, BLayoutTc,
                                  DLayoutTc>;
  CUDA_CHECK(cudaFuncSetAttribute(
      tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(SharedTc)));

  using ALayoutMma = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{},
      make_shape(Int<kLogicalM>{}, Int<kK>{}), LayoutLeft{}));
  using BLayoutMma = ALayoutMma;
  using CLayoutMma = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{},
      make_shape(Int<kLogicalM>{}, Int<kLogicalN>{}), LayoutLeft{}));
  auto mma_kernel = mma_sync_kernel<ALayoutMma, BLayoutMma, CLayoutMma>;
  size_t mma_smem = (kLogicalM * kK + kLogicalN * kK +
                     kLogicalM * kLogicalN) * sizeof(BF16);

  // Correctness: only the first 16 rows of the padded UMMA output matter.
  tc_kernel<<<1, kThreads, sizeof(SharedTc)>>>(d_a, d_b, d_c_tc, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  mma_kernel<<<1, kThreads, mma_smem>>>(d_a, d_b, d_c_mma, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got_tc(kTcgenM * kLogicalN);
  std::vector<BF16> got_mma(kLogicalM * kLogicalN);
  CUDA_CHECK(cudaMemcpy(got_tc.data(), d_c_tc,
                        got_tc.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_mma.data(), d_c_mma,
                        got_mma.size() * sizeof(BF16), cudaMemcpyDeviceToHost));

  int bad_tc = 0, bad_mma = 0;
  double max_diff = 0.0;
  for (int m = 0; m < kLogicalM; ++m) {
    for (int n = 0; n < kLogicalN; ++n) {
      float ref = 0.f;
      for (int k = 0; k < kK; ++k)
        ref += float(h_a[m * kK + k]) * float(h_b[n * kK + k]);
      float tc = got_tc[m * kLogicalN + n];
      float mm = float(got_mma[m * kLogicalN + n]);
      if (std::fabs(tc - ref) > 1e-4f) ++bad_tc;
      if (std::fabs(mm - ref) > 0.25f) ++bad_mma;
      max_diff = std::max(max_diff, double(std::fabs(tc - mm)));
    }
  }

  float ns_tc = time_kernel(tc_kernel, 1, kThreads, sizeof(SharedTc), repeats,
                            d_a, d_b, d_c_tc);
  float ns_mma = time_kernel(mma_kernel, 1, kThreads, mma_smem, repeats, d_a,
                             d_b, d_c_mma);
  std::printf("tcgen05_64x16x16: %.2f ns / padded MMA, %.2f ns / logical 16x16\n",
              ns_tc, ns_tc / 4.0f);
  std::printf("mma_sync_16x16:   %.2f ns / logical 16x16\n", ns_mma);
  std::printf("tcgen05/mma raw=%.3fx useful-work-normalized=%.3fx\n",
              ns_tc / ns_mma, (ns_tc / 4.0f) / ns_mma);
  std::printf("correctness: tcgen05 bad=%d, mma_sync bad=%d, max(tc-mma)=%.3g\n",
              bad_tc, bad_mma, max_diff);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c_mma));
  CUDA_CHECK(cudaFree(d_c_tc));
  return bad_tc || bad_mma ? 1 : 0;
}
