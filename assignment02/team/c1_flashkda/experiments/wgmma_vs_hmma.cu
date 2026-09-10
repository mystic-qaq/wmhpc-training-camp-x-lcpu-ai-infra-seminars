// Compare a WGMMA 64x16x16 BF16 product against four independent
// SM80-HMMA 16x16x16 products.  The orientation matches the useful
// V-split K2 operation:
//
//     state[V,D] @ k_chunk[16,D]^T = result[V,16]
//
// For V=64, one WGMMA warpgroup can cover the four 16-row HMMA tiles in a
// single instruction per K=16 slice.  This is deliberately a standalone
// microbenchmark.  It answers whether the opcode/warpgroup machinery is
// promising before taking on the much larger K2 layout and epilogue rewrite.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// CUTLASS gates WGMMA on the SM90a feature macro.  CUDA 13's SM103 target
// does not define __CUDA_ARCH_FEAT_SM90_ALL even though B300 still exposes
// the compatible WGMMA instruction family.  Define the gate only during the
// device compilation pass; defining it on the host pass would break the
// host/device annotations in the CUTLASS headers.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define CUTE_ARCH_MMA_SM90A_ENABLED
#endif

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

constexpr int M = 64;
constexpr int N = 16;
constexpr int K = 128;
constexpr int Threads = 128;

static void check(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

#define CUDA_CHECK(x) check((x), #x)

using Wmma = SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
using WmmaTiled = decltype(make_tiled_mma(Wmma{}));
using WmmaALayout = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF16>{}, make_shape(Int<M>{}, Int<K>{})));
using WmmaBLayout = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<BF16>{}, make_shape(Int<N>{}, Int<K>{})));

struct WmmaShared {
  alignas(128) ArrayEngine<BF16, cosize_v<WmmaALayout>> a;
  alignas(128) ArrayEngine<BF16, cosize_v<WmmaBLayout>> b;

  CUTE_DEVICE auto tensor_a() { return make_tensor(make_smem_ptr(a.begin()), WmmaALayout{}); }
  CUTE_DEVICE auto tensor_b() { return make_tensor(make_smem_ptr(b.begin()), WmmaBLayout{}); }
};

template <class MmaKernel>
__global__ void wgmma_kernel(BF16 const* g_a, BF16 const* g_b, float* g_c,
                              int repeats) {
  extern __shared__ char raw[];
  auto& smem = *reinterpret_cast<WmmaShared*>(raw);

  auto g_a_tensor = make_tensor(
      make_gmem_ptr(g_a),
      Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  auto g_b_tensor = make_tensor(
      make_gmem_ptr(g_b),
      Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  cooperative_copy<Threads>(threadIdx.x, g_a_tensor, smem.tensor_a());
  cooperative_copy<Threads>(threadIdx.x, g_b_tensor, smem.tensor_b());
  __syncthreads();

  MmaKernel mma = MmaKernel{};
  auto thr_mma = mma.get_thread_slice(threadIdx.x);
  auto t_a = thr_mma.partition_A(smem.tensor_a());
  auto t_b = thr_mma.partition_B(smem.tensor_b());
  auto g_c_tensor = make_tensor(
      make_gmem_ptr(g_c),
      Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>{});
  auto t_c = thr_mma.partition_C(g_c_tensor);
  auto t_acc = thr_mma.make_fragment_C(t_c);
  clear(t_acc);

  for (int r = 0; r < repeats; ++r) {
    warpgroup_arrive();
    gemm(mma, t_a, t_b, t_acc);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
  }

  // Make the compiler keep the accumulator alive.  The timed steady state is
  // dominated by the repeated WGMMA operations above.
  if (threadIdx.x == 0) {
    g_c[0] = 1.0f;
  }
  __syncthreads();
}

using HmmaLayout = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<16>{}, Int<K>{}),
    LayoutLeft{}));
using HmmaCLayout = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<16>{}, Int<16>{}),
    LayoutLeft{}));

struct HmmaShared {
  alignas(128) ArrayEngine<BF16, 4 * cosize_v<HmmaLayout>> a;
  alignas(128) ArrayEngine<BF16, cosize_v<HmmaLayout>> b;

  CUTE_DEVICE auto tensor_a(int warp) {
    return make_tensor(make_smem_ptr(a.begin() + warp * cosize_v<HmmaLayout>),
                       HmmaLayout{});
  }
  CUTE_DEVICE auto tensor_b() { return make_tensor(make_smem_ptr(b.begin()), HmmaLayout{}); }
};

__global__ void hmma_kernel(BF16 const* g_a, BF16 const* g_b, float* g_c,
                            int repeats) {
  extern __shared__ char raw[];
  auto& smem = *reinterpret_cast<HmmaShared*>(raw);

  // Each warp owns one 16-row A tile; B is shared by all four warps.
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  auto g_a_tile = make_tensor(
      make_gmem_ptr(g_a + warp * 16 * K),
      Layout<Shape<Int<16>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  auto g_b_tensor = make_tensor(
      make_gmem_ptr(g_b),
      Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>{});
  if (warp < 4) {
    cooperative_copy<32>(lane, g_a_tile, smem.tensor_a(warp));
  }
  // Have all four warps cooperatively initialize the B tile, once in total.
  if (warp < 4) {
    auto b_src = make_tensor(
        make_gmem_ptr(g_b + (warp * 4) * K),
        Layout<Shape<Int<4>, Int<K>>, Stride<Int<K>, Int<1>>>{});
    auto b_dst = smem.tensor_b();
    for (int i = lane; i < N * K; i += 128) {
      b_dst(i) = g_b[i];
    }
  }
  __syncthreads();

  if (warp < 4) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    auto a = smem.tensor_a(warp);
    auto b = smem.tensor_b();
    auto thr = mma.get_slice(lane);
    auto a_ref0 = local_tile(a, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
    auto b_ref0 = local_tile(b, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
    using AFrag = decltype(thr.partition_fragment_A(a_ref0));
    using BFrag = decltype(thr.partition_fragment_B(b_ref0));
    AFrag a_frags[K / 16];
    BFrag b_frags[K / 16];
    auto c_dummy = make_tensor(
        make_smem_ptr(smem.a.begin()),
        HmmaCLayout{});
    auto c_frag = thr.make_fragment_C(thr.partition_C(c_dummy));
    auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_a = copy_a.get_thread_slice(lane);
    auto thr_b = copy_b.get_thread_slice(lane);
    // Match K2's setup: LDSM loads each 16x16 operand tile once, then the
    // timed loop reuses the register fragments for repeated HMMA products.
    for (int k = 0; k < K / 16; ++k) {
      auto a_ref = local_tile(a, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k));
      auto b_ref = local_tile(b, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k));
      a_frags[k] = thr.partition_fragment_A(a_ref);
      b_frags[k] = thr.partition_fragment_B(b_ref);
      auto a_copy = make_fragment_like<BF16>(a_frags[k]);
      auto b_copy = make_fragment_like<BF16>(b_frags[k]);
      copy(copy_a, thr_a.partition_S(a_ref), thr_a.retile_D(a_copy));
      copy(copy_b, thr_b.partition_S(b_ref), thr_b.retile_D(b_copy));
      transform(a_copy, a_frags[k], identity{});
      transform(b_copy, b_frags[k], identity{});
    }
    clear(c_frag);

    for (int r = 0; r < repeats; ++r) {
      // HMMA's atom has K=16.  Spell out the eight slices so that this
      // comparison has the same K=128 useful work as the WGMMA path.
      gemm(thr, a_frags[0](_, _, Int<0>{}), b_frags[0](_, _, Int<0>{}), c_frag);
      gemm(thr, a_frags[1](_, _, Int<0>{}), b_frags[1](_, _, Int<0>{}), c_frag);
      gemm(thr, a_frags[2](_, _, Int<0>{}), b_frags[2](_, _, Int<0>{}), c_frag);
      gemm(thr, a_frags[3](_, _, Int<0>{}), b_frags[3](_, _, Int<0>{}), c_frag);
      gemm(thr, a_frags[4](_, _, Int<0>{}), b_frags[4](_, _, Int<0>{}), c_frag);
      gemm(thr, a_frags[5](_, _, Int<0>{}), b_frags[5](_, _, Int<0>{}), c_frag);
      gemm(thr, a_frags[6](_, _, Int<0>{}), b_frags[6](_, _, Int<0>{}), c_frag);
      gemm(thr, a_frags[7](_, _, Int<0>{}), b_frags[7](_, _, Int<0>{}), c_frag);
    }
    if (lane == 0) {
      g_c[warp * N] = 1.0f;
    }
  }
  __syncthreads();
}

template <class Kernel>
float time_kernel(Kernel kernel, size_t smem, BF16* a, BF16* b, float* c,
                  int repeats, int launches) {
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  for (int i = 0; i < 10; ++i) kernel<<<1, Threads, smem>>>(a, b, c, repeats);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < launches; ++i) kernel<<<1, Threads, smem>>>(a, b, c, repeats);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1.e6f / (launches * repeats);
}

int main(int argc, char** argv) {
  int repeats = argc > 1 ? std::atoi(argv[1]) : 200;
  int launches = argc > 2 ? std::atoi(argv[2]) : 50;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d repeats=%d launches=%d\n",
              prop.name, prop.major, prop.minor, repeats, launches);
  if (prop.major != 10) return 3;

  std::vector<BF16> ha(M * K), hb(N * K);
  for (int i = 0; i < M * K; ++i) ha[i] = BF16(float((i * 3) % 11 - 5));
  for (int i = 0; i < N * K; ++i) hb[i] = BF16(float((i * 5) % 7 - 3));
  BF16 *a = nullptr, *b = nullptr;
  float* c = nullptr;
  CUDA_CHECK(cudaMalloc(&a, ha.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&b, hb.size() * sizeof(BF16)));
  CUDA_CHECK(cudaMalloc(&c, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(a, ha.data(), ha.size() * sizeof(BF16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, hb.data(), hb.size() * sizeof(BF16), cudaMemcpyHostToDevice));

  float wg = time_kernel(wgmma_kernel<WmmaTiled>, sizeof(WmmaShared), a, b, c,
                         repeats, launches);
  float hm = time_kernel(hmma_kernel, sizeof(HmmaShared), a, b, c,
                         repeats, launches);
  // Both kernels compute 64*16*128 multiply-adds per iteration.  The ratio is
  // therefore directly comparable; no padding correction is needed here.
  std::printf("shape=%dx%dx%d wgmma=%.3f ns/64x16 product hmma=%.3f ns/64x16 product ratio=%.3f\n",
              M, N, K, wg, hm, wg / hm);

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
  return 0;
}
