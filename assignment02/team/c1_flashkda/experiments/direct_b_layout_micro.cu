#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cutlass/bfloat16.h>

#include <cstdio>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

template <int V>
using UForBLayout = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<16>{}, Int<V>{}), LayoutLeft{}));

template <int V>
__global__ void direct_b_test(uint32_t* result) {
  extern __shared__ __align__(128) unsigned char smem[];
  auto mma = make_tiled_mma(
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
      Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
  int lane = threadIdx.x;
  auto thr_mma = mma.get_slice(lane);

  auto ub = make_tensor(make_smem_ptr(reinterpret_cast<BF16*>(smem)), UForBLayout<V>{});
  auto ub_tile = local_tile(ub, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
  auto b_ref = thr_mma.partition_fragment_B(ub_tile);
  auto b_store = b_ref;

  // Give every B register a distinct bit pattern.  If the B-layout shared
  // publication and the ldmatrix B load agree, the four uint32 values must
  // survive this round trip exactly.
  uint32_t* b_store_ptr = reinterpret_cast<uint32_t*>(&b_store(0));
  for (int i = 0; i < 4; ++i) {
    b_store_ptr[i] = 0x10000000u + uint32_t(lane * 0x100 + i * 0x11 + 7);
  }

  auto store_b = make_tiled_copy_B(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
  auto thr_store_b = store_b.get_thread_slice(lane);
  copy(store_b, thr_store_b.retile_S(b_store), thr_store_b.partition_D(ub_tile));
  __syncwarp();

  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
  auto thr_copy_b = copy_b.get_thread_slice(lane);
  auto b_loaded = make_fragment_like<BF16>(b_ref);
  auto b_view = thr_copy_b.retile_D(b_loaded);
  copy(copy_b, thr_copy_b.partition_S(ub_tile), b_view);
  __syncwarp();

  uint32_t* b_loaded_ptr = reinterpret_cast<uint32_t*>(&b_loaded(0));
  for (int i = 0; i < 4; ++i) {
    result[lane * 8 + i] = b_store_ptr[i];
    result[lane * 8 + 4 + i] = b_loaded_ptr[i];
  }
}

int main() {
  constexpr int V = 32;
  std::vector<uint32_t> result(32 * 8);
  uint32_t* device_result = nullptr;
  cudaMalloc(&device_result, result.size() * sizeof(uint32_t));
  direct_b_test<V><<<1, 32, 2048>>>(device_result);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
    return 2;
  }
  cudaMemcpy(result.data(), device_result, result.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int mismatches = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int i = 0; i < 4; ++i) {
      if (result[lane * 8 + i] != result[lane * 8 + 4 + i]) ++mismatches;
    }
  }
  std::printf("direct B STSM/LDSM round-trip mismatched uint32 registers=%d\n", mismatches);
  cudaFree(device_result);
  return mismatches == 0 ? 0 : 1;
}
