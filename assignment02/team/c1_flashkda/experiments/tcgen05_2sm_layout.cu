#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

int main() {
  using Atom = SM100_MMA_F16BF16_2x1SM_SS<BF16, BF16, float, 128, 16,
                                         UMMA::Major::K, UMMA::Major::K>;
  using TiledMMA = decltype(make_tiled_mma(Atom{}));
  TiledMMA mma{};
  auto shape_a = partition_shape_A(mma, make_shape(Int<128>{}, Int<16>{}));
  auto shape_b = partition_shape_B(mma, make_shape(Int<16>{}, Int<16>{}));
  auto shape_c = partition_shape_C(mma, make_shape(Int<128>{}, Int<16>{}));
  auto layout_a = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, shape_a);
  auto layout_b = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<BF16>{}, shape_b);
  auto full = make_tensor(make_gmem_ptr((BF16*)nullptr),
                          Layout<Shape<Int<128>, Int<16>>,
                                 Stride<Int<16>, Int<1>>>{});
  auto local = make_tensor(make_gmem_ptr((BF16*)nullptr),
                           Layout<Shape<Int<64>, Int<16>>,
                                  Stride<Int<16>, Int<1>>>{});
  auto slice0 = mma.get_slice(Int<0>{});
  auto slice1 = mma.get_slice(Int<1>{});
  auto bfull = make_tensor(make_gmem_ptr((BF16*)nullptr),
                           Layout<Shape<Int<16>, Int<16>>,
                                  Stride<Int<16>, Int<1>>>{});
  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
  auto cluster_layout = tiled_divide(make_layout(cluster_shape),
                                     make_tile(typename TiledMMA::AtomThrID{}));
  print("mma: "); print(mma); print("\n");
  print("shape_a: "); print(shape_a); print("\n");
  print("shape_b: "); print(shape_b); print("\n");
  print("shape_c: "); print(shape_c); print("\n");
  print("layout_a: "); print(layout_a); print("\n");
  print("layout_b: "); print(layout_b); print("\n");
  print("cosize a/b: "); print(cosize_v<decltype(layout_a)>); print("/"); print(cosize_v<decltype(layout_b)>); print("\n");
  print("full C slice0: "); print(slice0.partition_C(full)); print("\n");
  print("local C slice0: "); print(slice0.partition_C(local)); print("\n");
  print("full C slice1: "); print(slice1.partition_C(full)); print("\n");
  print("local C slice1: "); print(slice1.partition_C(local)); print("\n");
  print("full A slice0: "); print(slice0.partition_A(full)); print("\n");
  print("full A slice1: "); print(slice1.partition_A(full)); print("\n");
  print("full B slice0: "); print(slice0.partition_B(bfull)); print("\n");
  print("full B slice1: "); print(slice1.partition_B(bfull)); print("\n");
  print("cluster layout: "); print(cluster_layout); print(" sizes ");
  print(size<0>(cluster_layout)); print(","); print(size<1>(cluster_layout));
  print(","); print(size<2>(cluster_layout)); print(","); print(size<3>(cluster_layout)); print("\n");
  using HmmaU = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{},
      make_shape(Int<16>{}, Int<32>{}), LayoutLeft{}));
  print("hmma U layout: "); print(HmmaU{}); print(" cosize ");
  print(cosize_v<HmmaU>); print("\n");
  return 0;
}
