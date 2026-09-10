#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <iostream>

using namespace cute;

template <int V>
void dump_v() {
  using BF16 = cutlass::bfloat16_t;
  auto mma = make_tiled_mma(
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
      Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
  std::cout << "\nV=" << V << "\n";
  std::cout << "mma A TV: "; print(mma.get_layoutA_TV()); std::cout << "\n";
  std::cout << "mma B TV: "; print(mma.get_layoutB_TV()); std::cout << "\n";
  std::cout << "mma C TV: "; print(mma.get_layoutC_TV()); std::cout << "\n";

  auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
  auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
  auto copy_c = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
  auto store_c = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
  std::cout << "copy A tiled: "; print(decltype(copy_a)::TiledLayout_TV{}); std::cout << "\n";
  std::cout << "copy B tiled: "; print(decltype(copy_b)::TiledLayout_TV{}); std::cout << "\n";
  std::cout << "copy C tiled: "; print(decltype(copy_c)::TiledLayout_TV{}); std::cout << "\n";
  std::cout << "store C tiled: "; print(decltype(store_c)::TiledLayout_TV{}); std::cout << "\n";

  using VO = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<16>{}, Int<V>{}), LayoutLeft{}));
  using UForB = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{}, make_shape(Int<V>{}, Int<16>{}), LayoutLeft{}));
  std::cout << "VO: "; print(VO{}); std::cout << "\n";
  std::cout << "VO layout: "; print(VO{}.layout()); std::cout << "\n";
  std::cout << "VO offsets: " << VO{}(0,0) << "," << VO{}(0,1) << ","
            << VO{}(1,0) << "," << VO{}(15,15) << "\n";
  std::cout << "UForB layout: "; print(UForB{}.layout()); std::cout << "\n";

  auto vo = make_tensor(make_smem_ptr(static_cast<BF16*>(nullptr)), VO{});
  auto ub = make_tensor(make_smem_ptr(static_cast<BF16*>(nullptr)), UForB{});
  auto vo_tile = local_tile(vo, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
  auto ub_tile = local_tile(ub, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
  auto thr_mma = mma.get_slice(0);
  auto c_frag = thr_mma.make_fragment_C(thr_mma.partition_C(vo_tile));
  auto bf_frag = make_fragment_like<BF16>(c_frag);
  auto cs = store_c.get_slice(0);
  auto bs = copy_b.get_thread_slice(0);
  std::cout << "C frag: "; print(bf_frag.layout()); std::cout << "\n";
  std::cout << "store source: "; print(cs.retile_S(bf_frag).layout()); std::cout << "\n";
  std::cout << "store VO dest: "; print(cs.partition_D(vo_tile).layout()); std::cout << "\n";
  std::cout << "store UForB dest: "; print(cs.partition_D(ub_tile).layout()); std::cout << "\n";
  std::cout << "B VO source: "; print(bs.partition_S(vo_tile).layout()); std::cout << "\n";
  std::cout << "B UForB source: "; print(bs.partition_S(ub_tile).layout()); std::cout << "\n";
}

int main() {
  dump_v<16>();
  dump_v<32>();
  dump_v<64>();
  return 0;
}
