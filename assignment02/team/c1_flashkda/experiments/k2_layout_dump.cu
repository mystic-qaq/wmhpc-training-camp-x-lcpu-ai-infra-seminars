#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <iostream>

#include "../FlashKDA/csrc/smxx/fwd_kernel2.cuh"

using namespace cute;

template <int V>
void dump() {
  using BF16 = cutlass::bfloat16_t;
  using QK = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{},
      make_shape(Int<16>{}, Int<128>{}), LayoutLeft{}));
  using VO = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{},
      make_shape(Int<16>{}, Int<V>{}), LayoutLeft{}));
  using State = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<BF16>{},
      make_shape(Int<128>{}, Int<V>{}), LayoutLeft{}));
  using TState = decltype(tile_to_shape(
      GMMA::Layout_MN_INTER_Atom<BF16>{},
      make_shape(Int<V>{}, Int<128>{}), LayoutRight{}));
  using TMAQK = decltype(composition(
      QK{}.layout_a(), QK{}.offset(), prepend(QK{}.layout_b())));
  using TMAVO = decltype(composition(
      VO{}.layout_a(), VO{}.offset(), prepend(VO{}.layout_b())));
  using TMAState = decltype(composition(
      State{}.layout_a(), State{}.offset(), prepend(State{}.layout_b())));
  using TMATState = decltype(composition(
      TState{}.layout_a(), TState{}.offset(), prepend(TState{}.layout_b())));
  std::cout << "V=" << V << "\n";
  std::cout << "QK="; print(QK{}.layout()); std::cout << "\n";
  std::cout << "VO="; print(VO{}.layout()); std::cout << "\n";
  std::cout << "State="; print(State{}.layout()); std::cout << "\n";
  std::cout << "TState="; print(TState{}.layout()); std::cout << "\n";
  std::cout << "TMAQK="; print(TMAQK{}); std::cout << "\n";
  std::cout << "TMAVO="; print(TMAVO{}); std::cout << "\n";
  std::cout << "TMAState="; print(TMAState{}); std::cout << "\n";
  std::cout << "TMATState="; print(TMATState{}); std::cout << "\n";
  auto state = State{};
  auto tstate = TState{};
  std::cout << "State offsets (k,v): "
            << state(0, 0) << "," << state(0, 1) << ","
            << state(1, 0) << "," << state(16, 0) << ","
            << state(0, 16) << "\n";
  std::cout << "TState offsets (v,k): "
            << tstate(0, 0) << "," << tstate(0, 1) << ","
            << tstate(1, 0) << "," << tstate(16, 0) << ","
            << tstate(0, 16) << "\n";
}

int main() {
  dump<128>();
  dump<64>();
  dump<32>();
  return 0;
}
