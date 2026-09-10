#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <iostream>

using namespace cute;

template<class Op>
void dump_op(char const* name) {
  std::cout << name << " ValID=";
  print(typename Copy_Traits<Op>::ValID{});
  std::cout << " upcast=";
  print(upcast<32>(typename Copy_Traits<Op>::ValID{}));
  std::cout << "\n";
}

int main() {
  using Mma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<cutlass::bfloat16_t, cutlass::bfloat16_t,
                           float, 64, 16, UMMA::Major::K, UMMA::Major::K>{}));
  auto t = make_tensor<UMMA::tmem_frg_1sm<float>>(
      make_shape(make_shape(Int<64>{}, Int<16>{}), Int<1>{}, Int<1>{}));
  std::cout << "tmem layout=";
  print(t.layout());
  std::cout << "\ncoalesced=";
  print(coalesce(t.layout()));
  std::cout << "\n";
  dump_op<SM100_TMEM_LOAD_32dp32b1x>("32dp32b1x");
  dump_op<SM100_TMEM_LOAD_16dp32b1x>("16dp32b1x");
  dump_op<SM100_TMEM_LOAD_16dp32b2x>("16dp32b2x");
  dump_op<SM100_TMEM_LOAD_16dp32b4x>("16dp32b4x");
  dump_op<SM100_TMEM_LOAD_16dp128b1x>("16dp128b1x");
  dump_op<SM100_TMEM_LOAD_16dp256b1x>("16dp256b1x");
  return 0;
}
