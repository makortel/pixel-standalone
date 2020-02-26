#ifndef rawtodigi_kokkos_h_
#define rawtodigi_kokkos_h_

#include <Kokkos_Macros.hpp>

class Input;
class Output;

namespace kokkos {
  KOKKOS_FUNCTION void rawtodigi(const Input* input,
                                 Output* output,
                                 const uint32_t wordCounter,
                                 bool useQualityInfo,
                                 bool includeErrors,
                                 bool debug,
                                 const int32_t index);
}  // end namespace kokkos

#endif  // rawtodigi_kokkos_h_
