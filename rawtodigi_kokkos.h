#ifndef rawtodigi_kokkos_h_
#define rawtodigi_kokkos_h_

#include <Kokkos_Macros.hpp>

#include "kokkosConfig.h"

class Input;
class Output;

namespace KOKKOS_NAMESPACE {
  KOKKOS_FUNCTION void rawtodigi(const Input* input,
                                 Output* output,
                                 const uint32_t wordCounter,
                                 bool useQualityInfo,
                                 bool includeErrors,
                                 bool debug,
                                 const int32_t index);
}  // namespace KOKKOS_NAMESPACE

#endif  // rawtodigi_kokkos_h_
