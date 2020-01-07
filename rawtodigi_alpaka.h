#ifndef rawtodigi_alpaka_h_
#define rawtodigi_alpaka_h_

#include "alpakaConfig.h"
#include "input.h"
#include "output.h"
#include "pixelgpudetails.h"

namespace ALPAKA_ARCHITECTURE {

  struct rawtodigi_kernel {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const Input* input,
                                  Output* output,
                                  bool useQualityInfo,
                                  bool includeErrors,
                                  bool debug) const;
  };

  // explicit template instantiation declaration for ALPAKA_ARCHITECTURE::Acc
  extern template ALPAKA_FN_ACC void rawtodigi_kernel::operator()(
      Acc const& acc, const Input* input, Output* output, bool useQualityInfo, bool includeErrors, bool debug) const;

}  // namespace ALPAKA_ARCHITECTURE

#endif  // rawtodigi_alpaka_h_
