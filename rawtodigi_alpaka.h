#ifndef rawtodigi_alpaka_h_
#define rawtodigi_alpaka_h_

#include "alpakaConfig.h"

class Input;
class Output;

namespace ALPAKA_ARCHITECTURE_NAMESPACE {

  struct rawtodigi_kernel {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const Input* input,
                                  Output* output,
                                  bool useQualityInfo,
                                  bool includeErrors,
                                  bool debug) const;
  };

  // explicit template instantiation declaration for ALPAKA_ACCELERATOR_NAMESPACE::Acc
  extern template ALPAKA_FN_ACC void rawtodigi_kernel::operator()(ALPAKA_ACCELERATOR_NAMESPACE::Acc const& acc,
                                                                  const Input* input,
                                                                  Output* output,
                                                                  bool useQualityInfo,
                                                                  bool includeErrors,
                                                                  bool debug) const;

}  // namespace ALPAKA_ARCHITECTURE_NAMESPACE

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace ALPAKA_ARCHITECTURE_NAMESPACE;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // rawtodigi_alpaka_h_
