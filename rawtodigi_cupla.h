#ifndef rawtodigi_cupla_h_
#define rawtodigi_cupla_h_

/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */
#include <cuda_to_cupla.hpp>

class Input;
class Output;

namespace CUPLA_ACCELERATOR_NAMESPACE {

  struct rawtodigi_kernel {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc,
                                  const Input* input,
                                  Output* output,
                                  bool useQualityInfo,
                                  bool includeErrors,
                                  bool debug) const;
  };

  // explicit template instantiation declaration for cupla::CUPLA_ACCELERATOR_NAMESPACE::Acc
  extern template ALPAKA_FN_ACC void rawtodigi_kernel::operator()(cupla::CUPLA_ACCELERATOR_NAMESPACE::Acc const& acc,
                                                                  const Input* input,
                                                                  Output* output,
                                                                  bool useQualityInfo,
                                                                  bool includeErrors,
                                                                  bool debug) const;

}  // namespace CUPLA_ACCELERATOR_NAMESPACE

#endif  // rawtodigi_cupla_h_
