#ifndef rawtodigi_cupla_h_
#define rawtodigi_cupla_h_

/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */
#include <cuda_to_cupla.hpp>

#include "GPUSimpleVector.h"
#include "input.h"
#include "output.h"
#include "pixelgpudetails.h"

namespace cupla {

  void rawtodigi(const Input *input_d,
                 Output *output_d,
                 const uint32_t wordCounter,
                 bool useQualityInfo,
                 bool includeErrors,
                 bool debug,
                 cudaStream_t stream);

}  // namespace cupla

#endif  // rawtodigi_cupla_h_
