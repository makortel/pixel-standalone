#ifndef rawtodigi_oneapi_h_
#define rawtodigi_oneapi_h_

#include <CL/sycl.hpp>

#include "GPUSimpleVector.h"
#include "input.h"
#include "output.h"
#include "pixelgpudetails.h"

namespace oneapi {

  void rawtodigi(const Input *input_d,
                 Output *output_d,
                 const uint32_t wordCounter,
                 bool useQualityInfo,
                 bool includeErrors,
                 bool debug,
                 cl::sycl::queue queue);

}  // namespace oneapi

#endif  // rawtodigi_oneapi_h_
