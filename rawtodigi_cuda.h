#ifndef rawtodigi_cuda_h_
#define rawtodigi_cuda_h_

#include <cuda_runtime.h>

#include "GPUSimpleVector.h"
#include "input.h"
#include "output.h"
#include "pixelgpudetails.h"

namespace cuda {

  void rawtodigi(const Input *input_d,
                 Output *output_d,
                 const uint32_t wordCounter,
                 bool useQualityInfo,
                 bool includeErrors,
                 bool debug,
                 cudaStream_t stream);

}  // namespace cuda

#endif  // rawtodigi_cuda_h_
