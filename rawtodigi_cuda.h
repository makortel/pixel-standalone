#ifndef rawtodigi_cuda_h_
#define rawtodigi_cuda_h_

#include <cuda_runtime.h>

class Input;
class Output;

namespace cuda {

  __global__ void rawtodigi_kernel(
      const Input* input, Output* output, bool useQualityInfo, bool includeErrors, bool debug);

}  // namespace cuda

#endif  // rawtodigi_cuda_h_
