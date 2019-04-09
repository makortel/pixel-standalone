#ifndef RAWTODIGI_CUDA_H
#define RAWTODIGI_CUDA_H

#include <cuda_runtime.h>

#include "pixelgpudetails.h"
#include "GPUSimpleVector.h"
#include "input.h"
#include "output.h"

namespace cuda {
  void rawtodigi(const Input *input_d, Output *output_d,
                 const uint32_t wordCounter,
                 bool useQualityInfo, bool includeErrors, bool debug, cudaStream_t stream);
}

#endif
