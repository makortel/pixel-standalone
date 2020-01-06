#ifndef output_h_
#define output_h_

#ifdef DIGI_NAIVE
#include <vector>
#elif defined DIGI_CUDA || defined DIGI_ALPAKA || defined DIGI_CUPLA || defined DIGI_KOKKOS || defined DIGI_ONEAPI
#include "GPUSimpleVector.h"
#endif

#include "pixelgpudetails.h"

struct alignas(128) Output {
  uint16_t xx[pixelgpudetails::MAX_FED_WORDS];
  uint16_t yy[pixelgpudetails::MAX_FED_WORDS];
  uint16_t adc[pixelgpudetails::MAX_FED_WORDS];
  uint16_t moduleInd[pixelgpudetails::MAX_FED_WORDS];
  uint16_t clus[pixelgpudetails::MAX_FED_WORDS];
  uint32_t digi[pixelgpudetails::MAX_FED_WORDS];
  uint32_t rawIdArr[pixelgpudetails::MAX_FED_WORDS];

#ifdef DIGI_NAIVE
  std::vector<PixelErrorCompact> err;
#elif defined DIGI_CUDA || defined DIGI_ALPAKA || defined DIGI_CUPLA || defined DIGI_KOKKOS || defined DIGI_ONEAPI
  PixelErrorCompact err_d[pixelgpudetails::MAX_FED_WORDS];
  GPU::SimpleVector<PixelErrorCompact> err;
#endif
};

#endif  // output_h_
