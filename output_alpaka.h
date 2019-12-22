#ifndef OUTPUT_ALPAKA_H
#define OUTPUT_ALPAKA_H

#include <vector>

#include "pixelgpudetails.h"

#include "GPUSimpleVector_alpaka.h"


struct alignas(128) Output {
  uint16_t xx[pixelgpudetails::MAX_FED_WORDS];
  uint16_t yy[pixelgpudetails::MAX_FED_WORDS];
  uint16_t adc[pixelgpudetails::MAX_FED_WORDS];
  uint16_t moduleInd[pixelgpudetails::MAX_FED_WORDS];
  uint16_t clus[pixelgpudetails::MAX_FED_WORDS];
  uint32_t digi[pixelgpudetails::MAX_FED_WORDS];
  uint32_t rawIdArr[pixelgpudetails::MAX_FED_WORDS];

  PixelErrorCompact err_d[pixelgpudetails::MAX_FED_WORDS];

};

#endif
