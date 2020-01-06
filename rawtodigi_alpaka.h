#ifndef rawtodigi_alpaka_h
#define rawtodigi_alpaka_h

#include "pixelgpudetails.h"
#include "input.h"
#include "output.h"

#include "alpakaConfig.h"

namespace ALPAKA_ARCHITECTURE {

  void rawtodigi(const Input *input_d,
                 Output *output_d,
                 const uint32_t wordCounter,
                 bool useQualityInfo,
                 bool includeErrors,
                 bool debug,
                 Queue queue);

}  // namespace ALPAKA_ARCHITECTURE

#endif  // rawtodigi_alpaka_h
