#ifndef RAWTODIGI_ALPAKA_H
#define RAWTODIGI_ALPAKA_H

#include "pixelgpudetails.h"
#include "input.h"
#include "output.h"

#include "alpakaConfig.h"

namespace ALPAKA_ARCHITECTURE{

  namespace Alpaka {
    
    void rawtodigi(const Input *input_d, Output *output_d,
                  const uint32_t wordCounter,
                  bool useQualityInfo, bool includeErrors, bool debug, Queue queue);
  }; //end alpaka
}; // end ALPAKA_ARCHITECTURE
#endif
