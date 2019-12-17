#ifndef RAWTODIGI_ALPAKA_H
#define RAWTODIGI_ALPAKA_H

#include "pixelgpudetails.h"
#include "input.h"
#include "output.h"

#include "alpakaConfig.h"

namespace Alpaka{

  namespace ALPAKA_ARCHITECTURE{
    
    void rawtodigi(const Input *input_d, Output *output_d,
                  const uint32_t wordCounter,
                  bool useQualityInfo, bool includeErrors, bool debug, Queue queue);
  }; //end ALPAKA_ARCHITECTURE
}; // end Alpaka 
#endif
