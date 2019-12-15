#ifndef RAWTODIGI_ALPAKASERIAL_H
#define RAWTODIGI_ALPAKASERIAL_H

/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */

#include "alpakaConfig.h"
#include "pixelgpudetails.h"
#include "GPUSimpleVector.h"
#include "input.h"
#include "output.h"


namespace alpaka {
  
  void rawtodigi(const Input *input_d, Output *output_d,
                const uint32_t wordCounter,
                bool useQualityInfo, bool includeErrors, bool debug, CpuSerial<1u>::QueueSync queue);
}

#endif
