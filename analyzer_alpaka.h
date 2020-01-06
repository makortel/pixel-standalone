#ifndef analyzer_alpaka_h
#define analyzer_alpaka_h

#include "alpakaConfig.h"

class Input;
class Output;

namespace Alpaka {
  namespace CPU_SERIAL {
    void analyze(Input const& input, Output& output, double& totaltime);
  }

  namespace CPU_TBB {
    void analyze(Input const& input, Output& output, double& totaltime);
  }

  namespace GPU_CUDA {
    void analyze(Input const& input, Output& output, double& totaltime);
  }
}  // namespace Alpaka

#endif  // analyzer_alpaka_h
