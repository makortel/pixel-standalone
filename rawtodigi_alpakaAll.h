#ifndef rawtodigi_alpakaAll_h
#define rawtodigi_alpakaAll_h

#include "alpakaConfig.h"
#include "input.h"

namespace gpuClustering {
  constexpr uint32_t MaxNumModules = 2000;
  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules
}  // namespace gpuClustering

namespace {
  int countModules(const uint16_t* id, int numElements) {
    int nmod = 0;
    for (int i = 0; i < numElements; ++i) {
      if (id[i] == gpuClustering::InvId)
        continue;
      auto j = i - 1;
      while (j >= 0 and id[j] == gpuClustering::InvId) {
        --j;
      }
      if (j < 0 or id[j] != id[i]) {
        ++nmod;
      }
    }
    return nmod;
  }
}  // namespace

namespace Alpaka {
  namespace CPU_SERIAL {
    void rawtodigi(Input const& input);
  }

  namespace CPU_TBB {
    void rawtodigi(Input const& input);
  }

  namespace GPU_CUDA {
    void rawtodigi(Input const& input);
  }
}  // namespace Alpaka

#endif  // rawtodigi_alpakaAll_h
