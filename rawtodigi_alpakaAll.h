#ifndef RAWTODIGI_ALPAKAALL_h
#define RAWTODIGI_ALPAKAALL_h

#include <alpaka/alpaka.hpp>

namespace {
  constexpr int NLOOPS = 100;
}

namespace gpuClustering {
  constexpr uint32_t MaxNumModules  = 2000;
  constexpr uint16_t InvId          = 9999;         // must be > MaxNumModules
}

namespace {
  int countModules(const uint16_t *id, int numElements) {
  int nmod = 0;
  for(int i=0; i<numElements; ++i) {
    if(id[i] == gpuClustering::InvId)
      continue;
    auto j = i-1;
    while(j >= 0 and id[j] == gpuClustering::InvId) {
      --j;
    }
    if(j < 0 or id[j] != id[i]) {
      ++nmod;
    }
  }
  return nmod;
  }  
}

namespace Alpaka {
  namespace CPU_SERIAL{
  void rawtodigi();
  }
}

namespace Alpaka{
    namespace CPU_TBB{
      void rawtodigi();
    }
}

namespace Alpaka{
    namespace GPU_CUDA{
      void rawtodigi();
    }
}


#endif // RAWTODIGI_ALPAKAALL_h
