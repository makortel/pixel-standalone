#ifndef RAWTODIGI_ALPAKAGPU_H
#define RAWTODIGI_ALPAKAGPU_H

#define ALPAKA_ARCHITECTURE GPU_CUDA

#include "alpakaConfigGpu.h"

namespace GPU_CUDA{

    using Accelerator = GpuCuda<1u>;
    using Acc = Accelerator::Acc;
    using Dim = Accelerator::Dim;
    using Idx = Accelerator::Idx;
    using DevHost = Accelerator::DevHost;
    using DevAcc = Accelerator::DevAcc;
    using PltfHost = Accelerator::PltfHost;
    using PltfAcc = Accelerator::PltfAcc;
    using Extent = Accelerator::Extent;
    using QueueAsync = Accelerator::QueueAsync;
    using QueueSync = Accelerator::QueueSync;
    using Vec = Accelerator::Vec;
    using WorkDiv = Accelerator::WorkDiv;
};

#include "rawtodigi_alpaka.h"
// #undef ALPAKA_ARCHITECTURE
#endif