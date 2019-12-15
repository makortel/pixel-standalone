#ifndef RAWTODIGI_ALPAKASER_H
#define RAWTODIGI_ALPAKASER_H

#define ALPAKA_ARCHITECTURE CPU_SERIAL

#include "alpakaConfigSer.h"

namespace CPU_SERIAL{

    using Accelerator = CpuSerial<1u>;
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