#define ALPAKA_ARCHITECTURE GPU_CUDA

#include <alpaka/standalone/GpuCudaRt.hpp>
#include <alpaka/alpaka.hpp>

using Dim = alpaka::dim::DimInt<1u>;
using Idx = uint64_t;
using Extent = uint64_t;
using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
using DevHost = alpaka::dev::DevCpu;
using DevAcc = alpaka::dev::Dev<Acc>;
using PltfHost = alpaka::pltf::Pltf<DevHost>;
using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
using Queue = alpaka::queue::QueueCudaRtNonBlocking;
using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
using Vec = alpaka::vec::Vec<Dim, Idx>;

#include "rawtodigi_alpakaAll.cc"