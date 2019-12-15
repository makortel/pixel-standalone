#ifndef AlpakaConfigTbb_h
#define AlpakaConfigTbb_h

#include <alpaka/standalone/CpuTbbBlocks.hpp>
#include <alpaka/alpaka.hpp>

template <uint64_t TDim>
struct CpuParallelTBB {
  using Dim = alpaka::dim::DimInt<TDim>;
  using Idx = uint64_t;
  using Extent = uint64_t;
  using Acc = alpaka::acc::AccCpuTbbBlocks<Dim, Extent>;
  using DevHost = alpaka::dev::DevCpu;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using QueueAsync = alpaka::queue::QueueCpuAsync;
  using QueueSync = alpaka::queue::QueueCpuSync;
  using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
  using Vec = alpaka::vec::Vec<Dim, Idx>;
};

#endif