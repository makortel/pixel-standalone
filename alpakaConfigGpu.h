#ifndef AlpakaConfigGpu_h
#define AlpakaConfigGpu_h
#include <alpaka/standalone/GpuCudaRt.hpp>
#include <alpaka/alpaka.hpp>

template <uint64_t TDim>
struct GpuCuda {
  using Dim = alpaka::dim::DimInt<TDim>;
  using Idx = uint64_t;
  using Extent = uint64_t;
  using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
  using DevHost = alpaka::dev::DevCpu;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using QueueAsync = alpaka::queue::QueueCudaRtAsync;
  using QueueSync = alpaka::queue::QueueCudaRtSync;
  using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
  using Vec = alpaka::vec::Vec<Dim, Idx>;
};

#endif