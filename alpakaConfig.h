#ifndef AlpakaConfig_h
#define AlpakaConfig_h

#include <alpaka/alpaka.hpp>

namespace Alpaka {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define ALPAKA_ARCHITECTURE GPU_CUDA
  namespace GPU_CUDA {
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
  } // namespace GPU_CUDA
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define ALPAKA_ARCHITECTURE CPU_SERIAL
  namespace CPU_SERIAL {
    using Dim = alpaka::dim::DimInt<1u>;
    using Idx = uint64_t;
    using Extent = uint64_t;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using DevHost = alpaka::dev::DevCpu;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::QueueCpuBlocking;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    using Vec = alpaka::vec::Vec<Dim, Idx>;
  }  // namespace CPU_SERIAL
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#define ALPAKA_ARCHITECTURE CPU_TBB
  namespace CPU_TBB {
    using Dim = alpaka::dim::DimInt<1u>;
    using Idx = uint64_t;
    using Extent = uint64_t;
    using Acc = alpaka::acc::AccCpuTbbBlocks<Dim, Extent>;
    using DevHost = alpaka::dev::DevCpu;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::QueueCpuNonBlocking;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    using Vec = alpaka::vec::Vec<Dim, Idx>;
  }  // namespace CPU_TBB
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

}  // namespace Alpaka

#endif  // AlpakaConfig_h
