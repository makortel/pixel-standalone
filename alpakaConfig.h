#ifndef alpakaConfig_h_
#define alpakaConfig_h_

#include <alpaka/alpaka.hpp>

namespace alpaka_common {
  using Dim = alpaka::dim::DimInt<1u>;
  using Idx = uint32_t;
  using Extent = uint32_t;
  using DevHost = alpaka::dev::DevCpu;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
  using Vec = alpaka::vec::Vec<Dim, Idx>;
}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define ALPAKA_ACC_GPU_CUDA_BACKEND
namespace alpaka_cuda_async {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCudaRtNonBlocking;
}  // namespace alpaka_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_BACKEND
#define ALPAKA_ARCHITECTURE alpaka_cuda_async
#endif  // ALPAKA_ACC_GPU_CUDA_BACKEND

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_BACKEND
namespace alpaka_serial_sync {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccCpuSerial<Dim, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCpuBlocking;
}  // namespace alpaka_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_BACKEND
#define ALPAKA_ARCHITECTURE alpaka_serial_sync
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_BACKEND

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_TBB_T_SEQ_BACKEND
namespace alpaka_tbb_async {
  using namespace alpaka_common;
  using Acc = alpaka::acc::AccCpuTbbBlocks<Dim, Extent>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCpuNonBlocking;
}  // namespace alpaka_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_BACKEND
#define ALPAKA_ARCHITECTURE alpaka_tbb_async
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_BACKEND

#endif  // alpakaConfig_h_
