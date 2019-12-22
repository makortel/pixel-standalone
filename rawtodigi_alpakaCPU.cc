#ifdef SERIAL
#define ALPAKA_ARCHITECTURE CPU_SERIAL

#include <alpaka/standalone/CpuSerial.hpp>
#include <alpaka/alpaka.hpp>

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

#elif defined TBB
#define ALPAKA_ARCHITECTURE CPU_TBB

#include <alpaka/standalone/CpuTbbBlocks.hpp>
#include <alpaka/alpaka.hpp>

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
#endif

#include "rawtodigi_alpakaAll.cc"
