  #include <alpaka/standalone/CpuSerial.hpp>
  #include <alpaka/alpaka.hpp>
  
  using Dim = alpaka::dim::DimInt<1u>;
  using Idx = std::size_t;
  using DevHost = alpaka::dev::DevCpu;
  using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using QueueAcc = alpaka::queue::QueueCpuSync;

  #include "rawtodigi_alpaka.cc"