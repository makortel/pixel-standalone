#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "input.h"
#include "output.h"

#include <Kokkos_Core.hpp>
#include "rawtodigi_kokkos.h"

namespace {
  constexpr int NLOOPS = 100;
}

namespace gpuClustering {
  constexpr uint32_t MaxNumModules  = 2000;
  constexpr uint16_t InvId          = 9999;         // must be > MaxNumModules
}

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

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  int totaltime = 0;

  std::unique_ptr<Output> output;
  for(int i=0; i<NLOOPS; ++i) {
    output = std::make_unique<Output>();

    const auto wordCounter = input.wordCounter;

#ifdef DIGI_KOKKOS_CUDA
    // Rather non-idiomatic use of Kokkos::View...
    Kokkos::View<Input, Kokkos::CudaSpace> input_d{"input_d"};
    Kokkos::View<Input, Kokkos::CudaSpace>::HostMirror input_h = Kokkos::create_mirror_view(input_d);
    std::memcpy(input_h.data(), &input, sizeof(Input));

    Kokkos::View<Output, Kokkos::CudaSpace> output_d{"output_d"};
    Kokkos::View<Output, Kokkos::CudaSpace>::HostMirror output_h = Kokkos::create_mirror_view(output_d);
    output_h(0).err.construct(pixelgpudetails::MAX_FED_WORDS, output_d.data()->err_d);

    auto start = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(input_d, input_h);

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0, input.wordCounter),
                         KOKKOS_LAMBDA (const size_t i) {
                           kokkos::rawtodigi(input_d, output_d, wordCounter,
                                             true, true, false, i);
      });
    Kokkos::fence(); // I don't know if parallel_for is synchronous or not
    Kokkos::deep_copy(output_h, output_d);
    Kokkos::fence();

    auto stop = std::chrono::high_resolution_clock::now();

    output_h(0).err.set_data(output_h(0).err_d);
    std::memcpy(output.get(), output_h.data(), sizeof(Output));
    output->err.set_data(output->err_d);

#else // DIGI_KOKKOS_CUDA
    // Rather non-idiomatic use of Kokkos::View...
    Kokkos::View<Input, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> input_h{&input};
    Kokkos::View<Output, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> output_h{output.get()};

    auto start = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_for(Kokkos::RangePolicy<
#ifdef DIGI_KOKKOS_SERIAL
                         Kokkos::Serial
#elif defined DIGI_KOKKOS_OPENMP
                         Kokkos::OpenMP
#endif // DIGI_KOKKOS_SERIAL
                         >(0, input.wordCounter),
                         KOKKOS_LAMBDA (const size_t i) {
                           kokkos::rawtodigi(input_h, output_h, wordCounter,
                                             true, true, false, i);
      });
    Kokkos::fence();
    auto stop = std::chrono::high_resolution_clock::now();
#endif // DIGI_KOKKOS_CUDA

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
  }
  
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime)/NLOOPS) << " us"
            << std::endl;

  Kokkos::finalize();
  
  return 0;
}
