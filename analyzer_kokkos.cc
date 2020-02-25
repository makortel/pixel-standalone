#include <Kokkos_Core.hpp>

#include "analyzer_kokkos.h"
#include "input.h"
#include "output.h"
#include "rawtodigi_kokkos.h"

namespace {
  constexpr int NLOOPS = 100;
}

#ifdef DIGI_KOKKOS_SERIAL
using KokkosExecSpace = Kokkos::Serial;
#elif defined DIGI_KOKKOS_OPENMP
using KokkosExecSpace = Kokkos::OpenMP
#elif defined DIGI_KOKKOS_CUDA
using KokkosExecSpace = Kokkos::Cuda
#endif

namespace kokkos {
  void initialize(int& argc, char** argv) { Kokkos::initialize(argc, argv); }

  void analyze(Input const& input, Output& output, double& totaltime) {
    totaltime = 0.;

    for (int i = 0; i < NLOOPS; ++i) {
      output = Output();
      const auto wordCounter = input.wordCounter;

#ifdef DIGI_KOKKOS_CUDA
      // Rather non-idiomatic use of Kokkos::View...
      Kokkos::View<Input, Kokkos::CudaSpace> input_d{"input_d"};
      Kokkos::View<Input, Kokkos::CudaSpace>::HostMirror input_h = Kokkos::create_mirror_view(input_d);
      std::memcpy(input_h.data(), &input, sizeof(Input));

      Kokkos::View<Output, Kokkos::CudaSpace> output_d{"output_d"};
      Kokkos::View<Output, Kokkos::CudaSpace>::HostMirror output_h = Kokkos::create_mirror_view(output_d);
      output_h(0).err.construct(pixelgpudetails::MAX_FED_WORDS, output_d.data()->err_d);
#else   // not DIGI_KOKKOS_CUDA \
        // Rather non-idiomatic use of Kokkos::View...
      Kokkos::View<const Input, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > input_d{&input};
      Kokkos::View<Output, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > output_d{&output};
#endif  // DIGI_KOKKOS_CUDA

      auto start = std::chrono::high_resolution_clock::now();
#ifdef DIGI_KOKKOS_CUDA
      Kokkos::deep_copy(input_d, input_h);
#endif  // DIGI_KOKKOS_CUDA

      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(0, input.wordCounter), KOKKOS_LAMBDA(const size_t i) {
            kokkos::rawtodigi(input_d.data(), output_d.data(), wordCounter, true, true, false, i);
          });
#ifdef DIGI_KOKKOS_CUDA
      Kokkos::fence();  // I don't know if parallel_for is synchronous or not
      Kokkos::deep_copy(output_h, output_d);
#endif  // DIGI_KOKKOS_CUDA
      Kokkos::fence();

      auto stop = std::chrono::high_resolution_clock::now();

#ifdef DIGI_KOKKOS_CUDA
      output_h(0).err.set_data(output_h(0).err_d);
      std::memcpy(output.get(), output_h.data(), sizeof(Output));
      output->err.set_data(output->err_d);
#endif  // DIGI_KOKKOS_CUDA

      auto diff = stop - start;
      auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      totaltime += time;
    }

    totaltime /= NLOOPS;
  }

  void finalize() { Kokkos::finalize(); }
}  // namespace kokkos
