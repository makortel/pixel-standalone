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
using KokkosExecSpace = Kokkos::OpenMP;
#elif defined DIGI_KOKKOS_CUDA
using KokkosExecSpace = Kokkos::Cuda;
#endif

namespace kokkos {
  void initialize(int& argc, char** argv) { Kokkos::initialize(argc, argv); }

  void analyze(Input const& input, Output& output, double& totaltime) {
    totaltime = 0.;

    for (int i = 0; i < NLOOPS; ++i) {
      output = Output();
      const auto wordCounter = input.wordCounter;

      // Rather non-idiomatic use of Kokkos::View...
      Kokkos::View<Input, KokkosExecSpace> input_d{"input_d"};
      Kokkos::View<Input, KokkosExecSpace>::HostMirror input_h = Kokkos::create_mirror_view(input_d);
      std::memcpy(input_h.data(), &input, sizeof(Input));

      Kokkos::View<Output, KokkosExecSpace> output_d{"output_d"};
      Kokkos::View<Output, KokkosExecSpace>::HostMirror output_h = Kokkos::create_mirror_view(output_d);
      output_h.data()->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d.data()->err_d);

      // could I still use unmanaged somehow to avoid the memcpy above and below?
      //Kokkos::View<const Input, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > input_d{&input};
      //Kokkos::View<Output, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > output_d{&output};

      auto start = std::chrono::high_resolution_clock::now();
      Kokkos::deep_copy(input_d, input_h);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(0, input.wordCounter), KOKKOS_LAMBDA(const size_t i) {
            kokkos::rawtodigi(input_d.data(), output_d.data(), wordCounter, true, true, false, i);
          });
      Kokkos::fence();  // I don't know if parallel_for is synchronous or not
      Kokkos::deep_copy(output_h, output_d);
      Kokkos::fence();

      auto stop = std::chrono::high_resolution_clock::now();

      output_h.data()->err.set_data(output_h.data()->err_d);
      std::memcpy(&output, output_h.data(), sizeof(Output));
      output.err.set_data(output.err_d);

      auto diff = stop - start;
      auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      totaltime += time;
    }

    totaltime /= NLOOPS;
  }

  void finalize() { Kokkos::finalize(); }
}  // namespace kokkos
