#ifndef analyzer_kokkos_h
#define analyzer_kokkos_h

#include "input.h"
#include "output.h"
#include "loops.h"
#include "rawtodigi_kokkos.h"
#include "kokkosConfig.h"

namespace kokkos_serial {
  template <typename MemorySpace>
  void analyze(InputKokkos<MemorySpace>& input, Output& output, double& totaltime) {
    totaltime = 0.;

    for (int i = 0; i < NLOOPS; ++i) {
      output = Output();

      const auto wordCounter = input.GetHostWordCounter();

      Kokkos::View<Output, KokkosExecSpace> output_d{"output_d"};
      Kokkos::View<Output, KokkosExecSpace>::HostMirror output_h = Kokkos::create_mirror_view(output_d);
      output_h.data()->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d.data()->err_d);

      // could I still use unmanaged somehow to avoid the memcpy above and below?
      //Kokkos::View<const Input, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > input_d{&input};
      //Kokkos::View<Output, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > output_d{&output};

      auto start = std::chrono::high_resolution_clock::now();
      input.DeepCopyH2D();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(0, wordCounter), KOKKOS_LAMBDA(const size_t i) {
            rawtodigi(input, output_d.data(), wordCounter, true, true, false, i);
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
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;
  }
}  // namespace kokkos

namespace kokkos_openmp {
  template <typename MemorySpace>
  void analyze(InputKokkos<MemorySpace>& input, Output& output, double& totaltime) {
    totaltime = 0.;

    for (int i = 0; i < NLOOPS; ++i) {
      output = Output();

      const auto wordCounter = input.GetHostWordCounter();

      Kokkos::View<Output, KokkosExecSpace> output_d{"output_d"};
      Kokkos::View<Output, KokkosExecSpace>::HostMirror output_h = Kokkos::create_mirror_view(output_d);
      output_h.data()->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d.data()->err_d);

      // could I still use unmanaged somehow to avoid the memcpy above and below?
      //Kokkos::View<const Input, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > input_d{&input};
      //Kokkos::View<Output, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > output_d{&output};

      auto start = std::chrono::high_resolution_clock::now();
      input.DeepCopyH2D();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(0, wordCounter), KOKKOS_LAMBDA(const size_t i) {
            rawtodigi(input, output_d.data(), wordCounter, true, true, false, i);
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
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;
  }
}  // namespace kokkos

namespace kokkos_cuda {
  template <typename MemorySpace>
  void analyze(InputKokkos<MemorySpace>& input, Output& output, double& totaltime) {
    totaltime = 0.;

    for (int i = 0; i < NLOOPS; ++i) {
      output = Output();

      const auto wordCounter = input.GetHostWordCounter();

      Kokkos::View<Output, KokkosExecSpace> output_d{"output_d"};
      Kokkos::View<Output, KokkosExecSpace>::HostMirror output_h = Kokkos::create_mirror_view(output_d);
      output_h.data()->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d.data()->err_d);

      // could I still use unmanaged somehow to avoid the memcpy above and below?
      //Kokkos::View<const Input, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > input_d{&input};
      //Kokkos::View<Output, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > output_d{&output};

      auto start = std::chrono::high_resolution_clock::now();
      input.DeepCopyH2D();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(0, wordCounter), KOKKOS_LAMBDA(const size_t i) {
            rawtodigi(input, output_d.data(), wordCounter, true, true, false, i);
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
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;
  }
}  // namespace kokkos

#endif
