#ifndef kokkosConfig_h_
#define kokkosConfig_h_

#include <Kokkos_Core.hpp>

#ifdef DIGI_KOKKOS_SERIAL
using KokkosExecSpace = Kokkos::Serial;
#define KOKKOS_NAMESPACE kokkos_serial
#elif defined DIGI_KOKKOS_OPENMP
using KokkosExecSpace = Kokkos::OpenMP;
#define KOKKOS_NAMESPACE kokkos_openmp
#elif defined DIGI_KOKKOS_CUDA
using KokkosExecSpace = Kokkos::Cuda;
#define KOKKOS_NAMESPACE kokkos_cuda
#endif

namespace kokkos_common {
  void initialize(const Kokkos::InitArguments& arguments);
  void finalize();
}  // namespace kokkos_common

#endif
