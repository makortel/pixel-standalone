#include "kokkosConfig.h"

namespace kokkos_common {
  void initialize(const Kokkos::InitArguments& arguments) {
    Kokkos::initialize(arguments);
  }

  void finalize() {
    Kokkos::finalize();
  }
}
