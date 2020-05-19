#ifndef analyzer_kokkos_h
#define analyzer_kokkos_h

class Input;
class Output;

namespace kokkos_serial {
  void analyze(Input const& input, Output& output, double& totaltime);
}  // namespace kokkos_serial
namespace kokkos_openmp {
  void analyze(Input const& input, Output& output, double& totaltime);
}  // namespace kokkos_openmp
namespace kokkos_cuda {
  void analyze(Input const& input, Output& output, double& totaltime);
}  // namespace kokkos_cuda

#endif
