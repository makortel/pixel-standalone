#ifndef analyzer_kokkos_h
#define analyzer_kokkos_h

class Input;
class Output;

namespace kokkos_serial {
  void analyze(Input const& input, Output& output, double& totaltime);
}  // namespace kokkos
namespace kokkos_openmp {
  void analyze(Input const& input, Output& output, double& totaltime);
}  // namespace kokkos
namespace kokkos_cuda {
  void analyze(Input const& input, Output& output, double& totaltime);
}  // namespace kokkos

#endif
