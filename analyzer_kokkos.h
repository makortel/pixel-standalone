#ifndef analyzer_kokkos_h
#define analyzer_kokkos_h

class Input;
class Output;

namespace kokkos {
  void initialize(int& argc, char** argv);
  void analyze(Input const& input, Output& output, double& totaltime);
  void finalize();
}  // namespace kokkos

#endif
