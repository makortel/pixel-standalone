#ifndef analyzer_cuda_h_
#define analyzer_cuda_h_

class Input;
class Output;

namespace cuda {
  void analyze(Input const& input, Output& output, double& totaltime);
}

#endif  // analyzer_cuda_h_
