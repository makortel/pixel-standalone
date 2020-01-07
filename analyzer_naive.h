#ifndef analyzer_naive_h_
#define analyzer_naive_h_

#include <memory>

class Input;
class Output;

namespace naive {
  void analyze(Input const& input, std::unique_ptr<Output>& output, double& totaltime);
}

#endif
