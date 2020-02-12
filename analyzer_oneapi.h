#ifndef analyzer_oneapi_h_
#define analyzer_oneapi_h_

#include <CL/sycl.hpp>

class Input;
class Output;

namespace oneapi {
  void analyze(cl::sycl::device device, Input const &input, Output &output, double &totaltime);
}

#endif  // analyzer_oneapi_h_
