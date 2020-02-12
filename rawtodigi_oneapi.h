#ifndef rawtodigi_oneapi_h_
#define rawtodigi_oneapi_h_

#include <CL/sycl.hpp>

class Input;
class Output;

namespace oneapi {

  void rawtodigi(const Input* input,
                 Output* output,
                 const uint32_t wordCounter,
                 bool useQualityInfo,
                 bool includeErrors,
                 bool debug,
                 bool first,
                 cl::sycl::queue queue);

}  // namespace oneapi

#endif  // rawtodigi_oneapi_h_
