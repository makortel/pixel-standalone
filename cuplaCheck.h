#ifndef HeterogeneousCore_CUDAUtilities_cuplaCheck_h
#define HeterogeneousCore_CUDAUtilities_cuplaCheck_h

#include <iostream>

/* Do NOT include other headers that use CUDA runtime functions or variables
 * (see above) before this include.
 * The reason for this is that cupla renames CUDA host functions and device build in
 * variables by using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>` (driver functions,
 * etc.).
 */
#include <cuda_to_cupla.hpp>

inline
bool cuplaCheck_(const char* file, int line, const char* cmd, cuplaError_t result)
{
  //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
  if (result == cuplaSuccess)
    return true;

  const char* message = cuplaGetErrorString(result);
  std::cerr << file << ", line " << line << ": " << message << std::endl;
  abort();
  return false;
}

#define cuplaCheck(ARG) (cuplaCheck_(__FILE__, __LINE__, #ARG, (ARG)))

#endif // HeterogeneousCore_CUDAUtilities_cuplaCheck_h
