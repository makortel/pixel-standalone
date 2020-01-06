#ifndef cudaCheck_h_
#define cudaCheck_h_

#include <iostream>

#if defined DIGI_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

inline bool cudaCheck_(const char* file, int line, const char* cmd, CUresult result) {
  //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
  if (result == CUDA_SUCCESS)
    return true;

  const char* error;
  const char* message;
  cuGetErrorName(result, &error);
  cuGetErrorString(result, &message);
  std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
  abort();
  return false;
}

#elif defined DIGI_CUPLA
/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */
#include <cuda_to_cupla.hpp>
#endif

inline bool cudaCheck_(const char* file, int line, const char* cmd, cudaError_t result) {
  //std::cerr << file << ", line " << line << ": " << cmd << std::endl;
  if (result == cudaSuccess)
    return true;

  const char* error = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);
  std::cerr << file << ", line " << line << ": " << error << ": " << message << std::endl;
  abort();
  return false;
}

#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))

#endif  // cudaCheck_h_
