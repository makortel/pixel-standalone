#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */
#include <cuda_to_cupla.hpp>

#include "cupla_check.h"
#include "input.h"
#include "loops.h"
#include "modules.h"
#include "output.h"
#include "rawtodigi_cupla.h"

namespace CUPLA_ACCELERATOR_NAMESPACE {

  void analyze(Input const &input, Output &output, double &totaltime) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    totaltime = 0;

    for (int i = 0; i <= NLOOPS; ++i) {
      output = Output();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      Input *input_d, *input_h;
      CUPLA_CHECK(cudaMalloc((void **)&input_d, sizeof(Input)));
      CUPLA_CHECK(cudaMallocHost((void **)&input_h, sizeof(Input)));
      std::memcpy(input_h, &input, sizeof(Input));

      Output *output_d, *output_h;
      CUPLA_CHECK(cudaMalloc((void **)&output_d, sizeof(Output)));
      CUPLA_CHECK(cudaMallocHost((void **)&output_h, sizeof(Output)));
      output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);
#else   // ALPAKA_ACC_GPU_CUDA_ENABLED
      const Input *input_d = &input;
      Output *output_d = &output;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

      auto start = std::chrono::high_resolution_clock::now();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      CUPLA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(Input), cudaMemcpyHostToDevice, stream));
      CUPLA_CHECK(cudaMemcpyAsync(output_d, output_h, sizeof(Output), cudaMemcpyHostToDevice, stream));
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

      const int threadsPerBlock = 512;
      const int blocks = (input.wordCounter + threadsPerBlock - 1) / threadsPerBlock;
      if (i == 0) {
        std::cout << "blocks per grid: " << blocks << ", threads per block: " << threadsPerBlock << std::endl;
      }
      CUPLA_KERNEL_OPTI(CUPLA_ACCELERATOR_NAMESPACE::rawtodigi_kernel)
      (blocks, threadsPerBlock, 0, stream)(input_d, output_d, true, true, i == 0);
      CUPLA_CHECK(cudaGetLastError());

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      CUPLA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(Output), cudaMemcpyDeviceToHost, stream));
#endif
      CUPLA_CHECK(cudaStreamSynchronize(stream));
      auto stop = std::chrono::high_resolution_clock::now();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      output_h->err.set_data(output_h->err_d);
      std::memcpy(&output, output_h, sizeof(Output));
      output.err.set_data(output.err_d);

      CUPLA_CHECK(cudaFree(output_d));
      CUPLA_CHECK(cudaFree(input_d));
      CUPLA_CHECK(cudaFreeHost(output_h));
      CUPLA_CHECK(cudaFreeHost(input_h));
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

      auto diff = stop - start;
      auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;

    CUPLA_CHECK(cudaStreamDestroy(stream));
  }

}  // namespace CUPLA_ACCELERATOR_NAMESPACE
