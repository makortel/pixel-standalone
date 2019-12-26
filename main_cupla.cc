#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "input.h"
#include "modules.h"
#include "output.h"

/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */
#include <cuda_to_cupla.hpp>
#include "cudaCheck.h"
#include "rawtodigi_cupla.h"

namespace {
  constexpr int NLOOPS = 100;
}

int main(int argc, char **argv) {

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int totaltime = 0;

  std::unique_ptr<Output> output;
  for(int i=0; i<NLOOPS; ++i) {
    output = std::make_unique<Output>();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    Input *input_d, *input_h;
    cudaCheck(cudaMalloc((void **) &input_d, sizeof(Input)));
    cudaCheck(cudaMallocHost((void **) &input_h, sizeof(Input)));
    std::memcpy(input_h, &input, sizeof(Input));

    Output *output_d, *output_h;
    cudaCheck(cudaMalloc((void **) &output_d, sizeof(Output)));
    cudaCheck(cudaMallocHost((void **) &output_h, sizeof(Output)));
    output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);
#else // ALPAKA_ACC_GPU_CUDA_ENABLED
    Input *input_d = &input;
    Output *output_d = output.get();
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

    auto start = std::chrono::high_resolution_clock::now();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    cudaCheck(cudaMemcpyAsync(input_d, input_h, sizeof(Input), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(output_d, output_h, sizeof(Output), cudaMemcpyHostToDevice, stream));
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

    cupla::rawtodigi(input_d, output_d,
                    input.wordCounter,
                    true, true, false, stream);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    cudaCheck(cudaMemcpyAsync(output_h, output_d, sizeof(Output), cudaMemcpyDeviceToHost, stream));
#endif
    cudaCheck(cudaStreamSynchronize(stream));
    auto stop = std::chrono::high_resolution_clock::now();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    output_h->err.set_data(output_h->err_d);
    std::memcpy(output.get(), output_h, sizeof(Output));
    output->err.set_data(output->err_d);

    cudaFree(output_d);
    cudaFree(input_d);
    cudaFreeHost(output_h);
    cudaFreeHost(input_h);
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
  }
  
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime)/NLOOPS) << " us"
            << std::endl;

  cudaStreamDestroy(stream);
  
  return 0;
}
