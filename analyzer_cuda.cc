#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>

#include "cudaCheck.h"
#include "input.h"
#include "launch.h"
#include "modules.h"
#include "output.h"
#include "rawtodigi_cuda.h"

namespace {
  constexpr int NLOOPS = 100;
}

namespace cuda {

  void analyze(Input const &input, Output &output, double &totaltime) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    totaltime = 0;

    for (int i = 0; i < NLOOPS; ++i) {
      output = Output();

      Input *input_d, *input_h;
      cudaCheck(cudaMalloc(&input_d, sizeof(Input)));
      cudaCheck(cudaMallocHost(&input_h, sizeof(Input)));
      std::memcpy(input_h, &input, sizeof(Input));

      Output *output_d, *output_h;
      cudaCheck(cudaMalloc(&output_d, sizeof(Output)));
      cudaCheck(cudaMallocHost(&output_h, sizeof(Output)));
      output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

      auto start = std::chrono::high_resolution_clock::now();
      cudaCheck(cudaMemcpyAsync(input_d, input_h, sizeof(Input), cudaMemcpyDefault, stream));
      cudaCheck(cudaMemcpyAsync(output_d, output_h, sizeof(Output), cudaMemcpyDefault, stream));

      const int threadsPerBlock = 512;
      const int blocks = (input.wordCounter + threadsPerBlock - 1) / threadsPerBlock;
      launch(cuda::rawtodigi_kernel, {blocks, threadsPerBlock, 0, stream}, input_d, output_d, true, true, false);

      cudaCheck(cudaMemcpyAsync(output_h, output_d, sizeof(Output), cudaMemcpyDefault, stream));
      cudaCheck(cudaStreamSynchronize(stream));
      auto stop = std::chrono::high_resolution_clock::now();

      output_h->err.set_data(output_h->err_d);
      std::memcpy(&output, output_h, sizeof(Output));
      output.err.set_data(output.err_d);

      cudaFree(output_d);
      cudaFree(input_d);
      cudaFreeHost(output_h);
      cudaFreeHost(input_h);

      auto diff = stop - start;
      totaltime += std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    }

    totaltime /= NLOOPS;

    cudaStreamDestroy(stream);
  }

}  // namespace cuda
