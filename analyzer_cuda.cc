#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>

#include "cute/check.h"
#include "cute/launch.h"
#include "input.h"
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

    for (int i = 0; i <= NLOOPS; ++i) {
      output = Output();

      Input *input_d, *input_h;
      CUTE_CHECK(cudaMalloc(&input_d, sizeof(Input)));
      CUTE_CHECK(cudaMallocHost(&input_h, sizeof(Input)));
      std::memcpy(input_h, &input, sizeof(Input));

      Output *output_d, *output_h;
      CUTE_CHECK(cudaMalloc(&output_d, sizeof(Output)));
      CUTE_CHECK(cudaMallocHost(&output_h, sizeof(Output)));
      output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

      auto start = std::chrono::high_resolution_clock::now();
      CUTE_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(Input), cudaMemcpyDefault, stream));
      CUTE_CHECK(cudaMemcpyAsync(output_d, output_h, sizeof(Output), cudaMemcpyDefault, stream));

      const int threadsPerBlock = 512;
      const int blocks = (input.wordCounter + threadsPerBlock - 1) / threadsPerBlock;
      cute::launch(cuda::rawtodigi_kernel, {blocks, threadsPerBlock, 0, stream}, input_d, output_d, true, true, i == 0);

      CUTE_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(Output), cudaMemcpyDefault, stream));
      CUTE_CHECK(cudaStreamSynchronize(stream));
      auto stop = std::chrono::high_resolution_clock::now();

      output_h->err.set_data(output_h->err_d);
      std::memcpy(&output, output_h, sizeof(Output));
      output.err.set_data(output.err_d);

      CUTE_CHECK(cudaFree(output_d));
      CUTE_CHECK(cudaFree(input_d));
      CUTE_CHECK(cudaFreeHost(output_h));
      CUTE_CHECK(cudaFreeHost(input_h));

      auto diff = stop - start;
      auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;

    CUTE_CHECK(cudaStreamDestroy(stream));
  }

}  // namespace cuda
