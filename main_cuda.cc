#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>

#include "cudaCheck.h"
#include "input.h"
#include "modules.h"
#include "output.h"
#include "rawtodigi_cuda.h"

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
  for (int i = 0; i < NLOOPS; ++i) {
    output = std::make_unique<Output>();

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

    cuda::rawtodigi(input_d, output_d, input.wordCounter, true, true, false, stream);

    cudaCheck(cudaMemcpyAsync(output_h, output_d, sizeof(Output), cudaMemcpyDefault, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    auto stop = std::chrono::high_resolution_clock::now();

    output_h->err.set_data(output_h->err_d);
    std::memcpy(output.get(), output_h, sizeof(Output));
    output->err.set_data(output->err_d);

    cudaFree(output_d);
    cudaFree(input_d);
    cudaFreeHost(output_h);
    cudaFreeHost(input_h);

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
  }

  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime) / NLOOPS) << " us" << std::endl;

  cudaStreamDestroy(stream);

  return 0;
}
