#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "input.h"
#include "output.h"

#ifdef DIGI_NAIVE
#include "rawtodigi_naive.h"
#elif defined DIGI_CUDA
#include <cuda_runtime.h>
#include "rawtodigi_cuda.h"
#include "cudaCheck.h"
#endif

namespace {
  constexpr int NLOOPS = 100;
}

namespace gpuClustering {
  constexpr uint32_t MaxNumModules  = 2000;
  constexpr uint16_t InvId          = 9999;         // must be > MaxNumModules
}

int countModules(const uint16_t *id, int numElements) {
  int nmod = 0;
  for(int i=0; i<numElements; ++i) {
    if(id[i] == gpuClustering::InvId)
      continue;
    auto j = i-1;
    while(j >= 0 and id[j] == gpuClustering::InvId) {
      --j;
    }
    if(j < 0 or id[j] != id[i]) {
      ++nmod;
    }
  }
  return nmod;
}

int main() {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

#ifdef DIGI_CUDA
  cudaStream_t stream;
  cudaStreamCreate(&stream);
#endif

  int totaltime = 0;

  std::unique_ptr<Output> output;
  for(int i=0; i<NLOOPS; ++i) {
    output = std::make_unique<Output>();

#ifdef DIGI_NAIVE
    auto start = std::chrono::high_resolution_clock::now();
    naive::rawtodigi(&input.cablingMap,
                     input.wordCounter,
                     input.word,
                     input.fedId,
                     output->xx,
                     output->yy,
                     output->adc,
                     output->digi,
                     output->rawIdArr,
                     output->moduleInd,
                     &output->err,
                     true, true, false);
    auto stop = std::chrono::high_resolution_clock::now();
#elif defined DIGI_CUDA
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

    cuda::rawtodigi(input_d, output_d,
                    input.wordCounter,
                    true, true, false, stream);

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
#endif

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
  }
  
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime)/NLOOPS) << " us"
            << std::endl;

#ifdef DIGI_CUDA
  cudaStreamDestroy(stream);
#endif
  
  return 0;
}
