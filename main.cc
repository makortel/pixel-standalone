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
#include "cudaCheck.h"
#include "rawtodigi_cuda.h"
#elif defined DIGI_CUPLA
/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */
#include <cuda_to_cupla.hpp>
#include "cudaCheck.h"
#include "rawtodigi_cupla.h"
#elif defined DIGI_KOKKOS
#include <Kokkos_Core.hpp>
#include "rawtodigi_kokkos.h"
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

int main(int argc, char **argv) {
#ifdef DIGI_KOKKOS
  Kokkos::initialize(argc, argv);
#endif

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

#if defined DIGI_CUDA or defined DIGI_CUPLA
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
#elif defined DIGI_CUPLA
#if defined ALPAKA_ACC_GPU_CUDA_ENABLED
    Input *input_d, *input_h;
    cudaCheck(cudaMalloc((void **) &input_d, sizeof(Input)));
    cudaCheck(cudaMallocHost((void **) &input_h, sizeof(Input)));
    std::memcpy(input_h, &input, sizeof(Input));

    Output *output_d, *output_h;
    cudaCheck(cudaMalloc((void **) &output_d, sizeof(Output)));
    cudaCheck(cudaMallocHost((void **) &output_h, sizeof(Output)));
    output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

    auto start = std::chrono::high_resolution_clock::now();
    cudaCheck(cudaMemcpyAsync(input_d, input_h, sizeof(Input), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(output_d, output_h, sizeof(Output), cudaMemcpyHostToDevice, stream));

    cupla::rawtodigi(input_d, output_d,
                    input.wordCounter,
                    true, true, false, stream);

    cudaCheck(cudaMemcpyAsync(output_h, output_d, sizeof(Output), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    auto stop = std::chrono::high_resolution_clock::now();

    output_h->err.set_data(output_h->err_d);
    std::memcpy(output.get(), output_h, sizeof(Output));
    output->err.set_data(output->err_d);

    cudaFree(output_d);
    cudaFree(input_d);
    cudaFreeHost(output_h);
    cudaFreeHost(input_h);
#else // ALPAKA_ACC_GPU_CUDA_ENABLED
    auto start = std::chrono::high_resolution_clock::now();
    cupla::rawtodigi(&input, output.get(),
                    input.wordCounter,
                    true, true, false, stream);
    cudaCheck(cudaStreamSynchronize(stream));
    auto stop = std::chrono::high_resolution_clock::now();
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
#elif defined DIGI_KOKKOS
    const auto wordCounter = input.wordCounter;

#if defined DIGI_KOKKOS_CUDA
    // Rather non-idiomatic use of Kokkos::View...
    Kokkos::View<Input, Kokkos::CudaSpace> input_d{"input_d"};
    Kokkos::View<Input, Kokkos::CudaSpace>::HostMirror input_h = Kokkos::create_mirror_view(input_d);
    std::memcpy(input_h.data(), &input, sizeof(Input));

    Kokkos::View<Output, Kokkos::CudaSpace> output_d{"output_d"};
    Kokkos::View<Output, Kokkos::CudaSpace>::HostMirror output_h = Kokkos::create_mirror_view(output_d);
    output_h(0).err.construct(pixelgpudetails::MAX_FED_WORDS, output_d.data()->err_d);

    auto start = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(input_d, input_h);

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0, input.wordCounter),
                         KOKKOS_LAMBDA (const size_t i) {
                           kokkos::rawtodigi(input_d, output_d, wordCounter,
                                             true, true, false, i);
      });
    Kokkos::fence(); // I don't know if parallel_for is synchronous or not
    Kokkos::deep_copy(output_h, output_d);
    Kokkos::fence();

    auto stop = std::chrono::high_resolution_clock::now();

    output_h(0).err.set_data(output_h(0).err_d);
    std::memcpy(output.get(), output_h.data(), sizeof(Output));
    output->err.set_data(output->err_d);

#else // DIGI_KOKKOS_CUDA
    // Rather non-idiomatic use of Kokkos::View...
    Kokkos::View<Input, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> input_h{&input};
    Kokkos::View<Output, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> output_h{output.get()};

    auto start = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_for(Kokkos::RangePolicy<
#ifdef DIGI_KOKKOS_SERIAL
                         Kokkos::Serial
#elif defined DIGI_KOKKOS_OPENMP
                         Kokkos::OpenMP
#endif
                         >(0, input.wordCounter),
                         KOKKOS_LAMBDA (const size_t i) {
                           kokkos::rawtodigi(input_h, output_h, wordCounter,
                                             true, true, false, i);
      });
    Kokkos::fence();
    auto stop = std::chrono::high_resolution_clock::now();

#endif // DIGI_KOKKOS_CUDA


#endif

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
  }
  
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime)/NLOOPS) << " us"
            << std::endl;

#if defined DIGI_CUDA or defined DIGI_CUPLA
  cudaStreamDestroy(stream);
#elif defined DIGI_KOKKOS
  Kokkos::finalize();
#endif
  
  return 0;
}
