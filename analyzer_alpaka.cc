#include <algorithm>
#include <cstdio>

#include "alpakaConfig.h"
#include "analyzer_alpaka.h"
#include "input.h"
#include "output.h"
#include "pixelgpudetails.h"
#include "rawtodigi_alpaka.h"

namespace {
  constexpr int NLOOPS = 100;
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void analyze(Input const& input, Output& output, double& totaltime) {
    const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    const DevAcc device(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    const Vec size(1u);

    Queue queue(device);

    totaltime = 0.;

    for (int i = 0; i < NLOOPS; i++) {
      output = Output();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      auto input_dBuf = alpaka::mem::buf::alloc<Input, Idx>(device, size);
      Input* input_d = alpaka::mem::view::getPtrNative(input_dBuf);
      auto input_hBuf = alpaka::mem::buf::alloc<Input, Idx>(host, size);
      alpaka::mem::buf::prepareForAsyncCopy(input_hBuf);
      Input* input_h = alpaka::mem::view::getPtrNative(input_hBuf);
      std::memcpy(input_h, &input, sizeof(Input));

      auto output_dBuf = alpaka::mem::buf::alloc<Output, Idx>(device, size);
      Output* output_d = alpaka::mem::view::getPtrNative(output_dBuf);
      auto output_hBuf = alpaka::mem::buf::alloc<Output, Idx>(host, size);
      alpaka::mem::buf::prepareForAsyncCopy(output_hBuf);
      Output* output_h = alpaka::mem::view::getPtrNative(output_hBuf);
      output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);
#else
      Input const* input_d = &input;
      Output* output_d = &output;
#endif

      auto start = std::chrono::high_resolution_clock::now();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      alpaka::mem::view::copy(queue, input_dBuf, input_hBuf, size);
      alpaka::mem::view::copy(queue, output_dBuf, output_hBuf, size);
#endif

      Vec elementsPerThread(Vec::all(1));
      Vec threadsPerBlock(Vec::all(512));
      Vec blocksPerGrid(Vec::all((input.wordCounter + 512 - 1) / 512));
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
      // on the GPU, run with 512 threads in parallel per block, each looking at a single element
      // on the CPU, run serially with a single thread per block, over 512 elements
      std::swap(threadsPerBlock, elementsPerThread);
#endif
      const WorkDiv workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

      alpaka::queue::enqueue(
          queue,
          alpaka::kernel::createTaskKernel<Acc>(workDiv, rawtodigi_kernel(), input_d, output_d, true, true, false));

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      alpaka::mem::view::copy(queue, output_hBuf, output_dBuf, size);
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

      alpaka::wait::wait(queue);

      auto stop = std::chrono::high_resolution_clock::now();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      output_h->err.set_data(output_h->err_d);
      std::memcpy(&output, output_h, sizeof(Output));
      output.err.set_data(output.err_d);
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

      auto diff = stop - start;
      totaltime += std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    }

    totaltime /= NLOOPS;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
