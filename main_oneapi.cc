#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "input.h"
#include "output.h"
#include "rawtodigi_oneapi.h"

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

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  cl::sycl::default_selector device_selector;
  cl::sycl::device device{device_selector};
  cl::sycl::context ctx{device};
  cl::sycl::queue queue{device};

  int totaltime = 0;

  std::unique_ptr<Output> output;
  for(int i=0; i<NLOOPS; ++i) {
    output = std::make_unique<Output>();

    Input *input_d, *input_h;
    input_d = (Input *) cl::sycl::malloc_device(sizeof(Input), device, ctx);
    input_h = (Input *) cl::sycl::malloc_host(sizeof(Input), ctx);
    std::memcpy(input_h, &input, sizeof(Input));

    Output *output_d, *output_h;
    output_d = (Output *) cl::sycl::malloc_device(sizeof(Output), device, ctx);
    output_h = (Output *) cl::sycl::malloc_host(sizeof(Output), ctx);
    output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

    auto start = std::chrono::high_resolution_clock::now();
    queue.memcpy((void*)input_d, (void*)input_h, sizeof(Input));
    queue.memcpy((void*)output_d, (void*)output_h, sizeof(Output));

    oneapi::rawtodigi(input_d, output_d,
                      input.wordCounter,
                      true, true, false, queue);

    queue.memcpy((void*)(output_h), (void*)(output_d), sizeof(Output));
    queue.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    output_h->err.set_data(output_h->err_d);
    std::memcpy(output.get(), output_h, sizeof(Output));
    output->err.set_data(output->err_d);

    cl::sycl::free(output_d, ctx);
    cl::sycl::free(input_d, ctx);
    cl::sycl::free(output_h, ctx);
    cl::sycl::free(input_h, ctx);

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
  }
  
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime)/NLOOPS) << " us"
            << std::endl;

  return 0;
}
