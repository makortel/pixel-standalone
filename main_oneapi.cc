#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "input.h"
#include "modules.h"
#include "output.h"
#include "rawtodigi_oneapi.h"

namespace {
  constexpr int NLOOPS = 100;
}

int main(int argc, char **argv) {

  cl::sycl::default_selector device_selector;
  cl::sycl::device device{device_selector};
  cl::sycl::context ctx{device};
  cl::sycl::queue queue{device};
  std::cout << "Running on SYCL device " << device.get_info<cl::sycl::info::device::name>() << std::endl;

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  int totaltime = 0;

  std::unique_ptr<Output> output;
#ifdef DIGI_ONEAPI_WORKAROUND
  output = std::make_unique<Output>();
#endif  // DIGI_ONEAPI_WORKAROUND
  for(int i=0; i<NLOOPS; ++i) {
#ifndef DIGI_ONEAPI_WORKAROUND
    output = std::make_unique<Output>();
#endif  // ! DIGI_ONEAPI_WORKAROUND

    auto input_d = (Input *) cl::sycl::malloc_device(sizeof(Input), device, ctx);
    if (input_d == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Input) << " bytes of device memory" << std::endl;
      exit(1);
    }
    auto input_h = (Input *) cl::sycl::malloc_host(sizeof(Input), ctx);
    if (input_h == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Input) << " bytes of host memory" << std::endl;
      exit(1);
    }
    std::memcpy(input_h, &input, sizeof(Input));

    auto output_d = (Output *) cl::sycl::malloc_device(sizeof(Output), device, ctx);
    if (output_d == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Output) << " bytes of device memory" << std::endl;
      exit(1);
    }
    auto output_h = (Output *) cl::sycl::malloc_host(sizeof(Output), ctx);
    if (output_h == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Output) << " bytes of host memory" << std::endl;
      exit(1);
    }
    output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

    auto start = std::chrono::high_resolution_clock::now();
    queue.memcpy((void*)input_d, (void*)input_h, sizeof(Input));
    queue.memcpy((void*)output_d, (void*)output_h, sizeof(Output));

    oneapi::rawtodigi(input_d, output_d,
                      input.wordCounter,
                      true, true, false, queue);

    queue.memcpy((void*)output_h, (void*)output_d, sizeof(Output));
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
