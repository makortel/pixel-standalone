#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <CL/sycl.hpp>

#include "input.h"
#include "modules.h"
#include "output.h"
#include "rawtodigi_oneapi.h"

namespace {
  constexpr int NLOOPS = 100;
}

namespace oneapi {

  void exception_handler(cl::sycl::exception_list exceptions) {
    for (auto const &exc_ptr : exceptions) {
      try {
        std::rethrow_exception(exc_ptr);
      } catch (cl::sycl::exception const &e) {
        std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
      }
    }
  }

  void analyze(cl::sycl::device device, Input const &input, Output &output, double &totaltime) {
    cl::sycl::queue queue{device, exception_handler};

    totaltime = 0;

    for (int i = 0; i <= NLOOPS; ++i) {
      output = Output{};

      auto input_d = (Input *)cl::sycl::malloc_device(sizeof(Input), queue);
      if (input_d == nullptr) {
        std::cerr << "oneAPI failed to allocate " << sizeof(Input) << " bytes of device memory" << std::endl;
        exit(1);
      }
      auto input_h = (Input *)cl::sycl::malloc_host(sizeof(Input), queue);
      if (input_h == nullptr) {
        std::cerr << "oneAPI failed to allocate " << sizeof(Input) << " bytes of host memory" << std::endl;
        exit(1);
      }
      std::memcpy(input_h, &input, sizeof(Input));

      auto output_d = (Output *)cl::sycl::malloc_device(sizeof(Output), queue);
      if (output_d == nullptr) {
        std::cerr << "oneAPI failed to allocate " << sizeof(Output) << " bytes of device memory" << std::endl;
        exit(1);
      }
      auto output_h = (Output *)cl::sycl::malloc_host(sizeof(Output), queue);
      if (output_h == nullptr) {
        std::cerr << "oneAPI failed to allocate " << sizeof(Output) << " bytes of host memory" << std::endl;
        exit(1);
      }
      output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

      auto start = std::chrono::high_resolution_clock::now();
      queue.memcpy(input_d, input_h, sizeof(Input));
      queue.memcpy(output_d, output_h, sizeof(Output));

      rawtodigi(input_d, output_d, input.wordCounter, true, true, i == 0, queue);

      queue.memcpy(output_h, output_d, sizeof(Output));
      queue.wait_and_throw();
      auto stop = std::chrono::high_resolution_clock::now();

      output_h->err.set_data(output_h->err_d);
      std::memcpy(&output, output_h, sizeof(Output));
      output.err.set_data(output.err_d);

      cl::sycl::free(output_d, queue);
      cl::sycl::free(input_d, queue);
      cl::sycl::free(output_h, queue);
      cl::sycl::free(input_h, queue);

      auto diff = stop - start;
      auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;
  }

}  // namespace oneapi
