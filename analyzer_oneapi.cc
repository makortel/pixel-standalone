#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <CL/sycl.hpp>

#include "input.h"
#include "loops.h"
#include "modules.h"
#include "output.h"
#include "rawtodigi_oneapi.h"

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
#if __SYCL_COMPILER_VERSION <= 20200118
    // Intel oneAPI beta 4
    cl::sycl::ordered_queue queue{device, exception_handler};
#else
    // Intel SYCL branch
    cl::sycl::queue queue{device, exception_handler, cl::sycl::property::queue::in_order()};
#endif

    totaltime = 0;

    for (int i = 0; i <= NLOOPS; ++i) {
      output = Output{};

#if __SYCL_COMPILER_VERSION <= 20200118
      auto input_d = (Input *)cl::sycl::malloc_device(sizeof(Input), queue.get_device(), queue.get_context());
#else
      auto input_d = (Input *)cl::sycl::malloc_device(sizeof(Input), queue);
#endif
      if (input_d == nullptr) {
        std::cerr << "oneAPI runtime failed to allocate " << sizeof(Input) << " bytes of device memory" << std::endl;
        exit(1);
      }
#if __SYCL_COMPILER_VERSION <= 20200118
      auto input_h = (Input *)cl::sycl::malloc_host(sizeof(Input), queue.get_context());
#else
      auto input_h = (Input *)cl::sycl::malloc_host(sizeof(Input), queue);
#endif
      if (input_h == nullptr) {
        std::cerr << "oneAPI runtime failed to allocate " << sizeof(Input) << " bytes of host memory" << std::endl;
        exit(1);
      }
      std::memcpy(input_h, &input, sizeof(Input));

#if __SYCL_COMPILER_VERSION <= 20200118
      auto output_d = (Output *)cl::sycl::malloc_device(sizeof(Output), queue.get_device(), queue.get_context());
#else
      auto output_d = (Output *)cl::sycl::malloc_device(sizeof(Output), queue);
#endif
      if (output_d == nullptr) {
        std::cerr << "oneAPI runtime failed to allocate " << sizeof(Output) << " bytes of device memory" << std::endl;
        exit(1);
      }
#if __SYCL_COMPILER_VERSION <= 20200118
      auto output_h = (Output *)cl::sycl::malloc_host(sizeof(Output), queue.get_context());
#else
      auto output_h = (Output *)cl::sycl::malloc_host(sizeof(Output), queue);
#endif
      if (output_h == nullptr) {
        std::cerr << "oneAPI runtime failed to allocate " << sizeof(Output) << " bytes of host memory" << std::endl;
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

#if __SYCL_COMPILER_VERSION <= 20200118
      cl::sycl::free(output_d, queue.get_context());
      cl::sycl::free(input_d, queue.get_context());
      cl::sycl::free(output_h, queue.get_context());
      cl::sycl::free(input_h, queue.get_context());
#else
      cl::sycl::free(output_d, queue);
      cl::sycl::free(input_d, queue);
      cl::sycl::free(output_h, queue);
      cl::sycl::free(input_h, queue);
#endif

      auto diff = stop - start;
      auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;
  }

}  // namespace oneapi
