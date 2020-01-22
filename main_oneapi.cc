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

enum DeviceType { default_device = 0, host_device, cpu_device, gpu_device };

void exception_handler(cl::sycl::exception_list exceptions) {
  for (auto const &exc_ptr : exceptions) {
    try {
      std::rethrow_exception(exc_ptr);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  DeviceType device_type = default_device;
  if (argc > 1) {
    if (std::strcmp(argv[1], "--host") == 0)
      device_type = host_device;
    else if (std::strcmp(argv[1], "--cpu") == 0)
      device_type = cpu_device;
    else if (std::strcmp(argv[1], "--gpu") == 0)
      device_type = gpu_device;
    else
      std::cout << "Ignoring unknown option " << argv[1] << std::endl;
  }

  cl::sycl::device device;
  switch (device_type) {
    case host_device: {
      cl::sycl::host_selector device_selector;
      device = cl::sycl::device{device_selector};
      break;
    }
    case cpu_device: {
      cl::sycl::cpu_selector device_selector;
      device = cl::sycl::device{device_selector};
      break;
    }
    case gpu_device: {
      cl::sycl::gpu_selector device_selector;
      device = cl::sycl::device{device_selector};
      break;
    }
    case default_device:
    default: {
      cl::sycl::default_selector device_selector;
      device = cl::sycl::device{device_selector};
    }
  }

  cl::sycl::queue queue{device, exception_handler};
  cl::sycl::context ctx = (device_type == cpu_device) ? cl::sycl::context{device} : queue.get_context();
  std::cout << "Running on SYCL device " << device.get_info<cl::sycl::info::device::name>() << ", driver version "
            << device.get_info<cl::sycl::info::device::driver_version>() << std::endl;

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  int totaltime = 0;

  std::unique_ptr<Output> output;
#ifdef DIGI_ONEAPI_WORKAROUND
  output = std::make_unique<Output>();
#endif  // DIGI_ONEAPI_WORKAROUND
  for (int i = 0; i <= NLOOPS; ++i) {
#ifndef DIGI_ONEAPI_WORKAROUND
    output = std::make_unique<Output>();
#endif  // ! DIGI_ONEAPI_WORKAROUND

    auto input_d = (Input *)cl::sycl::malloc_device(sizeof(Input), device, ctx);
    if (input_d == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Input) << " bytes of device memory" << std::endl;
      exit(1);
    }
    auto input_h = (Input *)cl::sycl::malloc_host(sizeof(Input), ctx);
    if (input_h == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Input) << " bytes of host memory" << std::endl;
      exit(1);
    }
    std::memcpy(input_h, &input, sizeof(Input));

    auto output_d = (Output *)cl::sycl::malloc_device(sizeof(Output), device, ctx);
    if (output_d == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Output) << " bytes of device memory" << std::endl;
      exit(1);
    }
    auto output_h = (Output *)cl::sycl::malloc_host(sizeof(Output), ctx);
    if (output_h == nullptr) {
      std::cerr << "oneAPI failed to allocate " << sizeof(Output) << " bytes of host memory" << std::endl;
      exit(1);
    }
    output_h->err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

    auto start = std::chrono::high_resolution_clock::now();
    queue.memcpy((void *)input_d, (void *)input_h, sizeof(Input));
    queue.memcpy((void *)output_d, (void *)output_h, sizeof(Output));

    oneapi::rawtodigi(input_d, output_d, input.wordCounter, true, true, false, i == 0, queue);

    queue.memcpy((void *)output_h, (void *)output_d, sizeof(Output));
    queue.wait_and_throw();
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
    if (i != 0) {
      totaltime += time;
    }
  }

  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime) / NLOOPS) << " us" << std::endl;

  return 0;
}
