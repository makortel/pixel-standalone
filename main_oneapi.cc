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

enum DeviceType { default_device = 0, host_device, cpu_device, gpu_device, cuda_device };

void exception_handler(cl::sycl::exception_list exceptions) {
  for (auto const &exc_ptr : exceptions) {
    try {
      std::rethrow_exception(exc_ptr);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
    }
  }
}

class cuda_selector : public cl::sycl::device_selector {
public:
  int operator()(const cl::sycl::device & device) const override {
    using namespace cl::sycl::info;
    std::string const& name = device.get_info<cl::sycl::info::device::name>();
    std::string const& vendor = device.get_info<cl::sycl::info::device::vendor>();
    if (device.is_gpu() and (name.find("NVIDIA") != std::string::npos or vendor.find("NVIDIA") != std::string::npos)) {
      return 1;
    };
    return -1;
  }
};

int main(int argc, char **argv) {
  DeviceType device_type = default_device;
  if (argc > 1) {
    if (std::strcmp(argv[1], "--host") == 0)
      device_type = host_device;
    else if (std::strcmp(argv[1], "--cpu") == 0)
      device_type = cpu_device;
    else if (std::strcmp(argv[1], "--gpu") == 0)
      device_type = gpu_device;
    else if (std::strcmp(argv[1], "--cuda") == 0)
      device_type = cuda_device;
    else
      std::cout << "Ignoring unknown option " << argv[1] << std::endl;
  }

  cl::sycl::device device;
  switch (device_type) {
    case host_device: {
      cl::sycl::host_selector selector;
      device = cl::sycl::device{selector};
      break;
    }
    case cpu_device: {
      cl::sycl::cpu_selector selector;
      device = cl::sycl::device{selector};
      break;
    }
    case gpu_device: {
      cl::sycl::gpu_selector selector;
      device = cl::sycl::device{selector};
      break;
    }
    case cuda_device: {
      cuda_selector selector;
      device = cl::sycl::device{selector};
      break;
    }
    case default_device:
    default: {
      cl::sycl::default_selector selector;
      device = cl::sycl::device{selector};
    }
  }

  cl::sycl::queue queue{device, exception_handler};
  std::cout << "Running on SYCL device " << device.get_info<cl::sycl::info::device::name>() << ", driver version "
            << device.get_info<cl::sycl::info::device::driver_version>() << std::endl;

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  int totaltime = 0;

  std::unique_ptr<Output> output;
  for (int i = 0; i <= NLOOPS; ++i) {
    output = std::make_unique<Output>();

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

    oneapi::rawtodigi(input_d, output_d, input.wordCounter, true, true, i == 0, queue);

    queue.memcpy(output_h, output_d, sizeof(Output));
    queue.wait_and_throw();
    auto stop = std::chrono::high_resolution_clock::now();

    output_h->err.set_data(output_h->err_d);
    std::memcpy(output.get(), output_h, sizeof(Output));
    output->err.set_data(output->err_d);

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

  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime) / NLOOPS) << " us" << std::endl;

  return 0;
}
