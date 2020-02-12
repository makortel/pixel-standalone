#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <CL/sycl.hpp>

#include "analyzer_oneapi.h"
#include "input.h"
#include "modules.h"
#include "output.h"

namespace {
  constexpr int NLOOPS = 100;
}

enum DeviceType { default_device = 0, host_device, cpu_device, gpu_device, cuda_device };

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
  /*
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
  */

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  std::unique_ptr<Output> output = std::make_unique<Output>();
  double totaltime = 0;

  auto devices = cl::sycl::device::get_devices();
  for (auto & device: devices) {
    std::cout << std::endl;
    std::cout << "Running on SYCL device " << device.get_info<cl::sycl::info::device::name>() << ", driver version "
              << device.get_info<cl::sycl::info::device::driver_version>() << std::endl;
    oneapi::analyze(device, input, *output, totaltime);
    std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
              << (static_cast<double>(totaltime) / NLOOPS) << " us" << std::endl;
  }

  return 0;
}
