#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include <CL/sycl.hpp>

#include "analyzer_oneapi.h"
#include "input.h"
#include "modules.h"
#include "output.h"

class cuda_selector : public cl::sycl::device_selector {
public:
  int operator()(const cl::sycl::device & device) const override {
    std::string const& name = device.get_info<cl::sycl::info::device::name>();
    std::string const& vendor = device.get_info<cl::sycl::info::device::vendor>();
    if (device.is_gpu() and (name.find("NVIDIA") != std::string::npos or vendor.find("NVIDIA") != std::string::npos)) {
      return 1000;
    };
    return -1;
  }
};

template <typename T>
void get_devices(T selector, cl::sycl::vector_class<cl::sycl::device> & devices) {
  auto const& all_devices = cl::sycl::device::get_devices();
  auto size = std::count_if(all_devices.begin(), all_devices.end(), [selector](cl::sycl::device device){ return selector(device) > 0; });
  devices.reserve(devices.size() + size);
  std::copy_if(all_devices.begin(), all_devices.end(), std::back_inserter(devices), [selector](cl::sycl::device device){ return selector(device) > 0; });
}

template <typename T>
cl::sycl::vector_class<cl::sycl::device> get_devices(T selector) {
  cl::sycl::vector_class<cl::sycl::device> devices;
  get_devices(selector, devices);
  return devices;
}

int main(int argc, char **argv) {
  cl::sycl::vector_class<cl::sycl::device> devices;
  bool selected = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--default") == 0) {
      devices.emplace_back(cl::sycl::default_selector{});
      selected = true;
    }
    else if (std::strcmp(argv[i], "--host") == 0) {
      devices.emplace_back(cl::sycl::host_selector{});
      selected = true;
    }
    else if (std::strcmp(argv[i], "--cpu") == 0) {
      ::get_devices(cl::sycl::cpu_selector{}, devices);
      selected = true;
    }
    else if (std::strcmp(argv[i], "--gpu") == 0) {
      ::get_devices(cl::sycl::gpu_selector{}, devices);
      selected = true;
    }
    else if (std::strcmp(argv[i], "--cuda") == 0) {
      ::get_devices(cuda_selector{}, devices);
      selected = true;
    }
    else
      std::cout << "Ignoring unknown option " << argv[i] << std::endl;
  }

  // run on all devices by default
  if (not selected) {
    devices = cl::sycl::device::get_devices();
  }

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  std::unique_ptr<Output> output = std::make_unique<Output>();
  double totaltime = 0;

  for (auto & device: devices) {
    std::cout << std::endl;
    std::cout << "Running on SYCL device " << device.get_info<cl::sycl::info::device::name>() << ", driver version "
              << device.get_info<cl::sycl::info::device::driver_version>() << std::endl;
    oneapi::analyze(device, input, *output, totaltime);
    std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
              << totaltime << " us" << std::endl;
  }

  return 0;
}
