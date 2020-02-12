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

enum class DeviceType { all_devices = -1, default_device = 0, host_device, cpu_device, gpu_device, cuda_device };

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
cl::sycl::vector_class<cl::sycl::device> get_devices(T selector) {
  auto const& all_devices = cl::sycl::device::get_devices();
  cl::sycl::vector_class<cl::sycl::device> devices;
  auto size = std::count_if(all_devices.begin(), all_devices.end(), [selector](cl::sycl::device device){ return selector(device) > 0; });
  devices.reserve(size);
  std::copy_if(all_devices.begin(), all_devices.end(), std::back_inserter(devices), [selector](cl::sycl::device device){ return selector(device) > 0; });
  return devices;
}

int main(int argc, char **argv) {
  DeviceType device_type = DeviceType::all_devices;
  if (argc > 1) {
    if (std::strcmp(argv[1], "--all") == 0)
      device_type = DeviceType::all_devices;
    else if (std::strcmp(argv[1], "--default") == 0)
      device_type = DeviceType::default_device;
    else if (std::strcmp(argv[1], "--host") == 0)
      device_type = DeviceType::host_device;
    else if (std::strcmp(argv[1], "--cpu") == 0)
      device_type = DeviceType::cpu_device;
    else if (std::strcmp(argv[1], "--gpu") == 0)
      device_type = DeviceType::gpu_device;
    else if (std::strcmp(argv[1], "--cuda") == 0)
      device_type = DeviceType::cuda_device;
    else
      std::cout << "Ignoring unknown option " << argv[1] << std::endl;
  }

  cl::sycl::vector_class<cl::sycl::device> devices;

  switch (device_type) {
    case DeviceType::all_devices: {
      devices = cl::sycl::device::get_devices();
      break;
    }
    case DeviceType::default_device: {
      cl::sycl::default_selector selector;
      devices.push_back(cl::sycl::device{selector});
      break;
    }
    case DeviceType::host_device: {
      cl::sycl::host_selector selector;
      devices.push_back(cl::sycl::device{selector});
      break;
    }
    case DeviceType::cpu_device: {
      cl::sycl::cpu_selector selector;
      devices = ::get_devices(selector);
      break;
    }
    case DeviceType::gpu_device: {
      cl::sycl::gpu_selector selector;
      devices = ::get_devices(selector);
      break;
    }
    case DeviceType::cuda_device: {
      cuda_selector selector;
      devices = ::get_devices(selector);
      break;
    }
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
              << (static_cast<double>(totaltime) / NLOOPS) << " us" << std::endl;
  }

  return 0;
}
