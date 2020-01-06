#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "analyzer_alpaka.h"
#include "input.h"
#include "modules.h"
#include "output.h"

int main(int argc, char** argv) {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  std::unique_ptr<Output> output;
  double totaltime = 0;

  output = std::make_unique<Output>();
  std::cout << "\nRunning with the CPU serial backend..." << std::endl;
  Alpaka::CPU_SERIAL::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;

  output = std::make_unique<Output>();
  std::cout << "\nRunning with the CPU TBB parallel backend..." << std::endl;
  Alpaka::CPU_TBB::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;

  output = std::make_unique<Output>();
  std::cout << "\nRunning with the GPU CUDA backend..." << std::endl;
  Alpaka::GPU_CUDA::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;

  return 0;
}
