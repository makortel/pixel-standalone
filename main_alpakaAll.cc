#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "analyzer_alpaka.h"
#include "input.h"

using namespace Alpaka;

int main(int argc, char **argv) {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  std::cout << "\nRunning with the CPU serial backend..." << std::endl;
  CPU_SERIAL::analyze(input);

  std::cout << "\nRunning with the CPU TBB parallel backend..." << std::endl;
  CPU_TBB::analyze(input);

  std::cout << "\nRunning with the GPU CUDA backend..." << std::endl;
  GPU_CUDA::analyze(input);

  return 0;
}
