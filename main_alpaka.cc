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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the CPU serial backend..." << std::endl;
  alpaka_serial_sync::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_BACKEND

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the CPU TBB parallel backend..." << std::endl;
  alpaka_tbb_async::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_BACKEND

#ifdef ALPAKA_ACC_GPU_CUDA_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the GPU CUDA backend..." << std::endl;
  alpaka_cuda_async::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_GPU_CUDA_BACKEND

  return 0;
}
