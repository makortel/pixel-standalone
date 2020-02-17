#include <iostream>
#include <memory>

#include "analyzer_cupla.h"
#include "input.h"
#include "modules.h"
#include "output.h"

int main(int argc, char** argv) {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  std::unique_ptr<Output> output;
  double totaltime = 0;

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the blocking serial CPU backend..." << std::endl;
  cupla_seq_seq_sync::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the non-blocking serial CPU backend..." << std::endl;
  cupla_seq_seq_async::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the non-blocking TBB CPU backend..." << std::endl;
  cupla_tbb_seq_async::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the non-blocking OpenMP 2.0 blocks CPU backend..." << std::endl;
  cupla_omp2_seq_async::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the non-blocking OpenMP 4.0 CPU backend..." << std::endl;
  cupla_omp4_omp4_async::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND

#ifdef ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the non-blocking CUDA GPU backend..." << std::endl;
  cupla_cuda_async::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;
#endif  // ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND

  return 0;
}
