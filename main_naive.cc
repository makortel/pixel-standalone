#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "input.h"
#include "modules.h"
#include "output.h"

#include "rawtodigi_naive.h"

namespace {
  constexpr int NLOOPS = 100;
}

int main(int argc, char **argv) {

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  int totaltime = 0;

  std::unique_ptr<Output> output;
  for(int i=0; i<NLOOPS; ++i) {
    output = std::make_unique<Output>();

    auto start = std::chrono::high_resolution_clock::now();
    naive::rawtodigi(&input.cablingMap,
                     input.wordCounter,
                     input.word,
                     input.fedId,
                     output->xx,
                     output->yy,
                     output->adc,
                     output->digi,
                     output->rawIdArr,
                     output->moduleInd,
                     &output->err,
                     true, true, false);
    auto stop = std::chrono::high_resolution_clock::now();

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
  }
  
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime)/NLOOPS) << " us"
            << std::endl;

  return 0;
}
