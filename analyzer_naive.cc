#include <chrono>
#include <memory>

#include "input.h"
#include "loops.h"
#include "output.h"
#include "rawtodigi_naive.h"

namespace naive {
  void analyze(Input const& input, std::unique_ptr<Output>& output, double& totaltime) {
    for (int i = 0; i <= NLOOPS; ++i) {
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
                       true,
                       true,
                       i == 0);
      auto stop = std::chrono::high_resolution_clock::now();

      auto diff = stop - start;
      auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      if (i != 0) {
        totaltime += time;
      }
    }

    totaltime /= NLOOPS;
  }
}  // namespace naive
