#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "analyzer_alpaka.h"
#include "input.h"
#include "modules.h"
#include "output.h"

int main() {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  std::unique_ptr<Output> output = std::make_unique<Output>();
  double totaltime = 0;

  Alpaka::ALPAKA_ARCHITECTURE::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;

  return 0;
}
