#include <cstring>
#include <iostream>
#include <memory>

#include "analyzer_naive.h"
#include "input.h"
#include "modules.h"
#include "output.h"

int main(int argc, char **argv) {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  double totaltime = 0;

  std::unique_ptr<Output> output;
  naive::analyze(input, output, totaltime);

  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;

  return 0;
}
