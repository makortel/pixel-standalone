#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "analyzer_kokkos.h"
#include "input.h"
#include "modules.h"
#include "output.h"

int main(int argc, char **argv) {
  kokkos::initialize(argc, argv);

  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  std::unique_ptr<Output> output;
  double totaltime = 0;

  output = std::make_unique<Output>();
  //std::cout << "\nRunning with the serial CPU backend..." << std::endl;
  kokkos::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in " << totaltime << " us"
            << std::endl;

  kokkos::finalize();

  return 0;
}
