#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "analyzer_alpaka.h"
#include "input.h"
#include "output.h"

int main() {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  Alpaka::ALPAKA_ARCHITECTURE::analyze(input);

  return 0;
}
