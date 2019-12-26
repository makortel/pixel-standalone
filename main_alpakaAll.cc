#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "rawtodigi_alpakaAll.h"

using namespace Alpaka;

int main(int argc, char **argv) {

    Input input = read_input();
    std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

    CPU_SERIAL::rawtodigi(input);

    CPU_TBB::rawtodigi(input);
    
    GPU_CUDA::rawtodigi(input);

    return 0;
}
