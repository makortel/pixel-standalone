#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

// #include "input.h"
// #include "output.h"

#include "rawtodigi_alpakaAll.h"

int main(){


    CPU_SERIAL::Alpaka::rawtodigi();

    CPU_TBB::Alpaka::rawtodigi();

    GPU_CUDA::Alpaka::rawtodigi();

    

    return 0;
}