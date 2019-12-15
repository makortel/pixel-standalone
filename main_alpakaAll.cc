#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

// #include "input.h"
// #include "output.h"

#include "rawtodigi_alpakaAll.h"

int main(){

    Alpaka::CPU_SERIAL::rawtodigi();

    Alpaka::CPU_TBB::rawtodigi();
    
    Alpaka::GPU_CUDA::rawtodigi();

    return 0;
}