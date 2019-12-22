#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

// #include "input.h"
// #include "output.h"

#include "rawtodigi_alpakaAll.h"
using namespace Alpaka;
int main(){

    CPU_SERIAL::rawtodigi();

    CPU_TBB::rawtodigi();
    
    GPU_CUDA::rawtodigi();

    return 0;
}
