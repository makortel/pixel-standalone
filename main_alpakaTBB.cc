#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "input.h"
#include "output.h"


/* Do NOT include other headers that use CUDA runtime functions or variables
 * before this include, because cupla renames CUDA host functions and device
 * built-in variables using macros and macro functions.
 * Do NOT include other specific includes such as `<cuda.h>`, etc.
 */

#include "rawtodigi_alpakaSerial.h"

namespace {
  constexpr int NLOOPS = 100;
}

namespace gpuClustering {
  constexpr uint32_t MaxNumModules  = 2000;
  constexpr uint16_t InvId          = 9999;         // must be > MaxNumModules
}

int countModules(const uint16_t *id, int numElements) {
  int nmod = 0;
  for(int i=0; i<numElements; ++i) {
    if(id[i] == gpuClustering::InvId)
      continue;
    auto j = i-1;
    while(j >= 0 and id[j] == gpuClustering::InvId) {
      --j;
    }
    if(j < 0 or id[j] != id[i]) {
      ++nmod;
    }
  }
  return nmod;
}

int main(){
    
    std::cout << "ALPAKA" << std::endl;

    Input input = read_input();
    std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;
    
    //Selecting the device to execute on
    using CpuParallelTBB = CpuParallelTBB<1u>;
    using Idx = CpuParallelTBB::Idx;

    CpuParallelTBB::DevHost const devHost(alpaka::pltf::getDevByIdx<CpuParallelTBB::PltfHost>(0u));
    CpuParallelTBB::DevAcc const devAcc(alpaka::pltf::getDevByIdx<CpuParallelTBB::PltfAcc>(0u));

    //Creating a queue on the device
    CpuParallelTBB::QueueSync queue(devAcc); //Synchronus
     
    int totaltime = 0;

    std::unique_ptr<Output> output;

    for(int i=0; i < NLOOPS; i++){
    
    output = std::make_unique<Output>();

    auto const input_dBuf(alpaka::mem::buf::alloc<Input, Idx>(devAcc, sizeof(Input)));
    auto const input_hBuf(alpaka::mem::buf::alloc<Input, Idx>(devHost, sizeof(Input)));

    auto* pinput_dBuf(alpaka::mem::view::getPtrNative(input_dBuf));
    pinput_dBuf = &input;

    auto output_dBuf(alpaka::mem::buf::alloc<Output, Idx>(devAcc, sizeof(Output)));
    auto const output_hBuf(alpaka::mem::buf::alloc<Output, Idx>(devHost, sizeof(Output)));

    auto* poutput_dBuf(alpaka::mem::view::getPtrNative(output_dBuf));

    poutput_dBuf = output.get();

    // std::cout << pinput_dBuf << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    alpaka::rawtodigi(pinput_dBuf, poutput_dBuf, input.wordCounter, false, false, false, queue);

    //Sync queue?

    auto stop = std::chrono::high_resolution_clock::now();
    
    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;
    
    }//end for i to NLOOPS

  

      std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime)/NLOOPS) << " us"
            << std::endl;
   


    


    return 0;
}