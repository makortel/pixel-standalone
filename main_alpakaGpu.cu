#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "input.h"
#include "output.h"

#include "rawtodigi_alpakaGpu.h"

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
    // using CpuSerial = CpuSerial<1u>;
    // using Idx = CpuSerial::Idx;
    using namespace GPU_CUDA;

    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    Idx const elements(1);
    Vec const extent(elements);

    // //Creating a queue on the device
    QueueSync queue(devAcc); //Synchronus
     
    int totaltime = 0;

    std::unique_ptr<Output> output;

    for(int i=0; i < NLOOPS; i++){
    
    output = std::make_unique<Output>();

    auto const input_dBuf(alpaka::mem::buf::alloc<Input, Idx>(devAcc, extent ));
    auto const input_hBuf(alpaka::mem::buf::alloc<Input, Idx>(devHost, extent));

    auto* pinput_dBuf(alpaka::mem::view::getPtrNative(input_dBuf));
    pinput_dBuf = &input;

    auto output_dBuf(alpaka::mem::buf::alloc<Output, Idx>(devAcc, extent));
    auto const output_hBuf(alpaka::mem::buf::alloc<Output, Idx>(devHost, extent));

    auto* poutput_dBuf(alpaka::mem::view::getPtrNative(output_dBuf));

    poutput_dBuf = output.get();

    // std::cout << pinput_dBuf << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    Alpaka::rawtodigi(pinput_dBuf, poutput_dBuf, input.wordCounter, true, true, true, queue);

    // //Sync queue?

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