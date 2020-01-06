#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>

#include "input.h"
#include "output.h"

#include "rawtodigi_alpaka.h"

namespace {
  constexpr int NLOOPS = 100;
}

namespace gpuClustering {
  constexpr uint32_t MaxNumModules = 2000;
  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules
}  // namespace gpuClustering

int countModules(const uint16_t *id, int numElements) {
  int nmod = 0;
  for (int i = 0; i < numElements; ++i) {
    if (id[i] == gpuClustering::InvId)
      continue;
    auto j = i - 1;
    while (j >= 0 and id[j] == gpuClustering::InvId) {
      --j;
    }
    if (j < 0 or id[j] != id[i]) {
      ++nmod;
    }
  }
  return nmod;
}

int main() {
  Input input = read_input();
  std::cout << "Got " << input.cablingMap.size << " for cabling, wordCounter " << input.wordCounter << std::endl;

  using namespace Alpaka::ALPAKA_ARCHITECTURE;

  DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
  DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
  Idx const elements(1);
  Vec const extent(elements);

  Queue queue(devAcc);

  int totaltime = 0;

  std::unique_ptr<Output> output;

  for (int i = 0; i < NLOOPS; i++) {
    output = std::make_unique<Output>();

    using ViewInput = alpaka::mem::view::ViewPlainPtr<DevHost, Input, Dim, Idx>;
    ViewInput input_hBuf(&input, devHost, extent);

    using BufDevInput = alpaka::mem::buf::Buf<DevAcc, Input, Dim, Idx>;
    BufDevInput input_dBuf(alpaka::mem::buf::alloc<Input, Idx>(devAcc, extent));

    alpaka::mem::view::copy(queue, input_dBuf, input_hBuf, extent);

    // auto output_dBuf(alpaka::mem::buf::alloc<Output, Idx>(devAcc, extent));
    // auto const output_hBuf(alpaka::mem::buf::alloc<Output, Idx>(devHost, extent));

    using ViewOutput = alpaka::mem::view::ViewPlainPtr<DevHost, Output, Dim, Idx>;
    ViewOutput output_hBuf(output.get(), devHost, extent);

    using BufDevOutput = alpaka::mem::buf::Buf<DevAcc, Output, Dim, Idx>;
    BufDevOutput output_dBuf(alpaka::mem::buf::alloc<Output, Idx>(devAcc, extent));

    auto start = std::chrono::high_resolution_clock::now();

    rawtodigi(alpaka::mem::view::getPtrNative(input_dBuf),
              alpaka::mem::view::getPtrNative(output_dBuf),
              input.wordCounter,
              true,
              true,
              true,
              queue);

    alpaka::mem::view::copy(queue, output_hBuf, output_dBuf, extent);

    alpaka::wait::wait(queue);

    auto stop = std::chrono::high_resolution_clock::now();

    auto diff = stop - start;
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    totaltime += time;

  }  //end for i to NLOOPS

  std::cout << "Output: " << countModules(output->moduleInd, input.wordCounter) << " modules in "
            << (static_cast<double>(totaltime) / NLOOPS) << " us" << std::endl;

  return 0;
}
