#include <algorithm>
#include <cstdio>

#include "alpakaConfig.h"
#include "analyzer_alpaka.h"
#include "input.h"
#include "output.h"
#include "pixelgpudetails.h"
#include "rawtodigi_alpaka.h"

namespace {
  constexpr int NLOOPS = 100;
}

namespace ALPAKA_ARCHITECTURE {

  void analyze(Input const& input, Output& output, double& totaltime) {
    const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    const DevAcc device(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    const Vec size(1u);

    Queue queue(device);

    totaltime = 0.;

    for (int i = 0; i < NLOOPS; i++) {
      using ViewInput = alpaka::mem::view::ViewPlainPtr<DevHost, const Input, Dim, Idx>;
      ViewInput input_hBuf(&input, host, size);

      auto input_dBuf(alpaka::mem::buf::alloc<Input, Idx>(device, size));
      Input* input_d = alpaka::mem::view::getPtrNative(input_dBuf);

      using ViewOutput = alpaka::mem::view::ViewPlainPtr<DevHost, Output, Dim, Idx>;
      ViewOutput output_hBuf(&output, host, size);

      auto output_dBuf(alpaka::mem::buf::alloc<Output, Idx>(device, size));
      Output* output_d = alpaka::mem::view::getPtrNative(output_dBuf);
      output.err.construct(pixelgpudetails::MAX_FED_WORDS, output_d->err_d);

      auto start = std::chrono::high_resolution_clock::now();

      alpaka::mem::view::copy(queue, input_dBuf, input_hBuf, size);
      alpaka::mem::view::copy(queue, output_dBuf, output_hBuf, size);

      rawtodigi(input_d,
                output_d,
                input.wordCounter,
                true,
                true,
                false,
                queue);

      alpaka::mem::view::copy(queue, output_hBuf, output_dBuf, size);

      alpaka::wait::wait(queue);

      auto stop = std::chrono::high_resolution_clock::now();

      auto diff = stop - start;
      totaltime += std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    }

    totaltime /= NLOOPS;
  }

}  // namespace ALPAKA_ARCHITECTURE
