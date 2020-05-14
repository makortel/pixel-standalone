#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "analyzer_kokkos.h"
#include "input.h"
#include "kokkosConfig.h"
#include "modules.h"
#include "output.h"

namespace {
  void print_help(std::string const& name) {
    std::cout
        << name << ": [--numberOfThreads NT]"
        << "\n\n"
        << "Options\n"
        << " --numberOfThreads   Number of threads to use (default -1 for all cores(?)1)\n"
        << std::endl;
  }
}  // namespace

int main(int argc, char **argv) {
  std::vector<std::string> args(argv, argv + argc);
  int numberOfThreads = -1;
  for (auto i = args.begin() + 1, e = args.end(); i != e; ++i) {
    if (*i == "-h" or *i == "--help") {
      print_help(args.front());
      return EXIT_SUCCESS;
    } else if (*i == "--numberOfThreads") {
      ++i;
      numberOfThreads = std::stoi(*i);
    }
  }

  {
    Kokkos::InitArguments arguments;
    arguments.num_threads = numberOfThreads;
    kokkos_common::initialize(arguments);
  }
 
  InputKokkos<KokkosMemSpace> input;
  input.ReadInput();

  std::cout << "Got " << input.GetHostCablingMapSize() << " for cabling, wordCounter " << input.GetHostWordCounter() << std::endl;

  std::unique_ptr<Output> output;
  double totaltime = 0;

#ifdef DIGI_KOKKOS_SERIAL
  totaltime = 0;
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the serial CPU backend..." << std::endl;
  kokkos_serial::analyze<KokkosMemSpace>(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.GetHostWordCounter()) << " modules in " << totaltime << " us"
            << std::endl;
#endif

#ifdef DIGI_KOKKOS_OPENMP
  totaltime = 0;
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the OpenMP backend..." << std::endl;
  kokkos_openmp::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.GetHostWordCounter()) << " modules in " << totaltime << " us"
            << std::endl;
#endif

#ifdef DIGI_KOKKOS_CUDA
  totaltime = 0;
  output = std::make_unique<Output>();
  std::cout << "\nRunning with the CUDA backend..." << std::endl;
  kokkos_cuda::analyze(input, *output, totaltime);
  std::cout << "Output: " << countModules(output->moduleInd, input.GetHostWordCounter()) << " modules in " << totaltime << " us"
            << std::endl;
#endif

  kokkos_common::finalize();

  return 0;
}
