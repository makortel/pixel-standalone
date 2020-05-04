#ifndef input_h_
#define input_h_

#include <fstream>

#include "pixelgpudetails.h"

#ifdef DIGI_KOKKOS
#include <Kokkos_Core.hpp>
#endif

struct alignas(128) Input {
  SiPixelFedCablingMapGPU cablingMap;
  unsigned int word[pixelgpudetails::MAX_FED_WORDS];
  unsigned char fedId[pixelgpudetails::MAX_FED_WORDS];
  unsigned int wordCounter;
};

#ifdef DIGI_KOKKOS
template <typename MemorySpace>
struct KokkosInput {
  KokkosSiPixelFedCablingMap<MemorySpace> cablingMap;
  Kokkos::View<unsigned int [pixelgpudetails::MAX_FED_WORDS], MemorySpace> word;
  Kokkos::View<unsigned char [pixelgpudetails::MAX_FED_WORDS], MemorySpace> fedId;
  Kokkos::View<unsigned int [1], MemorySpace> wordCounter;
};
#endif

inline Input read_input() {
  Input ret;

  std::ifstream file{"dump.bin", std::ios::binary};
  file.read(reinterpret_cast<char *>(&ret.cablingMap), sizeof(SiPixelFedCablingMapGPU));
  file.read(reinterpret_cast<char *>(&ret.wordCounter), sizeof(unsigned int));
  file.read(reinterpret_cast<char *>(&ret.word), sizeof(unsigned int) * ret.wordCounter);
  file.read(reinterpret_cast<char *>(&ret.fedId), sizeof(unsigned char) * ret.wordCounter / 2);

  return ret;
}

#endif  // input_h_
