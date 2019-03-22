#ifndef INPUT_H
#define INPUT_H

#include <fstream>

#include "pixelgpudetails.h"

struct alignas(128) Input {
  SiPixelFedCablingMapGPU cablingMap;
  unsigned int word[pixelgpudetails::MAX_FED_WORDS];
  unsigned char fedId[pixelgpudetails::MAX_FED_WORDS];
  unsigned int wordCounter;
};

Input read_input() {
  Input ret;

  std::ifstream file{"dump.bin", std::ios::binary};
  file.read(reinterpret_cast<char *>(&ret.cablingMap), sizeof(SiPixelFedCablingMapGPU));
  file.read(reinterpret_cast<char *>(&ret.wordCounter), sizeof(unsigned int));
  file.read(reinterpret_cast<char *>(&ret.word), sizeof(unsigned int)*ret.wordCounter);
  file.read(reinterpret_cast<char *>(&ret.fedId), sizeof(unsigned char)*ret.wordCounter/2);

  return ret;
}

#endif
