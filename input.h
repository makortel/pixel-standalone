#ifndef input_h_
#define input_h_

#include <fstream>
#include <iostream>

#include "pixelgpudetails.h"

#ifdef DIGI_KOKKOSVIEW
#include <Kokkos_Core.hpp>
#endif

struct alignas(128) Input {
  SiPixelFedCablingMapGPU cablingMap;
  unsigned int word[pixelgpudetails::MAX_FED_WORDS];
  unsigned char fedId[pixelgpudetails::MAX_FED_WORDS];
  unsigned int wordCounter;
};

inline Input read_input() {
  Input ret;

  std::ifstream file{"dump.bin", std::ios::binary};
  file.read(reinterpret_cast<char*>(&ret.cablingMap), sizeof(SiPixelFedCablingMapGPU));
  file.read(reinterpret_cast<char*>(&ret.wordCounter), sizeof(unsigned int));
  file.read(reinterpret_cast<char*>(&ret.word), sizeof(unsigned int) * ret.wordCounter);
  file.read(reinterpret_cast<char*>(&ret.fedId), sizeof(unsigned char) * ret.wordCounter / 2);

  return ret;
}

#ifdef DIGI_KOKKOSVIEW
template <typename MemorySpace>
class InputKokkosDevice {
public:
  InputKokkosDevice() {
    cablingMap.fed = Kokkos::View<unsigned int*, MemorySpace>("fed", pixelgpudetails::MAX_SIZE);
    cablingMap.link = Kokkos::View<unsigned int*, MemorySpace>("link", pixelgpudetails::MAX_SIZE);
    cablingMap.roc = Kokkos::View<unsigned int*, MemorySpace>("roc", pixelgpudetails::MAX_SIZE);
    cablingMap.RawId = Kokkos::View<unsigned int*, MemorySpace>("RawId", pixelgpudetails::MAX_SIZE);
    cablingMap.rocInDet = Kokkos::View<unsigned int*, MemorySpace>("rocInDet", pixelgpudetails::MAX_SIZE);
    cablingMap.moduleId = Kokkos::View<unsigned int*, MemorySpace>("moduleId", pixelgpudetails::MAX_SIZE);
    cablingMap.badRocs = Kokkos::View<unsigned char*, MemorySpace>("badRocs", pixelgpudetails::MAX_SIZE);
    cablingMap.size = Kokkos::View<unsigned int*, MemorySpace>("size", 1);

    fedId = Kokkos::View<unsigned char*, MemorySpace>("fedId", pixelgpudetails::MAX_FED_WORDS);
    word = Kokkos::View<unsigned int*, MemorySpace>("word", pixelgpudetails::MAX_FED_WORDS);
    wordCounter = Kokkos::View<unsigned int*, MemorySpace>("wordCounter", 1);
  }

  KOKKOS_INLINE_FUNCTION
  const SiPixelFedCablingMapKokkosDevice<MemorySpace>& getCablingMap() const { return this->cablingMap; }
  KOKKOS_INLINE_FUNCTION
  const Kokkos::View<unsigned char*, MemorySpace>& getFedId() const { return this->fedId; }
  KOKKOS_INLINE_FUNCTION
  const Kokkos::View<unsigned int*, MemorySpace>& getWord() const { return this->word; }
  KOKKOS_INLINE_FUNCTION
  const Kokkos::View<unsigned int*, MemorySpace>& getWordCounter() const { return this->wordCounter; }

private:
private:
  SiPixelFedCablingMapKokkosDevice<MemorySpace> cablingMap;
  Kokkos::View<unsigned char*, MemorySpace> fedId;
  Kokkos::View<unsigned int*, MemorySpace> word;
  Kokkos::View<unsigned int*, MemorySpace> wordCounter;
};

template <typename MemorySpace>
class InputKokkosHost {
public:
  InputKokkosHost() = delete;
  ~InputKokkosHost() = default;
  InputKokkosHost(InputKokkosDevice<MemorySpace>& input_d) : input_d{input_d} {
    auto cablingMap_d = input_d.getCablingMap();
    auto word_d = input_d.getWord();
    auto fedId_d = input_d.getFedId();
    auto wordCounter_d = input_d.getWordCounter();

    // Initialize host mirror based on device view
    this->cablingMap.fed = Kokkos::create_mirror_view(cablingMap_d.fed);
    this->cablingMap.link = Kokkos::create_mirror_view(cablingMap_d.link);
    this->cablingMap.roc = Kokkos::create_mirror_view(cablingMap_d.roc);
    this->cablingMap.RawId = Kokkos::create_mirror_view(cablingMap_d.RawId);
    this->cablingMap.rocInDet = Kokkos::create_mirror_view(cablingMap_d.rocInDet);
    this->cablingMap.moduleId = Kokkos::create_mirror_view(cablingMap_d.moduleId);
    this->cablingMap.badRocs = Kokkos::create_mirror_view(cablingMap_d.badRocs);
    this->cablingMap.size = Kokkos::create_mirror_view(cablingMap_d.size);

    this->word = Kokkos::create_mirror_view(word_d);
    this->fedId = Kokkos::create_mirror_view(fedId_d);
    this->wordCounter = Kokkos::create_mirror_view(wordCounter_d);
  }

  void readInput(const Input& raw_input) {
    // Loops are required since View is not guaranteed to have contiguous memory access
    for (int i = 0; i < pixelgpudetails::MAX_SIZE; i++) {
      cablingMap.fed(i) = raw_input.cablingMap.fed[i];
      cablingMap.link(i) = raw_input.cablingMap.link[i];
      cablingMap.roc(i) = raw_input.cablingMap.roc[i];
      cablingMap.RawId(i) = raw_input.cablingMap.RawId[i];
      cablingMap.rocInDet(i) = raw_input.cablingMap.rocInDet[i];
      cablingMap.moduleId(i) = raw_input.cablingMap.moduleId[i];
      cablingMap.badRocs(i) = raw_input.cablingMap.badRocs[i];
    }
    cablingMap.size(0) = raw_input.cablingMap.size;
    wordCounter(0) = raw_input.wordCounter;
    for (int i = 0; i < wordCounter(0); i++) {
      word(i) = raw_input.word[i];
    }
    for (int i = 0; i < wordCounter(0) / 2; i++) {
      fedId(i) = raw_input.fedId[i];
    }
  }

  void copyToDevice() {
    auto cablingMap_d = input_d.getCablingMap();
    auto word_d = input_d.getWord();
    auto fedId_d = input_d.getFedId();
    auto wordCounter_d = input_d.getWordCounter();

    Kokkos::deep_copy(cablingMap_d.fed, cablingMap.fed);
    Kokkos::deep_copy(cablingMap_d.link, cablingMap.link);
    Kokkos::deep_copy(cablingMap_d.roc, cablingMap.roc);
    Kokkos::deep_copy(cablingMap_d.RawId, cablingMap.RawId);
    Kokkos::deep_copy(cablingMap_d.rocInDet, cablingMap.rocInDet);
    Kokkos::deep_copy(cablingMap_d.moduleId, cablingMap.moduleId);
    Kokkos::deep_copy(cablingMap_d.badRocs, cablingMap.badRocs);
    Kokkos::deep_copy(cablingMap_d.size, cablingMap.size);

    Kokkos::deep_copy(word_d, word);
    Kokkos::deep_copy(fedId_d, fedId);
    Kokkos::deep_copy(wordCounter_d, wordCounter);
  }

private:
  InputKokkosDevice<MemorySpace>& input_d;
  SiPixelFedCablingMapKokkosHost<MemorySpace> cablingMap;
  typename Kokkos::View<unsigned int*, MemorySpace>::HostMirror word;
  typename Kokkos::View<unsigned char*, MemorySpace>::HostMirror fedId;
  typename Kokkos::View<unsigned int*, MemorySpace>::HostMirror wordCounter;
};
#endif  // DIGI_KOKKOSVIEW

#endif  // input_h_
