#ifndef input_h_
#define input_h_

#include <fstream>
#include <iostream>

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

inline Input read_input() {
  Input ret;

  std::ifstream file{"dump.bin", std::ios::binary};
  file.read(reinterpret_cast<char *>(&ret.cablingMap), sizeof(SiPixelFedCablingMapGPU));
  file.read(reinterpret_cast<char *>(&ret.wordCounter), sizeof(unsigned int));
  file.read(reinterpret_cast<char *>(&ret.word), sizeof(unsigned int) * ret.wordCounter);
  file.read(reinterpret_cast<char *>(&ret.fedId), sizeof(unsigned char) * ret.wordCounter / 2);

  return ret;
}

#ifdef DIGI_KOKKOS
template <typename MemorySpace>
struct InputKokkosDevice{
  SiPixelFedCablingMapKokkosDevice<MemorySpace> cablingMap;
  Kokkos::View<unsigned int [pixelgpudetails::MAX_FED_WORDS], MemorySpace> word;
  Kokkos::View<unsigned char [pixelgpudetails::MAX_FED_WORDS], MemorySpace> fedId;
  Kokkos::View<unsigned int [1], MemorySpace> wordCounter;
};

template <typename MemorySpace>
struct InputKokkosHost{
  SiPixelFedCablingMapKokkosHost<MemorySpace> cablingMap;
  typename Kokkos::View<unsigned int [pixelgpudetails::MAX_FED_WORDS], MemorySpace>::HostMirror word;
  typename Kokkos::View<unsigned char [pixelgpudetails::MAX_FED_WORDS], MemorySpace>::HostMirror fedId;
  typename Kokkos::View<unsigned int [1], MemorySpace>::HostMirror wordCounter;
};

template <typename MemorySpace>
class InputKokkos {
  public:
    InputKokkos() {CreateMirrorView();}
    ~InputKokkos() = default;

    InputKokkos& operator=(const InputKokkos&) = delete;

    void ReadInput() {
      Input raw_input = read_input();

      // Initialize host mirror based on raw data
      // Loops are required since View is not guaranteed to have contiguous memory access
      for (int i = 0; i < pixelgpudetails::MAX_SIZE; i++) {
        input_h.cablingMap.fed(i) = raw_input.cablingMap.fed[i];
        input_h.cablingMap.link(i) = raw_input.cablingMap.link[i];
        input_h.cablingMap.roc(i) = raw_input.cablingMap.roc[i];
        input_h.cablingMap.RawId(i) = raw_input.cablingMap.RawId[i];
        input_h.cablingMap.rocInDet(i) = raw_input.cablingMap.rocInDet[i];
        input_h.cablingMap.moduleId(i) = raw_input.cablingMap.moduleId[i];
        input_h.cablingMap.badRocs(i) = raw_input.cablingMap.badRocs[i];
      }
      input_h.cablingMap.size(0) = raw_input.cablingMap.size;

      input_h.wordCounter(0) = raw_input.wordCounter;

      for (int i = 0; i < input_h.wordCounter(0); i++) {
        input_h.word(i) = raw_input.word[i];
      }

      for (int i = 0; i < input_h.wordCounter(0) / 2; i++) {
        input_h.fedId(i) = raw_input.fedId[i];
      }
      DeepCopyH2D();
    }

    void DeepCopyH2D() {
      Kokkos::deep_copy (input_d.cablingMap.fed, input_h.cablingMap.fed);
      Kokkos::deep_copy (input_d.cablingMap.link, input_h.cablingMap.link);
      Kokkos::deep_copy (input_d.cablingMap.roc, input_h.cablingMap.roc);
      Kokkos::deep_copy (input_d.cablingMap.RawId, input_h.cablingMap.RawId);
      Kokkos::deep_copy (input_d.cablingMap.rocInDet, input_h.cablingMap.rocInDet);
      Kokkos::deep_copy (input_d.cablingMap.moduleId, input_h.cablingMap.moduleId);
      Kokkos::deep_copy (input_d.cablingMap.badRocs, input_h.cablingMap.badRocs);
      Kokkos::deep_copy (input_d.cablingMap.size, input_h.cablingMap.size);

      Kokkos::deep_copy (input_d.word, input_h.word);
      Kokkos::deep_copy (input_d.fedId, input_h.fedId);
      Kokkos::deep_copy (input_d.wordCounter, input_h.wordCounter);
    }
    void DeepCopyD2H() {
      Kokkos::deep_copy (input_h.cablingMap.fed, input_d.cablingMap.fed);
      Kokkos::deep_copy (input_h.cablingMap.link, input_d.cablingMap.link);
      Kokkos::deep_copy (input_h.cablingMap.roc, input_d.cablingMap.roc);
      Kokkos::deep_copy (input_h.cablingMap.RawId, input_d.cablingMap.RawId);
      Kokkos::deep_copy (input_h.cablingMap.rocInDet, input_d.cablingMap.rocInDet);
      Kokkos::deep_copy (input_h.cablingMap.moduleId, input_d.cablingMap.moduleId);
      Kokkos::deep_copy (input_h.cablingMap.badRocs, input_d.cablingMap.badRocs);
      Kokkos::deep_copy (input_h.cablingMap.size, input_d.cablingMap.size);

      Kokkos::deep_copy (input_h.word, input_d.word);
      Kokkos::deep_copy (input_h.fedId, input_d.fedId);
      Kokkos::deep_copy (input_h.wordCounter, input_d.wordCounter);
    }

    unsigned int GetHostCablingMapSize() {return input_h.cablingMap.size(0);}
    unsigned int GetHostWordCounter() {return input_h.wordCounter(0);}


    KOKKOS_INLINE_FUNCTION
    const Kokkos::View<unsigned char [pixelgpudetails::MAX_FED_WORDS], MemorySpace>& GetDeviceFedId() const {return input_d.fedId;}
    KOKKOS_INLINE_FUNCTION
    const Kokkos::View<unsigned int [pixelgpudetails::MAX_FED_WORDS], MemorySpace>& GetDeviceWord() const {return input_d.word;}
    KOKKOS_INLINE_FUNCTION
    const SiPixelFedCablingMapKokkosDevice<MemorySpace>& GetDeviceCablingMap() const {return input_d.cablingMap;}

  private:
    void CreateMirrorView() {
      input_h.cablingMap.fed = Kokkos::create_mirror_view(input_d.cablingMap.fed);
      input_h.cablingMap.link = Kokkos::create_mirror_view(input_d.cablingMap.link);
      input_h.cablingMap.roc = Kokkos::create_mirror_view(input_d.cablingMap.roc);
      input_h.cablingMap.RawId = Kokkos::create_mirror_view(input_d.cablingMap.RawId);
      input_h.cablingMap.rocInDet = Kokkos::create_mirror_view(input_d.cablingMap.rocInDet);
      input_h.cablingMap.moduleId = Kokkos::create_mirror_view(input_d.cablingMap.moduleId);
      input_h.cablingMap.badRocs = Kokkos::create_mirror_view(input_d.cablingMap.badRocs);
      input_h.cablingMap.size = Kokkos::create_mirror_view(input_d.cablingMap.size);

      input_h.word = Kokkos::create_mirror_view(input_d.word);
      input_h.fedId = Kokkos::create_mirror_view(input_d.fedId);
      input_h.wordCounter = Kokkos::create_mirror_view(input_d.wordCounter);
    }

    InputKokkosHost<MemorySpace> input_h;
    InputKokkosDevice<MemorySpace> input_d;
};
#endif

#endif  // input_h_
