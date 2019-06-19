TARGETS = naive cuda cupla-cuda-async cupla-seq-seq-async cupla-seq-seq-sync cupla-tbb-seq-async cupla-omp2-seq-async kokkos-serial kokkos-openmp kokkos-cuda

.PHONY: all debug clean $(TARGETS)

CXX := g++-8
CXX_FLAGS := -O2 -std=c++14 -ftemplate-depth-512
CXX_DEBUG := -g

NVCC := /usr/local/cuda-10.1/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr -w --generate-code arch=compute_50,code=sm_50
NVCC_DEBUG := -g -lineinfo

ALPAKA_BASE := $(HOME)/src/alpaka/alpaka
CUPLA_BASE  := $(HOME)/src/alpaka/cupla
CUPLA_FLAGS := -DALPAKA_DEBUG=0 -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include

all: $(TARGETS)

# Recommended to include only after the first target
# https://github.com/kokkos/kokkos/wiki/Compiling#42-using-kokkos-makefile-system
ifdef KOKKOS_BASE
  include $(KOKKOS_BASE)/Makefile.kokkos
  CXX_KOKKOS := $(CXX)
  ifeq ($(KOKKOS_INTERNAL_USE_CUDA),1)
    CXX_KOKKOS := $(NVCC_WRAPPER)
    KOKKOS_CXXFLAGS += --expt-relaxed-constexpr
  endif
endif


debug: $(TARGETS:%=debug-%)

clean:
	rm -f $(TARGETS:%=main-%) $(TARGETS:%=debug-%)

# Naive CPU implementation
naive: main-naive

main-naive: main_naive.cc rawtodigi_naive.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE -o $@ main_naive.cc

debug-naive: main_naive.cc rawtodigi_naive.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE $(CXX_DEBUG) -o $@ main_naive.cc

# CUDA implementation
cuda: main-cuda

main-cuda: main_cuda.cc rawtodigi_cuda.cu rawtodigi_cuda.h
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUDA -o $@ main_cuda.cc rawtodigi_cuda.cu

debug-cuda: main_cuda.cc rawtodigi_cuda.cu rawtodigi_cuda.h
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUDA $(NVCC_DEBUG) -o $@ main_cuda.cc rawtodigi_cuda.cu

# Alpaka/cupla implementation, with the CUDA GPU async backend
cupla-cuda-async: main-cupla-cuda-async

main-cupla-cuda-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/GpuCudaRt.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-cuda-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/GpuCudaRt.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) $(NVCC_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the serial CPU async backend
cupla-seq-seq-async: main-cupla-seq-seq-async

main-cupla-seq-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-seq-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the serial CPU sync backend
cupla-seq-seq-sync: main-cupla-seq-seq-sync

main-cupla-seq-seq-sync: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-seq-seq-sync: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the TBB blocks backend
cupla-tbb-seq-async: main-cupla-tbb-seq-async

main-cupla-tbb-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuTbbBlocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc -ltbb -lrt

debug-cupla-tbb-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuTbbBlocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc -ltbb -lrt

# Alpaka/cupla implementation, with the OpenMP 2 blocks backend
cupla-omp2-seq-async: main-cupla-omp2-seq-async

main-cupla-omp2-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuOmp2Blocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -fopenmp -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-omp2-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/standalone/CpuOmp2Blocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -fopenmp $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc


ifdef KOKKOS_BASE
# Kokkos implementation, serial backend
kokkos-serial: main-kokkos-serial

main-kokkos-serial: main_kokkos.cc rawtodigi_kokkos.h
	$(CXX_KOKKOS) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL -o $@ main_kokkos.cc $(KOKKOS_LIBS)

# Kokkos implementation, OpenMP backend
kokkos-openmp: main-kokkos-openmp

main-kokkos-openmp: main_kokkos.cc rawtodigi_kokkos.h
	$(CXX_KOKKOS) $(CXX_FLAGS)$(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_OPENMP -o $@ main_kokkos.cc $(KOKKOS_LIBS)

# Kokkos implementation, CUDA backend
kokkos-cuda: main-kokkos-cuda

main-kokkos-cuda: main_kokkos.cc rawtodigi_kokkos.h
	$(CXX_KOKKOS) $(CXX_FLAGS)$(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_CUDA -o $@ main_kokkos.cc $(KOKKOS_LIBS)

else
kokkos-serial:

kokkos-openmp:

kokkos-cuda:

endif
