TARGETS = naive alpakaSer alpakaTBB alpakaGpuCuda cuda cupla-cuda-async cupla-seq-seq-async cupla-seq-seq-sync cupla-tbb-seq-async cupla-omp2-seq-async kokkos-serial kokkos-openmp kokkos-cuda

.PHONY: all debug clean $(TARGETS)

EIGEN_ROOT := /data/cmssw/slc7_amd64_gcc820/external/eigen/e4c107b451c52c9ab2d7b7fa4194ee35332916ec-pafccj/include/eigen3
TBB_ROOT := /data/cmssw/slc7_amd64_gcc820/external/tbb/2019_U3-pafccj
BOOST :=  

CXX := g++ 
CXX_FLAGS := -O2 -std=c++14 -ftemplate-depth-512 -I/data/cmssw/slc7_amd64_gcc820/external/boost/1.67.0-pafccj/include/ -I$(TBB_ROOT)/include -L$(TBB_ROOT)/lib
CXX_DEBUG := -g



NVCC := /data/cmssw/slc7_amd64_gcc820/external/cuda/10.1.105-pafccj2/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr -w -lineinfo -DALPAKA_CUDA_ARCH=60:70:75 -I/data/cmssw/slc7_amd64_gcc820/external/boost/1.67.0-pafccj/include/ -I$(TBB_ROOT) -L$(TBB_ROOT)/lib
NVCC_DEBUG := -g -lineinfo

BASE := /data/user/wredjeb
ALPAKA_BASE := $(BASE)/cupla/alpaka
CUPLA_BASE  := $(BASE)/cupla
CUPLA_FLAGS := -DALPAKA_DEBUG=0 -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include

all: $(TARGETS)

# Recommended to include only after the first target
# https://github.com/kokkos/kokkos/wiki/Compiling#42-using-kokkos-makefile-system
# ifdef KOKKOS_BASE
#   include $(KOKKOS_BASE)/Makefile.kokkos
#   CXX_KOKKOS := $(CXX)
#   ifeq ($(KOKKOS_INTERNAL_USE_CUDA),1)
#     CXX_KOKKOS := $(NVCC_WRAPPER)
#     KOKKOS_CXXFLAGS += --expt-relaxed-constexpr
#   endif
# endif


debug: $(TARGETS:%=debug-%)

clean:
	rm -f $(TARGETS:%=main-%) $(TARGETS:%=debug-%)

# Naive CPU implementation
naive: main-naive

main-naive: main_naive.cc rawtodigi_naive.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE -o $@ main_naive.cc

debug-naive: main_naive.cc rawtodigi_naive.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE $(CXX_DEBUG) -o $@ main_naive.cc

#ALPAKA implementation

alpakaSer: main-alpaka-ser

main-alpaka-ser: main_alpaka.cc rawtodigi_alpaka.cc
	$(CXX) $(CXX_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DDIGI_ALPAKA -DALPAKA_ARCHITECTURE=CPU_SERIAL -I$(ALPAKA_BASE)/include/alpaka/ $(CUPLA_FLAGS) -o $@ $^

alpakaTBB: main-alpaka-tbb

main-alpaka-tbb: main_alpaka.cc rawtodigi_alpaka.cc
	$(CXX) $(CXX_FLAGS) -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DDIGI_ALPAKA -DALPAKA_ARCHITECTURE=CPU_TBB -I$(ALPAKA_BASE)/include/alpaka/ $(CUPLA_FLAGS) -o $@ $^ -ltbb -lrt -pthread 

alpakaGpuCuda: main-alpaka-gpu

main-alpaka-gpu: main_alpaka.cc rawtodigi_alpaka.cc
	$(NVCC) -x cu $(NVCC_FLAGS) -DALPAKA_ACC_GPU_CUDA_ENABLED -DDIGI_ALPAKA -DALPAKA_ARCHITECTURE=GPU_CUDA -I$(ALPAKA_BASE)/include/alpaka/ $(CUPLA_FLAGS) -o $@ $^

#Alpaka/cupla implementation, with the CUDA GPU async backend
cupla-cuda-async: main-cupla-cuda-async

main-cupla-cuda-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/GpuCudaRt.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-cuda-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/GpuCudaRt.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) $(NVCC_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the serial CPU async backend
cupla-seq-seq-async: main-cupla-seq-seq-async

main-cupla-seq-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-seq-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the serial CPU sync backend
cupla-seq-seq-sync: main-cupla-seq-seq-sync

main-cupla-seq-seq-sync: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-seq-seq-sync: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the TBB blocks backend
cupla-tbb-seq-async: main-cupla-tbb-seq-async

main-cupla-tbb-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuTbbBlocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc -ltbb -lrt

debug-cupla-tbb-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuTbbBlocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc -ltbb -lrt

# Alpaka/cupla implementation, with the OpenMP 2 blocks backend
cupla-omp2-seq-async: main-cupla-omp2-seq-async

main-cupla-omp2-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuOmp2Blocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -fopenmp -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-omp2-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "$(CUPLA_BASE)/include/cupla/config/CpuOmp2Blocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -fopenmp $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

ifdef KOKKOS_BASE
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -I$(ALPAKA_BASE)/include/alpaka/ $(CUPLA_FLAGS) -o $@ main_alpakaSerial.cc rawtodigi_alpakaSer.cc
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
