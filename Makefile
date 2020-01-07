TARGETS = naive cuda alpaka cupla kokkos oneapi
BUILD   = build

.PHONY: all debug clean $(TARGETS)

# general rules and targets
all: $(TARGETS)

debug: $(TARGETS:%=%-debug)

clean:
	rm -r -f test-* debug-* $(BUILD)

$(BUILD):
	mkdir -p $(BUILD)

# configure external tool here
BOOST_BASE  :=
TBB_BASE    :=
CUDA_BASE   := /usr/local/cuda
ALPAKA_BASE := /usr/local/alpaka/alpaka
CUPLA_BASE  := /usr/local/alpaka/cupla

# host compiler
CXX := g++
CXX_FLAGS := -O2 -std=c++14
CXX_DEBUG := -g

# CUDA compiler
NVCC := $(CUDA_BASE)/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr -w --generate-code arch=compute_35,code=sm_35
NVCC_DEBUG := -g -lineinfo

# boost flags
ifdef BOOST_BASE
BOOST_CXX_FLAGS := -I$(BOOST_BASE)/include
else
BOOST_CXX_FLAGS :=
endif

# TBB flags
ifdef TBB_BASE
TBB_CXX_FLAGS := -I$(TBB_BASE)/include
TBB_LD_FLAGS  := -L$(TBB_BASE)/lib -ltbb -lrt
else
TBB_CXX_FLAGS :=
TBB_LD_FLAGS  := -ltbb -lrt
endif

# alpaka flags
ALPAKA_FLAGS := -DALPAKA_DEBUG=0 -I$(ALPAKA_BASE)/include $(BOOST_CXX_FLAGS)

# cupla flags
CUPLA_FLAGS  := $(ALPAKA_FLAGS) -I$(CUPLA_BASE)/include

# oneAPI flags
ONEAPI_CXX := $(shell which dpcpp 2> /dev/null)

# Kokkos flags
# recommended to include only after the first target, see
# https://github.com/kokkos/kokkos/wiki/Compiling#42-using-kokkos-makefile-system
ifdef KOKKOS_BASE
  include $(KOKKOS_BASE)/Makefile.kokkos
  ifeq ($(KOKKOS_INTERNAL_USE_CUDA),1)
    CXX_KOKKOS := $(NVCC_WRAPPER)
    KOKKOS_CXXFLAGS += --expt-relaxed-constexpr
  else
    CXX_KOKKOS := $(CXX)
  endif
endif

# color highlights for ANSI terminals
GREEN  := '\033[32m'
RED    := '\033[31m'
YELLOW := '\033[38;5;220m'
RESET  := '\033[0m'

# Naive CPU implementation
naive: test-naive
	@echo -e $(GREEN)naive targets built$(RESET)

naive-debug: debug-naive
	@echo -e $(GREEN)naive debug targets built$(RESET)

test-naive: main_naive.cc rawtodigi_naive.h analyzer_naive.cc analyzer_naive.h modules.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE -o $@ main_naive.cc analyzer_naive.cc

debug-naive: main_naive.cc rawtodigi_naive.h analyzer_naive.cc analyzer_naive.h modules.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE $(CXX_DEBUG) -o $@ main_naive.cc

ifdef CUDA_BASE
# CUDA implementation
cuda: test-cuda
	@echo -e $(GREEN)CUDA targets built$(RESET)

cuda-debug: debug-cuda
	@echo -e $(GREEN)CUDA debug targets built$(RESET)

$(BUILD)/rawtodigi_cuda.o: rawtodigi_cuda.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUDA -o $@ -x cu -c $<

$(BUILD)/analyzer_cuda.o: analyzer_cuda.cc | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUDA -I$(CUDA_BASE)/include -o $@ -c $<

$(BUILD)/main_cuda.o: main_cuda.cc | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUDA -I$(CUDA_BASE)/include -o $@ -c $<

test-cuda: $(BUILD)/main_cuda.o $(BUILD)/analyzer_cuda.o $(BUILD)/rawtodigi_cuda.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ -L$(CUDA_BASE)/lib64 -lcudart -lcuda

debug-cuda: main_cuda.cc rawtodigi_cuda.cu rawtodigi_cuda.h
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUDA $(NVCC_DEBUG) -o $@ main_cuda.cc rawtodigi_cuda.cu
else
cuda:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), CUDA targets will not be built

cuda-debug:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), CUDA debug targets will not be built

endif

ifdef ALPAKA_BASE
alpaka: test-alpaka-serial test-alpaka-tbb test-alpaka-cuda test-alpaka
	@echo -e $(GREEN)Alpaka targets built $(RESET)

# Alpaka implementation with compile-time device choice

test-alpaka-serial: main_alpaka.cc rawtodigi_alpaka.cc rawtodigi_alpaka.h analyzer_alpaka.cc analyzer_alpaka.h modules.h
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_FLAGS) -o $@ main_alpaka.cc rawtodigi_alpaka.cc analyzer_alpaka.cc

test-alpaka-tbb: main_alpaka.cc rawtodigi_alpaka.cc rawtodigi_alpaka.h analyzer_alpaka.cc analyzer_alpaka.h modules.h
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_FLAGS) $(TBB_CXX_FLAGS) -o $@ main_alpaka.cc rawtodigi_alpaka.cc analyzer_alpaka.cc $(TBB_LD_FLAGS) -pthread

ifdef CUDA_BASE
test-alpaka-cuda: main_alpaka.cc rawtodigi_alpaka.cc rawtodigi_alpaka.h analyzer_alpaka.cc analyzer_alpaka.h modules.h
	$(NVCC) -x cu $(NVCC_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_FLAGS) -o $@ main_alpaka.cc analyzer_alpaka.cc rawtodigi_alpaka.cc

else
test-alpaka-cuda:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Alpaka targets using CUDA will not be built

endif

# Alpaka implementation with run-time device choice
$(BUILD)/main_alpakaAll.o: main_alpakaAll.cc analyzer_alpaka.h modules.h $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA $(ALPAKA_FLAGS) -pthread -o $@ -c $<

$(BUILD)/rawtodigi_alpaka.serial.o: rawtodigi_alpaka.cc rawtodigi_alpaka.h alpakaConfig.h input.h output.h pixelgpudetails.h $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_FLAGS) -pthread -o $@ -c $<

$(BUILD)/rawtodigi_alpaka.tbb.o: rawtodigi_alpaka.cc rawtodigi_alpaka.h alpakaConfig.h input.h output.h pixelgpudetails.h $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_FLAGS) $(TBB_CXX_FLAGS) -pthread -o $@ -c $<

$(BUILD)/rawtodigi_alpaka.cuda.o: rawtodigi_alpaka.cc rawtodigi_alpaka.h alpakaConfig.h input.h output.h pixelgpudetails.h $(BUILD)
	$(NVCC) -x cu $(NVCC_FLAGS)  -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_FLAGS) -Xcompiler -pthread -o $@ -c $<

$(BUILD)/analyzer_alpaka.serial.o: analyzer_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h output.h pixelgpudetails.h $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_FLAGS) -pthread -o $@ -c $<

$(BUILD)/analyzer_alpaka.tbb.o: analyzer_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h output.h pixelgpudetails.h $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_FLAGS) $(TBB_CXX_FLAGS) -pthread -o $@ -c $<

$(BUILD)/analyzer_alpaka.cuda.o: analyzer_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h output.h pixelgpudetails.h $(BUILD)
	$(NVCC) -x cu $(NVCC_FLAGS)  -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_FLAGS) -Xcompiler -pthread -o $@ -c $<

test-alpaka: $(BUILD)/main_alpakaAll.o $(BUILD)/rawtodigi_alpaka.serial.o $(BUILD)/rawtodigi_alpaka.tbb.o $(BUILD)/rawtodigi_alpaka.cuda.o $(BUILD)/analyzer_alpaka.serial.o $(BUILD)/analyzer_alpaka.tbb.o $(BUILD)/analyzer_alpaka.cuda.o
	$(NVCC) $(NVCC_FLAGS) -DDIGI_ALPAKA $(ALPAKA_FLAGS) -Xcompiler -pthread $(TBB_LD_FLAGS) -o $@ $+
else
alpaka:
	@echo -e $(YELLOW)Alpaka not found$(RESET), Alpaka targets will not be built

endif

ifdef CUPLA_BASE
cupla: test-cupla-cuda-async test-cupla-seq-seq-async test-cupla-seq-seq-sync test-cupla-tbb-seq-async test-cupla-omp2-seq-async
	@echo -e $(GREEN)Cupla targets built$(RESET)

cupla-debug: debug-cupla-cuda-async debug-cupla-seq-seq-async debug-cupla-seq-seq-sync debug-cupla-tbb-seq-async debug-cupla-omp2-seq-async
	@echo -e $(GREEN)Cupla debug targets built$(RESET)

ifdef CUDA_BASE
# Alpaka/cupla implementation, with the CUDA GPU async backend
test-cupla-cuda-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -include "cupla/config/GpuCudaRt.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-cuda-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -include "cupla/config/GpuCudaRt.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) $(NVCC_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

else
test-cupla-cuda-async:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Cupla targets using CUDA will not be built

debug-cupla-cuda-async:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Cupla debug targets using CUDA will not be built

endif

# Alpaka/cupla implementation, with the serial CPU async backend
test-cupla-seq-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-seq-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the serial CPU sync backend
test-cupla-seq-seq-sync: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-seq-seq-sync: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuSerial.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the TBB blocks backend
test-cupla-tbb-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuTbbBlocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) $(TBB_CXX_FLAGS) -pthread -o $@ main_cupla.cc rawtodigi_cupla.cc $(TBB_LD_FLAGS)

debug-cupla-tbb-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuTbbBlocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) $(TBB_CXX_FLAGS) -pthread $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc $(TBB_LD_FLAGS)

# Alpaka/cupla implementation, with the OpenMP 2 blocks backend
test-cupla-omp2-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuOmp2Blocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -fopenmp -o $@ main_cupla.cc rawtodigi_cupla.cc

debug-cupla-omp2-seq-async: main_cupla.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -include "cupla/config/CpuOmp2Blocks.hpp" -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -fopenmp $(CXX_DEBUG) -o $@ main_cupla.cc rawtodigi_cupla.cc

else
cupla:
	@echo -e $(YELLOW)Cupla not found$(RESET), Cupla targets will not be built

cupla-debug:
	@echo -e $(YELLOW)Cupla not found$(RESET), Cupla debug targets will not be built

endif

ifdef KOKKOS_BASE
kokkos: test-kokkos-serial test-kokkos-openmp test-kokkos-cuda
	@echo -e $(GREEN)Kokkos targets built$(RESET)

kokkos-debug:

# Kokkos implementation, serial backend
test-kokkos-serial: main_kokkos.cc rawtodigi_kokkos.h
	$(CXX_KOKKOS) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL -o $@ main_kokkos.cc $(KOKKOS_LIBS)

# Kokkos implementation, OpenMP backend
test-kokkos-openmp: main_kokkos.cc rawtodigi_kokkos.h
	$(CXX_KOKKOS) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_OPENMP -o $@ main_kokkos.cc $(KOKKOS_LIBS)

# Kokkos implementation, CUDA backend
test-kokkos-cuda: main_kokkos.cc rawtodigi_kokkos.h
	$(CXX_KOKKOS) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_CUDA -o $@ main_kokkos.cc $(KOKKOS_LIBS)

else
kokkos:
	@echo -e $(YELLOW)Kokkos not found$(RESET), Kokkos targets will not be built

kokkos-debug:

endif

ifdef ONEAPI_CXX
oneapi: test-oneapi
	@echo -e $(GREEN)oneAPI targets built$(RESET)

oneapi-debug: debug-oneapi
	@echo -e $(GREEN)oneAPI debug targets built$(RESET)

# oneAPI implementation
test-oneapi: main_oneapi.cc rawtodigi_oneapi.cc rawtodigi_oneapi.h
	$(ONEAPI_CXX) -O2 -std=c++14 -DDIGI_ONEAPI -DDIGI_ONEAPI_WORKAROUND -o $@ main_oneapi.cc rawtodigi_oneapi.cc

debug-oneapi: main_oneapi.cc rawtodigi_oneapi.cc rawtodigi_oneapi.h
	$(ONEAPI_CXX) -g -O2 -std=c++14 -DDIGI_ONEAPI -DDIGI_ONEAPI_WORKAROUND -o $@ main_oneapi.cc rawtodigi_oneapi.cc

else
oneapi:
	@echo -e $(YELLOW)Intel oneAPI not found$(RESET), oneAPI targets will not be built

oneapi-debug:
	@echo -e $(YELLOW)Intel oneAPI not found$(RESET), oneAPI debug targets will not be built
endif
