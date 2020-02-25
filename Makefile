TARGETS = naive cuda alpaka cupla kokkos oneapi
BUILD   = build
DEBUG   = build/debug

.PHONY: all debug clean $(TARGETS)

# general rules and targets
all: $(TARGETS)

debug: $(TARGETS:%=%-debug)

clean:
	rm -r -f test-* debug-* $(BUILD) Kokkos*.o libkokkos.a $(DEBUG) env.sh

$(BUILD):
	mkdir -p $(BUILD)

$(DEBUG):
	mkdir -p $(DEBUG)

# configure external tool here
BOOST_BASE  :=
TBB_BASE    :=
CUDA_BASE   := /usr/local/cuda
ALPAKA_BASE := /usr/local/alpaka/alpaka
CUPLA_BASE  := /usr/local/alpaka/cupla
ONEAPI_BASE := /opt/intel/inteloneapi/compiler/latest/linux
DPCT_BASE   := /opt/intel/inteloneapi/dpcpp-ct/latest

# host compiler
CXX := g++
CXX_FLAGS := -O2 -std=c++14
CXX_DEBUG := -g
OMP_CXX_FLAGS := -fopenmp -foffload=disable
OMP_LD_FLAGS  := -fopenmp -foffload=disable

# CUDA compiler
ifdef CUDA_BASE
NVCC := $(CUDA_BASE)/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr -w --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_60,code=sm_60 --generate-code arch=compute_70,code=sm_70
NVCC_DEBUG := -g -lineinfo

# CUDA flags for the host linker
CUDA_LIBDIR   := $(CUDA_BASE)/lib64
CUDA_LD_FLAGS := -L$(CUDA_LIBDIR) -lcudart -lcuda
endif

# boost flags
ifdef BOOST_BASE
BOOST_CXX_FLAGS := -I$(BOOST_BASE)/include
else
BOOST_CXX_FLAGS :=
endif

# TBB flags
ifdef TBB_BASE
TBB_LIBDIR    := $(TBB_BASE)/lib
TBB_CXX_FLAGS := -I$(TBB_BASE)/include
TBB_LD_FLAGS  := -L$(TBB_LIBDIR) -ltbb -lrt
else
TBB_LIBDIR    :=
TBB_CXX_FLAGS :=
TBB_LD_FLAGS  := -ltbb -lrt
endif

# alpaka flags
ifdef ALPAKA_BASE
ALPAKA_CXX_FLAGS  := $(BOOST_CXX_FLAGS) -I$(ALPAKA_BASE)/include
ALPAKA_NVCC_FLAGS := $(BOOST_CXX_FLAGS) -I$(ALPAKA_BASE)/include -Xcompiler -pthread
ALPAKA_LD_FLAGS   := -pthread
ALPAKA_DEBUG      := -DALPAKA_DEBUG=1
endif

# cupla flags
ifdef CUPLA_BASE
CUPLA_LIBDIR     := $(CUPLA_BASE)/lib
CUPLA_CXX_FLAGS  := $(ALPAKA_CXX_FLAGS) -I$(CUPLA_BASE)/include
CUPLA_NVCC_FLAGS := $(ALPAKA_NVCC_FLAGS) -I$(CUPLA_BASE)/include
CUPLA_LD_FLAGS   := $(ALPAKA_LD_FLAGS) -L$(CUPLA_LIBDIR)
CUPLA_DEBUG      := $(ALPAKA_DEBUG)
endif

# oneAPI flags
ifdef ONEAPI_BASE
ifneq ($(wildcard $(ONEAPI_BASE)/lib/libsycl.so),)
ONEAPI_LIBDIR := $(ONEAPI_BASE)/lib
else ifneq ($(wildcard $(ONEAPI_BASE)/lib64/libsycl.so),)
ONEAPI_LIBDIR := $(ONEAPI_BASE)/lib64
else
ONEAPI_BASE :=
endif
endif
ifdef ONEAPI_BASE
ONEAPI_CXX   := $(ONEAPI_BASE)/bin/clang++
ONEAPI_FLAGS := -fsycl -I$(DPCT_BASE)/include -Wno-unknown-cuda-version
ifdef CUDA_BASE
ONEAPI_CUDA_PLUGIN := $(wildcard $(ONEAPI_BASE)/lib64/libpi_cuda.so)
ONEAPI_CUDA_FLAGS  := -fsycl-targets=nvptx64-nvidia-cuda-sycldevice --cuda-path=$(CUDA_BASE)
endif
endif

# Kokkos flags
# recommended to include only after the first target, see
# https://github.com/kokkos/kokkos/wiki/Compiling#42-using-kokkos-makefile-system
ifdef KOKKOS_BASE
  KOKKOS_DEVICES := "Serial,OpenMP"
  KOKKOS_ARCH := "SNB"
  KOKKOS_CXX_STANDARD := c++14
  # In theory this would be the right way, but the mechanism seems to be broken in Kokkos ATM
  #KOKKOS_CUDA_OPTIONS := "enable_constexpr"
  # Stuff in Makefile.kokkos uses KOKKOS_PATH
  KOKKOS_PATH := $(KOKKOS_BASE)
  # If CUDA is enabled, Makefile.kokkos complains on non-nvcc compiler
  CXX_ORIG := $(CXX)
  ifneq ($(wildcard $(CUDA_BASE)),)
    KOKKOS_DEVICES := "$(KOKKOS_DEVICES),Cuda"
    KOKKOS_ARCH := "$(KOKKOS_ARC),Volta70"
    CXX := $(KOKKOS_BASE)/bin/nvcc_wrapper
    KOKKOS_CXXFLAGS += --expt-relaxed-constexpr
  endif
  include $(KOKKOS_BASE)/Makefile.kokkos
endif

# color highlights for ANSI terminals
GREEN  := '\033[32m'
RED    := '\033[31m'
YELLOW := '\033[38;5;220m'
WHITE  := '\033[97m'
RESET  := '\033[0m'

# force the recreation of the environment file any time the Makefile is updated, before building any other target
-include environment
.PHONY: environment

environment: env.sh

env.sh: Makefile
	@echo '#! /bin/bash' > $@
	@echo -n 'export LD_LIBRARY_PATH=' >> $@
ifdef TBB_LIBDIR
	@echo -n '$(TBB_LIBDIR):' >> $@
endif
ifdef CUDA_LIBDIR
	@echo -n '$(CUDA_LIBDIR):' >> $@
endif
ifdef CUPLA_LIBDIR
	@echo -n '$(CUPLA_LIBDIR):' >> $@
endif
ifdef ONEAPI_LIBDIR
	@echo -n '$(ONEAPI_LIBDIR):' >> $@
endif
	@echo '$$LD_LIBRARY_PATH' >> $@
	@echo -e $(GREEN)Environment file$(RESET) regenerated, load the new envirnment with
	@echo
	@echo -e \ \ $(WHITE)source env.sh$(RESET)
	@echo


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
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -DDIGI_CUDA -o $@ main_cuda.cc rawtodigi_cuda.cu
else
cuda:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), CUDA targets will not be built

cuda-debug:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), CUDA debug targets will not be built

endif

ifdef ALPAKA_BASE
alpaka: test-alpaka-serial test-alpaka-tbb test-alpaka-omp2 test-alpaka-omp4 test-alpaka-cuda test-alpaka
	@echo -e $(GREEN)Alpaka targets built$(RESET)

alpaka-debug: debug-alpaka-serial debug-alpaka-tbb debug-alpaka-omp2 debug-alpaka-omp4 debug-alpaka-cuda debug-alpaka
	@echo -e $(GREEN)Alpaka debug targets built$(RESET)

# Alpaka implementation, compiled for the CPU serial backend
$(BUILD)/rawtodigi_alpaka.serial.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) -o $@ -c $<

$(BUILD)/analyzer_alpaka.serial.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_alpaka.serial.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@ -c $<

test-alpaka-serial: $(BUILD)/main_alpaka.serial.o $(BUILD)/analyzer_alpaka.serial.o $(BUILD)/rawtodigi_alpaka.serial.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(ALPAKA_LD_FLAGS)

$(DEBUG)/rawtodigi_alpaka.serial.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

$(DEBUG)/analyzer_alpaka.serial.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

$(DEBUG)/main_alpaka.serial.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

debug-alpaka-serial: $(DEBUG)/main_alpaka.serial.o $(DEBUG)/analyzer_alpaka.serial.o $(DEBUG)/rawtodigi_alpaka.serial.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(ALPAKA_LD_FLAGS)

# Alpaka implementation, compiled for the CPU parallel TBB backend
$(BUILD)/rawtodigi_alpaka.tbb.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

$(BUILD)/analyzer_alpaka.tbb.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_alpaka.tbb.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

test-alpaka-tbb: $(BUILD)/main_alpaka.tbb.o $(BUILD)/analyzer_alpaka.tbb.o $(BUILD)/rawtodigi_alpaka.tbb.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(TBB_LD_FLAGS) $(ALPAKA_LD_FLAGS)

$(DEBUG)/rawtodigi_alpaka.tbb.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/analyzer_alpaka.tbb.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/main_alpaka.tbb.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

debug-alpaka-tbb: $(DEBUG)/main_alpaka.tbb.o $(DEBUG)/analyzer_alpaka.tbb.o $(DEBUG)/rawtodigi_alpaka.tbb.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(TBB_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# Alpaka implementation, compiled for the CPU parallel OpenMP 2.0 backend
$(BUILD)/rawtodigi_alpaka.omp2.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/analyzer_alpaka.omp2.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_alpaka.omp2.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

test-alpaka-omp2: $(BUILD)/main_alpaka.omp2.o $(BUILD)/analyzer_alpaka.omp2.o $(BUILD)/rawtodigi_alpaka.omp2.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

$(DEBUG)/rawtodigi_alpaka.omp2.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/analyzer_alpaka.omp2.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/main_alpaka.omp2.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

debug-alpaka-omp2: $(DEBUG)/main_alpaka.omp2.o $(DEBUG)/analyzer_alpaka.omp2.o $(DEBUG)/rawtodigi_alpaka.omp2.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# Alpaka implementation, compiled for the CPU parallel OpenMP 4.0 backend
$(BUILD)/rawtodigi_alpaka.omp4.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/analyzer_alpaka.omp4.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_alpaka.omp4.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

test-alpaka-omp4: $(BUILD)/main_alpaka.omp4.o $(BUILD)/analyzer_alpaka.omp4.o $(BUILD)/rawtodigi_alpaka.omp4.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

$(DEBUG)/rawtodigi_alpaka.omp4.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/analyzer_alpaka.omp4.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/main_alpaka.omp4.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

debug-alpaka-omp4: $(DEBUG)/main_alpaka.omp4.o $(DEBUG)/analyzer_alpaka.omp4.o $(DEBUG)/rawtodigi_alpaka.omp4.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

ifdef CUDA_BASE
# Alpaka implementation, compiled for the GPU CUDA backend
$(BUILD)/rawtodigi_alpaka.cuda.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) -o $@ -x cu -dc $<

$(BUILD)/analyzer_alpaka.cuda.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) -o $@ -x cu -dc $<

$(BUILD)/alpaka.dlink.o: $(BUILD)/rawtodigi_alpaka.cuda.o $(BUILD)/analyzer_alpaka.cuda.o
	$(NVCC) $(NVCC_FLAGS) -o $@ -dlink $^

$(BUILD)/main_alpaka.cuda.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@ -c $<

test-alpaka-cuda: $(BUILD)/main_alpaka.cuda.o $(BUILD)/analyzer_alpaka.cuda.o $(BUILD)/rawtodigi_alpaka.cuda.o $(BUILD)/alpaka.dlink.o
	$(CXX) $(CXX_FLAGS) $(ALPAKA_CXX_FLAGS) -o $@ $+ $(CUDA_LD_FLAGS) $(ALPAKA_LD_FLAGS)

$(DEBUG)/rawtodigi_alpaka.cuda.o: rawtodigi_alpaka.cc GPUSimpleVector.h alpakaConfig.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) $(ALPAKA_DEBUG) -o $@ -x cu -dc $<

$(DEBUG)/analyzer_alpaka.cuda.o: analyzer_alpaka.cc GPUSimpleVector.h alpakaConfig.h analyzer_alpaka.h input.h output.h pixelgpudetails.h rawtodigi_alpaka.h | $(DEBUG)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ENABLED $(ALPAKA_NVCC_FLAGS) $(ALPAKA_DEBUG) -o $@ -x cu -dc $<

$(DEBUG)/alpaka.dlink.o: $(DEBUG)/rawtodigi_alpaka.cuda.o $(DEBUG)/analyzer_alpaka.cuda.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -o $@ -dlink $^

$(DEBUG)/main_alpaka.cuda.o: main_alpaka.cc analyzer_alpaka.h alpakaConfig.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

debug-alpaka-cuda: $(DEBUG)/main_alpaka.cuda.o $(DEBUG)/analyzer_alpaka.cuda.o $(DEBUG)/rawtodigi_alpaka.cuda.o $(DEBUG)/alpaka.dlink.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ $+ $(CUDA_LD_FLAGS) $(ALPAKA_LD_FLAGS)

# Alpaka implementation with run-time device choice
$(BUILD)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@ -c $<

test-alpaka: $(BUILD)/main_alpaka.o $(BUILD)/rawtodigi_alpaka.serial.o $(BUILD)/rawtodigi_alpaka.tbb.o $(BUILD)/rawtodigi_alpaka.omp2.o $(BUILD)/rawtodigi_alpaka.omp4.o $(BUILD)/rawtodigi_alpaka.cuda.o $(BUILD)/analyzer_alpaka.serial.o $(BUILD)/analyzer_alpaka.tbb.o $(BUILD)/analyzer_alpaka.omp2.o $(BUILD)/analyzer_alpaka.omp4.o $(BUILD)/analyzer_alpaka.cuda.o $(BUILD)/alpaka.dlink.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(CUDA_LD_FLAGS) $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

$(DEBUG)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

debug-alpaka: $(DEBUG)/main_alpaka.o $(DEBUG)/rawtodigi_alpaka.serial.o $(DEBUG)/rawtodigi_alpaka.tbb.o $(DEBUG)/rawtodigi_alpaka.omp2.o $(DEBUG)/rawtodigi_alpaka.omp4.o $(DEBUG)/rawtodigi_alpaka.cuda.o $(DEBUG)/analyzer_alpaka.serial.o $(DEBUG)/analyzer_alpaka.tbb.o $(DEBUG)/analyzer_alpaka.omp2.o $(DEBUG)/analyzer_alpaka.omp4.o $(DEBUG)/analyzer_alpaka.cuda.o $(DEBUG)/alpaka.dlink.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(CUDA_LD_FLAGS) $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

else
test-alpaka-cuda:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Alpaka targets using CUDA will not be built

debug-alpaka-cuda:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Alpaka debug targets using CUDA will not be built

# Alpaka implementation with run-time device choice, with the CUDA backend disabled
$(BUILD)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) -o $@ -c $<

test-alpaka: $(BUILD)/main_alpaka.o $(BUILD)/rawtodigi_alpaka.serial.o $(BUILD)/rawtodigi_alpaka.tbb.o $(BUILD)/rawtodigi_alpaka.omp2.o $(BUILD)/rawtodigi_alpaka.omp4.o $(BUILD)/analyzer_alpaka.serial.o $(BUILD)/analyzer_alpaka.tbb.o $(BUILD)/analyzer_alpaka.omp2.o $(BUILD)/analyzer_alpaka.omp4.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

$(DEBUG)/main_alpaka.o: main_alpaka.cc analyzer_alpaka.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ALPAKA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(ALPAKA_CXX_FLAGS) $(ALPAKA_DEBUG) -o $@ -c $<

debug-alpaka: $(DEBUG)/main_alpaka.o $(DEBUG)/rawtodigi_alpaka.serial.o $(DEBUG)/rawtodigi_alpaka.tbb.o $(DEBUG)/rawtodigi_alpaka.omp2.o $(DEBUG)/rawtodigi_alpaka.omp4.o $(DEBUG)/analyzer_alpaka.serial.o $(DEBUG)/analyzer_alpaka.tbb.o $(DEBUG)/analyzer_alpaka.omp2.o $(DEBUG)/analyzer_alpaka.omp4.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(TBB_LD_FLAGS) $(OMP_LD_FLAGS) $(ALPAKA_LD_FLAGS)

endif

else
alpaka:
	@echo -e $(YELLOW)Alpaka not found$(RESET), Alpaka targets will not be built

endif

ifdef CUPLA_BASE
cupla: test-cupla-cuda-async test-cupla-seq-seq-async test-cupla-seq-seq-sync test-cupla-tbb-seq-async test-cupla-omp2-seq-async test-cupla
	@echo -e $(GREEN)Cupla targets built$(RESET)

cupla-debug: debug-cupla-cuda-async debug-cupla-seq-seq-async debug-cupla-seq-seq-sync debug-cupla-tbb-seq-async debug-cupla-omp2-seq-async debug-cupla
	@echo -e $(GREEN)Cupla debug targets built$(RESET)

ifdef CUDA_BASE
# Alpaka/Cupla implementation, with the CUDA GPU async backend
$(BUILD)/analyzer_cupla.cuda.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_NVCC_FLAGS) -o $@ -x cu -dc $<

$(BUILD)/rawtodigi_cupla.cuda.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_NVCC_FLAGS) -o $@ -x cu -dc $<

$(BUILD)/cupla.dlink.o: $(BUILD)/analyzer_cupla.cuda.o $(BUILD)/rawtodigi_cupla.cuda.o
	$(NVCC) $(NVCC_FLAGS) -o $@ -dlink $(BUILD)/analyzer_cupla.cuda.o $(BUILD)/rawtodigi_cupla.cuda.o

$(BUILD)/main_cupla.cuda.o: main_cupla.cc analyzer_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) -o $@ -c $<

test-cupla-cuda-async: $(BUILD)/main_cupla.cuda.o $(BUILD)/analyzer_cupla.cuda.o $(BUILD)/rawtodigi_cupla.cuda.o $(BUILD)/cupla.dlink.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CUPLA_LD_FLAGS) $(CUDA_LD_FLAGS) -lcupla-cuda

$(DEBUG)/analyzer_cupla.cuda.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(DEBUG)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_NVCC_FLAGS) $(CUPLA_DEBUG) -o $@ -x cu -dc $<

$(DEBUG)/rawtodigi_cupla.cuda.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(DEBUG)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_NVCC_FLAGS) $(CUPLA_DEBUG) -o $@ -x cu -dc $<

$(DEBUG)/cupla.dlink.o: $(DEBUG)/analyzer_cupla.cuda.o $(DEBUG)/rawtodigi_cupla.cuda.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEBUG) -o $@ -dlink $(DEBUG)/analyzer_cupla.cuda.o $(DEBUG)/rawtodigi_cupla.cuda.o

$(DEBUG)/main_cupla.cuda.o: main_cupla.cc analyzer_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) -o $@ -c $<

debug-cupla-cuda-async: $(DEBUG)/main_cupla.cuda.o $(DEBUG)/analyzer_cupla.cuda.o $(DEBUG)/rawtodigi_cupla.cuda.o $(DEBUG)/cupla.dlink.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $^ $(CUPLA_LD_FLAGS) $(CUDA_LD_FLAGS) -lcupla-cuda

# Alpaka/Cupla implementation with run-time device choice
$(BUILD)/main_cupla.o: main_cupla.cc analyzer_cupla.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

test-cupla: $(BUILD)/main_cupla.o $(BUILD)/rawtodigi_cupla.seq-seq-sync.o $(BUILD)/rawtodigi_cupla.seq-seq-async.o $(BUILD)/rawtodigi_cupla.tbb-seq-async.o $(BUILD)/rawtodigi_cupla.omp2-seq-async.o $(BUILD)/rawtodigi_cupla.omp4-async.o $(BUILD)/rawtodigi_cupla.cuda.o $(BUILD)/analyzer_cupla.seq-seq-sync.o $(BUILD)/analyzer_cupla.seq-seq-async.o $(BUILD)/analyzer_cupla.tbb-seq-async.o $(BUILD)/analyzer_cupla.omp2-seq-async.o $(BUILD)/analyzer_cupla.omp4-async.o $(BUILD)/analyzer_cupla.cuda.o $(BUILD)/cupla.dlink.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(CUPLA_LD_FLAGS) $(CUDA_LD_FLAGS) -lcupla-cuda -lcupla-serial -lcupla-tbb -lcupla-omp2-blocks -lcupla-omp4 $(TBB_LD_FLAGS) $(OMP_LD_FLAGS)

$(DEBUG)/main_cupla.o: main_cupla.cc analyzer_cupla.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND -DALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

debug-cupla: $(DEBUG)/main_cupla.o $(DEBUG)/rawtodigi_cupla.seq-seq-sync.o $(DEBUG)/rawtodigi_cupla.seq-seq-async.o $(DEBUG)/rawtodigi_cupla.tbb-seq-async.o $(DEBUG)/rawtodigi_cupla.omp2-seq-async.o $(DEBUG)/rawtodigi_cupla.omp4-async.o $(DEBUG)/rawtodigi_cupla.cuda.o $(DEBUG)/analyzer_cupla.seq-seq-sync.o $(DEBUG)/analyzer_cupla.seq-seq-async.o $(DEBUG)/analyzer_cupla.tbb-seq-async.o $(DEBUG)/analyzer_cupla.omp2-seq-async.o $(DEBUG)/analyzer_cupla.omp4-async.o $(DEBUG)/analyzer_cupla.cuda.o $(DEBUG)/cupla.dlink.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(CUPLA_LD_FLAGS) -lcupla-cuda -lcupla-serial -lcupla-tbb -lcupla-omp2-blocks -lcupla-omp4 $(CUDA_LD_FLAGS) $(TBB_LD_FLAGS) $(OMP_LD_FLAGS)

else
test-cupla-cuda-async:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Cupla targets using CUDA will not be built

debug-cupla-cuda-async:
	@echo -e $(YELLOW)NVIDIA CUDA not found$(RESET), Cupla debug targets using CUDA will not be built

# Alpaka/Cupla implementation with run-time device choice, without the CUDA backend
$(BUILD)/main_cupla.o: main_cupla.cc analyzer_cupla.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

test-cupla: $(BUILD)/main_cupla.o $(BUILD)/rawtodigi_cupla.seq-seq-sync.o $(BUILD)/rawtodigi_cupla.seq-seq-async.o $(BUILD)/rawtodigi_cupla.tbb-seq-async.o $(BUILD)/rawtodigi_cupla.omp2-seq-async.o $(BUILD)/rawtodigi_cupla.omp4-async.o $(BUILD)/analyzer_cupla.seq-seq-sync.o $(BUILD)/analyzer_cupla.seq-seq-async.o $(BUILD)/analyzer_cupla.tbb-seq-async.o $(BUILD)/analyzer_cupla.omp2-seq-async.o $(BUILD)/analyzer_cupla.omp4-async.o $(BUILD)/cupla.dlink.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(CUPLA_LD_FLAGS) -lcupla-serial -lcupla-tbb -lcupla-omp2-blocks -lcupla-omp4 $(CUDA_LD_FLAGS) $(TBB_LD_FLAGS) $(OMP_LD_FLAGS)

$(DEBUG)/main_cupla.o: main_cupla.cc analyzer_cupla.h input.h pixelgpudetails.h modules.h output.h GPUSimpleVector.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND -DALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

debug-cupla: $(DEBUG)/main_cupla.o $(DEBUG)/rawtodigi_cupla.seq-seq-sync.o $(DEBUG)/rawtodigi_cupla.seq-seq-async.o $(DEBUG)/rawtodigi_cupla.tbb-seq-async.o $(DEBUG)/rawtodigi_cupla.omp2-seq-async.o $(DEBUG)/rawtodigi_cupla.omp4-async.o $(DEBUG)/analyzer_cupla.seq-seq-sync.o $(DEBUG)/analyzer_cupla.seq-seq-async.o $(DEBUG)/analyzer_cupla.tbb-seq-async.o $(DEBUG)/analyzer_cupla.omp2-seq-async.o $(DEBUG)/analyzer_cupla.omp4-async.o $(DEBUG)/cupla.dlink.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $+ $(CUPLA_LD_FLAGS) $(CUDA_LD_FLAGS) -lcupla-serial -lcupla-tbb -lcupla-omp2-blocks -lcupla-omp4 $(TBB_LD_FLAGS) $(OMP_LD_FLAGS)

endif

# Alpaka/Cupla implementation, with the serial CPU async backend
$(BUILD)/analyzer_cupla.seq-seq-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) -o $@ -c $<

$(BUILD)/rawtodigi_cupla.seq-seq-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_cupla.seq-seq-async.o: main_cupla.cc analyzer_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) -o $@ -c $<

test-cupla-seq-seq-async: $(BUILD)/main_cupla.seq-seq-async.o $(BUILD)/analyzer_cupla.seq-seq-async.o $(BUILD)/rawtodigi_cupla.seq-seq-async.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-serial

$(DEBUG)/analyzer_cupla.seq-seq-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) -o $@ -c $<

$(DEBUG)/rawtodigi_cupla.seq-seq-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) -o $@ -c $<

$(DEBUG)/main_cupla.seq-seq-async.o: main_cupla.cc analyzer_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) -o $@ -c $<

debug-cupla-seq-seq-async: $(DEBUG)/main_cupla.seq-seq-async.o $(DEBUG)/analyzer_cupla.seq-seq-async.o $(DEBUG)/rawtodigi_cupla.seq-seq-async.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-serial

# Alpaka/Cupla implementation, with the serial CPU sync backend
$(BUILD)/analyzer_cupla.seq-seq-sync.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(CUPLA_CXX_FLAGS) -o $@ -c $<

$(BUILD)/rawtodigi_cupla.seq-seq-sync.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(CUPLA_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_cupla.seq-seq-sync.o: main_cupla.cc analyzer_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(CUPLA_CXX_FLAGS) -o $@ -c $<

test-cupla-seq-seq-sync: $(BUILD)/main_cupla.seq-seq-sync.o $(BUILD)/analyzer_cupla.seq-seq-sync.o $(BUILD)/rawtodigi_cupla.seq-seq-sync.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-serial

$(DEBUG)/analyzer_cupla.seq-seq-sync.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) -o $@ -c $<

$(DEBUG)/rawtodigi_cupla.seq-seq-sync.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) -o $@ -c $<

$(DEBUG)/main_cupla.seq-seq-sync.o: main_cupla.cc analyzer_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) -o $@ -c $<

debug-cupla-seq-seq-sync: $(DEBUG)/main_cupla.seq-seq-sync.o $(DEBUG)/analyzer_cupla.seq-seq-sync.o $(DEBUG)/rawtodigi_cupla.seq-seq-sync.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-serial

# Alpaka/Cupla implementation, with the TBB blocks backend
$(BUILD)/analyzer_cupla.tbb-seq-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

$(BUILD)/rawtodigi_cupla.tbb-seq-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_cupla.tbb-seq-async.o: main_cupla.cc analyzer_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(TBB_CXX_FLAGS) -o $@ -c $<

test-cupla-tbb-seq-async: $(BUILD)/main_cupla.tbb-seq-async.o $(BUILD)/analyzer_cupla.tbb-seq-async.o $(BUILD)/rawtodigi_cupla.tbb-seq-async.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-tbb $(TBB_LD_FLAGS)

$(DEBUG)/analyzer_cupla.tbb-seq-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/rawtodigi_cupla.tbb-seq-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/main_cupla.tbb-seq-async.o: main_cupla.cc analyzer_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(TBB_CXX_FLAGS) -o $@ -c $<

debug-cupla-tbb-seq-async: $(DEBUG)/main_cupla.tbb-seq-async.o $(DEBUG)/analyzer_cupla.tbb-seq-async.o $(DEBUG)/rawtodigi_cupla.tbb-seq-async.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-tbb $(TBB_LD_FLAGS)

# Alpaka/Cupla implementation, with the OpenMP 2.0 block-parallel backend
$(BUILD)/analyzer_cupla.omp2-seq-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/rawtodigi_cupla.omp2-seq-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_cupla.omp2-seq-async.o: main_cupla.cc analyzer_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

test-cupla-omp2-seq-async: $(BUILD)/main_cupla.omp2-seq-async.o $(BUILD)/analyzer_cupla.omp2-seq-async.o $(BUILD)/rawtodigi_cupla.omp2-seq-async.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-omp2-blocks $(OMP_LD_FLAGS)

$(DEBUG)/analyzer_cupla.omp2-seq-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/rawtodigi_cupla.omp2-seq-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/main_cupla.omp2-seq-async.o: main_cupla.cc analyzer_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

debug-cupla-omp2-seq-async: $(DEBUG)/main_cupla.omp2-seq-async.o $(DEBUG)/analyzer_cupla.omp2-seq-async.o $(DEBUG)/rawtodigi_cupla.omp2-seq-async.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-omp2-blocks $(OMP_LD_FLAGS)

# Alpaka/Cupla implementation, with the OpenMP 4.0 parallel backend
$(BUILD)/analyzer_cupla.omp4-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_BT_OMP4_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/rawtodigi_cupla.omp4-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_BT_OMP4_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

$(BUILD)/main_cupla.omp4-async.o: main_cupla.cc analyzer_cupla.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_BT_OMP4_BACKEND $(CUPLA_CXX_FLAGS) $(OMP_CXX_FLAGS) -o $@ -c $<

test-cupla-omp4-async: $(BUILD)/main_cupla.omp4-async.o $(BUILD)/analyzer_cupla.omp4-async.o $(BUILD)/rawtodigi_cupla.omp4-async.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CUPLA_LD_FLAGS) $(OMP_LD_FLAGS) -lcupla-omp4

$(DEBUG)/analyzer_cupla.omp4-async.o: analyzer_cupla.cc analyzer_cupla.h rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_BT_OMP4_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/rawtodigi_cupla.omp4-async.o: rawtodigi_cupla.cc rawtodigi_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_BT_OMP4_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 -DALPAKA_ACC_CPU_BT_OMP4_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

$(DEBUG)/main_cupla.omp4-async.o: main_cupla.cc analyzer_cupla.h | $(DEBUG)
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_CUPLA -DALPAKA_ACC_CPU_BT_OMP4_BACKEND $(CUPLA_CXX_FLAGS) $(CUPLA_DEBUG) $(OMP_CXX_FLAGS) -o $@ -c $<

debug-cupla-omp4-async: $(DEBUG)/main_cupla.omp4-async.o $(DEBUG)/analyzer_cupla.omp4-async.o $(DEBUG)/rawtodigi_cupla.omp4-async.o
	$(CXX) $(CXX_FLAGS) $(CXX_DEBUG) -o $@ $^ $(CUPLA_LD_FLAGS) -lcupla-omp4 $(OMP_LD_FLAGS)

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
$(BUILD)/rawtodigi_kokkos.serial.o: rawtodigi_kokkos.cc rawtodigi_kokkos.h pixelgpudetails.h input.h output.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS) -o $@ -c $<

$(BUILD)/analyzer_kokkos.serial.o: analyzer_kokkos.cc analyzer_kokkos.h input.h output.h rawtodigi_kokkos.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS) -o $@ -c $<

$(BUILD)/main_kokkos.serial.o: main_kokkos.cc analyzer_kokkos.h input.h modules.h output.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL -o $@ -c $<

# KOKKOS_CXXFLAGS needed for -fopenmp in the final link
test-kokkos-serial: $(BUILD)/main_kokkos.serial.o $(BUILD)/analyzer_kokkos.serial.o $(BUILD)/rawtodigi_kokkos.serial.o | $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS)

# Kokkos implementation, OpenMP backend
$(BUILD)/rawtodigi_kokkos.omp.o: rawtodigi_kokkos.cc rawtodigi_kokkos.h pixelgpudetails.h input.h output.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_OPENMP $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS) -o $@ -c $<

$(BUILD)/analyzer_kokkos.omp.o: analyzer_kokkos.cc analyzer_kokkos.h input.h output.h rawtodigi_kokkos.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_OPENMP $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS) -o $@ -c $<

$(BUILD)/main_kokkos.omp.o: main_kokkos.cc analyzer_kokkos.h input.h modules.h output.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_OPENMP -o $@ -c $<

test-kokkos-openmp:  $(BUILD)/main_kokkos.omp.o $(BUILD)/analyzer_kokkos.omp.o $(BUILD)/rawtodigi_kokkos.omp.o | $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS)

# Kokkos implementation, CUDA backend
$(BUILD)/rawtodigi_kokkos.cuda.o: rawtodigi_kokkos.cc rawtodigi_kokkos.h pixelgpudetails.h input.h output.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS) -o $@ -c $<

$(BUILD)/analyzer_kokkos.cuda.o: analyzer_kokkos.cc analyzer_kokkos.h input.h output.h rawtodigi_kokkos.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS) -o $@ -c $<

$(BUILD)/main_kokkos.cuda.o: main_kokkos.cc analyzer_kokkos.h input.h modules.h output.h | $(BUILD)
	$(CXX) $(CXX_FLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DDIGI_KOKKOS -DDIGI_KOKKOS_SERIAL -o $@ -c $<

test-kokkos-cuda:  $(BUILD)/main_kokkos.cuda.o $(BUILD)/analyzer_kokkos.cuda.o $(BUILD)/rawtodigi_kokkos.cuda.o | $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(KOKKOS_CXXFLAGS) $(KOKKOS_CXXLDFLAGS) $(KOKKOS_LIBS)

else
kokkos:
	@echo -e $(YELLOW)Kokkos not found$(RESET), Kokkos targets will not be built

kokkos-debug:

endif

ifdef ONEAPI_BASE
oneapi: test-oneapi test-oneapi-cuda
	@echo -e $(GREEN)oneAPI targets built$(RESET)

oneapi-debug: debug-oneapi debug-oneapi-cuda
	@echo -e $(GREEN)oneAPI debug targets built$(RESET)

# oneAPI implementation
test-oneapi: main_oneapi.cc analyzer_oneapi.cc analyzer_oneapi.h rawtodigi_oneapi.cc rawtodigi_oneapi.h
	$(ONEAPI_CXX) $(ONEAPI_FLAGS) $(CXX_FLAGS) -DDIGI_ONEAPI -o $@ main_oneapi.cc analyzer_oneapi.cc rawtodigi_oneapi.cc

debug-oneapi: main_oneapi.cc analyzer_oneapi.cc analyzer_oneapi.h rawtodigi_oneapi.cc rawtodigi_oneapi.h
	$(ONEAPI_CXX) $(ONEAPI_FLAGS) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ONEAPI -o $@ main_oneapi.cc analyzer_oneapi.cc rawtodigi_oneapi.cc

ifdef ONEAPI_CUDA_PLUGIN
test-oneapi-cuda: main_oneapi.cc analyzer_oneapi.cc analyzer_oneapi.h rawtodigi_oneapi.cc rawtodigi_oneapi.h
	$(ONEAPI_CXX) $(ONEAPI_FLAGS) $(ONEAPI_CUDA_FLAGS) $(CXX_FLAGS) -DDIGI_ONEAPI -o $@ main_oneapi.cc analyzer_oneapi.cc rawtodigi_oneapi.cc

debug-oneapi-cuda: main_oneapi.cc analyzer_oneapi.cc analyzer_oneapi.h rawtodigi_oneapi.cc rawtodigi_oneapi.h
	$(ONEAPI_CXX) $(ONEAPI_FLAGS) $(ONEAPI_CUDA_FLAGS) $(CXX_FLAGS) $(CXX_DEBUG) -DDIGI_ONEAPI -o $@ main_oneapi.cc analyzer_oneapi.cc rawtodigi_oneapi.cc

else
test-oneapi-cuda:
	@echo -e $(YELLOW)NVIDIA CUDA support not found$(RESET), oneAPI targets using CUDA will not be built

debug-oneapi-cuda:
	@echo -e $(YELLOW)NVIDIA CUDA support not found$(RESET), oneAPI debug targets using CUDA will not be built

endif

else
oneapi:
	@echo -e $(YELLOW)Intel oneAPI not found$(RESET), oneAPI targets will not be built

oneapi-debug:
	@echo -e $(YELLOW)Intel oneAPI not found$(RESET), oneAPI debug targets will not be built

endif
