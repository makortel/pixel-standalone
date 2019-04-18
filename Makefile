TARGETS = naive cuda cupla-cuda cupla-serial cupla-tbb

.PHONY: all debug clean $(TARGETS)

CXX := g++-7
CXX_FLAGS := -O2 -std=c++14 -ftemplate-depth-512
CXX_DEBUG := -g

NVCC := /usr/local/cuda-10.0/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr
NVCC_DEBUG := -g -lineinfo

ALPAKA_BASE := $(HOME)/src/alpaka/alpaka
CUPLA_BASE  := $(HOME)/src/alpaka/cupla
CUPLA_FLAGS := -DALPAKA_DEBUG=0 -DCUPLA_STREAM_ASYNC_ENABLED=1 -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include -L$(CUPLA_BASE)/lib

all: $(TARGETS)

debug: $(TARGETS:%=debug-%)

clean:
	rm -f $(TARGETS:%=main-%) $(TARGETS:%=debug-%)

# Naive CPU implementation
naive: main-naive

main-naive: main.cc rawtodigi_naive.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE -o main-naive main.cc

debug-naive: main.cc rawtodigi_naive.h
	$(CXX) $(CXX_FLAGS) -DDIGI_NAIVE $(CXX_DEBUG) -o debug-naive main.cc

# CUDA implementation
cuda: main-cuda

main-cuda: main.cc rawtodigi_cuda.cu rawtodigi_cuda.h
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUDA -o main-cuda main.cc rawtodigi_cuda.cu

debug-cuda: main.cc rawtodigi_cuda.cu rawtodigi_cuda.h
	$(NVCC) $(NVCC_FLAGS) -DDIGI_CUDA $(NVCC_DEBUG) -o debug-cuda main.cc rawtodigi_cuda.cu

# Alpaka/cupla implementation, with CUDA backend
cupla-cuda: main-cupla-cuda

main-cupla-cuda: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED $(CUPLA_FLAGS) -lcupla-cuda -o main-cupla-cuda main.cc rawtodigi_cupla.cc

debug-cupla-cuda: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED $(CUPLA_FLAGS) -lcupla-cuda $(NVCC_DEBUG) -o debug-cupla-cuda main.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with serial backend
cupla-serial: main-cupla-serial

main-cupla-serial: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(CUPLA_FLAGS) -pthread -o main-cupla-serial main.cc rawtodigi_cupla.cc -lcupla-serial

debug-cupla-serial: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o debug-cupla-serial main.cc rawtodigi_cupla.cc -lcupla-serial

# Alpaka/cupla implementation, with TBB backend
cupla-tbb: main-cupla-tbb

main-cupla-tbb: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(CUPLA_FLAGS) -pthread -o main-cupla-tbb main.cc rawtodigi_cupla.cc -lcupla-tbb -ltbb -lrt

debug-cupla-tbb: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o debug-cupla-tbb main.cc rawtodigi_cupla.cc -lcupla-tbb -ltbb -lrt
