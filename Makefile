TARGETS = naive cuda cupla-cuda-async cupla-seq-seq-async cupla-seq-seq-sync cupla-tbb-seq-async

.PHONY: all debug clean $(TARGETS)

CXX := g++-7
CXX_FLAGS := -O2 -std=c++14 -ftemplate-depth-512
CXX_DEBUG := -g

NVCC := /usr/local/cuda-10.0/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr
NVCC_DEBUG := -g -lineinfo

ALPAKA_BASE := $(HOME)/src/alpaka/alpaka
CUPLA_BASE  := $(HOME)/src/alpaka/cupla
CUPLA_FLAGS := -DALPAKA_DEBUG=0 -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include -L$(CUPLA_BASE)/lib

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

# Alpaka/cupla implementation, with the CUDA GPU async backend
cupla-cuda-async: main-cupla-cuda-async

main-cupla-cuda-async: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -lcupla-cuda-async -o main-cupla-cuda-async main.cc rawtodigi_cupla.cc

debug-cupla-cuda-async: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(NVCC) -x cu -w $(NVCC_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -lcupla-cuda-async $(NVCC_DEBUG) -o debug-cupla-cuda-async main.cc rawtodigi_cupla.cc

# Alpaka/cupla implementation, with the serial CPU async backend
cupla-seq-seq-async: main-cupla-seq-seq-async

main-cupla-seq-seq-async: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -o main-cupla-seq-seq-async main.cc rawtodigi_cupla.cc -lcupla-seq-seq-async

debug-cupla-seq-seq-async: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o debug-cupla-seq-seq-async main.cc rawtodigi_cupla.cc -lcupla-seq-seq-async

# Alpaka/cupla implementation, with the serial CPU sync backend
cupla-seq-seq-sync: main-cupla-seq-seq-sync

main-cupla-seq-seq-sync: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread -o main-cupla-seq-seq-sync main.cc rawtodigi_cupla.cc -lcupla-seq-seq-sync

debug-cupla-seq-seq-sync: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o debug-cupla-seq-seq-sync main.cc rawtodigi_cupla.cc -lcupla-seq-seq-sync

# Alpaka/cupla implementation, with the TBB blocks backend
cupla-tbb-seq-async: main-cupla-tbb-seq-async

main-cupla-tbb-seq-async: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread -o main-cupla-tbb-seq-async main.cc rawtodigi_cupla.cc -lcupla-tbb-seq-async -ltbb -lrt

debug-cupla-tbb-seq-async: main.cc rawtodigi_cupla.cc rawtodigi_cupla.h
	$(CXX) $(CXX_FLAGS) -DDIGI_CUPLA -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $(CUPLA_FLAGS) -pthread $(CXX_DEBUG) -o debug-cupla-tbb-seq-async main.cc rawtodigi_cupla.cc -lcupla-tbb-seq-async -ltbb -lrt

