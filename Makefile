TARGETS = naive cuda

.PHONY: default all debug clean $(TARGETS)

CXXFLAGS := -O2 -std=c++17
CXX := g++

CUDAFLAGS := -O2 -std=c++14 --expt-relaxed-constexpr
NVCC := nvcc

default: naive

all: $(TARGETS)

debug: $(TARGETS:%=debug-%)

clean:
	rm -f $(TARGETS:%=main-%) $(TARGETS:%=debug-%)

naive: main-naive

main-naive: main.cc rawtodigi_naive.h
	$(CXX) $(CXXFLAGS) -DDIGI_NAIVE -o main-naive main.cc

debug-naive: main.cc rawtodigi_naive.h
	$(CXX) $(CXXFLAGS) -DDIGI_NAIVE -g -o debug-naive main.cc

cuda: main-cuda

main-cuda: main.cc rawtodigi_cuda.cu rawtodigi_cuda.h
	$(NVCC) $(CUDAFLAGS) -DDIGI_CUDA -o main-cuda main.cc rawtodigi_cuda.cu

debug-cuda: main.cc rawtodigi_cuda.cu rawtodigi_cuda.h
	$(NVCC) $(CUDAFLAGS) -DDIGI_CUDA -g -lineinfo -o debug-cuda main.cc rawtodigi_cuda.cu
