CXXFLAGS := -O2 -std=c++17
CXX := g++

CUDAFLAGS := -O2 -std=c++14 --expt-relaxed-constexpr
NVCC:= nvcc

default: naive

naive:
	$(CXX) $(CXXFLAGS) -DDIGI_NAIVE -o main-naive main.cc

debug-naive:
	$(CXX) $(CXXFLAGS) -DDIGI_NAIVE -g -o main-naive main.cc

cuda:
	$(NVCC) $(CUDAFLAGS) -DDIGI_CUDA -o main-cuda main.cc rawtodigi_cuda.cu

debug-cuda:
	$(NVCC) $(CUDAFLAGS) -DDIGI_CUDA -g -G -o main-cuda main.cc rawtodigi_cuda.cu
