# Alpaka and Cupla

## Alpaka
From the Alpaka [README](https://github.com/ComputationalRadiationPhysics/alpaka/blob/develop/README.md):

> The alpaka library is a header-only C++11 abstraction library for accelerator development.
> 
> Its aim is to provide performance portability across accelerators through the abstraction
> of the underlying levels of parallelism.
> 
> It is platform independent and supports the concurrent and cooperative use of multiple devices
> such as the hosts CPU as well as attached accelerators as for instance CUDA GPUs and Xeon Phis
> (currently native execution only). A multitude of accelerator back-end variants using CUDA,
> OpenMP (2.0/4.0), Boost.Fiber, std::thread and also serial execution is provided and can be
> selected depending on the device. Only one implementation of the user kernel is required by
> representing them as function objects with a special interface. There is no need to write
> special CUDA, OpenMP or custom threading code. Accelerator back-ends can be mixed within a
> device queue. The decision which accelerator back-end executes which kernel can be made at
> runtime.

Relevant links:
  - [Alpaka](https://github.com/ComputationalRadiationPhysics/alpaka) on GitHub
  - Alpaka's [documentation](https://github.com/ComputationalRadiationPhysics/alpaka/blob/develop/doc/markdown/user/Introduction.md)

## Cupla

From the Cupla [README]():

> Cupla is a simple user interface for the platform independent parallel kernel acceleration library
> Alpaka. It follows a similar concept as the NVIDIA® CUDA® API by providing a software layer to manage
> accelerator devices. Alpaka is used as backend for Cupla.

Relevant links:
  - [Cupla](https://github.com/ComputationalRadiationPhysics/cupla) on GitHub
  - Cupla's [porting guide](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/doc/PortingGuide.md)
  - Cupla's [tuning guide](https://github.com/ComputationalRadiationPhysics/cupla/blob/master/doc/TuningGuide.md)


# Building Alpaka and Cupla without CMake

## Set up the environment
```bash
BASE=$PWD
export CUDA_ROOT=/usr/local/cuda
export ALPAKA_ROOT=$BASE/alpaka
export CUPLA_ROOT=$BASE/cupla

CXX=/usr/bin/g++
CXXFLAGS="-m64 -std=c++14 -g -O2 -DALPAKA_DEBUG=0 -I$CUDA_ROOT/include -I$ALPAKA_ROOT/include -I$CUPLA_ROOT/include"
HOST_FLAGS="-fopenmp -pthread -fPIC -ftemplate-depth-512 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes -Wno-reorder -Wno-sign-compare"

NVCC="$CUDA_ROOT/bin/nvcc"
NVCC_FLAGS="-ccbin $CXX -w -lineinfo --expt-extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_35,code=sm_35 --use_fast_math --ftz=false --cudart shared"
```

## Download Alpaka and Cupla
```bash
git clone git@github.com:ComputationalRadiationPhysics/alpaka.git -b develop $ALPAKA_ROOT
git clone git@github.com:ComputationalRadiationPhysics/cupla.git  -b dev     $CUPLA_ROOT
```

## Remove the embedded version of Alpaka from Cupla
```bash
cd $CUPLA_ROOT
git config core.sparsecheckout true
echo -e '/*\n!/alpaka' > .git/info/sparse-checkout
git read-tree -mu HEAD
```

# Building an example with Cupla

## Using the serial CPU backend, with synchronous kernel launches
```bash
cd $BASE
$CXX -include cupla/config/CpuSerial.hpp -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -o vectorAdd-seq-seq-sync
./vectorAdd-seq-seq-sync
```

## Using the TBB blocks CPU backend, with asynchronous kernel launches
```bash
cd $BASE
$CXX -include cupla/config/CpuTbbBlocks.hpp -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -ltbbmalloc -ltbb -o vectorAdd-tbb-seq-async
./vectorAdd-tbb-seq-async
```

## Using the CUDA GPU backend, with asynchronous kernel launches
```bash
cd $BASE
$NVCC -include cupla/config/GpuCudaRt.hpp -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -o vectorAdd-cuda-async
LD_LIBRARY_PATH=$CUDA_ROOT/lib64 ./vectorAdd-cuda-async
```
