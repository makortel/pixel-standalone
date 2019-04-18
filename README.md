# Pixel raw2digi test program

The purpose of this test program is to experiment with various
"performance portability" approaches.

## Current implementations

| Implementation | Executable           | `make` target | `#ifdef` macros                                    |
|----------------|----------------------|---------------|----------------------------------------------------|
| Naive CPU      | `main-naive`         |`naive`        | `DIGI_NAIVE`                                       |
| CUDA           | `main-cuda`          |`cuda`         | `DIGI_CUDA`                                        |
| Cupla on CPU   | `main-cupla-serial`  |`cupla-serial` | `DIGI_CUPLA`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` |
| Cupla on CPU   | `main-cupla-tbb`     |`cupla-tbb`    | `DIGI_CUPLA`, `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED` |
| Cupla on GPU   | `main-cupla-cuda`    |`cupla-cuda`   | `DIGI_CUPLA`, `ALPAKA_ACC_GPU_CUDA_ENABLED`        |

### Naive CPU

The only requirements for "naive CPU" are `g++` supporting C++ 17 in the `$PATH`.

### CUDA

The CUDA test program requires a recent CUDA version (`nvcc`
supporting C++14 and `--expt-relaxed-constexpr`) and a machine with
GPU.

### Cupla

The Cupla test program can be compiled for different backends; so far it has
been tested with the CUDA, serial, and TBB backends.
The CUDA backend requires CUDA 9.2 or 10.0, and has been tested with gcc 7.
The TBB backend requires a small patch tu Cupla itself (see [`cupla.patch`]).

Rather than using the advertised `CMake`-based approach, one can build a shared
library for each Cupla backend, and link it directly with the target program:
```bash
export CUDA_ROOT=/usr/local/cuda
export ALPAKA_ROOT=$HOME/src/alpaka/alpaka
export CUPLA_ROOT=$HOME/src/alpaka/cupla

git clone git@github.com:ComputationalRadiationPhysics/alpaka.git -b 0.3.5 $ALPAKA_ROOT
git clone git@github.com:ComputationalRadiationPhysics/cupla.git  -b 0.1.1 $CUPLA_ROOT
( cd $CUPLA_ROOT; patch -p1 ) < cupla.patch

mkdir -p $CUPLA_ROOT/lib
FILES="$CUPLA_ROOT/src/*.cpp $CUPLA_ROOT/src/manager/*.cpp"

CXX_FLAGS="-m64 -std=c++11 -g -O2 -DALPAKA_DEBUG=0 -DCUPLA_STREAM_ASYNC_ENABLED=1 -I$CUDA_ROOT/include -I$ALPAKA_ROOT/include -I$CUPLA_ROOT/include"
HOST_FLAGS="-fPIC -ftemplate-depth-512 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes -Wno-reorder -Wno-sign-compare"
NVCC_FLAGS="-lineinfo --expt-extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_50,code=sm_50 --use_fast_math --ftz=false --cudart shared"

mkdir -p $CUPLA_ROOT/build/cuda
cd $CUPLA_ROOT/build/cuda
for FILE in $FILES; do
  nvcc -DALPAKA_ACC_GPU_CUDA_ENABLED $CXX_FLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu -c $FILE -o $(basename $FILE).o
done
nvcc -DALPAKA_ACC_GPU_CUDA_ENABLED $CXX_FLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -shared *.o -o $CUPLA_ROOT/lib/libcupla-cuda.so

mkdir -p $CUPLA_ROOT/build/serial
cd $CUPLA_ROOT/build/serial
for FILE in $FILES; do
  g++ -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o
done
g++ -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-serial.so

mkdir -p $CUPLA_ROOT/build/tbb
cd $CUPLA_ROOT/build/tbb
for FILE in $FILES; do
  g++ -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o
done
g++ -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED $CXX_FLAGS $HOST_FLAGS -shared *.o -ltbbmalloc -ltbb -lpthread -lrt -o $CUPLA_ROOT/lib/libcupla-tbb.so
```

## How to add a new implementation?

- Copy of (e.g.) the `rawtodigi_naive.h` for the new *thing* (with name of the *thing* after the underscore)
- Enclose all specifics under `#ifdef DIGI_THING`
- Add new build rules to `Makefile`
- Update `README.md`, including the requirements to compile and run with the *thing*
