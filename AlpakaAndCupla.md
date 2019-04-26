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
export CUDA_ROOT=/usr/local/cuda-10.0
export ALPAKA_ROOT=$BASE/alpaka
export CUPLA_ROOT=$BASE/cupla

CXX="/usr/bin/g++-7"
CXXFLAGS="-m64 -std=c++11 -g -O2 -DALPAKA_DEBUG=0 -I$CUDA_ROOT/include -I$ALPAKA_ROOT/include -I$CUPLA_ROOT/include"
HOST_FLAGS="-fopenmp -pthread -fPIC -ftemplate-depth-512 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes -Wno-reorder -Wno-sign-compare"

NVCC="$CUDA_ROOT/bin/nvcc"
NVCC_FLAGS="-ccbin $CXX -lineinfo --expt-extended-lambda --expt-relaxed-constexpr --generate-code arch=compute_50,code=sm_50 --use_fast_math --ftz=false --cudart shared"
```

## Download alpaka and cupla
```bash
git clone git@github.com:ComputationalRadiationPhysics/alpaka.git -b 0.3.5 $ALPAKA_ROOT
git clone git@github.com:ComputationalRadiationPhysics/cupla.git  -b 0.1.1 $CUPLA_ROOT
( cd $CUPLA_ROOT; patch -p1 ) < cupla.patch
```

## Build cupla
```bash
mkdir -p $CUPLA_ROOT/lib
FILES="$CUPLA_ROOT/src/*.cpp $CUPLA_ROOT/src/manager/*.cpp"
```

### Build cupla for the serial CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/seq-seq-async $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/seq-seq-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-seq-seq-async.so
```

### Build cupla for the serial CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/seq-seq-sync $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/seq-seq-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-seq-seq-sync.so
```

### Build cupla for the std::threads CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/seq-threads-async $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/seq-threads-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-seq-threads-async.so
```

### Build cupla for the std::threads CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/seq-threads-sync $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/seq-threads-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-seq-threads-sync.so
```

### Build cupla for the OpenMP 2.0 threads CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/seq-omp2-async $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/seq-omp2-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-seq-omp2-async.so
```

### Build cupla for the OpenMP 2.0 threads CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/seq-omp2-sync $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/seq-omp2-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-seq-omp2-sync.so
```

### Build cupla for the OpenMP 2.0 blocks CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/omp2-seq-async $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/omp2-seq-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-omp2-seq-async.so
```

### Build cupla for the OpenMP 2.0 blocks CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/omp2-seq-sync $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/omp2-seq-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -o $CUPLA_ROOT/lib/libcupla-omp2-seq-sync.so
```

### Build cupla for the TBB blocks CPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/tbb-seq-async $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/tbb-seq-async

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -ltbbmalloc -ltbb -o $CUPLA_ROOT/lib/libcupla-tbb-seq-async.so
```

### Build cupla for the TBB blocks CPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/tbb-seq-sync $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/tbb-seq-sync

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS -c $FILE -o $(basename $FILE).o &
done
wait
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -ltbbmalloc -ltbb -o $CUPLA_ROOT/lib/libcupla-tbb-seq-sync.so
```

### Build cupla for the CUDA GPU backend, with asynchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/cuda-async $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/cuda-async

for FILE in $FILES; do
  $NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu -c $FILE -o $(basename $FILE).o &
done
wait
# $NVCC $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -shared *.o -o $CUPLA_ROOT/lib/libcupla-cuda-async.so
# or
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -L$CUDA_ROOT/lib64 -lcudart -o $CUPLA_ROOT/lib/libcupla-cuda-async.so
```

### Build cupla for the CUDA GPU backend, with synchronous kernel launches
```bash
mkdir -p $CUPLA_ROOT/build/cuda-sync $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build/cuda-sync

for FILE in $FILES; do
  $NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu -c $FILE -o $(basename $FILE).o &
done
wait
# $NVCC $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -shared *.o -o $CUPLA_ROOT/lib/libcupla-cuda-sync.so
# or
$CXX $CXXFLAGS $HOST_FLAGS -shared *.o -L$CUDA_ROOT/lib64 -lcudart -o $CUPLA_ROOT/lib/libcupla-cuda-sync.so
```


# Building an example with Cupla

### Using the serial CPU backend, with synchronous kernel launches
```bash
cd $BASE
$CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=0 $CXXFLAGS $HOST_FLAGS $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -L$CUPLA_ROOT/lib -lcupla-seq-seq-sync -o vectorAdd-seq-seq-sync
LD_LIBRARY_PATH=$CUPLA_ROOT/lib ./vectorAdd-seq-seq-sync
```

### Using the TBB blocks CPU backend, with asynchronous kernel launches
```bash
cd $BASE
$CXX -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $HOST_FLAGS $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -L$CUPLA_ROOT/lib -lcupla-tbb-seq-async -ltbbmalloc -ltbb -o vectorAdd-tbb-seq-async
LD_LIBRARY_PATH=$CUPLA_ROOT/lib ./vectorAdd-tbb-seq-async
```

### Using the CUDA GPU backend, with asynchronous kernel launches
```bash
cd $BASE
$NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED -DCUPLA_STREAM_ASYNC_ENABLED=1 $CXXFLAGS $NVCC_FLAGS -Xcompiler "$HOST_FLAGS" -x cu $CUPLA_ROOT/example/CUDASamples/vectorAdd/src/vectorAdd.cpp -L$CUPLA_ROOT/lib -lcupla-cuda-async -o vectorAdd-cuda-async
LD_LIBRARY_PATH=$CUPLA_ROOT/lib:$CUDA_ROOT/lib64 ./vectorAdd-cuda-async
```
