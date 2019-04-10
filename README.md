# Pixel raw2digi test program

The purpose of this test program is to experiment with various
"performance portability" approaches.

## Current implementations

| Implementation | Execuable            | `make` target | `#ifdef` macro |
|----------------|----------------------|---------------|----------------|
| Naive CPU      | `main-naive`         |`naive`        | `DIGI_NAIVE`   |
| CUDA           | `main-cuda`          |`cuda`         | `DIGI_CUDA`    |
| Cupla on CPU   | `main-cupla-serial`  |`cupla-serial` | `DIGI_CUPLA`   |
| Cupla on GPU   | `main-cupla-cuda`    |`cupla-cuda`   | `DIGI_CUPLA`   |

### Naive CPU

The only requirements for "naive CPU" are `g++` supporting C++ 17 in the `$PATH`.

### CUDA

The CUDA test program requires a recent CUDA version (`nvcc`
supporting C++14 and `--expt-relaxed-constexpr`) and a machine with
GPU.

### Cupla

The Cupla test program can be compiled for different backends; so far it has
been tested with the CUDA backend (`-DALPAKA_ACC_GPU_CUDA_ENABLED`) and the
serial CPU backend (`-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`). The CUDA backend
requires CUDA 9.2 or CUDA 10.0, and has been tested with gcc 8.

In fact, the Cupla libraries need to be built before they can be used:
```bash
export ALPAKA_ROOT=$HOME/src/alpaka/alpaka
export CUPLA_ROOT=$HOME/src/alpaka/cupla

git clone git@github.com:ComputationalRadiationPhysics/alpaka.git -b 0.3.5 $ALPAKA_ROOT
git clone git@github.com:ComputationalRadiationPhysics/cupla.git  -b 0.1.1 $CUPLA_ROOT

mkdir -p $CUPLA_ROOT/build $CUPLA_ROOT/lib
cd $CUPLA_ROOT/build
FILES="$CUPLA_ROOT/src/*.cpp $CUPLA_ROOT/src/manager/*.cpp"

for FILE in $FILES; do
  $NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED $NVCC_FLAGS -x cu -c $FILE -o cuda_$(basename $FILE).o
done
$NVCC -DALPAKA_ACC_GPU_CUDA_ENABLED $NVCC_FLAGS -shared cuda_*.o -o $CUPLA_ROOT/lib/libcupla-cuda.so

for FILE in $FILES; do
  $CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $CXX_FLAGS -c $FILE -o serial_$(basename $FILE).o
done
$CXX -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $CXX_FLAGS -shared serial_*.o -o $CUPLA_ROOT/lib/libcupla-serial.so

```

## How to add a new implementation?

- Copy of (e.g.) the `rawtodigi_naive.h` for the new *thing* (with name of the *thing* after the underscore)
- Enclose all specifics under `#ifdef DIGI_THING`
- Add new build rules to `Makefile`
- Update `README.md`, including the requirements to compile and run with the *thing*
