# Pixel raw2digi test program

The purpose of this test program is to experiment with various
"performance portability" approaches.

## Current implementations

| Implementation | `make` target         | Executable (also `make` target) | `#ifdef` macros                                                                       |
|----------------|-----------------------|---------------------------------|---------------------------------------------------------------------------------------|
| Naive CPU      | `naive`               | `main-naive`                    | `DIGI_NAIVE`                                                                          |
| CUDA           | `cuda`                | `main-cuda`                     | `DIGI_CUDA`                                                                           |
| Alpaka on CPU  | `alpaka`              | `main-alpaka-ser` (sync)        | `DIGI_ALPAKA`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`                                   |
|                |                       | `main-alpaka-tbb` (async)       | `DIGI_ALPAKA`, `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED`                                   |
| Alpaka on GPU  |                       | `main-alpaka-gpu` (async)       | `DIGI_ALPAKA`, `ALPAKA_ACC_GPU_CUDA_ENABLED`                                          |
| Alpaka on all  |                       | `main-alpaka`                   | `DIGI_ALPAKA`, `ALPAKA_ACC_*`                                                         |
| Cupla on CPU   | `cupla`               | `main-cupla-seq-seq-async`      | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`    |
|                |`                      | `main-cupla-seq-seq-sync`       | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=0`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`    |
|                |                       | `main-cupla-tbb-seq-async`      | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED`    |
|                |                       | `main-cupla-opm2-seq-async`     | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED`   |
| Cupla on GPU   |                       | `main-cupla-cuda-async`         | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_GPU_CUDA_ENABLED`           |
| Kokkos on CPU  | `kokkos`              | `main-kokkos-serial`            | `DIGI_KOKKOS`, `DIGI_KOKKOS_SERIAL`                                                   |
|                |                       | `main-kokkos-openmp`            | `DIGI_KOKKOS`, `DIGI_KOKKOS_OPENMP`                                                   |
| Kokkos on GPU  |                       | `main-kokkos-cuda`              | `DIGI_KOKKOS`, `DIGI_KOKKOS_CUDA`                                                     |
| Intel oneAPI   | `oneapi`              | `main-oneapi`                   | `DIGI_ONEAPI`                                                                         |


The per-technology targets build all the executables of that
technology. For finer-grained compilation, use the executable names
directly as make targets.

### Naive CPU

The only requirements for "naive CPU" are `g++` supporting C++ 17 in the `$PATH`.

### CUDA

The CUDA test program requires a recent CUDA version (`nvcc`
supporting C++14 and `--expt-relaxed-constexpr`) and a machine with
GPU.

### Alpaka [release 0.4.0](https://github.com/ComputationalRadiationPhysics/alpaka/tree/release-0.4.0)

The Alpaka test program can be compiled for different backends; so far it has been
tested with the CUDA, serial, and TBB backends.
The CUDA backend requires CUDA 9.2 through 10.2, and has been tested with gcc 7.x
and gcc 8.x.

The backend is chosen at compile time setting one of the `ALPAKA_ACC_*` preprocessor
symbols. The `main-alpaka` binary tries to exercise all available backends.

See [here](AlpakaAndCupla.md) for instructions on installing Alpaka.

### Cupla [dev](https://github.com/ComputationalRadiationPhysics/cupla/tree/dev) branch

The Cupla test program can be compiled for different backends; so far it has been
tested with the CUDA, serial, TBB and OpenMP backends.
The CUDA backend requires CUDA 9.2 through 10.2, and has been tested with gcc 7.x
and gcc 8.x.

Rather than using the advertised `CMake`-based approach, one can build Cupla as
a header-only library, as described [here](AlpakaAndCupla.md).

### Kokkos

#### Install Kokkos
```bash
# In some directory
git clone https://github.com/kokkos/kokkos.git
cd kokkos
export KOKKOS_SRC=$PWD

# Define installation path
export KOKKOS_BASE=...

# In some other, temporary directory
# CPU-only
$KOKKOS_SRC/generate_makefile.bash --prefix=$KOKKOS_BASE --cxxstandard=c++17 --with-openmp --with-pthread --with-serial [--with-cuda=PATH_TO_CUDA]
# with CUDA,
# $KOKKOS_SRC/generate_makefile.bash --prefix=$KOKKOS_BASE --cxxstandard=c++14 --with-openmp --with-pthread --with-serial --with-cuda=<PATH_TO_CUDA> --arch=Pascal60 --with-cuda-options=enable_lambda
make kokkoslib
make install
```

In principle the following fix is needed, unless `$KOKKOS_SRC` is kept alive as well
* Edit `$KOKKOS_BASE/Makefile.kokkos` to fix the path to `nvcc_wrapper` to point to `$KOKKOS_BASE` instead of `$KOKKOS_SRC`

### Intel oneAPI

The beta version of Intel oneAPI can be obtained from https://software.intel.com/en-us/oneapi .

The tests programs choose an OpenCL device at runtime using the `cl::sycl::default_selector`.

## How to add a new implementation?

- Copy of (e.g.) the `rawtodigi_naive.h` for the new *thing* (with name of the *thing* after the underscore)
- Enclose all specifics under `#ifdef DIGI_THING`
- Add new build rules to `Makefile`
- Update `README.md`, including the requirements to compile and run with the *thing*
