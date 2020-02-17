# Pixel raw2digi test program

The purpose of this test program is to experiment with various
"performance portability" frameworks and libraries.

## Current implementations

| Implementation | `make` target         | Executable (also `make` target) | `#ifdef` macros                                                                       |
|----------------|-----------------------|---------------------------------|---------------------------------------------------------------------------------------|
| Naive CPU      | `naive`               | `test-naive`                    | `DIGI_NAIVE`                                                                          |
| CUDA           | `cuda`                | `test-cuda`                     | `DIGI_CUDA`                                                                           |
| Alpaka         | `alpaka`              | `test-alpaka`                   | `DIGI_ALPAKA`, `ALPAKA_ACC_*`                                                         |
|  - only on CPU |                       | `test-alpaka-ser` (sync)        | `DIGI_ALPAKA`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`                                   |
|                |                       | `test-alpaka-tbb` (async)       | `DIGI_ALPAKA`, `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED`                                   |
|  - only on GPU |                       | `test-alpaka-gpu` (async)       | `DIGI_ALPAKA`, `ALPAKA_ACC_GPU_CUDA_ENABLED`                                          |
| Cupla          | `cupla`               | `test-cupla`                    | `DIGI_CUPLA`, `ALPAKA_ACC_*`                                                          |
|  - only on CPU |                       | `test-cupla-seq-seq-async`      | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`    |
|                |                       | `test-cupla-seq-seq-sync`       | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=0`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`    |
|                |                       | `test-cupla-tbb-seq-async`      | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED`    |
|                |                       | `test-cupla-opm2-seq-async`     | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED`   |
|  - only on GPU |                       | `test-cupla-cuda-async`         | `DIGI_CUPLA`, `CUPLA_STREAM_ASYNC_ENABLED=1`, `ALPAKA_ACC_GPU_CUDA_ENABLED`           |
| Kokkos on CPU  | `kokkos`              | `test-kokkos-serial`            | `DIGI_KOKKOS`, `DIGI_KOKKOS_SERIAL`                                                   |
|                |                       | `test-kokkos-openmp`            | `DIGI_KOKKOS`, `DIGI_KOKKOS_OPENMP`                                                   |
| Kokkos on GPU  |                       | `test-kokkos-cuda`              | `DIGI_KOKKOS`, `DIGI_KOKKOS_CUDA`                                                     |
| Intel oneAPI   | `oneapi`              | `test-oneapi`                   | `DIGI_ONEAPI`                                                                         |



The per-technology targets build all the executables of that
technology. For finer-grained compilation, use the executable names
directly as make targets.

### Naive CPU

The only requirements for "naive CPU" are `g++` supporting C++17 in the `$PATH`.

### CUDA

The CUDA test program requires a recent CUDA version (`nvcc`
supporting C++14 and `--expt-relaxed-constexpr`) and a machine with
an NVIDIA GPU.
By deafult, the binaries target SM 3.5, 5.0,  6.0 and 7.0. Different targets can
be added in the `Makefile`.

### Alpaka [release 0.4.0](https://github.com/ComputationalRadiationPhysics/alpaka/tree/0.4.0)

The Alpaka test program can be compiled for different backends; so far it has been
tested with the CUDA, serial, and TBB backends.
The CUDA backend requires CUDA 9.2 through 10.2, and has been tested with gcc 7.x
and gcc 8.x.

The backend is chosen at compile time setting one of the `ALPAKA_ACC` preprocessor
symbols. The `test-alpaka` binary tries to exercise all available backends.

See [the instructions](https://patatrack.web.cern.ch/patatrack/wiki/AlpakaAndCupla/)
on the Patatrack Wiki for installing Alpaka and Cupla.

### Cupla [release 0.2.0](https://github.com/ComputationalRadiationPhysics/cupla/tree/0.2.0)

The Cupla test program can be compiled for different backends; so far it has been
tested with the CUDA, serial, TBB and OpenMP backends.
The CUDA backend requires CUDA 9.2 through 10.2, and has been tested with gcc 7.x
and gcc 8.x.

The backend is chosen at compile time setting one of the `ALPAKA_ACC` preprocessor
symbols. The `test-cupla` binary tries to exercise all available backends.

See [the instructions](https://patatrack.web.cern.ch/patatrack/wiki/AlpakaAndCupla/)
on the Patatrack Wiki for installing Alpaka and Cupla.

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
