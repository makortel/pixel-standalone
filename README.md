# Pixel raw2digi test program

The purpose of this test program is to experiment with various
"performance portability" frameworks and libraries.

## Overall structure

The test programs are divided in three units
* `main_*.cc`: contains `main()`, reads input, prints total timing. Plays the role of the "experiment framework".
* `analyzer_*.cc`: plays the role of a framework module (even though the "event loop" is there). Calls the memory tranfers (if necessary), and the computational kernel.
* `rawtodigi_*.cc`: contains the computational kernel (which is mostly just shuffling bytes around memory)


## Current implementations

| Implementation | `make` target         | Executable (also `make` target) | `#ifdef` macros                                                                       |
|----------------|-----------------------|---------------------------------|---------------------------------------------------------------------------------------|
| Naive CPU      | `naive`               | `test-naive`                    | `DIGI_NAIVE`                                                                          |
| CUDA           | `cuda`                | `test-cuda`                     | `DIGI_CUDA`                                                                           |
| Alpaka         | `alpaka`              | `test-alpaka`                   | `DIGI_ALPAKA`, `ALPAKA_ACC_*`                                                         |
|  - only on CPU |                       | `test-alpaka-ser` (sync)        | `DIGI_ALPAKA`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`                                   |
|                |                       | `test-alpaka-tbb` (async)       | `DIGI_ALPAKA`, `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED`                                   |
|                |                       | `test-alpaka-omp2` (async)      | `DIGI_ALPAKA`, `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED`                                  |
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
|  - OpenCL      |                       | `test-oneapi-opencl`            | `DIGI_ONEAPI`                                                                         |
|  - CUDA        |                       | `test-oneapi-cuda`              | `DIGI_ONEAPI`                                                                         |


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

### Alpaka [release 0.4.0](https://github.com/alpaka-group/alpaka/tree/0.4.0)

The Alpaka test program can be compiled for different backends; so far it has been
tested with the CUDA, serial, and TBB backends.
The CUDA backend requires CUDA 9.2 through 10.2, and has been tested with gcc 7.x
and gcc 8.x.

The backend is chosen at compile time setting one of the `ALPAKA_ACC` preprocessor
symbols. The `test-alpaka` binary tries to exercise all available backends.

See [the instructions](https://patatrack.web.cern.ch/patatrack/wiki/AlpakaAndCupla/)
on the Patatrack Wiki for installing Alpaka and Cupla.

### Cupla [release 0.2.0](https://github.com/alpaka-group/cupla/tree/0.2.0)

The Cupla test program can be compiled for different backends; so far it has been
tested with the CUDA, serial, TBB and OpenMP backends.
The CUDA backend requires CUDA 9.2 through 10.2, and has been tested with gcc 7.x
and gcc 8.x.

The backend is chosen at compile time setting one of the `ALPAKA_ACC` preprocessor
symbols. The `test-cupla` binary tries to exercise all available backends.

See [the instructions](https://patatrack.web.cern.ch/patatrack/wiki/AlpakaAndCupla/)
on the Patatrack Wiki for installing Alpaka and Cupla.

### Kokkos [release 3.0.00](https://github.com/kokkos/kokkos/tree/3.0.00)

The Kokkos test programs require Kokkos' source. Run something along
the following before compiling any of the test programs
```bash
# In some directory
git clone --branch 3.0.00 https://github.com/kokkos/kokkos.git
export KOKKOS_BASE=$PWD/kokkos
```

If CUDA is enabled (i.e. `$CUDA_BASE` points to an existing
directory), the `$CUDA_BASE/bin` should be put in `$PATH` before
compilation. In addition, all Kokkos test programs need to be run on a
machine with GPU.

If Kokkos is enabled (i.e.`$KOKKOS_BASE` is set), everything will be
compiled with `nvcc_wrapper` (also other targets than Kokkos test
programs).

Note that Kokkos' runtime library gets built on the fly, and the
`libkokkos.a` and the intermediate object files are placed to the
working directory.

### Intel oneAPI

The test program relies on the In-Order Queues and Unified Shared Memory extensions
to SYCL 1.2.1, which are currently available in Intel oneAPI toolchain and in the
LLVM SYCL branch.

The beta version of Intel oneAPI can be obtained from
https://software.intel.com/en-us/oneapi .

The in-development version of the LLVM compiler with support for SYCL, Intel's
extensions, and Codeplay's CUDA backend is available on GitHub at
https://github.com/intel/llvm/ .
See [the instructions](https://patatrack.web.cern.ch/patatrack/wiki/SYCL/)
on the Patatrack Wiki for building the SYCL toolchain.

The test program should run on any available SYCL device, and can select it at
runtime based on the command line options.

## How to add a new implementation?

- Copy of (e.g.) the `rawtodigi_naive.h` for the new *thing* (with name of the *thing* after the underscore)
- Enclose all specifics under `#ifdef DIGI_THING`
- Add new build rules to `Makefile`
- Update `README.md`, including the requirements to compile and run with the *thing*
