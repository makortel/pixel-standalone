# Pixel raw2digi test program

The purpose of this test program is to experiment with various
"performance portability" approaches.

## Current implementations

| Implementation | Executable                   | `make` target         | `#ifdef` macros                                                                       |
|----------------|------------------------------|-----------------------|---------------------------------------------------------------------------------------|
| Naive CPU      | `main-naive`                 |`naive`                | `DIGI_NAIVE`                                                                          |
| CUDA           | `main-cuda`                  |`cuda`                 | `DIGI_CUDA`                                                                           |
| Cupla on CPU   | `main-cupla-seq-seq-async`   |`cupla-seq-seq-async`  | `DIGI_CUPLA`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`, `CUPLA_STREAM_ASYNC_ENABLED=1`    |
|                | `main-cupla-seq-seq-sync`    |`cupla-seq-seq-sync`   | `DIGI_CUPLA`, `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`, `CUPLA_STREAM_ASYNC_ENABLED=0`    |
|                | `main-cupla-tbb-seq-async`   |`cupla-tbb-seq-async`  | `DIGI_CUPLA`, `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED`, `CUPLA_STREAM_ASYNC_ENABLED=1`    |
| Cupla on GPU   | `main-cupla-cuda-async`      |`cupla-cuda-async`     | `DIGI_CUPLA`, `ALPAKA_ACC_GPU_CUDA_ENABLED`,        `CUPLA_STREAM_ASYNC_ENABLED=1`    |

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
library for each Cupla backend, and link it directly with the target program,
as described [here](AlpakaAndCupla.md).

## How to add a new implementation?

- Copy of (e.g.) the `rawtodigi_naive.h` for the new *thing* (with name of the *thing* after the underscore)
- Enclose all specifics under `#ifdef DIGI_THING`
- Add new build rules to `Makefile`
- Update `README.md`, including the requirements to compile and run with the *thing*
