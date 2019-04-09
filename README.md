# Pixel raw2digi test program

The purpose of this test program is to experiment with various
"performance portability" approaches.

## Current implementations

| Implementation | Execuable    | `make` target | `#ifdef` macro |
|----------------|--------------|---------------|----------------|
| Naive CPU      | `main-naive` |`naive`        | `DIGI_NAIVE`   |
| CUDA           | `main-cuda`  |`cuda`         | `DIGI_CUDA`    |

### Naive CPU

The only requirements for "naive CPU" are `g++` supporting C++ 17 in the `$PATH`.

### CUDA

The CUDA test program requires a recent CUDA version (`nvcc`
supporting C++14 and `--expt-relaxed-constexpr`) and a machine with
GPU.

## How to add a new implementation?

- Copy of (e.g.) the `rawtodigi_naive.h` for the new *thing* (with name of the *thing* after the underscore)
- Enclose all specifics under `#ifdef DIGI_THING`
- Add new build rules to `Makefile`
- Update `README.md`, including the requirements to compile and run with the *thing*
