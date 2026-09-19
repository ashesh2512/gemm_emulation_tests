# gemm_emulation_tests

An API for FP64 `dgemm` emulation on GPUs. It runs the vendor FP64 `dgemm` and one emulated method on the same problem, and compares both against a double-double reference.

## Building

Emulation libraries GEMMul8 and ozablas are git submodules under `external/`, so clone with:
```
git clone --recurse-submodules <gemm_emulation_tests url>
```

The build command should explicitly specify the backend (`cuda`, `hip`), backend path, GPU architecture, and library used for emulation, E.g. on MI250X GPUs, a build command using Ozaki II emulation via the Gemmul8 library looks like - 
```
make BACKEND=hip HIP_PATH=/opt/rocm-7.2.0 GPU_ARCH=gfx90a HAVE_GEMMUL8=1
```
Enabling `HAVE_GEMMUL8=1` or `HAVE_OZABLAS=1` makes the Makefile build the corresponding submodule first, with the same backend and architecture as the main binary. The toolchain paths have no defaults, so the build stops with an error if one is missing; `GPU_ARCH=auto` queries the device (if available) instead of naming an architecture, and `BACKEND` defaults to `auto`, which picks CUDA or HIP from whichever of `nvidia-smi` or `rocminfo` is on the path.

These are all the variables the Makefile reads; they can be set on the command line or in the environment:

| Variable | Default | Required when | Meaning |
| --- | --- | --- | --- |
| `BACKEND` | `auto` | always | `cuda`, `hip`, or `auto`. `auto` picks `cuda` if `nvidia-smi` is on the path, else `hip` if `rocminfo` is, else errors out. |
| `GPU_ARCH` | none | always | Target architecture, e.g. `90` for CUDA or `gfx90a` for HIP. `auto` reads it from `nvidia-smi` or `amd-smi`. |
| `CUDA_PATH` | none | `BACKEND=cuda` | Root of the CUDA toolkit. Its `bin` and `lib64` are prepended to `PATH` and `LD_LIBRARY_PATH`. |
| `HIP_PATH` | none | `BACKEND=hip` | Root of the ROCm install. Its `bin` and `lib` are prepended to `PATH` and `LD_LIBRARY_PATH`. |
| `HAVE_GEMMUL8` | `0` | never | Set to `1` to build the `external/GEMMul8` submodule and compile in the `gemmul8` method. |
| `HAVE_OZABLAS` | `0` | never | Set to `1` to build the `external/ozablas` submodule and compile in the `ozablas-ozaki1` and `ozablas-ozaki2` methods. |
| `HAVE_CUBLAS_OZAKI1` | `0` | never | Set to `1` to compile in the `cublas-ozaki1` method. CUDA only; it compiles out on HIP. |
| `INT8_ONLY` | `1` | never | Passed to the GEMMul8 submodule. `1` restricts its explicit instantiations to INT8, which builds much faster. |

The submodule locations are currently fixed at `external/GEMMul8` and `external/ozablas` and cannot be pointed elsewhere. The compiler is chosen by the backend, `nvcc` for CUDA and `hipcc` for HIP, and is not overridable.

## Running

It is possible to have multiple libraries linked at the same time for emulation. Hence `--method` is required and the run stops with an error without it, e.g.
```
./gemm_test --method=gemmul8 ...
```
The methods are `native`, the vendor FP64 `dgemm` from the GPU BLAS library, `cublas-ozaki1`, `ozablas-ozaki1`, `ozablas-ozaki2`, and `gemmul8` which only emulates via the `ozaki2` method. A method that was not compiled into the binary is rejected at startup.

The remaining options describe the problem being solved:

| Option | Default | Meaning |
| --- | --- | --- |
| `--m=<int>` | 1024 | Rows of A and C. |
| `--n=<int>` | 1024 | Columns of B and C. |
| `--k=<int>` | 1024 | Inner dimension, i.e. columns of A and rows of B. |
| `--phi=<double>` | 1.0 | Controls the entries of A and B. |
| `--moduli=<int>` | 2 | Number of moduli used by the Ozaki II methods, `ozablas-ozaki2` and `gemmul8`. |
| `--splits=<int>` | 2 | Number of splits used by the Ozaki I methods, `cublas-ozaki1` and `ozablas-ozaki1`. |
| `--auto-mantissa` | off | `cublas-ozaki1` only: ignore `--splits` and let cuBLAS pick the mantissa bit count itself. |
| `--warmups=<int>` | 2 | Untimed calls run before the timed one, for both native and emulated `gemm`. |
| `--device=<int>` | 0 | GPU to run on. |
| `--no-106bit-ref` | off | Skip the 106 bit reference. |
| `--verify-ref` | off | Sanity-check the 106 bit reference against a host FP128 one and report `ref-diff`. |

`--phi` sets how the random input matrices are generated. A negative value draws every entry from a standard normal distribution. A non-negative value uses `(rand - 0.5) * exp(randn * phi)` instead, so the exponent range of the entries widens as `phi` grows and the emulation has a harder time matching FP64. The seed is fixed in the code, so repeated runs with the same options give the same matrices.

The reference is a double-double matrix multiply on the GPU, carrying about 106 mantissa bits, so it is accurate to far more digits than FP64 can represent. The native `dgemm` and the chosen method are both measured against it, as a relative Frobenius norm and as the worst single element:
```
./gemm_test --method=cublas-ozaki1 --m=10240 --n=10240 --k=10240 --phi=4.0
```
The output has three sections:
```
Run
  method            : cublas-ozaki1
  m, n, k           : 10240, 10240, 10240
  phi               : 4
  warmups           : 2

Errors
                                  rel-frob  max-rel-elem
  64 bit native vs 106 bit    1.568254e-15  1.442610e-07
  64 bit emulated vs 106 bit  6.816338e-17  9.025597e-15
  emulated vs native          1.587247e-15  1.442610e-07

Performance
  time   [ms] (native | emulated) :  49.046 | 220.925
  memory [GB] (native | emulated) :   4.528 |   6.966
  energy [J]  (native | emulated) : 277.396 | 118.914
```
`Run` echoes the settings. It also prints a `note` line if the native `dgemm` was itself served by cuBLAS emulation, in which case the native error comes from a hand written FP64 kernel and the native performance columns read `n/a`.

`Errors` gives the error of the native FP64 `dgemm` against the 106 bit reference, the error of the selected method against the same reference, and finally the two FP64 results against each other without involving the reference at all. `rel-frob` is the relative Frobenius norm of the difference, `max-rel-elem` the worst single element. If the vendor `dgemm` emulated, a hand written FP64 kernel stands in for it so the native row still describes FP64.

`Performance` compares the native and emulated runs. `time` is the timed call in milliseconds after `--warmups` untimed calls; `memory` is the peak device memory in use during the run, sampled by polling free memory, so a workspace allocated and freed inside one call may be missed; `energy` is read from the on-board counter, NVML on NVIDIA and ROCm SMI on AMD, divided by the number of calls. All three native columns read `n/a` when the native `dgemm` emulated.

`--no-106bit-ref` skips the reference altogether, leaving only the `emulated vs native` row. It is a full double-double `gemm`, so on large problems it costs far more than the two `dgemm` calls being measured.
```
./gemm_test --no-106bit-ref ...
```

`--verify-ref` sanity-checks the reference itself. A and B are copied back to the host, the FP128 reference runs there under OpenMP, and the extra `ref-diff` line reports how far the two references are apart. The host result is never used to score the two `dgemm` calls; it exists only to confirm the 106 bit reference is trustworthy, so `--verify-ref` needs the reference and is rejected together with `--no-106bit-ref`:
```
OMP_NUM_THREADS=64 ./gemm_test --method=gemmul8 --verify-ref
```

## Machines

### Castner (NVIDIA GH200, `sm_90`)

Load the NVIDIA HPC SDK and build against cuBLAS with the Ozaki I emulation path enabled:
```
module purge
module load PrgEnv-nvidia nvidia/26.3
module load craype-arm-grace
make BACKEND=cuda CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/26.3 GPU_ARCH=90 HAVE_CUBLAS_OZAKI1=1
```

cuBLAS turns FP64 emulation on and off through the environment variable `CUBLAS_EMULATE_DOUBLE_PRECISION`, and the binary sets it based on `--method`: `1` for `cublas-ozaki1` and `0` for everything else, so it never has to be set by hand. The two runs below are therefore the pair to compare, same problem, emulation off then on for NVIDIA GPUs. `~/cudevmap.sh` is a bash wrapper script that sets CPU-GPU affinity, binding each rank to the CPU cores closest to the GPU it uses.

A: guaranteed-native baseline, `CUBLAS_EMULATE_DOUBLE_PRECISION=0`:
```
~/cudevmap.sh ./gemm_test --method=native --m=10240 --n=10240 --k=10240 --phi=4.0
```

B: the emulated run, `CUBLAS_EMULATE_DOUBLE_PRECISION=1`:
```
~/cudevmap.sh ./gemm_test --method=cublas-ozaki1 --m=10240 --n=10240 --k=10240 --phi=4.0 --splits=10
```

### Pinoak (AMD MI250X, `gfx90a`)

Load ROCm, then build with whichever emulation libraries are wanted:
```
module purge
module load PrgEnv-amd amd/7.2.0 rocm/7.2.0
module load craype-x86-trento
export ROCM_PATH=/opt/rocm-7.2.0
export HIP_PLATFORM=amd
make -j BACKEND=hip HIP_PATH=/opt/rocm-7.2.0 GPU_ARCH=gfx90a HAVE_GEMMUL8=1 HAVE_OZABLAS=1
```

`~/rocrmap.sh` is the ROCm equivalent bash script for CPU-GPU affinity:
```
~/rocrmap.sh ./gemm_test --method=ozablas-ozaki1 --m=10240 --n=10240 --k=10240 --phi=4.0 --splits=10
```
