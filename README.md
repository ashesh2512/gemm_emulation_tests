# gemm_emulation_tests

An API for FP64 `dgemm` emulation on GPUs. It runs the vendor FP64 `dgemm` and one emulated method on the same problem, measures both against a double-double reference computed on the GPU, and reports the errors side by side.

## Building

The build command should explicitly specify the backend (`cuda`, `hip`), backend path, GPU architecture, and library used for emulation, E.g. on MI250X GPUs, a build command using Ozaki II emulation via the Gemmul8 library looks like - 
```
make BACKEND=hip HIP_PATH=/opt/rocm-7.2.0 GPU_ARCH=gfx90a HAVE_GEMMUL8=1 GEMMUL8_PATH=/path/to/GEMMul8
```
The paths have no defaults, so the build stops with an error if one is missing; `GPU_ARCH=auto` queries the device instead of naming an architecture, and `BACKEND` defaults to `auto`, which picks CUDA or HIP from whichever of `nvidia-smi` or `rocminfo` is on the path. Enabling `HAVE_OZABLAS=1`, e.g., likewise requires `OZABLAS_PATH`.

These are all the variables the Makefile reads; they can be set on the command line or in the environment:

| Variable | Default | Required when | Meaning |
| --- | --- | --- | --- |
| `BACKEND` | `auto` | never | `cuda`, `hip`, or `auto`. `auto` picks `cuda` if `nvidia-smi` is on the path, else `hip` if `rocminfo` is, else errors out. |
| `GPU_ARCH` | none | always | Target architecture, e.g. `90` for CUDA or `gfx90a` for HIP. `auto` reads it from `nvidia-smi` or `amd-smi`. |
| `CUDA_PATH` | none | `BACKEND=cuda` | Root of the CUDA toolkit. Its `bin` and `lib64` are prepended to `PATH` and `LD_LIBRARY_PATH`. |
| `HIP_PATH` | none | `BACKEND=hip` | Root of the ROCm install. Its `bin` and `lib` are prepended to `PATH` and `LD_LIBRARY_PATH`. |
| `HAVE_GEMMUL8` | `0` | never | Set to `1` to compile in the `gemmul8` method. |
| `GEMMUL8_PATH` | none | `HAVE_GEMMUL8=1` | Root of GEMMul8; expects `include/` and `lib/libgemmul8.a`. |
| `HAVE_OZABLAS` | `0` | never | Set to `1` to compile in the `ozablas-ozaki1` and `ozablas-ozaki2` methods. |
| `OZABLAS_PATH` | none | `HAVE_OZABLAS=1` | Root of ozablas; expects `include/` and `build/src/libozablas.so`. |
| `HAVE_CUBLAS_OZAKI1` | `0` | never | Set to `1` to compile in the `cublas-ozaki1` method. CUDA only; it compiles out on HIP. |

The compiler is chosen by the backend, `nvcc` for CUDA and `hipcc` for HIP, and is not overridable. `make clean` skips all of the above, so it needs none of them.

## Running

It is possible to have multiple libraries linked at the same time for emulation. Hence `--method` is required and the run stops with an error without it, e.g.
```
./gemm_test --method=gemmul8
```
The methods are `native`, the vendor FP64 `dgemm` from the GPU BLAS library, `cublas-ozaki1`, `ozablas-ozaki1`, `ozablas-ozaki2`, and `gemmul8`. A method that was not compiled into the binary is rejected at startup.

The remaining options describe the problem being solved:

| Option | Default | Meaning |
| --- | --- | --- |
| `--m=<int>` | 1024 | Rows of A and C. |
| `--n=<int>` | 1024 | Columns of B and C. |
| `--k=<int>` | 1024 | Inner dimension, i.e. columns of A and rows of B. |
| `--phi=<double>` | 1.0 | Controls the entries of A and B. |
| `--moduli=<int>` | 2 | Number of moduli used by the Ozaki II methods, `ozablas-ozaki2` and `gemmul8`. |
| `--splits=<int>` | 2 | Number of splits used by the Ozaki I methods, `cublas-ozaki1` and `ozablas-ozaki1`. |
| `--verify-ref` | off | Also compute the host FP128 reference and report `ref-diff`. |

`--phi` sets how the random input matrices are generated. A negative value draws every entry from a standard normal distribution. A non-negative value uses `(rand - 0.5) * exp(randn * phi)` instead, so the exponent range of the entries widens as `phi` grows and the emulation has a harder time matching FP64. The seed is fixed in the code, so repeated runs with the same options give the same matrices.

The reference is a double-double matrix multiply on the GPU, carrying about 106 mantissa bits, so it is accurate to far more digits than FP64 can represent. The native `dgemm` and the chosen method are both measured against it, as a relative Frobenius norm and as the worst single element:
```
./gemm_test --method=gemmul8 --m=1024 --n=1024 --k=1024 --phi=2.0 --moduli=3
```
The run first echoes the settings, the mantissa bit count each `dgemm` call actually used, and the time for each, then prints the error table:
```
                relative      max
native   = 1.041317e-15  1.131069e-09
emulated = 5.097529e-13  3.970221e-06
diff     = 5.097528e-13  3.969089e-06
```
`native` is the error of the vendor FP64 `dgemm`, `emulated` the error of the selected method, and `diff` compares the two against each other.

The `mantissa` line is the check that the emulated path was really taken: it reports the bit count cuBLAS was configured with for the native and emulated calls, and `-1` means the query was unavailable, i.e. plain FP64 ran. The `time` line reports one warm run of each in milliseconds.

`--verify-ref` adds a cross-check: A and B are copied back to the host, the FP128 reference runs there under OpenMP, and the extra `ref-diff` line reports how far the two references are apart:
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
OMP_NUM_THREADS=64 ~/cudevmap.sh ./gemm_test --method=native --m=10240 --n=10240 --k=10240 --phi=4.0
```

B: the emulated run, `CUBLAS_EMULATE_DOUBLE_PRECISION=1`:
```
OMP_NUM_THREADS=64 ~/cudevmap.sh ./gemm_test --method=cublas-ozaki1 --m=10240 --n=10240 --k=10240 --phi=4.0 --splits=10
```

### Pinoak (AMD MI250X, `gfx90a`)

Load ROCm, then build against whichever emulation libraries are wanted. Each one needs its own path:
```
module purge
module load PrgEnv-amd amd/7.2.0 rocm/7.2.0
module load craype-x86-trento
export ROCM_PATH=/opt/rocm-7.2.0
export HIP_PLATFORM=amd
make BACKEND=hip HIP_PATH=/opt/rocm-7.2.0 GPU_ARCH=gfx90a HAVE_GEMMUL8=1 GEMMUL8_PATH=/home/users/sharmaas/GEMMul8 HAVE_OZABLAS=1 OZABLAS_PATH=/home/users/sharmaas/ozablas
```

`~/rocrmap.sh` is the ROCm equivalent bash script for CPU-GPU affinity:
```
OMP_NUM_THREADS=64 ~/rocrmap.sh ./gemm_test --method=ozablas-ozaki1 --m=10240 --n=10240 --k=10240 --phi=4.0 --splits=10
```
