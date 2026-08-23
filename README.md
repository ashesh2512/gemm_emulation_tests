The build command should explicitly specify the backend (`cuda`, `hip`), backend path, gpu architecture, and library used for emmulation, E.g. on MI250X GPUs, a build command using Ozaki II emulation via the Gemmul8 library looks like - 
```
make BACKEND=hip HIP_PATH=/opt/rocm-7.2.0 GPU_ARCH=gfx90a HAVE_GEMMUL8=1 GEMMUL8_PATH=/path/to/GEMMul8
```
None of these paths have defaults, so the build stops with an error if one is missing. `GPU_ARCH=auto` queries the device instead, and `BACKEND` defaults to `auto`. Enabling `HAVE_OZABLAS=1`, e.g., likewise requires `OZABLAS_PATH`.
It is possible to have multiple libraries linked at the same time for emmulation. Hence to run the test, it is important to specify the method, e.g.
```
OMP_NUM_THREADS=64 ./gemm_test --method=gemmul8
```
Other options for method are `reference` which is the GPU library for BLAS, `cublas-ozaki1`, `ozablas-ozaki1`, `ozablas-ozaki2`.

The remaining options describe the problem being solved:

| Option | Default | Meaning |
| --- | --- | --- |
| `--m=<int>` | 1024 | Rows of A and C. |
| `--n=<int>` | 1024 | Columns of B and C. |
| `--k=<int>` | 1024 | Inner dimension, i.e. columns of A and rows of B. |
| `--phi=<double>` | 1.0 | Controls the entries of A and B. |
| `--moduli=<int>` | 2 | Number of moduli used by the Ozaki II methods. |

`--phi` sets how the random input matrices are generated. A negative value draws every entry from a standard normal distribution. A non-negative value uses `(rand - 0.5) * exp(randn * phi)` instead, so the exponent range of the entries widens as `phi` grows and the emulation has a harder time matching FP64. The seed is fixed in the code, so repeated runs with the same options give the same matrices.

The reference is an FP128 matrix multiply on the host, so it is exact for these problem sizes. It runs under OpenMP, and `OMP_NUM_THREADS` sets the thread count. The native `dgemm` and the chosen method are both measured against it, as a relative Frobenius norm and as the worst single element:
```
OMP_NUM_THREADS=64 ./gemm_test --method=gemmul8 --m=1024 --n=1024 --k=1024 --phi=2.0 --moduli=3.0
```
```
                relative      max
native   = 1.041317e-15  1.131069e-09
emulated = 5.097529e-13  3.970221e-06
diff     = 5.097528e-13  3.969089e-06
```
`native` is the error of the vendor FP64 `dgemm`, `emulated` the error of the selected method, and `diff` compares the two against each other.
