The build command should explicitly specify the backend (`cuda`, `hip`), backend path, gpu architecture, and library used for emmulation, E.g. on MI250X GPUs, a build command using Ozaki II emulation via the Gemmul8 library looks like - 
```
make BACKEND=hip HIP_PATH=/opt/rocm-7.2.0 GPU_ARCH=gfx90a HAVE_GEMMUL8=1 
```
It is possible to have multiple libraries linked at the same time for emmulation. Hence to run the test, it is important to specify the method, e.g.
```
./gemm_test --method=gemmul8
```
Other options for method are `reference` which is the GPU library for BLAS, `cublas-ozaki1`, `ozablas-ozaki1`, `ozablas-ozaki2`.
