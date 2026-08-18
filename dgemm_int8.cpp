//=====================================================================
// HIP/CUDA portability shim
//
// The source below is written with HIP names.  On an NVIDIA platform the
// file is compiled by nvcc, so the HIP names are mapped onto their CUDA
// equivalents.  On an AMD platform (or hipcc targeting NVIDIA) the native
// HIP headers are used directly.
//=====================================================================
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <hipblas/hipblas.h>
#else
  #include <cuda_runtime.h>
  #include <cublas_v2.h>

  #define hipMalloc               cudaMalloc
  #define hipFree                 cudaFree
  #define hipMemcpy               cudaMemcpy
  #define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
  #define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
  #define hipDeviceReset          cudaDeviceReset

  #define hipblasHandle_t         cublasHandle_t
  #define hipblasCreate           cublasCreate
  #define hipblasDestroy          cublasDestroy
  #define hipblasDgemm            cublasDgemm
  #define HIPBLAS_OP_N            CUBLAS_OP_N
#endif

#include <gemmul8.hpp>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

void disp_mat(int m, int n, double *Mat) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%24.16e  ", Mat[j * m + i]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    const gemmul8::Backend backend = gemmul8::Backend::INT8;
    const int num_moduli           = 2;
    const bool fastmode            = false;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    const double alpha = 1.0;
    const double beta  = 0.0;

    const int m = 4, n = 3, k = 5;
    const int lda = m, ldb = k, ldc = m;
    std::vector<double> hA = {
        0x1.13491b78f7ff1p-1, 0x1.d5797d024f750p+0, -0x1.2121e4d9576a2p+1, 0x1.b96ec80cedfb6p-1,
        0x1.466a65212f053p-2, -0x1.4ec4a901fe3c4p+0, -0x1.bbff8c0e700a1p-2, 0x1.5ed8f2ba5f2dbp-2,
        0x1.ca08e9321d439p+1, 0x1.627ce99fd7ed1p+1, -0x1.599230c5450f8p+0, 0x1.84785f44e10f1p+1,
        0x1.73682ebd0c291p-1, -0x1.0245d3a33f7d8p-4, 0x1.6df2c829f659fp-1, -0x1.a3c53ea980203p-3,
        -0x1.fc7ec8b9281f7p-4, 0x1.7d5cd28a5e35bp+0, 0x1.68b67bfca10cfp+0, 0x1.6acd1f3bd1cafp+0};

    std::vector<double> hB = {
        0x1.57ce78e868ad7p-1, -0x1.351ddceb47a8bp+0, 0x1.6f39e78dc4de4p-1, 0x1.a1571993bf63bp+0,
        0x1.f4a0918ad43eep-2, 0x1.08e1a41eff3c4p+0, 0x1.742a49c7a8c1fp-1, -0x1.36b937c0e54f0p-2,
        0x1.2ceca451a1789p-2, -0x1.9316bb4db16cfp-1, 0x1.c6dbcad09ddd8p-1, -0x1.25a662f3a6d75p+0,
        -0x1.11a17e8d7e02fp+0, -0x1.9e769ce56b489p-1, -0x1.78de4dacf30d6p+1};

    std::vector<double> hC(lda * n, 0.0);
    std::vector<double> hC_exact(ldc * n, 0.0);

    double *A, *B, *C, *C_exact;
    hipMalloc(reinterpret_cast<void **>(&A), lda * k * sizeof(double));
    hipMalloc(reinterpret_cast<void **>(&B), ldb * n * sizeof(double));
    hipMalloc(reinterpret_cast<void **>(&C), ldc * n * sizeof(double));
    hipMalloc(reinterpret_cast<void **>(&C_exact), ldc * n * sizeof(double));

    hipMemcpy(A, hA.data(), lda * k * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(B, hB.data(), ldb * n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(C, hC.data(), ldc * n * sizeof(double), hipMemcpyHostToDevice);

    // reference result: native FP64 GEMM used as the baseline for the error norm
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                 m, n, k, &alpha, A, lda, B, ldb, &beta, C_exact, ldc);
    hipMemcpy(hC_exact.data(), C_exact, ldc * n * sizeof(double), hipMemcpyDeviceToHost);

    // When work == nullptr, the routine is used as a workspace-query call.
    // The requested BLAS-like operation is not executed; instead, GEMMul8 
    // computes and returns the required workspace sizes in bytes. 
    // calculate workspace size
    void *work = nullptr;
    std::vector<double> res =
        gemmul8::gemm<backend>(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
            num_moduli, fastmode, work);

    const size_t lwork = size_t(res[0]);

    // allocate workspace
    hipMalloc(&work, lwork);

    // run emulation
    gemmul8::gemm<backend>(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc,
        num_moduli, fastmode, work);

    // calculate error
    hipMemcpy(hC.data(), C, ldc * n * sizeof(double), hipMemcpyDeviceToHost);

    double nrm = 0.0;
    for (int i = 0; i < ldc * n; ++i) {
        const double err = hC_exact[i] - hC[i];
        nrm              = std::fma(err, err, nrm);
    }
    nrm = std::sqrt(nrm);

    // display results
    printf("===== input A: %d x %d =====\n", m, k);
    disp_mat(m, k, hA.data());

    printf("===== input B: %d x %d =====\n", k, n);
    disp_mat(k, n, hB.data());

    printf("===== exact C: %d x %d =====\n", m, n);
    disp_mat(m, n, hC_exact.data());

    printf("===== output C: %d x %d =====\n", m, n);
    disp_mat(m, n, hC.data());

    printf("error = %e\n", nrm);

    hipFree(work);
    hipFree(C_exact);
    hipFree(C);
    hipFree(B);
    hipFree(A);
    hipblasDestroy(handle);
    hipDeviceReset();
    return 0;
}
