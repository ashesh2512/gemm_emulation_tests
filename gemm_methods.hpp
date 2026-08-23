#pragma once

//=====================================================================
// HIP/CUDA portability shim
//
// Everything below is written with HIP names.  On an NVIDIA platform the
// sources are compiled by nvcc, so the HIP names are mapped onto their
// CUDA equivalents.  On an AMD platform (or hipcc targeting NVIDIA) the
// native HIP headers are used directly.
//=====================================================================
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <hipblas/hipblas.h>
  #include <hiprand/hiprand_kernel.h>
#else
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <curand_kernel.h>

  #define hipMalloc               cudaMalloc
  #define hipFree                 cudaFree
  #define hipMemcpy               cudaMemcpy
  #define hipMemset               cudaMemset
  #define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
  #define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
  #define hipDeviceReset          cudaDeviceReset

  #define hipblasHandle_t         cublasHandle_t
  #define hipblasOperation_t      cublasOperation_t
  #define hipblasCreate           cublasCreate
  #define hipblasDestroy          cublasDestroy
  #define hipblasDgemm            cublasDgemm
  #define HIPBLAS_OP_N            CUBLAS_OP_N
  #define HIPBLAS_OP_T            CUBLAS_OP_T
  #define HIPBLAS_OP_C            CUBLAS_OP_C

  #define hiprandState_t          curandState_t
  #define hiprand_init            curand_init
  #define hiprand_uniform_double  curand_uniform_double
  #define hiprand_normal_double   curand_normal_double
#endif

#include <cstddef>
#include <string>

enum class Method {
  Reference,      // native FP64 dgemm, also the baseline for the error norm
  CublasOzaki1,   // Ozaki I built directly on cuBLAS   (NVIDIA only)
  OzablasOzaki1,  // Ozaki I via ozablas
  OzablasOzaki2,  // Ozaki II via ozablas
  Gemmul8,        // Ozaki II via GEMMul8
  Count
};

struct Problem {
  hipblasOperation_t transa = HIPBLAS_OP_N;
  hipblasOperation_t transb = HIPBLAS_OP_N;

  int m = 0, n = 0, k = 0;
  double alpha = 1.0, beta = 0.0;

  const double *A = nullptr; int lda = 0;
  const double *B = nullptr; int ldb = 0;
  double *C       = nullptr; int ldc = 0;

  int num_moduli = 2;      // Ozaki II
  int num_splits = 2;      // Ozaki I
  bool fastmode  = false;  // Ozaki II
};

// Fills n device doubles: randn when phi < 0, else (rand - 0.5) * exp(randn * phi).
// Larger phi widens the exponent range and makes the emulated gemm harder.
void fill_random(double *x, size_t n, double phi, unsigned long long seed);

// ||x - y|| / ||x||, both device pointers, so the result is dimensionless.
double error_norm(const double *x, const double *y, size_t n);

// Worst single element of |x - y| / |x|, falling back to |x - y| where x is zero.
double max_error(const double *x, const double *y, size_t n);

// Host pointers, column-major, lda = m, ldb = k, ldc = m. Accumulates every dot
// product in FP128, so the result is the yardstick both gemms are measured against.
void gemm_fp128(int m, int n, int k, const double *A, const double *B, double *C);

const char *method_name(Method method);

// Throws std::invalid_argument listing the accepted names when there is no match.
Method method_from_name(const std::string &name);

// False when the library was not compiled in or the GPU is the wrong vendor.
bool method_available(Method method);

// work == nullptr: return the required workspace in bytes, compute nothing.
// otherwise:       compute C and return 0.
size_t gemm_run(Method method, hipblasHandle_t handle, const Problem &p, void *work);
