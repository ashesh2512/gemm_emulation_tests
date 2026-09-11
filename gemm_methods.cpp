#include "gemm_methods.hpp"

#if defined(HAVE_GEMMUL8)
  #include <gemmul8.hpp>
#endif

#if defined(HAVE_OZABLAS)
  #include <ozablas/ozablas.hpp>
  #include <ozablas/core/executor.hpp>
  #include <ozablas/core/workspace.hpp>
  #include <memory>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

__global__ void fill_random_kernel(double *x, size_t n, double phi, unsigned long long seed) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= n) return;

  // The index is the sequence number, so each element is independent of n.
  hiprandState_t state;
  hiprand_init(seed, i, 0, &state);

  const double rand  = hiprand_uniform_double(&state);
  const double randn = hiprand_normal_double(&state);

  x[i] = phi < 0.0 ? randn : (rand - 0.5) * exp(randn * phi);
}

__global__ void error_norm_kernel(const double *x, const double *y, size_t n, double *acc) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= n) return;

  const double err = x[i] - y[i];
  atomicAdd(&acc[0], err * err);
  atomicAdd(&acc[1], x[i] * x[i]);
}

__global__ void max_error_kernel(const double *x, const double *y, size_t n,
                                 unsigned long long *acc) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= n) return;

  const double err = fabs(x[i] - y[i]);
  const double rel = x[i] != 0.0 ? err / fabs(x[i]) : err;

  // rel is never negative, so its bit pattern orders the same way it does.
  atomicMax(acc, __double_as_longlong(rel));
}

}  // namespace

void fill_random(double *x, size_t n, double phi, unsigned long long seed) {
  const int block = 256;
  fill_random_kernel<<<(n + block - 1) / block, block>>>(x, n, phi, seed);
}

double error_norm(const double *x, const double *y, size_t n) {
  double *acc = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&acc), 2 * sizeof(double)));
  HIP_CHECK(hipMemset(acc, 0, 2 * sizeof(double)));

  const int block = 256;
  // The atomics sum in an arbitrary order, so the last digits can move run to run.
  error_norm_kernel<<<(n + block - 1) / block, block>>>(x, y, n, acc);

  double sum[2] = {0.0, 0.0};
  HIP_CHECK(hipMemcpy(sum, acc, 2 * sizeof(double), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(acc));

  return sum[1] > 0.0 ? std::sqrt(sum[0] / sum[1]) : std::sqrt(sum[0]);
}

double max_error(const double *x, const double *y, size_t n) {
  unsigned long long *acc = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&acc), sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(acc, 0, sizeof(unsigned long long)));

  const int block = 256;
  max_error_kernel<<<(n + block - 1) / block, block>>>(x, y, n, acc);

  unsigned long long bits = 0;
  HIP_CHECK(hipMemcpy(&bits, acc, sizeof(unsigned long long), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(acc));

  double out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

// __float128 is an x86-only GCC extension; on AArch64 long double is IEEE binary128.
#if defined(__aarch64__)
using fp128_t = long double;
#else
using fp128_t = __float128;
#endif

void gemm_ref(int m, int n, int k, const double *A, const double *B, double *C) {
  // Row-major copy of A so both operands are walked contiguously below.
  std::vector<double> At(size_t(m) * k);
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < m; ++i) At[size_t(i) * k + j] = A[size_t(j) * m + i];

#pragma omp parallel for schedule(static)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      const double *a = &At[size_t(i) * k];
      const double *b = &B[size_t(j) * k];

      fp128_t sum = 0;
      for (int l = 0; l < k; ++l)
        sum += static_cast<fp128_t>(a[l]) * static_cast<fp128_t>(b[l]);

      C[size_t(j) * m + i] = static_cast<double>(sum);
    }
  }
}

namespace {

// The _rn intrinsics are used throughout so no compiler can contract or reorder
// these away; the error-free transforms below are only exact as written.

// Knuth's TwoSum
__device__ void two_sum(double a, double b, double &s, double &e) {
  s = __dadd_rn(a, b);
  const double bb = __dsub_rn(s, a);
  e = __dadd_rn(__dsub_rn(a, __dsub_rn(s, bb)), __dsub_rn(b, bb));
}

__device__ void two_prod(double a, double b, double &p, double &e) {
  p = __dmul_rn(a, b);
  e = __fma_rn(a, b, -p);
}

__global__ void gemm_ref_kernel(int m, int n, int k, const double *A, const double *B,
                                double *C) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;

  // (hi, lo) is an unevaluated double-double, so the dot product carries ~106
  // mantissa bits against the 113 the host __float128 path has.
  double hi = 0.0, lo = 0.0;

  for (int l = 0; l < k; ++l) {
    double p, pe;
    two_prod(A[size_t(l) * m + i], B[size_t(j) * k + l], p, pe);

    double se;
    two_sum(hi, p, hi, se);
    lo = __dadd_rn(lo, __dadd_rn(se, pe));
  }

  C[size_t(j) * m + i] = __dadd_rn(hi, lo);
}

}  // namespace

void gemm_ref_gpu(int m, int n, int k, const double *A, const double *B, double *C) {
  const dim3 block(16, 16);
  const dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  gemm_ref_kernel<<<grid, block>>>(m, n, k, A, B, C);
}

const char *method_name(Method method) {
  switch (method) {
    case Method::Native:        return "native";
    case Method::CublasOzaki1:  return "cublas-ozaki1";
    case Method::OzablasOzaki1: return "ozablas-ozaki1";
    case Method::OzablasOzaki2: return "ozablas-ozaki2";
    case Method::Gemmul8:       return "gemmul8";
    default:
      fprintf(stderr, "error: bad gemm method %d\n", int(method));
      exit(1);
  }
}

Method method_from_name(const std::string &name) {
  for (int i = 0; i < int(Method::Count); ++i) {
    if (name == method_name(Method(i))) return Method(i);
  }

  std::string message = "unknown method '" + name + "'; expected one of:";
  for (int i = 0; i < int(Method::Count); ++i) {
    message += ' ';
    message += method_name(Method(i));
  }
  throw std::invalid_argument(message);
}

bool method_available(Method method) {
  switch (method) {
    case Method::Native:
      return true;

    case Method::CublasOzaki1:
#if defined(HAVE_CUBLAS_OZAKI1) && !defined(__HIP_PLATFORM_AMD__)
      return true;
#else
      return false;
#endif

    case Method::OzablasOzaki1:
    case Method::OzablasOzaki2:
#if defined(HAVE_OZABLAS)
      return true;
#else
      return false;
#endif

    case Method::Gemmul8:
#if defined(HAVE_GEMMUL8)
      return true;
#else
      return false;
#endif

    default:
      fprintf(stderr, "error: bad gemm method %d\n", int(method));
      exit(1);
  }
}

#if defined(HAVE_CUBLAS_OZAKI1) && !defined(__HIP_PLATFORM_AMD__)
namespace {

int *d_mantissa_bits = nullptr;

// mantissa_bits <= 0 selects the strategy under which cuBLAS declines to emulate on
// FP64-strong parts; there is no API that disables emulation outright.
void set_fp64_emulation(cublasHandle_t handle, int mantissa_bits) {
  const bool off = mantissa_bits <= 0;

  if (d_mantissa_bits == nullptr)
    cudaMalloc(reinterpret_cast<void **>(&d_mantissa_bits), sizeof(int));
  cudaMemset(d_mantissa_bits, 0xFF, sizeof(int));
  cublasSetFixedPointEmulationMantissaBitCountPointer(handle, d_mantissa_bits);

  cublasSetEmulationStrategy(handle, off ? CUBLAS_EMULATION_STRATEGY_DEFAULT
                                         : CUBLAS_EMULATION_STRATEGY_EAGER);

  if (off) {
    cublasSetFixedPointEmulationMantissaControl(
        handle, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
  } else {
    cublasSetFixedPointEmulationMantissaControl(
        handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
    cublasSetFixedPointEmulationMaxMantissaBitCount(handle, mantissa_bits);
  }
}

}  // namespace
#endif

void set_fp64_emulation_gate(bool enabled) {
#if defined(HAVE_CUBLAS_OZAKI1) && !defined(__HIP_PLATFORM_AMD__)
  setenv("CUBLAS_EMULATE_DOUBLE_PRECISION", enabled ? "1" : "0", 1);
#endif
}

int emulation_mantissa_bits() {
#if defined(HAVE_CUBLAS_OZAKI1) && !defined(__HIP_PLATFORM_AMD__)
  if (d_mantissa_bits == nullptr) return -1;

  int bits = -1;
  cudaMemcpy(&bits, d_mantissa_bits, sizeof(int), cudaMemcpyDeviceToHost);
  return bits;
#else
  return -1;
#endif
}

void gemm_run(Method method, hipblasHandle_t handle, const Problem &p) {
  switch (method) {

    case Method::Native: {
#if defined(HAVE_CUBLAS_OZAKI1) && !defined(__HIP_PLATFORM_AMD__)
      set_fp64_emulation(handle, 0);
#endif
      hipblasDgemm(handle, p.transa, p.transb, p.m, p.n, p.k,
                   &p.alpha, p.A, p.lda, p.B, p.ldb, &p.beta, p.C, p.ldc);
      return;
    }

#if defined(HAVE_CUBLAS_OZAKI1) && !defined(__HIP_PLATFORM_AMD__)
    case Method::CublasOzaki1: {
      // sliceCount = ceildiv(mantissaBitCount + 1, 8), so this pins num_splits slices.
      set_fp64_emulation(handle, 8 * p.num_splits - 1);

      cublasGemmEx(handle, p.transa, p.transb, p.m, p.n, p.k,
                   &p.alpha, p.A, CUDA_R_64F, p.lda,
                             p.B, CUDA_R_64F, p.ldb,
                   &p.beta,  p.C, CUDA_R_64F, p.ldc,
                   CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
      return;
    }
#endif

#if defined(HAVE_OZABLAS)
    case Method::OzablasOzaki1:
    case Method::OzablasOzaki2: {
  #if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
      std::shared_ptr<const ozablas::Executor> exec =
          std::make_shared<ozablas::HipExecutor>(0);
  #else
      std::shared_ptr<const ozablas::Executor> exec =
          std::make_shared<ozablas::CudaExecutor>(0);
  #endif

      // ozablas is row-major and only does C = A * B, so the column-major
      // problem is handed over as C^T = B^T * A^T with the operands swapped.
      if (method == Method::OzablasOzaki1) {
        ozablas::WorkspaceScheme1 ws(exec, p.n, p.m, p.k, p.num_splits);
        ozablas::ozaki_scheme1_gemm(ws, p.B, p.A, p.C);
      } else {
        ozablas::WorkspaceScheme2 ws(exec, p.n, p.m, p.k, p.num_moduli);
        ozablas::ozaki_scheme2_gemm(ws, p.B, p.A, p.C);
      }

      exec->synchronize();
      return;
    }
#endif

#if defined(HAVE_GEMMUL8)
    case Method::Gemmul8: {
      void *work = nullptr;
      HIP_CHECK(hipMalloc(&work, gemmul8::workSize(p.m, p.n, p.k, p.num_moduli)));

      gemmul8::gemm<gemmul8::Backend::INT8>(
          handle, p.transa, p.transb, p.m, p.n, p.k,
          &p.alpha, p.A, p.lda, p.B, p.ldb, &p.beta, p.C, p.ldc,
          p.num_moduli, p.fastmode, work);

      HIP_CHECK(hipFree(work));
      return;
    }
#endif

    default:
      fprintf(stderr, "error: method %d is not compiled into this binary\n", int(method));
      exit(1);
  }
}
