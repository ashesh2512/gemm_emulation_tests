#include "gemm_methods.hpp"

#if defined(HAVE_GEMMUL8)
  #include <gemmul8.hpp>
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
  hipMalloc(reinterpret_cast<void **>(&acc), 2 * sizeof(double));
  hipMemset(acc, 0, 2 * sizeof(double));

  const int block = 256;
  // The atomics sum in an arbitrary order, so the last digits can move run to run.
  error_norm_kernel<<<(n + block - 1) / block, block>>>(x, y, n, acc);

  double sum[2] = {0.0, 0.0};
  hipMemcpy(sum, acc, 2 * sizeof(double), hipMemcpyDeviceToHost);
  hipFree(acc);

  return sum[1] > 0.0 ? std::sqrt(sum[0] / sum[1]) : std::sqrt(sum[0]);
}

double max_error(const double *x, const double *y, size_t n) {
  unsigned long long *acc = nullptr;
  hipMalloc(reinterpret_cast<void **>(&acc), sizeof(unsigned long long));
  hipMemset(acc, 0, sizeof(unsigned long long));

  const int block = 256;
  max_error_kernel<<<(n + block - 1) / block, block>>>(x, y, n, acc);

  unsigned long long bits = 0;
  hipMemcpy(&bits, acc, sizeof(unsigned long long), hipMemcpyDeviceToHost);
  hipFree(acc);

  double out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

void gemm_fp128(int m, int n, int k, const double *A, const double *B, double *C) {
  // Row-major copy of A so both operands are walked contiguously below.
  std::vector<double> At(size_t(m) * k);
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < m; ++i) At[size_t(i) * k + j] = A[size_t(j) * m + i];

#pragma omp parallel for schedule(static)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      const double *a = &At[size_t(i) * k];
      const double *b = &B[size_t(j) * k];

      __float128 sum = 0;
      for (int l = 0; l < k; ++l)
        sum += static_cast<__float128>(a[l]) * static_cast<__float128>(b[l]);

      C[size_t(j) * m + i] = static_cast<double>(sum);
    }
  }
}

const char *method_name(Method method) {
  switch (method) {
    case Method::Reference:     return "reference";
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
    case Method::Reference:
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

size_t gemm_run(Method method, hipblasHandle_t handle, const Problem &p, void *work) {
  switch (method) {

    case Method::Reference: {
      if (work == nullptr) return 0;
      hipblasDgemm(handle, p.transa, p.transb, p.m, p.n, p.k,
                   &p.alpha, p.A, p.lda, p.B, p.ldb, &p.beta, p.C, p.ldc);
      return 0;
    }

#if defined(HAVE_CUBLAS_OZAKI1) && !defined(__HIP_PLATFORM_AMD__)
    case Method::CublasOzaki1: {
      // TODO: split A and B into exactly representable slices, accumulate the
      // pairwise cublasDgemm products, then sum in decreasing magnitude order.
      return 0;
    }
#endif

#if defined(HAVE_OZABLAS)
    case Method::OzablasOzaki1:
    case Method::OzablasOzaki2: {
      // TODO: create the ozablas handle, set the split count (Ozaki I) or the
      // modulus count and fastmode (Ozaki II), then call its Dgemm.
      return 0;
    }
#endif

#if defined(HAVE_GEMMUL8)
    case Method::Gemmul8: {
      // GEMMul8 uses the same call for both phases: a null workspace makes it
      // report the required sizes instead of multiplying.
      std::vector<double> res = gemmul8::gemm<gemmul8::Backend::INT8>(
          handle, p.transa, p.transb, p.m, p.n, p.k,
          &p.alpha, p.A, p.lda, p.B, p.ldb, &p.beta, p.C, p.ldc,
          p.num_moduli, p.fastmode, work);
      return work == nullptr ? size_t(res[0]) : 0;
    }
#endif

    default:
      fprintf(stderr, "error: method %d is not compiled into this binary\n", int(method));
      exit(1);
  }
}
