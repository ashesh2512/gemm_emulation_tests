#include "gemm_methods.hpp"

#if defined(HAVE_GEMMUL8)
  #include <gemmul8.hpp>
#endif

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

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
