#include "gemm_methods.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) try {
  bool method_given = false;
  Method method     = Method::Reference;

  Problem p;
  p.m = 1024; p.n = 1024; p.k = 1024;
  double phi = 1.0;
  p.num_moduli = 2;

  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--method=", 9) == 0) {
      method       = method_from_name(argv[i] + 9);
      method_given = true;
    }
    else if (strncmp(argv[i], "--m=", 4) == 0) p.m = atoi(argv[i] + 4);
    else if (strncmp(argv[i], "--n=", 4) == 0) p.n = atoi(argv[i] + 4);
    else if (strncmp(argv[i], "--k=", 4) == 0) p.k = atoi(argv[i] + 4);
    else if (strncmp(argv[i], "--phi=", 6) == 0) phi = atof(argv[i] + 6);
    else if (strncmp(argv[i], "--moduli=", 9) == 0) p.num_moduli = atoi(argv[i] + 9);
    else throw std::invalid_argument(std::string("unknown option '") + argv[i] + "'");
  }

  if (!method_given) throw std::invalid_argument("missing --method=<name>");
  if (p.m <= 0 || p.n <= 0 || p.k <= 0)
    throw std::invalid_argument("m, n and k must be positive");

  p.lda = p.m; p.ldb = p.k; p.ldc = p.m;

  const size_t len_a = size_t(p.lda) * p.k;
  const size_t len_b = size_t(p.ldb) * p.n;
  const size_t len_c = size_t(p.ldc) * p.n;

  hipblasHandle_t handle;
  hipblasCreate(&handle);

  double *A, *B, *C, *C_native, *C_exact;
  hipMalloc(reinterpret_cast<void **>(&A), len_a * sizeof(double));
  hipMalloc(reinterpret_cast<void **>(&B), len_b * sizeof(double));
  hipMalloc(reinterpret_cast<void **>(&C), len_c * sizeof(double));
  hipMalloc(reinterpret_cast<void **>(&C_native), len_c * sizeof(double));
  hipMalloc(reinterpret_cast<void **>(&C_exact), len_c * sizeof(double));

  // Fixed so a run is reproducible; the two values keep A and B different.
  fill_random(A, len_a, phi, 7774);
  fill_random(B, len_b, phi, 4777);

  // The FP128 reference runs on the host, so the operands have to come back.
  std::vector<double> hA(len_a), hB(len_b), hC(len_c);
  hipMemcpy(hA.data(), A, len_a * sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy(hB.data(), B, len_b * sizeof(double), hipMemcpyDeviceToHost);

  gemm_fp128(p.m, p.n, p.k, hA.data(), hB.data(), hC.data());
  hipMemcpy(C_exact, hC.data(), len_c * sizeof(double), hipMemcpyHostToDevice);

  p.A = A; p.B = B; p.C = C_native;

  gemm_run(Method::Reference, handle, p, reinterpret_cast<void *>(1));

  p.C = C;

  if (!method_available(method))
    throw std::invalid_argument(std::string("method '") + method_name(method) +
                                "' is not compiled into this binary");

  void *work = nullptr;
  const size_t lwork = gemm_run(method, handle, p, nullptr);
  if (lwork > 0) hipMalloc(&work, lwork);

  gemm_run(method, handle, p, work != nullptr ? work : reinterpret_cast<void *>(1));

  printf("method   = %s\n", method_name(method));
  printf("m,n,k    = %d,%d,%d\n", p.m, p.n, p.k);
  printf("phi      = %g\n", phi);
  printf("                relative      max\n");
  printf("native   = %e  %e\n", error_norm(C_exact, C_native, len_c),
                                max_error(C_exact, C_native, len_c));
  printf("emulated = %e  %e\n", error_norm(C_exact, C, len_c),
                                max_error(C_exact, C, len_c));
  printf("diff     = %e  %e\n", error_norm(C_native, C, len_c),
                                max_error(C_native, C, len_c));

  if (work != nullptr) hipFree(work);

  hipFree(C_exact);
  hipFree(C_native);
  hipFree(C);
  hipFree(B);
  hipFree(A);
  hipblasDestroy(handle);
  hipDeviceReset();
  return 0;
} catch (const std::exception &e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
