#include "gemm_methods.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void disp_mat(int m, int n, const double *Mat) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      printf("%24.16e  ", Mat[j * m + i]);
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char **argv) try {
  Problem p;
  p.m = 4; p.n = 3; p.k = 5;
  p.lda = p.m; p.ldb = p.k; p.ldc = p.m;
  p.num_moduli = 2;

  bool print_matrices = false;
  bool method_given   = false;
  Method method       = Method::Reference;

  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--method=", 9) == 0) {
      method       = method_from_name(argv[i] + 9);
      method_given = true;
    }
    else if (strcmp(argv[i], "--print") == 0)       print_matrices = true;
    else if (strncmp(argv[i], "--moduli=", 9) == 0) p.num_moduli = atoi(argv[i] + 9);
    else throw std::invalid_argument(std::string("unknown option '") + argv[i] + "'");
  }

  if (!method_given) throw std::invalid_argument("missing --method=<name>");

  hipblasHandle_t handle;
  hipblasCreate(&handle);

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

  std::vector<double> hC(p.ldc * p.n, 0.0);
  std::vector<double> hC_exact(p.ldc * p.n, 0.0);

  double *A, *B, *C;
  hipMalloc(reinterpret_cast<void **>(&A), p.lda * p.k * sizeof(double));
  hipMalloc(reinterpret_cast<void **>(&B), p.ldb * p.n * sizeof(double));
  hipMalloc(reinterpret_cast<void **>(&C), p.ldc * p.n * sizeof(double));

  hipMemcpy(A, hA.data(), p.lda * p.k * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(B, hB.data(), p.ldb * p.n * sizeof(double), hipMemcpyHostToDevice);

  p.A = A; p.B = B; p.C = C;

  // baseline for every error norm below
  gemm_run(Method::Reference, handle, p, reinterpret_cast<void *>(1));
  hipMemcpy(hC_exact.data(), C, p.ldc * p.n * sizeof(double), hipMemcpyDeviceToHost);

  if (print_matrices) {
    printf("===== input A: %d x %d =====\n", p.m, p.k);
    disp_mat(p.m, p.k, hA.data());
    printf("===== input B: %d x %d =====\n", p.k, p.n);
    disp_mat(p.k, p.n, hB.data());
    printf("===== exact C: %d x %d =====\n", p.m, p.n);
    disp_mat(p.m, p.n, hC_exact.data());
  }

  if (!method_available(method))
    throw std::invalid_argument(std::string("method '") + method_name(method) +
                                "' is not compiled into this binary");

  void *work = nullptr;
  const size_t lwork = gemm_run(method, handle, p, nullptr);
  if (lwork > 0) hipMalloc(&work, lwork);

  gemm_run(method, handle, p, work != nullptr ? work : reinterpret_cast<void *>(1));
  hipMemcpy(hC.data(), C, p.ldc * p.n * sizeof(double), hipMemcpyDeviceToHost);

  double nrm = 0.0;
  for (int j = 0; j < p.ldc * p.n; ++j) {
    const double err = hC_exact[j] - hC[j];
    nrm              = std::fma(err, err, nrm);
  }
  nrm = std::sqrt(nrm);

  if (print_matrices) {
    printf("===== output C (%s): %d x %d =====\n", method_name(method), p.m, p.n);
    disp_mat(p.m, p.n, hC.data());
  }

  printf("method = %s\n", method_name(method));
  printf("error  = %e\n", nrm);

  if (work != nullptr) hipFree(work);

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
