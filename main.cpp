#include "gemm_methods.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

// The first call pays for lazy module loading, so only the second one is timed.
float time_gemm(Method method, hipblasHandle_t handle, const Problem &p) {
  hipEvent_t t0, t1;
  hipEventCreate(&t0);
  hipEventCreate(&t1);

  gemm_run(method, handle, p);

  hipEventRecord(t0, 0);
  gemm_run(method, handle, p);
  hipEventRecord(t1, 0);
  hipEventSynchronize(t1);

  float ms = 0.0f;
  hipEventElapsedTime(&ms, t0, t1);
  hipEventDestroy(t0);
  hipEventDestroy(t1);
  return ms;
}

}  // namespace

int main(int argc, char **argv) try {
  bool method_given = false;
  Method method     = Method::Native;
  bool verify_ref   = false;

  Problem p;
  p.m = 1024; p.n = 1024; p.k = 1024;
  double phi = 1.0;
  p.num_moduli = 2;
  p.num_splits = 2;

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
    else if (strncmp(argv[i], "--splits=", 9) == 0) p.num_splits = atoi(argv[i] + 9);
    else if (strcmp(argv[i], "--verify-ref") == 0) verify_ref = true;
    else throw std::invalid_argument(std::string("unknown option '") + argv[i] + "'");
  }

  if (!method_given) throw std::invalid_argument("missing --method=<name>");
  if (p.m <= 0 || p.n <= 0 || p.k <= 0)
    throw std::invalid_argument("m, n and k must be positive");

  set_fp64_emulation_gate(method == Method::CublasOzaki1);

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

  gemm_ref_gpu(p.m, p.n, p.k, A, B, C_exact);

  // Off by default: the host reference costs two transfers plus an FP128 gemm.
  double verify_norm = 0.0, verify_max = 0.0;
  if (verify_ref) {
    std::vector<double> hA(len_a), hB(len_b), hC(len_c);
    hipMemcpy(hA.data(), A, len_a * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(hB.data(), B, len_b * sizeof(double), hipMemcpyDeviceToHost);

    gemm_ref(p.m, p.n, p.k, hA.data(), hB.data(), hC.data());

    double *C_exact_cpu;
    hipMalloc(reinterpret_cast<void **>(&C_exact_cpu), len_c * sizeof(double));
    hipMemcpy(C_exact_cpu, hC.data(), len_c * sizeof(double), hipMemcpyHostToDevice);

    verify_norm = error_norm(C_exact_cpu, C_exact, len_c);
    verify_max  = max_error(C_exact_cpu, C_exact, len_c);
    hipFree(C_exact_cpu);
  }

  p.A = A; p.B = B; p.C = C_native;

  gemm_run(Method::Native, handle, p);
  const int native_bits = emulation_mantissa_bits();
  const float native_ms = time_gemm(Method::Native, handle, p);

  p.C = C;

  if (!method_available(method))
    throw std::invalid_argument(std::string("method '") + method_name(method) +
                                "' is not compiled into this binary");

  gemm_run(method, handle, p);
  const int emulated_bits = emulation_mantissa_bits();
  const float emulated_ms = time_gemm(method, handle, p);

  printf("method   = %s\n", method_name(method));
  printf("m,n,k    = %d,%d,%d\n", p.m, p.n, p.k);
  printf("phi      = %g\n", phi);
  // -1 means cuBLAS ran native FP64 rather than the emulated path.
  printf("mantissa = %d native, %d emulated\n", native_bits, emulated_bits);
  printf("time     = %.3f ms native, %.3f ms emulated\n", native_ms, emulated_ms);
  printf("                relative      max\n");
  printf("native   = %e  %e\n", error_norm(C_exact, C_native, len_c),
                                max_error(C_exact, C_native, len_c));
  printf("emulated = %e  %e\n", error_norm(C_exact, C, len_c),
                                max_error(C_exact, C, len_c));
  printf("diff     = %e  %e\n", error_norm(C_native, C, len_c),
                                max_error(C_native, C, len_c));
  // Both references are exact to well under an FP64 ulp, so this should be ~1e-16.
  if (verify_ref) printf("ref-diff = %e  %e\n", verify_norm, verify_max);

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
