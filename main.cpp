#include "gemm_methods.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

// The first call pays for lazy module loading, so only the last one is timed.
float time_gemm(Method method, hipblasHandle_t handle, const Problem &p, int warmups) {
  hipEvent_t t0, t1;
  hipEventCreate(&t0);
  hipEventCreate(&t1);

  for (int i = 0; i < warmups; ++i) gemm_run(method, handle, p);

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
  bool use_ref      = true;
  bool verify_ref   = false;

  Problem p;
  p.m = 1024; p.n = 1024; p.k = 1024;
  double phi = 1.0;
  int warmups = 2;
  int device = 0;
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
    else if (strncmp(argv[i], "--warmups=", 10) == 0) warmups = atoi(argv[i] + 10);
    else if (strncmp(argv[i], "--device=", 9) == 0) device = atoi(argv[i] + 9);
    else if (strcmp(argv[i], "--no-106bit-ref") == 0) use_ref = false;
    else if (strcmp(argv[i], "--verify-ref") == 0) verify_ref = true;
    else throw std::invalid_argument(std::string("unknown option '") + argv[i] + "'");
  }

  if (!method_given) throw std::invalid_argument("missing --method=<name>");
  if (!method_available(method))
    throw std::invalid_argument(std::string("method '") + method_name(method) +
                                "' is not compiled into this binary");
  if (p.m <= 0 || p.n <= 0 || p.k <= 0)
    throw std::invalid_argument("m, n and k must be positive");
  if (warmups < 0) throw std::invalid_argument("warmups cannot be negative");
  if (device < 0) throw std::invalid_argument("device cannot be negative");
  if (verify_ref && !use_ref)
    throw std::invalid_argument("--verify-ref checks the 106 bit reference, so it "
                                "cannot be used with --no-106bit-ref");

  set_fp64_emulation_gate(method == Method::CublasOzaki1);

  HIP_CHECK(hipSetDevice(device));

  p.lda = p.m; p.ldb = p.k; p.ldc = p.m;

  const size_t len_a = size_t(p.lda) * p.k;
  const size_t len_b = size_t(p.ldb) * p.n;
  const size_t len_c = size_t(p.ldc) * p.n;

  hipblasHandle_t handle;
  hipblasCreate(&handle);

  double *A, *B, *C, *C_native, *C_exact = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&A), len_a * sizeof(double)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&B), len_b * sizeof(double)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&C), len_c * sizeof(double)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&C_native), len_c * sizeof(double)));
  if (use_ref)
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&C_exact), len_c * sizeof(double)));

  // Fixed so a run is reproducible; the two values keep A and B different.
  fill_random(A, len_a, phi, 7774);
  fill_random(B, len_b, phi, 4777);

  if (use_ref) gemm_ref_gpu(p.m, p.n, p.k, A, B, C_exact);

  // Off by default: the host reference costs two transfers plus an FP128 gemm.
  double verify_norm = 0.0, verify_max = 0.0;
  if (verify_ref) {
    std::vector<double> hA(len_a), hB(len_b), hC(len_c);
    HIP_CHECK(hipMemcpy(hA.data(), A, len_a * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hB.data(), B, len_b * sizeof(double), hipMemcpyDeviceToHost));

    gemm_ref(p.m, p.n, p.k, hA.data(), hB.data(), hC.data());

    double *C_exact_cpu;
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&C_exact_cpu), len_c * sizeof(double)));
    HIP_CHECK(hipMemcpy(C_exact_cpu, hC.data(), len_c * sizeof(double), hipMemcpyHostToDevice));

    verify_norm = error_norm(C_exact_cpu, C_exact, len_c);
    verify_max  = max_error(C_exact_cpu, C_exact, len_c);
    HIP_CHECK(hipFree(C_exact_cpu));
  }

  p.A = A; p.B = B; p.C = C_native;

  // total_bytes is a device constant, so one read serves both runs below.
  size_t free_bytes = 0, total_bytes = 0;
  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

  // Both gemms are polled the same way; a workspace allocated and freed inside a
  // single call is invisible to a before/after snapshot.
  MemoryHighWaterMonitor native_mem;
  native_mem.start(device);

  EnergyMonitor native_energy;
  native_energy.start(device);
  const float native_ms = time_gemm(Method::Native, handle, p, warmups);
  const double native_j = native_energy.stop() / (warmups+1);

  hipDeviceSynchronize();
  const size_t native_min_free_bytes = native_mem.stop();

  // this is used to determine if native GEMM call actually ran an emulation
  const int native_bits = emulation_mantissa_bits();

  p.C = C;

  MemoryHighWaterMonitor emulated_mem;
  emulated_mem.start(device);

  EnergyMonitor emulated_energy;
  emulated_energy.start(device);
  const float emulated_ms = time_gemm(method, handle, p, warmups);
  const double emulated_j = emulated_energy.stop() / (warmups+1);

  hipDeviceSynchronize();

  const size_t emulated_min_free_bytes = emulated_mem.stop();

  printf("Run\n");
  printf("  method            : %s\n", method_name(method));
  printf("  m, n, k           : %d, %d, %d\n", p.m, p.n, p.k);
  printf("  phi               : %g\n", phi);
  // Ozaki I is parameterised by splits, Ozaki II by moduli; native uses neither.
  if (method == Method::CublasOzaki1 || method == Method::OzablasOzaki1)
    printf("  splits            : %d\n", p.num_splits);
  else if (method == Method::OzablasOzaki2 || method == Method::Gemmul8)
    printf("  moduli            : %d\n", p.num_moduli);
  printf("  warmups           : %d\n", warmups);
  if (native_bits > -1)
    printf("  note              : native GEMM call used cuBLAS emulation\n");

  printf("\nErrors\n");
  printf("  %-26s  %12s  %12s\n", "", "rel-frob", "max-rel-elem");
  if (use_ref) {
    printf("  %-26s  %12e  %12e\n", "64 bit native vs 106 bit",
                                        error_norm(C_exact, C_native, len_c),
                                        max_error(C_exact, C_native, len_c));
    printf("  %-26s  %12e  %12e\n", "64 bit emulated vs 106 bit",
                                        error_norm(C_exact, C, len_c),
                                        max_error(C_exact, C, len_c));
  }
  printf("  %-26s  %12e  %12e\n", "emulated vs native",
                                      error_norm(C_native, C, len_c),
                                      max_error(C_native, C, len_c));
  // Both references are exact to well under an FP64 ulp, so this should be ~1e-16.
  if (verify_ref)
    printf("  %-26s  %12e  %12e\n", "ref-diff", verify_norm, verify_max);

  // Least free memory seen is most memory in use, so the peaks invert the samples.
  const size_t native_peak_bytes   = total_bytes - native_min_free_bytes;
  const size_t emulated_peak_bytes = total_bytes - emulated_min_free_bytes;
  const double gb = 1024.0 * 1024.0 * 1024.0;
  const double overhead_gb = emulated_peak_bytes > native_peak_bytes
      ? (emulated_peak_bytes - native_peak_bytes) / gb
      : 0.0;

  printf("\nPerformance\n");
  printf("  time   [ms] (native | emulated) : %10.3f | %10.3f\n", native_ms, emulated_ms);
  printf("  memory [GB] (native | emulated) : %10.3f | %10.3f\n", native_peak_bytes / gb,
                                                                 emulated_peak_bytes / gb);
  if (native_j >= 0.0 && emulated_j >= 0.0)
    printf("  energy [J]  (native | emulated) : %10.3f | %10.3f\n", native_j, emulated_j);
  else
    printf("  energy [J]  (native | emulated) : %10s | %10s\n", "n/a", "n/a");

  if (use_ref) HIP_CHECK(hipFree(C_exact));
  HIP_CHECK(hipFree(C_native));
  HIP_CHECK(hipFree(C));
  HIP_CHECK(hipFree(B));
  HIP_CHECK(hipFree(A));
  hipblasDestroy(handle);
  hipDeviceReset();
  return 0;
} catch (const std::exception &e) {
  fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
