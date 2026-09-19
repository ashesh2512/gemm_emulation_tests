#include "gemm_methods.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

struct RunResult {
  float ms;
  size_t peak_bytes;
  double joules;
  int mantissa_bits;  // -1 when the call was not emulated
};

// The first call pays for lazy module loading, so only the last one is timed.
// Memory is polled rather than snapshotted because a workspace allocated and
// freed inside a single call would be invisible to a before/after pair.
RunResult measure(Method method, hipblasHandle_t handle, const Problem &p, int warmups,
                  int device) {
  // hipMemGetInfo needs both out-pointers, but only the device wide total is used.
  size_t free_bytes = 0, total_bytes = 0;
  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

  MemoryHighWaterMonitor mem;
  mem.start(device);

  EnergyMonitor energy;
  energy.start(device);

  hipEvent_t t0, t1;
  hipEventCreate(&t0);
  hipEventCreate(&t1);

  for (int i = 0; i < warmups; ++i) gemm_run(method, handle, p);

  hipEventRecord(t0, 0);
  gemm_run(method, handle, p);
  hipEventRecord(t1, 0);
  hipEventSynchronize(t1);

  RunResult r;
  hipEventElapsedTime(&r.ms, t0, t1);
  hipEventDestroy(t0);
  hipEventDestroy(t1);

  r.joules = energy.stop() / (warmups+1);

  hipDeviceSynchronize();
  // Least free memory seen is most memory in use, so the peak inverts the sample.
  r.peak_bytes = total_bytes - mem.stop();

  r.mantissa_bits = emulation_mantissa_bits();
  return r;
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
    else if (strcmp(argv[i], "--auto-mantissa") == 0) p.auto_mantissa = true;
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
  if (p.auto_mantissa && method != Method::CublasOzaki1)
    throw std::invalid_argument("--auto-mantissa only applies to cublas-ozaki1");

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
  const RunResult native = measure(Method::Native, handle, p, warmups, device);

  // Only when the vendor dgemm emulated is its result not FP64, so only then is
  // it replaced by a hand written FP64 kernel for the error table.
  if (native.mantissa_bits > -1) {
    gemm_fp64_gpu(p.m, p.n, p.k, A, B, C_native);
    hipDeviceSynchronize();
  }

  p.C = C;
  const RunResult emulated = measure(method, handle, p, warmups, device);

  printf("Run\n");
  printf("  method            : %s\n", method_name(method));
  printf("  m, n, k           : %d, %d, %d\n", p.m, p.n, p.k);
  printf("  phi               : %g\n", phi);
  // Ozaki I is parameterised by splits, Ozaki II by moduli; native uses neither.
  if (method == Method::CublasOzaki1 && p.auto_mantissa)
    printf("  splits            : auto\n");
  else if (method == Method::CublasOzaki1 || method == Method::OzablasOzaki1)
    printf("  splits            : %d\n", p.num_splits);
  else if (method == Method::OzablasOzaki2 || method == Method::Gemmul8)
    printf("  moduli            : %d\n", p.num_moduli);
  printf("  warmups           : %d\n", warmups);
  if (native.mantissa_bits > -1)
    printf("  note              : native GEMM call used cuBLAS emulation, "
           "cuBLAS chose %d splits; native error is computed with a hand written "
           "FP64 kernel and native performance is reported as n/a\n",
           emulation_splits(native.mantissa_bits));
  // cuBLAS picked the bit count itself whenever the count was not pinned.
  if (p.auto_mantissa && method == Method::CublasOzaki1 && emulated.mantissa_bits > -1)
    printf("  note              : cuBLAS chose %d splits\n",
           emulation_splits(emulated.mantissa_bits));

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

  const double gb = 1024.0 * 1024.0 * 1024.0;

  printf("\nPerformance\n");
  // A native call that emulated says nothing about native cost.
  const bool native_perf = native.mantissa_bits <= -1;
  if (native_perf)
    printf("  time   [ms] (native | emulated) : %10.3f | %10.3f\n", native.ms, emulated.ms);
  else
    printf("  time   [ms] (native | emulated) : %10s | %10.3f\n", "n/a", emulated.ms);

  if (native_perf)
    printf("  memory [GB] (native | emulated) : %10.3f | %10.3f\n", native.peak_bytes / gb,
                                                                   emulated.peak_bytes / gb);
  else
    printf("  memory [GB] (native | emulated) : %10s | %10.3f\n", "n/a",
                                                                 emulated.peak_bytes / gb);

  if (native_perf)
    printf("  energy [J]  (native | emulated) : %10.3f | %10.3f\n", native.joules, emulated.joules);
  else
    printf("  energy [J]  (native | emulated) : %10s | %10.3f\n", "n/a", emulated.joules);

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
