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
  #define hipMemGetInfo           cudaMemGetInfo
  #define hipSetDevice            cudaSetDevice
  #define hipDeviceSynchronize    cudaDeviceSynchronize

  #define hipError_t              cudaError_t
  #define hipSuccess              cudaSuccess
  #define hipGetErrorString       cudaGetErrorString

  #define hipEvent_t              cudaEvent_t
  #define hipEventCreate          cudaEventCreate
  #define hipEventRecord          cudaEventRecord
  #define hipEventSynchronize     cudaEventSynchronize
  #define hipEventElapsedTime     cudaEventElapsedTime
  #define hipEventDestroy         cudaEventDestroy

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
#include <stdexcept>
#include <string>

// Wraps a HIP/CUDA runtime call and throws naming the call that failed.
#define HIP_CHECK(expr)                                                     \
  do {                                                                      \
    const hipError_t hip_check_err = (expr);                                \
    if (hip_check_err != hipSuccess)                                        \
      throw std::runtime_error(std::string(#expr) + " failed: " +           \
                               hipGetErrorString(hip_check_err));           \
  } while (0)

enum class Method {
  Native,         // native FP64 dgemm, also the baseline for the error norm
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

  // cuBLAS Ozaki I: ignore num_splits and let cuBLAS pick the mantissa bit count.
  bool auto_mantissa = false;
};

// Polls hipMemGetInfo() from a background host thread to catch transient
// peak GPU memory usage (e.g. cuBLAS emulation workspace that is allocated
// and freed internally within a single call, which a before/after snapshot
// would miss).
#include <atomic>
#include <thread>
class MemoryHighWaterMonitor {
public:
    void start(int device_id, std::chrono::microseconds poll_interval = std::chrono::microseconds(100)) {
        device_id_ = device_id;
        poll_interval_ = poll_interval;
        stop_.store(false);
        min_free_bytes_.store(SIZE_MAX);
        worker_ = std::thread([this]() {
            HIP_CHECK(hipSetDevice(device_id_));
            while (!stop_.load(std::memory_order_relaxed)) {
                size_t free_bytes = 0, total_bytes = 0;
                if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
                    size_t prev = min_free_bytes_.load(std::memory_order_relaxed);
                    while (free_bytes < prev &&
                           !min_free_bytes_.compare_exchange_weak(prev, free_bytes, std::memory_order_relaxed)) {}
                }
                std::this_thread::sleep_for(poll_interval_);
            }
        });
    }

    // Stops polling and returns the minimum free memory (bytes) observed.
    size_t stop() {
        stop_.store(true);
        worker_.join();
        return min_free_bytes_.load();
    }

private:
    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> min_free_bytes_{SIZE_MAX};
    int device_id_ = 0;
    std::chrono::microseconds poll_interval_{1};
};

// Reads the cumulative on-board energy counter (NVML on NVIDIA, ROCm SMI on
// AMD), so no sampling is needed. Both counters are device wide, not per process.
class EnergyMonitor {
 public:
  void start(int device_id);

  // Joules since start(), or -1 when the device exposes no energy counter.
  double stop();

 private:
  int device_id_ = 0;
  double start_joules_ = -1.0;
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
void gemm_ref(int m, int n, int k, const double *A, const double *B, double *C);

// Same problem and layout as gemm_ref, but device pointers and double-double
// accumulation (~106 mantissa bits). This is the reference every run uses.
void gemm_ref_gpu(int m, int n, int k, const double *A, const double *B, double *C);

const char *method_name(Method method);

// Throws std::invalid_argument listing the accepted names when there is no match.
Method method_from_name(const std::string &name);

// False when the library was not compiled in or the GPU is the wrong vendor.
bool method_available(Method method);

// Opens or shuts the cuBLAS FP64 emulation gate. cuBLAS latches this at library
// init, so it must run before any other cuBLAS call and cannot be changed after.
void set_fp64_emulation_gate(bool enabled);

// Mantissa bits the last gemm_run() retained, or -1 when it was not emulated.
int emulation_mantissa_bits();

// Splits cuBLAS breaks each operand into for the given mantissa bit count.
int emulation_splits(int mantissa_bits);

// Computes C. Any scratch memory the method needs is allocated and freed inside.
void gemm_run(Method method, hipblasHandle_t handle, const Problem &p);
