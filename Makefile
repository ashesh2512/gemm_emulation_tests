#===============
# options
#===============

BACKEND ?= auto

# CUDA_PATH, HIP_PATH and GPU_ARCH have no default; set them on the command line
# or in the environment. GPU_ARCH=auto opts in to querying the device for the
# architecture.

# Set to 1 to build the corresponding method into the binary.
HAVE_GEMMUL8 ?= 0
HAVE_OZABLAS ?= 0
HAVE_CUBLAS_OZAKI1 ?= 0

# Restrict GEMMul8 explicit instantiations to INT8.
INT8_ONLY ?= 1

# Host C++ compiler; must support C++20 (<bit>, <numbers>, ...).
HOST_CXX ?= g++-14

# GEMMul8 and ozablas are git submodules; `override` keeps them pinned to the checked-in copies.
override GEMMUL8_PATH := $(CURDIR)/external/GEMMul8
override OZABLAS_PATH := $(CURDIR)/external/ozablas
GEMMUL8_LIB := $(GEMMUL8_PATH)/lib/libgemmul8.a
OZABLAS_LIB := $(OZABLAS_PATH)/build/src/libozablas.so


# The clean targets just delete files, so skip toolchain detection and its hard errors.
ifeq ($(filter clean distclean,$(MAKECMDGOALS)),)

#===============
# Auto-detect backend (CUDA or HIP)
#===============

ifeq ($(BACKEND),auto)
ifneq ($(shell command -v nvidia-smi 2>/dev/null),)
BACKEND := cuda
else ifneq ($(shell command -v rocminfo 2>/dev/null),)
BACKEND := hip
else
$(error Neither NVIDIA (CUDA) nor AMD (ROCm) GPU environment detected!)
endif
endif

ifeq ($(origin GPU_ARCH),undefined)
$(error GPU_ARCH=<arch> is required (e.g. GPU_ARCH=90, GPU_ARCH=gfx942, or GPU_ARCH=auto to detect))
endif


#===============
# CUDA setup
#===============

ifeq ($(BACKEND),cuda)

ifeq ($(origin CUDA_PATH),undefined)
$(error BACKEND=cuda requires CUDA_PATH=<path to CUDA toolkit>)
endif

ifeq ($(GPU_ARCH),auto)
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
endif

export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

COMPILER := nvcc
LIBS := -lcublas -lcublasLt -lcurand -lcudart -lcuda -lnvidia-ml -ldl -lgomp
FLAGS := -ccbin $(HOST_CXX) -std=c++20 -O3 
FLAGS += -x cu
# nvcc does not take -fopenmp itself; it has to reach the host compiler.
FLAGS += -Xcompiler -fopenmp
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
SUBMAKE_TOOLCHAIN := CUDA_PATH=$(CUDA_PATH)
# GEMMul8's own makefiles never pass -ccbin; NVCC_PREPEND_FLAGS injects it without patching the submodule.
SUBMAKE_ENV := NVCC_PREPEND_FLAGS="-ccbin $(HOST_CXX)"
OZABLAS_CMAKE_ARGS := -DCMAKE_BUILD_TYPE=Release -DOZABLAS_ENABLE_CUDA=ON
OZABLAS_CMAKE_ARGS += -DCMAKE_CUDA_ARCHITECTURES=$(GPU_ARCH)
OZABLAS_CMAKE_ARGS += -DOZABLAS_BUILD_EXAMPLES=OFF
# nvcc has to be told which host compiler to use for both plain C++ and the CUDA host pass.
OZABLAS_CMAKE_ARGS += -DCMAKE_CXX_COMPILER=$(HOST_CXX)
OZABLAS_CMAKE_ARGS += -DCMAKE_CUDA_HOST_COMPILER=$(HOST_CXX)

endif


#===============
# HIP setup
#===============

ifeq ($(BACKEND),hip)

ifeq ($(origin HIP_PATH),undefined)
$(error BACKEND=hip requires HIP_PATH=<path to ROCm>)
endif

ifeq ($(GPU_ARCH),auto)
GPU_ARCH := $(shell amd-smi static --asic --csv 2>/dev/null | grep -o 'gfx[0-9]\+' | head -n 1)
endif

export PATH := $(HIP_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(HIP_PATH)/lib:$(LD_LIBRARY_PATH)

COMPILER := hipcc
LIBS := -lamd_smi -lrocm_smi64 -lamdhip64 -lhipblas -lhipblaslt -lhiprand -ldl
FLAGS := -std=c++20 -O3
FLAGS += -ffp-contract=off
FLAGS += -fopenmp
FLAGS += -Wno-unused-result -Wno-unused-command-line-argument -Wno-unused-value
FLAGS += -DOCML_BASIC_ROUNDED_OPERATIONS
ARCH := --offload-arch=$(GPU_ARCH)
SUBMAKE_TOOLCHAIN := HIP_PATH=$(HIP_PATH)
SUBMAKE_ENV :=
OZABLAS_CMAKE_ARGS := -DCMAKE_BUILD_TYPE=Release -DOZABLAS_ENABLE_HIP=ON
OZABLAS_CMAKE_ARGS += -DCMAKE_HIP_ARCHITECTURES=$(GPU_ARCH)
OZABLAS_CMAKE_ARGS += -DOZABLAS_BUILD_EXAMPLES=OFF
OZABLAS_CMAKE_ARGS += -DCMAKE_CXX_COMPILER=hipcc

endif


#===============
# Optional GEMM libraries
#===============

ifeq ($(HAVE_GEMMUL8),1)
FLAGS += -DHAVE_GEMMUL8 -I$(GEMMUL8_PATH)/include
# Passed via -Xlinker so the archive is not mistaken for a source file by -x cu
# (nvcc) or HIP (hipcc); nvcc rejects the -Wl, form outright.
LIBS += -Xlinker $(GEMMUL8_LIB)
DEPS += $(GEMMUL8_LIB)
endif

ifeq ($(HAVE_OZABLAS),1)
FLAGS += -DHAVE_OZABLAS -I$(OZABLAS_PATH)/include
LIBS += -L$(OZABLAS_PATH)/build/src -lozablas -Xlinker -rpath=$(OZABLAS_PATH)/build/src
DEPS += $(OZABLAS_LIB)
endif

ifeq ($(HAVE_CUBLAS_OZAKI1),1)
FLAGS += -DHAVE_CUBLAS_OZAKI1
endif

endif # no clean target requested


#===============
# Compile
#===============

TARGET := gemm_test
SRCS := main.cpp gemm_methods.cpp

all: INFO VERSION $(TARGET)

INFO:
	$(info BACKEND      : $(BACKEND))
ifeq ($(BACKEND),cuda)
	$(info CUDA_PATH    : $(CUDA_PATH))
endif
ifeq ($(BACKEND),hip)
	$(info HIP_PATH     : $(HIP_PATH))
endif
	$(info GPU_ARCH     : $(GPU_ARCH))
	$(info COMPILER     : $(COMPILER))
	$(info HAVE_GEMMUL8 : $(HAVE_GEMMUL8))
	$(info HAVE_OZABLAS : $(HAVE_OZABLAS))

$(TARGET): $(SRCS) gemm_methods.hpp $(DEPS)
	$(COMPILER) $(SRCS) $(FLAGS) $(ARCH) -o $@ $(LIBS)

$(GEMMUL8_LIB):
	$(SUBMAKE_ENV) $(MAKE) -C $(GEMMUL8_PATH) BACKEND=$(BACKEND) GPU_ARCH=$(GPU_ARCH) \
	  INT8_ONLY=$(INT8_ONLY) $(SUBMAKE_TOOLCHAIN)

$(OZABLAS_LIB):
	cmake -S $(OZABLAS_PATH) -B $(OZABLAS_PATH)/build $(OZABLAS_CMAKE_ARGS)
	cmake --build $(OZABLAS_PATH)/build -j

VERSION:
	$(COMPILER) --version

clean:
	rm -f *.o $(TARGET)

distclean: clean
	rm -rf $(GEMMUL8_PATH)/build $(GEMMUL8_PATH)/lib $(GEMMUL8_PATH)/compile_info
	rm -rf $(OZABLAS_PATH)/build
