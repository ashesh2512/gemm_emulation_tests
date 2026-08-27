#===============
# options
#===============

BACKEND ?= auto

# CUDA_PATH, HIP_PATH, GPU_ARCH, GEMMUL8_PATH and OZABLAS_PATH have no default;
# set them on the command line or in the environment. GPU_ARCH=auto opts in to
# querying the device for the architecture.

# Set to 1 to build the corresponding method into the binary.
HAVE_GEMMUL8 ?= 0
HAVE_OZABLAS ?= 0
HAVE_CUBLAS_OZAKI1 ?= 0


# `clean` just deletes files, so skip toolchain detection and its hard errors.
ifneq ($(MAKECMDGOALS),clean)

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
FLAGS := -ccbin g++-14 -std=c++20 -O3 
FLAGS += -x cu
# nvcc does not take -fopenmp itself; it has to reach the host compiler.
FLAGS += -Xcompiler -fopenmp
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

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
LIBS := -lamd_smi -lamdhip64 -lhipblas -lhipblaslt -lhiprand -ldl
FLAGS := -std=c++20 -O3
FLAGS += -ffp-contract=off
FLAGS += -fopenmp
FLAGS += -Wno-unused-result -Wno-unused-command-line-argument -Wno-unused-value
FLAGS += -DOCML_BASIC_ROUNDED_OPERATIONS
ARCH := --offload-arch=$(GPU_ARCH)

endif


#===============
# Optional GEMM libraries
#===============

ifeq ($(HAVE_GEMMUL8),1)
ifeq ($(origin GEMMUL8_PATH),undefined)
$(error HAVE_GEMMUL8=1 requires GEMMUL8_PATH=<path to GEMMul8>)
endif
FLAGS += -DHAVE_GEMMUL8 -I$(GEMMUL8_PATH)/include
# Passed via -Wl, so hipcc does not mistake the archive for a HIP source file.
LIBS += -Wl,$(GEMMUL8_PATH)/lib/libgemmul8.a
endif

ifeq ($(HAVE_OZABLAS),1)
ifeq ($(origin OZABLAS_PATH),undefined)
$(error HAVE_OZABLAS=1 requires OZABLAS_PATH=<path to ozablas>)
endif
FLAGS += -DHAVE_OZABLAS -I$(OZABLAS_PATH)/include
LIBS += -L$(OZABLAS_PATH)/build/src -lozablas -Wl,-rpath,$(OZABLAS_PATH)/build/src
endif

ifeq ($(HAVE_CUBLAS_OZAKI1),1)
FLAGS += -DHAVE_CUBLAS_OZAKI1
endif

endif # MAKECMDGOALS != clean


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

$(TARGET): $(SRCS) gemm_methods.hpp
	$(COMPILER) $(SRCS) $(FLAGS) $(ARCH) -o $@ $(LIBS)

VERSION:
	$(COMPILER) --version

clean:
	rm -f *.o $(TARGET)
