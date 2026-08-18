#===============
# options
#===============

CUDA_PATH ?= /usr/local/cuda
HIP_PATH ?= /opt/rocm
GEMMUL8_PATH ?= /home/users/sharmaas/GEMMul8
BACKEND ?= auto
GPU_ARCH ?= auto


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


#===============
# CUDA setup
#===============

ifeq ($(BACKEND),cuda)

ifeq ($(GPU_ARCH),auto)
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
endif

export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

COMPILER := nvcc
LIBS := -lcublas -lcublasLt -lcudart -lcuda -lnvidia-ml -ldl $(GEMMUL8_PATH)/lib/libgemmul8.a
FLAGS := -std=c++20 -O3 
FLAGS += -x cu
FLAGS += -I$(GEMMUL8_PATH)/include
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

endif


#===============
# HIP setup
#===============

ifeq ($(BACKEND),hip)

ifeq ($(GPU_ARCH),auto)
GPU_ARCH := $(shell amd-smi static --asic --csv | grep -o 'gfx[0-9]\+' | head -n 1)
endif

export PATH := $(HIP_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(HIP_PATH)/lib:$(LD_LIBRARY_PATH)

COMPILER := hipcc
LIBS := -lamd_smi -lamdhip64 -lhipblas -lhipblaslt -ldl -Wl,$(GEMMUL8_PATH)/lib/libgemmul8.a
FLAGS := -std=c++20 -O3
FLAGS += -ffp-contract=off
FLAGS += -Wno-unused-result -Wno-unused-command-line-argument -Wno-unused-value
FLAGS += -DOCML_BASIC_ROUNDED_OPERATIONS
FLAGS += -I$(GEMMUL8_PATH)/include
ARCH := --offload-arch=$(GPU_ARCH)

endif


#===============
# Compile
#===============

TARGET1 := dgemm_int8

all: INFO VERSION $(TARGET1)

INFO:
	$(info BACKEND      : $(BACKEND))
ifeq ($(BACKEND),cuda)
	$(info CUDA_PATH    : $(CUDA_PATH))
endif
ifeq ($(BACKEND),hip)
	$(info HIP_PATH     : $(HIP_PATH))
endif
	$(info GEMMUL8_PATH : $(GEMMUL8_PATH))
	$(info GPU_ARCH     : $(GPU_ARCH))
	$(info COMPILER     : $(COMPILER))

$(TARGET1): $(TARGET1).cpp
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

VERSION:
	$(COMPILER) --version

clean:
	rm -f *.o
	rm -f $(TARGET1)
