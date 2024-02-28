CC      := /usr/bin/g++
CUDA_PATH ?= "/usr/local/cuda-9.0"
NVCC    := $(CUDA_PATH)/bin/nvcc -ccbin $(CC)

CUDA_LDFLAGS := -lrt -m64 -lcuda -std=c++11
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50

all: createcntx

createcntx: cuda_cntxt.cu
	$(NVCC) ${CUDA_LDFLAGS} cuda_cntxt.cu -o createcntx 
clean: 
	rm -rf createcntx
