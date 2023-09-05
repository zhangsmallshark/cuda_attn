###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/soft/compilers/cudatoolkit/cuda-11.8.0

CUDNN_ROOT_DIR=/home/czh5/seq/cudnn_attn/cudnn-linux-x86_64-8.9.4.25_cuda11-archive


##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 --default-stream per-thread -Xcompiler -fopenmp -arch=sm_80 -gencode=arch=compute_80,code=sm_80
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include -I/home/czh5/seq/cudnn_attn/cutlass/include -I/home/czh5/seq/cudnn_attn/cutlass/tools/util/include -I/home/czh5/seq/cudnn_attn/cutlass/examples/common
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcublas

# CUDNN library directory:
CUDNN_LIB_DIR = -L$(CUDNN_ROOT_DIR)/lib
# CUDNN include directory:
CUDNN_INC_DIR = -I$(CUDNN_ROOT_DIR)/include
# CUDNN linking libraries:
CUDNN_LINK_LIBS = -lcudnn


##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
# EXE = $(OBJ_DIR)/cublas_gemm
# EXE = $(OBJ_DIR)/cutlass_gemm
EXE = $(OBJ_DIR)/tune_gemm

# Object files:
# OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/cuda_kernel.o
# OBJS = cudnn_attn.cpp
# OBJS = test1.cu
# OBJS = cublas_gemm.cpp
# OBJS = cutlass_gemm.cu
OBJS = tune_gemm.cu

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
# $(EXE) : $(OBJS)
# 	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CUDNN_INC_DIR) $(CUDNN_LIB_DIR) $(CUDNN_LINK_LIBS)

# Compile main .cpp file to object files:
# $(OBJ_DIR)/%.o : %.cpp
# 	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
# 	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)

