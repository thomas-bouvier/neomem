MAIN=rehearsal

SPACK_VIEW?=/mnt/view
TORCH_ROOT?=$(SPACK_VIEW)/lib/python3.10/site-packages/torch

PYTHON_INCLUDE=$(shell python -m pybind11 --includes)
TORCH_INCLUDE=-I$(TORCH_ROOT)/include -I$(TORCH_ROOT)/include/torch/csrc/api/include
MPI_INCLUDE=$(shell pkg-config --cflags-only-I ompi)
THALLIUM_INCLUDE=$(shell pkg-config --cflags-only-I thallium)

PYTHON_LIBS?=$(shell python3-config --ldflags --embed)
TORCH_LIBS?=-L$(TORCH_ROOT)/lib
MPI_LIBS?=$(shell pkg-config --libs ompi)
THALLIUM_LIBS?=$(shell pkg-config --libs thallium)

ifneq ($(WITHOUT_CUDA), 1)
	CUDA_INCLUDE=-I$(SPACK_VIEW)/include
	CUDA_LIBS?=-L$(SPACK_VIEW)/lib64 -lc10_cuda -ltorch_cuda -lcudart
else
	OPTS = -DWITHOUT_CUDA
endif

INCLUDES=$(PYTHON_INCLUDE) $(TORCH_INCLUDE) $(MPI_INCLUDE) $(THALLIUM_INCLUDE) $(CUDA_INCLUDE) $(shell python3-config --cflags --embed)
LIBS=$(PYTHON_LIBS) $(TORCH_LIBS) $(MPI_LIBS) $(THALLIUM_LIBS) $(CUDA_LIBS) -lc10 -ltorch -ltorch_cpu -ltorch_python -lmpi -lnl-3
EXT=$(shell python3-config --extension-suffix)


# https://github.com/chrischoy/MakePytorchPlusPlus
WITH_ABI := $(shell python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
CC=g++
# https://github.com/pytorch/pytorch/issues/36437
FLAGS=-O3 -Wall -std=c++17 -fPIC -Wl,--no-as-needed -D_GLIBCXX_USE_CXX11_ABI=$(WITH_ABI)

all:
	$(CC) $(OPTS) -shared $(FLAGS) $(INCLUDES) rehearsal.cpp stream_loader.cpp distributed_stream_loader.cpp -o $(MAIN)$(EXT) $(LIBS)
test:
	$(CC) $(OPTS) -g $(FLAGS) $(INCLUDES) distributed_stream_loader.cpp main.cpp -o $(MAIN) $(LIBS)
clean:
	rm -rf $(MAIN) $(MAIN)*.so *~