MAIN=rehearsal

TORCH_ROOT?=/opt/view/lib/python3.10/site-packages/torch

PYTHON_INCLUDE=$(shell python -m pybind11 --includes)
TORCH_INCLUDE=-I$(TORCH_ROOT)/include -I$(TORCH_ROOT)/include/torch/csrc/api/include
MPI_INCLUDE=$(shell pkg-config --cflags-only-I ompi)
THALLIUM_INCLUDE=$(shell pkg-config --cflags-only-I thallium)

PYTHON_LIBS?=$(shell python3-config --ldflags --embed)
TORCH_LIBS?=-L$(TORCH_ROOT)/lib
MPI_LIBS?=$(shell pkg-config --libs ompi)
THALLIUM_LIBS?=$(shell pkg-config --libs thallium)
CUDA_LIBS?=-L/opt/view/lib64

INCLUDES=$(PYTHON_INCLUDE) $(TORCH_INCLUDE) $(MPI_INCLUDE) $(THALLIUM_INCLUDE) $(shell python3-config --cflags --embed)
LIBS=$(PYTHON_LIBS) $(TORCH_LIBS) $(MPI_LIBS) $(THALLIUM_LIBS) $(CUDA_LIBS) -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lmpi -lcudart
EXT=$(shell python3-config --extension-suffix)

CC=g++
FLAGS=-O3 -Wall -std=c++17 -fPIC

all:
	$(CC) -shared $(FLAGS) $(INCLUDES) rehearsal.cpp stream_loader.cpp distributed_stream_loader.cpp -o $(MAIN)$(EXT) $(LIBS)
test:
	$(CC) -g $(FLAGS) $(INCLUDES) distributed_stream_loader.cpp main.cpp -o $(MAIN) $(LIBS)
clean:
	rm -rf $(MAIN) $(MAIN)*.so *~
