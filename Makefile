MAIN=rehearsal
SOURCES=rehearsal.cpp stream_loader.cpp
FLAGS=-O3 -Wall -shared -std=c++14 -fPIC
TORCH_ROOT=$(HOME)/.local/lib/python3.10/site-packages/torch
TORCH_INCLUDE=-I$(TORCH_ROOT)/include -I$(TORCH_ROOT)/include/torch/csrc/api/include
INCLUDES=$(shell python3 -m pybind11 --includes) $(TORCH_INCLUDE)
LIBS=-L$(TORCH_ROOT)/lib -ltorch
EXT=$(shell python3-config --extension-suffix)
CC=g++

all:
	$(CC) $(FLAGS) $(INCLUDES) $(SOURCES) -o $(MAIN)$(EXT) $(LIBS)
clean:
	rm -rf $(MAIN)*.so *~