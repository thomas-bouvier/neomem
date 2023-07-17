# CMake 3.19 needed to compile CUDA with standard 17
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(Neomem VERSION "0.0.1")
set(CMAKE_CXX_STANDARD 17)

option(WITHOUT_CUDA "Disable CUDA support" OFF)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# Find pybind11Config.cmake
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE CUSTOM_PYTHON_PYBIND11_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
set(pybind11_DIR ${CUSTOM_PYTHON_PYBIND11_PATH})

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})

# Pytorch
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

# https://github.com/pytorch/pytorch/issues/38122
set(TORCH_PYTHON_LIBRARIES "${TORCH_PATH}/lib/libtorch_python.so")

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(Thallium REQUIRED)

find_package(Nvtx REQUIRED)

if (NOT WITHOUT_CUDA)
  find_package(CUDA REQUIRED)
else ()
  add_definitions(-DWITHOUT_CUDA)
endif ()

execute_process(COMMAND python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
                OUTPUT_VARIABLE WITH_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(
  -O3
  -g
  -Wall
  -fPIC
  -Wl,--no-as-needed
  -D__ASSERT
  -D__DEBUG
)

if (NOT WITHOUT_CUDA)
  add_subdirectory(third_party/cuda-api-wrappers)
endif ()
add_subdirectory(src)
