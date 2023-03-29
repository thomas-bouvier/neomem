cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(Neomem VERSION "0.0.1")
set(CMAKE_CXX_STANDARD 17)

option(WITHOUT_CUDA "Disable CUDA support" OFF)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
link_directories(${Python_LIBRARY_DIRS})

find_package(PythonLibs REQUIRED)

find_package(Torch REQUIRED)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(Thallium REQUIRED)

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
  -D_GLIBCXX_USE_CXX11_ABI=${WITH_ABI}
  -D__ASSERT
  -D__DEBUG
)

add_subdirectory(src)
