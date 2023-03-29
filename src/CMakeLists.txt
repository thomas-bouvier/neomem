# Set sources and executable
set(LIBRARY_SOURCES
    engine_loader.cpp
    distributed_stream_loader.cpp
    rehearsal.cpp
)
add_library(neomem SHARED ${LIBRARY_SOURCES})
target_include_directories(neomem PUBLIC ${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(neomem PUBLIC ${Python_LIBRARIES} torch ${MPI_CXX_LIBRARIES} thallium)


set(EXAMPLE_SOURCES
    engine_loader.cpp
    distributed_stream_loader.cpp
    main.cpp
)
add_executable(neomem_example ${EXAMPLE_SOURCES})
target_include_directories(neomem_example PUBLIC ${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(neomem_example PUBLIC ${Python_LIBRARIES} torch ${MPI_CXX_LIBRARIES} thallium)
