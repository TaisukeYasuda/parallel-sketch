cmake_minimum_required(VERSION 2.8.12)
project(parallel-sketch)
find_package(CUDA QUIET REQUIRED)

# C++11
add_compile_options(-std=c++11)

################################################################################
#                                   CUDA                                       #
################################################################################
# CUDA compilation
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)
list(APPEND CUDA_NVCC_FLAGS "-O3 --gpu-architecture compute_61 -std=c++11")

# include headers
include_directories(include)
# set source files
set(CUDA_SOURCES
    src/cuda/sketches/oblivious_sketch.cu
    src/cuda/sketches/count_sketch.cu
    src/cuda/sketches/util.cpp
)

# make libraries
cuda_add_library(sketch_cuda ${CUDA_SOURCES})

# compile test cases
cuda_add_executable(test_sketch_cuda test/test_sketch_cuda.cu)
target_link_libraries(test_sketch_cuda sketch_cuda)

################################################################################
#                                    C++                                       #
################################################################################

# dependencies: comment out if these libraries are installed on your system
include_directories(deps/eigen-3.3.4/include/eigen3)

#set c++ source files
set(CXX_SOURCES
    src/cpp/sketches/oblivious_sketch.cpp
    src/cpp/sketches/adaptive_sketch.cpp
    src/cpp/sketches/gaussian_sketch.cpp
    src/cpp/sketches/count_sketch.cpp
    src/cpp/sketches/leverage_score_sketch.cpp
    src/cpp/sketches/util.cpp
)
# make libraries
add_library(sketch ${CXX_SOURCES})

# copy test cases to binary directory
file(COPY test/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# add results directory for test cases
file(MAKE_DIRECTORY build/results build/results/count_sketch)

# compile test code
add_executable(test_sketch test/test_sketch.cpp)
target_link_libraries(test_sketch sketch)

# cuda_add_executable(count_sketch_test_cuda test/count_sketch_test ${CUDA_SOURCES})
# cuda_add_executable(gaussian_sketch_test_cuda test/gaussian_sketch_test ${CUDA_SOURCES})

file(COPY test/regress.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
