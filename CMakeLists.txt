cmake_minimum_required(VERSION 2.8.12)
project(parallel-sketch)
find_package(CUDA QUIET REQUIRED)
find_package(OpenMP QUIET REQUIRED)

# C++11
add_compile_options(-std=c++11)
# include headers
include_directories(include)

################################################################################
#                                   CUDA                                       #
################################################################################
# CUDA compilation
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)
list(APPEND CUDA_NVCC_FLAGS "-O3 --gpu-architecture compute_61 -std=c++11")
list(APPEND CUDA_NVCC_FLAGS -lcublas_device)
set(CMAKE_CUDA_FLAGS "-Wall -fopenmp")

# set cuda source files
set(CUDA_SOURCES
    src/cuda/sketches/oblivious_sketch.cu
    src/cuda/sketches/adaptive_sketch.cu
    src/cuda/sketches/count_sketch_basic.cu
    src/cuda/sketches/gaussian_sketch_basic.cu
    src/cuda/sketches/leverage_score_sketch_opt.cu
    src/util.cpp
)

# make libraries
cuda_add_library(sketch_cuda ${CUDA_SOURCES})

# compile test cases
cuda_add_executable(test_sketch_cuda test/test_sketch_cuda.cu)
target_link_libraries(test_sketch_cuda sketch_cuda)
cuda_add_cublas_to_target(test_sketch_cuda)
find_cuda_helper_libs(cusolver)
target_link_libraries(test_sketch_cuda ${CUDA_cusolver_LIBRARY})
target_link_libraries(test_sketch_cuda gomp)

################################################################################
#                                    C++                                       #
################################################################################

# dependencies: comment out if these libraries are installed on your system
include_directories(deps/eigen-3.3.4/include/eigen3)

set(CMAKE_CXX_FLAGS "-Wall -fopenmp")

#set c++ source files
set(CXX_SOURCES
    src/cpp/sketches/oblivious_sketch.cpp
    src/cpp/sketches/adaptive_sketch.cpp
    src/cpp/sketches/gaussian_sketch.cpp
    src/cpp/sketches/count_sketch.cpp
    src/cpp/sketches/leverage_score_sketch.cpp
    src/cpp/sketches/count_min_sketch.cpp
    src/util.cpp
)
# make libraries
add_library(sketch ${CXX_SOURCES})

# compile test code
add_executable(test_sketch test/test_sketch.cpp)
target_link_libraries(test_sketch sketch)

################################################################################
#                                   Testing                                    #
################################################################################

# compile subspace embedding
add_executable(test_subspace_embedding
    test/test_subspace_embedding.cpp
    src/util.cpp
)

# compile mad sketch
add_executable(run_mad_sketch_omp
    src/cpp/applications/run_MAD_sketch.cpp
    src/cpp/applications/MAD_sketch_omp.cpp)
target_link_libraries(run_mad_sketch_omp sketch)

add_executable(run_mad_sketch
    src/cpp/applications/run_MAD_sketch.cpp
    src/cpp/applications/MAD_sketch.cpp)
target_link_libraries(run_mad_sketch sketch)

# copy test cases to binary directory
file(COPY test/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY
    test/regress.py
    test/gen_reference.py
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
