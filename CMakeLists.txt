project(parallel-sketch)
cmake_minimum_required(VERSION 2.8.12)
find_package(CUDA QUIET REQUIRED)

# C++11
add_compile_options(-std=c++11)

# dependencies: comment out if these libraries are installed on your system
include_directories(deps/eigen-3.3.4/include/eigen3)

# include headers
include_directories(include)
# set source files
set(CXX_SOURCES
    src/cpp/sketches/oblivious_sketch.cpp
    src/cpp/sketches/adaptive_sketch.cpp
    src/cpp/sketches/gaussian_sketch.cpp
    src/cpp/sketches/count_sketch.cpp
    src/cpp/sketches/leverage_score_sketch.cpp
)
set(CUDA_SOURCES
    # src/cuda/sketches/oblivious_sketch.cu
    # src/cuda/sketches/adaptive_sketch.cu
    # src/cuda/sketches/gaussian_sketch.cu
    src/cuda/sketches/count_sketch.cu
    # src/cuda/sketches/leverage_score_sketch.cu
)
# make libraries
add_library(sketch ${CXX_SOURCES} ${CUDA_SOURCES})

# copy test cases to binary directory
file(COPY test/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# add results directory for test cases
file(MAKE_DIRECTORY build/results build/results/count_sketch)

# compile test code
add_executable(count_sketch_test test/count_sketch_test.cpp)
add_executable(gaussian_sketch_test test/gaussian_sketch_test.cpp)
add_executable(test_sketch test/test_sketch.cpp)
target_link_libraries(count_sketch_test sketch)
target_link_libraries(gaussian_sketch_test sketch)
target_link_libraries(test_sketch sketch)

# cuda_add_executable(count_sketch_test_cuda test/count_sketch_test ${CUDA_SOURCES})
# cuda_add_executable(gaussian_sketch_test_cuda test/gaussian_sketch_test ${CUDA_SOURCES})
