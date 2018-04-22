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
    src/sketches/oblivious_sketch.cpp
    src/sketches/adaptive_sketch.cpp
    src/sketches/gaussian_sketch.cpp
    src/sketches/count_sketch.cpp
    src/sketches/leverage_score_sketch.cpp
)
set(CUDA_SOURCES
    #src/sketches/count_sketch.cu
)

# copy test cases to binary directory
file(COPY test/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# compile test code
add_executable(count_sketch_test test/count_sketch_test ${CXX_SOURCES})
add_executable(gaussian_sketch_test test/gaussian_sketch_test ${CXX_SOURCES})

# cuda_add_executable()
