cmake_minimum_required(VERSION 2.8.12)
project(parallel-sketch)

# C++11
add_compile_options(-std=c++11)

# include headers
include_directories(deps/eigen-3.3.4/include/eigen3)
include_directories(src/sketches)
# set source files
set(SOURCES
    src/sketches/oblivious_sketch.cpp
    src/sketches/adaptive_sketch.cpp
    src/sketches/gaussian_projection.cpp
    src/sketches/count_sketch.cpp
    src/sketches/uniform_sampling_sketch.cpp
    src/sketches/leverage_score_sketch.cpp
    test/count_sketch_test.cpp
)

# copy test cases to binary directory
file(COPY test/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# compile test code
add_executable(sketch_test ${SOURCES})
