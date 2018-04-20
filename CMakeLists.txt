cmake_minimum_required(VERSION 2.8.12)
project(parallel-sketch)

# C++11
add_compile_options(-std=c++11)

# include headers
include_directories(src/sketches)
# set source files
set(SOURCES src/sketches/count_sketch.cpp test/count_sketch_test.cpp)

# compile test code
add_executable(test ${SOURCES})
