CXX = g++
CXXFLAGS = -Wall -g -std=c++11
SKETCH_DIR = ../src/sequential/sketches
SKETCH_OBJ = $(shell find $(SKETCH_DIR) -name "*.o")

test_obj: count_sketch_test.cpp
	$(CXX) $(CXXFLAGS) -I $(SKETCH_DIR) -c count_sketch_test.cpp -o count_sketch_test.o

test: count_sketch_test.o
	$(CXX) -o test count_sketch_test.o $(SKETCH_OBJ)

