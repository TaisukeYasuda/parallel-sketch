#include "sketch_cuda.h"
#include "util.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define INPUT 1
#define OUTPUT 2
#define SKETCH_TYPE 3

#define SEED 116
#define MAX_ROWS 10000

int main(int argc, char *argv[]) {
    std::cout << "Testing CUDA " << argv[SKETCH_TYPE] << std::endl;

    std::string test_dir = std::string(argv[INPUT]);
    std::string res_dir = std::string(argv[OUTPUT]);
    std::string sketch_type = std::string(argv[SKETCH_TYPE]);

    std::vector< std::vector<double> > *temp;
    temp = read_matrix(test_dir);

    size_t n = temp->size();
    size_t d = temp->at(0).size();
    size_t p;
    double eps = 0.5;
    sketch::par::sketch_interface<int*, int*> *S;

    p = 100; // @TODO don't hardcode

    int* A = (int*) malloc(sizeof(int) * n * d);
    int* device_result;
    int* device_input;
    size_t SA_size = sizeof(int) * p * d;
    size_t A_size = sizeof(int) * n * d;
    cudaMalloc((void **)&device_result, SA_size);
    cudaMalloc((void **)&device_input, A_size);
    /* initialize array */
    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < d; j++)
            A[i * n + j] = temp->at(i)[j];
    /* copy to device */
    cudaMemcpy(device_input, A, A_size, cudaMemcpyHostToDevice);

    S = new sketch::par::count_sketch<int*, int*>(p, n, SEED);
    S->sketch(&device_input, &device_result, n, d);

    /*
    if (sketch_type.compare("count_sketch") == 0) {
        p = sketch::par::count_sketch<M, M>::eps_approx_rows(n, d, eps);
        S = new sketch::par::count_sketch<M, M>(p, n, SEED);
    } else if (sketch_type.compare("gaussian_sketch") == 0) {
        p = sketch::par::gaussian_sketch<M, M>::eps_approx_rows(n, d, eps);
        S = new sketch::par::gaussian_sketch<M, M>(p, n, SEED);
    } else if (sketch_type.compare("leverage_score_sketch") == 0) {
        p = sketch::par::leverage_score_sketch<M, M>::eps_approx_rows(n, d, eps);
        S = new sketch::par::leverage_score_sketch<M, M>(p, n, SEED);
    } else {
        std::cerr << "Invalid sketch type." << std::endl;
        exit(1);
    }

    if (p > MAX_ROWS) {
        std::cerr << "Too many rows in the sketch." << std::endl;
        exit(1);
    }

    M A(n, d);
    M SA(p, d);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < d; j++) {
            A(i, j) = temp->at(i)[j];
        }
    }

    std::cout << "\tCreated sketch of size " << p << std::endl;
    S->sketch(&A, &SA);

    std::ofstream outfile;
    outfile.open(res_dir.c_str());
    outfile << SA << std::endl;
    outfile.close();
    */

    return 0;
}
