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
#define MAX_ROWS 500

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
    sketch::par::sketch_interface<double*, double*> *S;

    if (sketch_type.compare("count_sketch") == 0) {
        p = sketch::par::count_sketch<double*, double*>::eps_approx_rows(n, d, eps);
        if (p > MAX_ROWS) {
            p = MAX_ROWS;
            std::cout << "\tSketch size capped to " << p << std::endl;
        }
        S = new sketch::par::count_sketch<double*, double*>(p, n, SEED);
    } else if (sketch_type.compare("gaussian_sketch") == 0) {
        p = sketch::par::gaussian_sketch<double*, double*>::eps_approx_rows(n, d, eps);
        if (p > MAX_ROWS) {
            p = MAX_ROWS;
            std::cout << "\tSketch size capped to " << p << std::endl;
        }
        S = new sketch::par::gaussian_sketch<double*, double*>(p, n, SEED);
    /*} else if (sketch_type.compare("leverage_score_sketch") == 0) {
        p = sketch::par::leverage_score_sketch<float*, float*>::eps_approx_rows(n, d, eps);
        if (p > MAX_ROWS) {
            p = MAX_ROWS;
            std::cout << "\tSketch size capped to " << p << std::endl;
        }
        S = new sketch::par::leverage_score_sketch<float*, float*>(p, n, SEED);*/
    } else {
        std::cerr << "Invalid sketch type." << std::endl;
        exit(1);
    }

    size_t A_size = sizeof(double) * n * d;
    size_t SA_size = sizeof(double) * p * d;
    double* A = (double*) malloc(A_size);
    double* SA = (double*) malloc(SA_size);
    double* device_result;
    double* device_input;
    cudaMalloc((void **)&device_result, SA_size);
    cudaMalloc((void **)&device_input, A_size);
    /* initialize array */
    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < d; j++)
            A[i*d + j] = temp->at(i)[j];

    /* copy to device */
    cudaMemcpy(device_input, A, A_size, cudaMemcpyHostToDevice);

    S->sketch(&device_input, &device_result, n, d);

    /* copy from device */
    cudaMemcpy(SA, device_result, SA_size, cudaMemcpyDeviceToHost);

    std::ofstream outfile;
    outfile.open(res_dir.c_str());
    for (unsigned int i = 0; i < p; i++) {
        for (unsigned int j = 0; j < d-1; j++)
            outfile << SA[i*d + j] << ",";
        outfile << SA[i*d + d-1] << std::endl;
    }
    outfile << std::endl;
    outfile.close();

    return 0;
}
