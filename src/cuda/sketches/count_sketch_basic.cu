/*
 * Count Sketch
 *
 * Naive parallel implementation of the count sketch algorithm. For each
 * column, we select a uniformly random row and assign it a random sign.
 */

#include "sketch_cuda.h"
#include "util.hpp"
#include <random>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cublas_v2.h>

/* compute y = ax + y on device */
void gpu_cublas_daxpy(double *y, double a, double *x, size_t d) {
    const double alpha = a;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDaxpy(handle, d, &alpha, x, 1, y, 1);
    cublasDestroy(handle);
}

namespace sketch {

namespace par {

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n, unsigned int s) {
    /* allocate sketch on host */
    size_t S_size = n * sizeof(int);
    this->S = (int*)malloc(S_size);
    /* sample count sketch */
    this->seed = s;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> rand_row(0, p-1);
    std::uniform_int_distribution<int> rand_sign(0, 1);
    for (unsigned int j = 0; j < n; j++) {
        int i = rand_row(mt);
        int sign = rand_sign(mt) * 2 - 1;
        S[j] = i * sign;
    }
    /* allocate sketch on device and send */
    cudaMalloc(&this->S_device, S_size); // not used in this implementation
    cudaMemcpy(this->S_device, S, S_size, cudaMemcpyHostToDevice);
}

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n) : count_sketch<I, T>::count_sketch(p, n, random_seed()) {}

template <typename I, typename T>
void count_sketch<I, T>::sketch(I *A_device, T *SA_device, size_t n, size_t d) {
    size_t SA_size = n * d * sizeof(double);
    /* zero out SA */
    cudaMemset(*SA_device, 0, SA_size);
    /* hash each row */
    for (unsigned int i = 0; i < n; i++) {
        int signed_bucket = this->S[i];
        int sign = (signed_bucket > 0) - (signed_bucket < 0);
        int bucket = sign * signed_bucket;
        gpu_cublas_daxpy((*SA_device)+bucket*d, (double)sign, (*A_device)+i*d, d);
    }
}

template <typename I, typename T>
size_t count_sketch<I, T>::eps_approx_rows(size_t n, size_t d, double eps) {
    double delta = 0.01; // failure rate of 1/100
    return 6 * d*d / (delta * eps*eps);
}

template class count_sketch<double*, double* >;

}

}
