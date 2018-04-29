/*
 * Gaussian Sketch
 *
 * Naive parallel implementation of Gaussian sketch with CUDA. This
 * implementation is mostly just for making use of the GPU and doesn't attempt
 * to do any tricks. Gaussian matrix is sketched on host and only matrix
 * multiplication is done in parallel.
 */

#include "sketch_cuda.h"
#include "util.hpp"
#include <math.h>
#include <random>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cublas_v2.h>

/* compute C(p, d) = A(p, n) * B(n, d) on device, inputs in row major */
void gpu_cublas_mmul(const double *A, const double *B, double *C,
        size_t p, size_t n, size_t d) {
    int lda = d, ldb = n, ldc = d;
    const double alpha = 1;
    const double beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, p, n,
            &alpha, B, lda, A, ldb, &beta, C, ldc);
    cublasDestroy(handle);
}

namespace sketch {

namespace par {

template <typename I, typename T>
gaussian_sketch<I, T>::gaussian_sketch(size_t p, size_t n, unsigned int s) {
    /* allocate sketch on host */
    size_t S_size = sizeof(double) * p * n;
    this->S = (double*)malloc(S_size);
    this->_p = p;
    /* sample gaussians */
    this->seed = s;
    std::mt19937 mt(seed);
    double stddev = sqrt(1.0 / p);
    std::normal_distribution<> g(0, stddev);
    for (unsigned int i = 0; i < p; i++)
        for (unsigned int j = 0; j < n; j++)
            S[i*n + j] = g(mt);
    /* allocate sketch on device and send */
    cudaMalloc(&this->S_device, S_size);
    cudaMemcpy(this->S_device, S, S_size, cudaMemcpyHostToDevice);
}

template <typename I, typename T>
gaussian_sketch<I, T>::gaussian_sketch(size_t p, size_t n) : gaussian_sketch<I, T>::gaussian_sketch(p, n, random_seed()) {}

template <typename I, typename T>
void gaussian_sketch<I, T>::sketch(I *A_device, T *SA_device, size_t n, size_t d) {
    gpu_cublas_mmul(this->S_device, *A_device, *SA_device, this->_p, n, d);
}

/*
 * Note that the success probability is 1 - 1/n^2, but we require n to be at
 * least 10 before we sketched, so the success probability is at least 99/100.
 */
template <typename I, typename T>
size_t gaussian_sketch<I, T>::eps_approx_rows(size_t n, size_t d, double eps) {
    if (n < min_n) {
        std::string info;
        info = std::string("Too few rows. Expected at least ");
        info += std::to_string(min_n) + std::string(" but got ");
        info += std::to_string(n) + std::string(".");
        throw bad_dimension(info);
    } else {
        return (int) ceil(4 / (pow(eps, 2) / 2 - pow(eps, 3) / 3) * log(n));
    }
}

template class gaussian_sketch<double*, double* >;

}

}
