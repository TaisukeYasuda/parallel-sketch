#include "sketch_cuda.h"
#include "util.hpp"
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

namespace sketch {

namespace par {

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n, unsigned int s) {
    _n = n;
    cudaMalloc((void**)&this->S, n * sizeof(int));

    /*
    seed = s;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> rand_row(0, p-1);
    std::uniform_int_distribution<int> rand_sign(0, 1);
    int *temp = new int[n];

    for (unsigned int j = 0; j < n; j++) {
        unsigned int i = rand_row(mt); temp[j] = rand_sign(mt) * 2 - 1;
    }

    cudaMemcpy(S, temp, n, cudaMemcpyHostToDevice);

    delete[] temp;
    */
}

//@TODO cudaFree in destructor

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n) : count_sketch<I, T>::count_sketch(p, n, random_seed()) {}

// Assume in_matrix is on device
template<typename I, typename T>
__global__ void sketch_kernel(I *in_matrix, T *out_matrix, int *cols, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x,
        col = blockIdx.y * blockDim.y + threadIdx.y;

    /*
    __shared__ int shared_cols[n];
    __shared__ int signs[n];

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        int col_start = blockIdx.y * blockDim.y;
        for(int i = 0; i < n; i++)
            shared_cols[i] = fabs(cols[i]) - 1;

            if(cols[i] < 0)
                signs[i] = -1;
            else
                signs[i] = 1;
    }

    __syncthreads();

    double res = 0;

    for(int i = 0; i < n; i++) {
        if(shared_cols[i] == row) { //If 1 is in current row
            res += in_matrix(row, i) * signs[i];
        }
    }

    out_matrix(row, col) = res;
    */
}

template <typename I, typename T>
void count_sketch<I, T>::sketch(I *A, T *SA, size_t n, size_t d) {
    /*
    int rows = n, cols = d ;//TODO get columns of A matrix
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (rows + blockDim.x - 1) / blockDim.x,
        (cols + blockDim.y - 1) / blockDim.y);

    sketch_kernel<<<gridDim, blockDim>>>(A, SA, S, n);
    */

}

template <typename I, typename T>
size_t count_sketch<I, T>::eps_approx_rows(size_t n, size_t d, double eps) {
    double delta = 0.01; // failure rate of 1/100
    size_t k = 6 * d*d / (delta * eps*eps);
    return std::min(n, k);
}

template class count_sketch<int*, int* >;

}

}
