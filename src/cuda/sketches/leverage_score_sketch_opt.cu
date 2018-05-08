/*
 * Leverage Score Sampling Sketch
 *
 * Naive parallel implementation of leverage score sampling.
 */

#include "sketch_cuda.h"
#include "util.hpp"
#include <math.h>
#include <random>
#include <vector>
#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define LEVERAGE_BLOCK_SIZE 32

namespace sketch {

namespace par {

template <typename I, typename T>
leverage_score_sketch<I, T>::leverage_score_sketch(size_t p, size_t n, unsigned int s) {
    size_t D_size = sizeof(double) * p;
    size_t Omega_size = sizeof(int) * p;
    this->D = (double*)malloc(D_size); // diagonal rescaling
    this->Omega = (int*)malloc(Omega_size); // sampling
    this->seed = s;
    this->_p = p;
    this->_n = n;
    this->sketched = false;
}

template <typename I, typename T>
leverage_score_sketch<I, T>::leverage_score_sketch(size_t p, size_t n) : leverage_score_sketch<I, T>::leverage_score_sketch(p, n, random_seed()) {}

__global__ void initIdentityGPU(double *R, int d) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (y < d && x < d) {
        if (x == y)
            R[x * d + y] = 1;
        else
            R[x * d + y] = 0;
    }
}

void gpu_cublas_trans(cublasHandle_t handle, double *A, double *At, size_t n, size_t d) {
    const double alpha_t = 1;
    const double beta_t = 0;
    cublasDgeam(
        handle,        // cublas library context
        CUBLAS_OP_T,   // transpose it
        CUBLAS_OP_T,
        n,             // number of rows of A
        d,             // number of columns of A
        &alpha_t,      // scale A
        A,             // matrix to transpose
        d,             // leading dimension of A
        &beta_t,       // scale B by 0, doesn't have to be valid input
        NULL,          // B
        d,             // leading dimension of B
        At,            // result of transposing A
        n              // leading dimension of At
    );
}

template <typename I, typename T>
void leverage_score_sketch<I, T>::sketch(I *A, T *SA, size_t n, size_t d) {
    cublasHandle_t handle;
    cudaError_t cuda_status = cudaSuccess;
    cublasStatus_t cublas_status;
    if (!this->sketched) {
        // find a 1+1/36 subspace embedding SA_count
        size_t p_count = sketch::par::count_sketch<I, T>::eps_approx_rows(n, d, 1.0/36);
        double *SA_count,
               *SA_count_col; // SA_count in column major format
        if (p_count < n) {
            size_t SA_count_size = sizeof(double) * p_count * d;
            cudaMalloc(&SA_count, SA_count_size); // allocate space on device
            sketch::par::count_sketch<I, T> S_count(p_count, n);
            S_count.sketch(A, &SA_count, n, d);
        } else {
            p_count = n;
            size_t SA_count_size = sizeof(double) * p_count * d;
            cudaMalloc(&SA_count, SA_count_size); // allocate space on device
            cudaMemcpy(SA_count, *A, SA_count_size, cudaMemcpyDeviceToDevice);
        }
        cudaMalloc(&SA_count_col, sizeof(double) * p_count * d); // allocate space on device

        // find a QR decomposition of SA_count
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
        cusolverDnHandle_t cusolverH;
        cusolver_status = cusolverDnCreate(&cusolverH); // create handle
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        // set up QR decomposition call
        int lwork; // working space size
        int tau_size = p_count;
        double *d_work; // working space
        double *d_tau; // tau (representation of Q)
        int *dev_info; // info in gpu
        cuda_status = cudaMalloc((void**)&dev_info, sizeof(int));
        assert(cudaSuccess == cuda_status);
        cublasCreate(&handle);
        gpu_cublas_trans(handle, SA_count, SA_count_col, p_count, d);
        cusolver_status = cusolverDnDgeqrf_bufferSize( // query for space
            cusolverH,      // cuSolverDN library context
            p_count,        // number of rows of SA_count
            d,              // number of columns of SA_count
            SA_count_col,   // matrix to decompose
            p_count,        // leading dimension
            &lwork          // working space size
        );
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        cuda_status = cudaMalloc((void**)&d_work, sizeof(double) * lwork); // allocate working space
        assert(cudaSuccess == cuda_status);
        cuda_status = cudaMalloc((void**)&d_tau, sizeof(double) * tau_size); // allocate tau
        assert(cudaSuccess == cuda_status);
        // compute QR decomposition
        cusolver_status = cusolverDnDgeqrf(
            cusolverH,       // cuSolverDN library context
            p_count,         // number of rows of SA_count
            d,               // number of columns of SA_count
            SA_count_col,    // matrix to decompose
            p_count,         // leading dimension
            d_tau,           // tau
            d_work,          // working space
            lwork,           // working space size
            dev_info         // info
        );
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        cuda_status = cudaDeviceSynchronize();
        assert(cudaSuccess == cuda_status);
        if (cusolverH) cusolverDnDestroy(cusolverH); // done with cusolver
        if (d_work) cudaFree(d_work); // done with d_work
        // invert R by solving RX = I where R is triangular
        double *R_inv;
        const double alpha_t = 1;
        const double beta_t = 0;
        cuda_status = cudaMalloc((void**)&R_inv, sizeof(double) * d * d); // allocate R inverse
        assert(cudaSuccess == cuda_status);
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (d + blockDim.x - 1) / blockDim.x,
            (d + blockDim.y - 1) / blockDim.y);
        initIdentityGPU<<<gridDim, blockDim>>>(R_inv, d); // set R_inv to identity
        cublasCreate(&handle); // initialize cublas handle
        cublas_status = cublasDtrsm(
            handle,                     // cublas library handle
            CUBLAS_SIDE_LEFT,           // upper triangular matrix on left side
            CUBLAS_FILL_MODE_UPPER,     // upper triangular
            CUBLAS_OP_N,                // don't transpose
            CUBLAS_DIAG_NON_UNIT,       // diagonals are not 1
            d,                          // number of rows of R_inv
            d,                          // number of columns of R_inv
            &alpha_t,                   // scalar multiple is just 1
            SA_count_col,               // upper triangular matrix to solve
            p_count,                    // leading dimension of R
            R_inv,                      // identity at first, R_inv later
            d                           // leading dimension of R_inv
        );
        // compute RG for a gaussian sketch G
        size_t p_gauss;
        try {
            p_gauss = sketch::par::gaussian_sketch<I, T>::eps_approx_rows(d, d, 0.5);
        } catch (const sketch::par::bad_dimension& e) {
            p_gauss = d;
        }
        double* RG;
        if (p_gauss < d) {
            /* allocate sketch on host */
            size_t G_size = sizeof(double) * d * p_gauss;
            double *G = (double*)malloc(G_size);
            double *G_device;
            /* sample gaussians */
            std::mt19937 mt_gauss(this->seed);
            double stddev = sqrt(1.0 / p_gauss);
            std::normal_distribution<> g(0, stddev);
            for (unsigned int i = 0; i < d; i++)
                for (unsigned int j = 0; j < p_gauss; j++)
                    G[i*n + j] = g(mt_gauss);
            /* allocate sketch on device and send */
            cudaMalloc(&RG, G_size);
            cudaMalloc(&G_device, G_size);
            cudaMemcpy(G_device, G, G_size, cudaMemcpyHostToDevice);
            cublasDgemm(
                handle,        // cublas library context
                CUBLAS_OP_N,   // transposing is handled by switching order
                CUBLAS_OP_T,   // need to transpose R
                p_gauss,       // last outer dimension
                d,             // first outer dimension
                d,             // inner dimension
                &alpha_t,      // scale product
                G_device,      // second factor
                p_gauss,       // leading dimension of G
                R_inv,         // first factor
                d,             // leading dimension of A
                &beta_t,       //
                RG,            // result
                p_gauss        // leading dimension of RG
            );
            cuda_status = cudaDeviceSynchronize();
        } else {
            cudaMalloc(&RG, sizeof(double) * d * d); // allocate space for RG
            gpu_cublas_trans(handle, R_inv, RG, d, d); // just transpose R_inv
        }
        // compute ARG
        double* ARG;
        cudaMalloc(&ARG, sizeof(double) * n * p_gauss);
        cublasDgemm(
            handle,        // cublas library context
            CUBLAS_OP_N,   // transposing is handled by switching order
            CUBLAS_OP_N,   // doesn't matter here
            p_gauss,       // last outer dimension
            n,             // first outer dimension
            d,             // inner dimension
            &alpha_t,      // scale product
            RG,            // second factor
            p_gauss,       // leading dimension of RG_trans
            *A,            // first factor
            d,             // leading dimension of A
            &beta_t,       //
            ARG,           // result
            p_gauss        // leading dimension of ARG
        );
        cuda_status = cudaDeviceSynchronize();
        assert(cudaSuccess == cuda_status);
        // scaled row norms of ARG are estimates of leverage scores
        double beta = 4.0 / 7.0;
        double *q = (double*)malloc(sizeof(double) * n);
        double *cdf = (double*)malloc(sizeof(double) * n);
        double temp;
        double sum = 0.0;
        for (unsigned int i = 0; i < n; i++) {
            double *row =  ARG + i*p_gauss;
            cublas_status = cublasDnrm2(handle, p_gauss, row, 1, &temp);
            assert(CUBLAS_STATUS_SUCCESS == cublas_status);
            q[i] = beta * temp * temp / d;
            sum += q[i];
            cdf[i] = sum;
        }
        cublasDestroy(handle);
        double excess_probability = (1.0 - sum) / n;
        for (unsigned int i = 0; i < n; i++)
            cdf[i] += excess_probability * (i+1);

        // sample, q_i is the above plus the excess probability
        std::mt19937 mt(seed);
        std::uniform_real_distribution<double> unif(0, 1);
        for (unsigned int i = 0; i < this->_p; i++) {
            double score = unif(mt), q_j;
            unsigned int lo = 0, hi = n, mid;
            while (lo < hi) {
                mid = lo + (hi-lo) / 2;
                if (cdf[mid] < score) lo = mid + 1;
                else hi = mid;
            }
            unsigned int j = lo;
            this->Omega[i] = j;
            q_j = q[j] + excess_probability;
            double scale = 1.0 / sqrt(q_j * this->_p);
            this->D[i] = scale;
        }

        // mark as sketched
        this->sketched = true;
    }
    cudaMemset(*SA, 0, sizeof(double) * this->_p * d);
    cublasCreate(&handle);
    for (unsigned int i = 0; i < this->_p; i++) {
        unsigned int j = (this->Omega)[i];
        double scale = (this->D)[i];
        cublas_status = cublasDaxpy(
            handle,       // cublas library context
            d,            // length of vectors
            &scale,       // scaling factor
            (*A)+j*d,     // sampled A row
            1,            // leading dimension of row
            (*SA)+i*d,    // destination
            1             // leading dimension of result row
        );
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    }
    if (handle) cublasDestroy(handle);
}

template <typename I, typename T>
size_t leverage_score_sketch<I, T>::eps_approx_rows(
        size_t n, size_t d, double eps) {
    if (d <= 1) {
        std::string info;
        info = std::string("Too few columns. Expected at least 2 columns.");
        throw bad_dimension(info);
    } else {
        double delta = 0.01;
        double beta = 4.0 / 7.0;
        double c = 1 + log(2.0 / delta) / log(d * 1.0);
        return (int) ceil(c * 4.0 / 3.0 * d / beta * log(d) / (eps * eps));
    }
}

template class leverage_score_sketch<double*, double* >;

}

}
