/*
 * Leverage Score Sampling Sketch
 *
 * Naive sequential implementation of leverage score sampling.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include "sketch.hpp"
#include <math.h>
#include <random>
#include <iostream>

namespace sketch {

namespace seq {

template <typename I, typename T>
leverage_score_sketch<I, T>::leverage_score_sketch(size_t p, size_t n) {
    this->S = new Eigen::SparseMatrix<double>(p, n);
    std::random_device rd;
    this->seed = rd();
    this->_p = p;
    this->_n = n;
}

template <typename I, typename T>
void leverage_score_sketch<I, T>::sketch(I *A, T *SA) {
    // find a 1+1/36 subspace embedding
    size_t n = this->_n, d = A->cols();
    size_t p_count = sketch::seq::count_sketch<I, T>::eps_approx_rows(n, d, 1.0/36);
    sketch::seq::count_sketch<I, T> S_count(p_count, n);
    T SA_count(p_count, d);
    S_count.sketch(A, &SA_count);
    // find a QR decomposition of SA
    Eigen::ColPivHouseholderQR<T> qr(A->rows(), A->cols());
    qr.compute(SA_count);
    T R = qr.matrixQR().triangularView<Upper>();
    T Q = qr.matrixQ();
    std::cout << Q*R << std::endl;
    std::cout << std::endl;
    std::cout << SA_count << std::endl;
    //(*SA) = (*S) * (*A);
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
        double c = 1 + log(2.0 / delta) / log(d * 1.0);
        size_t k = (int) ceil(c * 4.0 / 3.0 * d * log(d) / (eps * eps));
        return std::min(n, k);
    }
}

template class leverage_score_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}

}
