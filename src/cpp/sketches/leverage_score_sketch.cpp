/*
 * Leverage Score Sampling Sketch
 *
 * Naive sequential implementation of leverage score sampling.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "sketch.hpp"
#include <math.h>
#include <random>

namespace sketch {

template <typename I, typename T>
leverage_score_sketch<I, T>::leverage_score_sketch(size_t p, size_t n) {
    S = new Eigen::SparseMatrix<double>(p, n);
    std::random_device rd;
    seed = rd();
}

template <typename I, typename T>
void leverage_score_sketch<I, T>::sketch(I *A, T *SA) {
    // @TODO compute leverage scores and sample
    (*SA) = (*S) * (*A);
}

template <typename I, typename T>
size_t leverage_score_sketch<I, T>::eps_approx_rows(
        double eps, size_t n, size_t d) {
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
