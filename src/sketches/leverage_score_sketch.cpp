/*
 * Leverage Score Sampling Sketch
 *
 * Naive sequential implementation of leverage score sampling.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "sketch.hpp"
#include <random>

namespace sketch{

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

template class leverage_score_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}
