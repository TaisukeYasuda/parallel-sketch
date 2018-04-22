/*
 * Uniform Sampling Sketch
 *
 * Naive sequential implementation of uniform sampling sketch.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "sketch.hpp"
#include <random>

namespace sketch{

template <typename I, typename T>
uniform_sampling_sketch<I, T>::uniform_sampling_sketch(size_t p, size_t n) {
    S = new Eigen::SparseMatrix<double>(p, n);
    std::random_device rd;
    seed = rd();
    // @TODO uniformly sample and scale
}

template <typename I, typename T>
void uniform_sampling_sketch<I, T>::sketch(I *A, T *SA) {
    (*SA) = (*S) * (*A);
}

template class uniform_sampling_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}
