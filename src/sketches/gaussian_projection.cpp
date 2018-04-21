/*
 * Gaussian Projection Sketch
 *
 * Naive sequential implementation of Gaussian projection.
 */

#include <Eigen/Dense>
#include "sketch.hpp"
#include <random>

namespace sketch{

template <typename I, typename T>
gaussian_projection<I, T>::gaussian_projection(size_t p, size_t n) {
    S = new Eigen::MatrixXd(p, n);
    std::random_device rd;
    seed = rd();
    // @TODO sample gaussian matrix
}

template <typename I, typename T>
void gaussian_projection<I, T>::sketch(I *A, T *SA) {
    (*SA) = (*S) * (*A);
}

template class gaussian_projection<Eigen::MatrixXd, Eigen::MatrixXd >;

}
