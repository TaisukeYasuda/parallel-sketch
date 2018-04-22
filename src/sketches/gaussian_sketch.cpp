/*
 * Gaussian Sketch
 *
 * Naive sequential implementation of Gaussian projection.
 */

#include <Eigen/Dense>
#include "sketch.hpp"
#include <math.h>
#include <random>

namespace sketch{

template <typename I, typename T>
gaussian_sketch<I, T>::gaussian_sketch(size_t p, size_t n) {
    S = new Eigen::MatrixXd(p, n);
    std::random_device rd;
    seed = rd();
    std::mt19937 mt(seed);
    double stddev = sqrt(1.0 / p);
    std::normal_distribution<> g(0, stddev);
    for (unsigned int i = 0; i < p; i++)
        for (unsigned int j = 0; j < n; j++)
            (*S)(i, j) = g(mt);
}

template <typename I, typename T>
void gaussian_sketch<I, T>::sketch(I *A, T *SA) {
    (*SA) = (*S) * (*A);
}

template <typename I, typename T>
void gaussian_sketch<I, T>::sketch_right(I *A, T *AS) {
    (*AS) = (*A) * (*S);
}

template class gaussian_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}
