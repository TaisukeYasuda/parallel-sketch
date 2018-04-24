/*
 * Gaussian Sketch
 *
 * Naive sequential implementation of Gaussian projection.
 */

#include <Eigen/Dense>
#include "sketch.hpp"
#include <math.h>
#include <random>
#include <algorithm>

namespace sketch {

namespace seq {

template <typename I, typename T>
gaussian_sketch<I, T>::gaussian_sketch(size_t n, size_t d, double eps) {
    size_t p = gaussian_sketch<I, T>::eps_approx_rows(n, d, eps);
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
        size_t k = (int) ceil(4 / (pow(eps, 2) / 2 - pow(eps, 3) / 3) * log(n));
        return std::min(n, k);
    }
}

template class gaussian_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}

}
