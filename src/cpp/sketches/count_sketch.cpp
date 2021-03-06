/*
 * Count Sketch
 *
 * Naive sequential implementation of the count sketch algorithm. For each
 * column, we select a uniformly random row and assign it a random sign.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "sketch.hpp"
#include "util.hpp"
#include <random>
#include <vector>

namespace sketch {

namespace seq {

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n, unsigned int s) {
    S = new Eigen::SparseMatrix<double>(p, n);
    seed = s;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> rand_row(0, p-1);
    std::uniform_int_distribution<int> rand_sign(0, 1);
    std::vector<Eigen::Triplet<double> > entries;
    for (unsigned int j = 0; j < n; j++) {
        int i = rand_row(mt);
        int sign = rand_sign(mt) * 2 - 1;
        entries.push_back(Eigen::Triplet<double>(i, j, sign));
    }
    S->setFromTriplets(entries.begin(), entries.end());
}

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n) : count_sketch<I, T>::count_sketch(p, n, random_seed()) {}

template <typename I, typename T>
void count_sketch<I, T>::sketch(I *A, T *SA) {
    (*SA) = (*S) * (*A);
}

template <typename I, typename T>
size_t count_sketch<I, T>::eps_approx_rows(size_t n, size_t d, double eps) {
    double delta = 0.01; // failure rate of 1/100
    return 6 * d*d / (delta * eps*eps);
}

template class count_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}

}
