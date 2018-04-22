/*
 * Count Sketch
 *
 * Naive sequential implementation of the count sketch algorithm. For each
 * column, we select a uniformly random row and assign it a random sign.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "sketch.hpp"
#include <random>
#include <vector>

namespace sketch{

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n) {
    S = new Eigen::SparseMatrix<double>(p, n);
    std::random_device rd;
    seed = rd();
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> rand_row(0, p-1);
    std::uniform_int_distribution<int> rand_sign(0, 1);
    std::vector<Eigen::Triplet<double> > entries;
    for (unsigned int j = 0; j < n; j++) {
        unsigned int i = rand_row(mt);
        entries.push_back(Eigen::Triplet<double>(i, j, rand_sign(mt) * 2 - 1));
    }
    S->setFromTriplets(entries.begin(), entries.end());
}

template <typename I, typename T>
void count_sketch<I, T>::sketch(I *A, T *SA) {
    (*SA) = (*S) * (*A);
}

template class count_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}
