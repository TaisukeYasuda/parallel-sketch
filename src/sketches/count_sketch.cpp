/*
 * Count Sketch
 *
 * Naive sequential implementation of the count sketch algorithm. For each
 * column, we select a uniformly random row and assign it a random sign.
 */

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "sketch.hpp"
#include <random>

namespace bnu = boost::numeric::ublas;

namespace sketch{

template <typename I, typename T>
count_sketch<I, T>::count_sketch(size_t p, size_t n) {
    S = new bnu::compressed_matrix<float>(p, n);
    std::random_device rd;
    seed = rd();
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> rand_row(0, p-1);
    std::uniform_int_distribution<int> rand_sign(0, 1);
    for (unsigned int i = 0; i < n; i++)
        (*S)(rand_row(mt), i) = rand_sign(mt) * 2 - 1;
}

template <typename I, typename T>
void count_sketch<I, T>::sketch(I *A, T *SA) {
    *SA = prod(*S, *A);
}

template class count_sketch<bnu::matrix<float>, bnu::matrix<float> >;

}
