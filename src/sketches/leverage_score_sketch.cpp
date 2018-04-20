/*
 * Leverage Score Sampling Sketch
 *
 * Naive sequential implementation of leverage score sampling.
 */

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "sketch.hpp"
#include <random>

namespace bnu = boost::numeric::ublas;

namespace sketch{

template <typename I, typename T>
leverage_score_sketch<I, T>::leverage_score_sketch(size_t p, size_t n) {
    S = new bnu::compressed_matrix<float>(p, n);
    std::random_device rd;
    seed = rd();
}

template <typename I, typename T>
void leverage_score_sketch<I, T>::sketch(I *A, T *SA) {
    // @TODO compute leverage scores and sample
    *SA = prod(*S, *A);
}

template class leverage_score_sketch<bnu::matrix<float>, bnu::matrix<float> >;

}
