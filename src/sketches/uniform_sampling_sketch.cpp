/*
 * Uniform Sampling Sketch
 *
 * Naive sequential implementation of uniform sampling sketch.
 */

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "sketch.hpp"
#include <random>

namespace bnu = boost::numeric::ublas;

namespace sketch{

template <typename I, typename T>
uniform_sampling_sketch<I, T>::uniform_sampling_sketch(size_t p, size_t n) {
    S = new bnu::compressed_matrix<float>(p, n);
    std::random_device rd;
    seed = rd();
    // @TODO uniformly sample and scale
}

template <typename I, typename T>
void uniform_sampling_sketch<I, T>::sketch(I *A, T *SA) {
    *SA = prod(*S, *A);
}

template class uniform_sampling_sketch<bnu::matrix<float>, bnu::matrix<float> >;

}
