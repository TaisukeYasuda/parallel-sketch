/*
 * Gaussian Projection Sketch
 *
 * Naive sequential implementation of Gaussian projection.
 */

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "sketch.hpp"
#include <random>

namespace bnu = boost::numeric::ublas;

namespace sketch{

template <typename I, typename T>
gaussian_projection<I, T>::gaussian_projection(size_t p, size_t n) {
    S = new bnu::matrix<float>(p, n);
    std::random_device rd;
    seed = rd();
    // @TODO sample gaussian matrix
}

template <typename I, typename T>
void gaussian_projection<I, T>::sketch(I *A, T *SA) {
    *SA = prod(*S, *A);
}

template class gaussian_projection<bnu::matrix<float>, bnu::matrix<float> >;

}
