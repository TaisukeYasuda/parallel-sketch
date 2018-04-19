#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "count_sketch.hpp"
#include <random>

namespace bnu = boost::numeric::ublas;

CountSketch::CountSketch(size_t p, size_t n) {
    S = bnu::compressed_matrix<float>(p, n);
    std::random_device rd;
    std::uniform_int_distribution<int> rand_row(0, p-1);
    for (unsigned int i = 0; i < n; i++) {
    }
}

void CountSketch::sketch(bnu::matrix<float> *A, bnu::matrix<float> *SA) {
}
