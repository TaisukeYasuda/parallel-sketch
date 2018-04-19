#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "count_sketch.hpp"
#include <random>

namespace bnu = boost::numeric::ublas;

count_sketch::count_sketch(size_t p, size_t n) {
    S = bnu::compressed_matrix<float>(p, n);
    std::random_device rd;
    seed = rd();
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> rand_row(0, p-1);
    std::uniform_int_distribution<int> rand_sign(0, 1);
    for (unsigned int i = 0; i < n; i++)
        S(rand_row(mt), i) = rand_sign(mt) * 2 - 1;
}

void count_sketch::sketch(bnu::matrix<float> *A, bnu::matrix<float> *SA) {
    *SA = prod(S, *A);
}
