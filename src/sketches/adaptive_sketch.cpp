#include <boost/numeric/ublas/matrix.hpp>
#include "sketch.hpp"

namespace bnu = boost::numeric::ublas;

namespace sketch{

template<typename I, typename T>
adaptive_sketch<I, T>::adaptive_sketch() {
}

template<typename I, typename T>
adaptive_sketch<I, T>::adaptive_sketch(std::size_t num_rows, std::size_t num_cols) {
}

template class adaptive_sketch<bnu::matrix<float>, bnu::matrix<float> >;

}
