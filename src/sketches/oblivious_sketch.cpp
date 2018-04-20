#include <boost/numeric/ublas/matrix.hpp>
#include "sketch.hpp"

namespace bnu = boost::numeric::ublas;

namespace sketch{

template<typename I, typename T>
oblivious_sketch<I, T>::oblivious_sketch() {

}

template<typename I, typename T>
oblivious_sketch<I, T>::oblivious_sketch(std::size_t num_rows, std::size_t num_cols) {

}

template class oblivious_sketch<bnu::matrix<float>, bnu::matrix<float> >;

}
