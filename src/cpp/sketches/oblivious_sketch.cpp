#include <Eigen/Dense>
#include "sketch.hpp"

namespace sketch{

template<typename I, typename T>
oblivious_sketch<I, T>::oblivious_sketch() {

}

template<typename I, typename T>
oblivious_sketch<I, T>::oblivious_sketch(size_t p, size_t d) {

}

template class oblivious_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}
