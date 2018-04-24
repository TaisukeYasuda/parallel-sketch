#include <Eigen/Dense>
#include "sketch.hpp"

namespace sketch {

namespace seq {

template<typename I, typename T>
adaptive_sketch<I, T>::adaptive_sketch() {
}

template<typename I, typename T>
adaptive_sketch<I, T>::adaptive_sketch(size_t n, size_t d, double eps) {
}

template class adaptive_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}

}
