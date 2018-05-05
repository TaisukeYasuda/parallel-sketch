#include "sketch_cuda.h"

namespace sketch {

namespace par {

template<typename I, typename T>
adaptive_sketch<I, T>::adaptive_sketch() {
}

template<typename I, typename T>
adaptive_sketch<I, T>::adaptive_sketch(size_t p, size_t n) {
}

template class adaptive_sketch<double*, double* >;

}

}
