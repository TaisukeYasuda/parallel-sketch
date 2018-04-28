#include "sketch_cuda.h"

namespace sketch {

namespace par {

template<typename I, typename T>
oblivious_sketch<I, T>::oblivious_sketch() {

}

template<typename I, typename T>
oblivious_sketch<I, T>::oblivious_sketch(size_t p, size_t n) {

}

template class oblivious_sketch<int*, int* >;

}

}
