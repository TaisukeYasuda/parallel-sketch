#include <boost/numeric/ublas/matrix.hpp>
#include "count_sketch.hpp"
#include <iostream>

namespace bnu = boost::numeric::ublas;

int main() {
    std::cout << "Running a stupid test" << std::endl;
    CountSketch S(3, 20);
    bnu::matrix<float> A(20, 5);
    S.sketch(&A, &A);
    return 0;
}
