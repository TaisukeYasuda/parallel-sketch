#include <boost/numeric/ublas/matrix.hpp>
#include "sketch.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>

namespace bnu = boost::numeric::ublas;

void print_matrix(bnu::matrix<float>& A) {
    for (unsigned i = 0; i < A.size1(); i++) {
        for (unsigned j = 0; j < A.size2(); j++)
            std::cout << A(i, j) << " ";
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "Running a stupid test" << std::endl;
    size_t p = 3, n = 20, d = 5;
    sketch::count_sketch<bnu::matrix<float>, bnu::matrix<float> > S(p, n);
    bnu::matrix<float> A(n, d);
    bnu::matrix<float> SA;

    std::string line;
    std::ifstream infile("random_matrices/small_test0.txt");
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < d; j++) {
            if (j < d-1) std::getline(infile, line, ',');
            else std::getline(infile, line, '\n');
            std::istringstream iss(line);
            float temp;
            iss >> temp;
            A(i, j) = temp;
        }
    }

    std::cout << "Printing A" << std::endl;
    print_matrix(A);
    std::cout << std::endl;

    S.sketch(&A, &SA);
    std::cout << "Printing SA" << std::endl;
    print_matrix(SA);
    std::cout << A.size1() << ' ' << A.size2();

    std::cout << std::endl;

    return 0;
}
