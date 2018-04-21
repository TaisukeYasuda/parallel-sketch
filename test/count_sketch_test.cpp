#include <Eigen/Dense>
#include "sketch.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>

int main() {
    std::cout << "Running a stupid test" << std::endl;
    size_t p = 3, n = 20, d = 5;
    sketch::count_sketch<Eigen::MatrixXd, Eigen::MatrixXd > S(p, n);
    Eigen::MatrixXd A(n, d);
    Eigen::MatrixXd SA;

    std::string line;
    std::ifstream infile("./data/random_matrices/small_test0.txt");
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < d; j++) {
            if (j < d-1) std::getline(infile, line, ',');
            else std::getline(infile, line, '\n');
            std::istringstream iss(line);
            double temp;
            iss >> temp;
            A(i, j) = temp;
        }
    }

    std::cout << "Printing A" << std::endl;
    std::cout << A << std::endl;

    S.sketch(&A, &SA);
    std::cout << "Printing SA" << std::endl;
    std::cout << SA << std::endl;

    return 0;
}
