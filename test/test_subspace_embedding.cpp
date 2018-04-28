/*
 * File for testing that the matrix SA is indeed a subspace embedding for A.
 * Checks the norm of Ax and SAx.
 */

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include "util.hpp"

#define INPUT_A 1 // original matrix
#define INPUT_SA 2 // sketched matrix
#define INPUT_x 3 // vector
#define INPUT_eps 4 // accuracy parameter

int main(int argc, char *argv[]) {
    typedef Eigen::MatrixXd M;

    std::cout << "Testing subspace embedding" << std::endl;

    std::string A_dir = std::string(argv[INPUT_A]);
    std::string SA_dir = std::string(argv[INPUT_SA]);
    std::string x_dir = std::string(argv[INPUT_x]);
    double eps = std::stof(argv[INPUT_eps]);

    std::vector< std::vector<double> > *temp_A;
    temp_A = read_matrix(A_dir);
    std::vector< std::vector<double> > *temp_SA;
    temp_SA = read_matrix(SA_dir);
    std::vector< std::vector<double> > *temp_x;
    temp_x = read_matrix(x_dir);

    size_t n = temp_A->size(),
           d = temp_A->at(0).size(),
           p = temp_SA->size();

    M A(n, d);
    M SA(p, d);
    M x(d, 1);

    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < d; j++)
            A(i, j) = temp_A->at(i)[j];
    for (unsigned int i = 0; i < p; i++)
        for (unsigned int j = 0; j < d; j++)
            SA(i, j) = temp_SA->at(i)[j];
    for (unsigned int i = 0; i < d; i++)
        x(i, 0) = temp_x->at(i)[0];

    std::cout << (A*x).squaredNorm() << std::endl;
    std::cout << (SA*x).squaredNorm() << std::endl;

    return 0;
}
