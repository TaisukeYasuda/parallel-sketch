#include <Eigen/Dense>
#include "sketch.hpp"
#include "sketch_cuda.h"
#include "read_matrices.cpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>

#define INPUT 1
#define OUTPUT 2
#define SKETCH_TYPE 3

int main(int argc, char *argv[]) {
    std::cout << "Testing " << argv[SKETCH_TYPE] << std::endl;

    std::string test_dir = std::string(argv[INPUT]);
    std::string res_dir = std::string(argv[OUTPUT]);
    std::string sketch_type = std::string(argv[SKETCH_TYPE]);

    std::vector< std::vector<double> > *temp;
    temp = read_matrix(test_dir);

    size_t n = temp->size(), p = 10, d = temp->at(0).size();
    sketch::seq::sketch_interface<Eigen::MatrixXd, Eigen::MatrixXd> *S;

    if (sketch_type.compare("count_sketch") == 0) {
        S = new sketch::seq::count_sketch<Eigen::MatrixXd, Eigen::MatrixXd>(p, n);
    } else if (sketch_type.compare("gaussian_sketch") == 0) {
        S = new sketch::seq::count_sketch<Eigen::MatrixXd, Eigen::MatrixXd>(p, n);
    } else {
        std::cerr << "Invalid sketch type." << std::endl;
        exit(1);
    }

    Eigen::MatrixXd A(n, d);
    Eigen::MatrixXd SA(p, d);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < d; j++) {
            A(i, j) = temp->at(i)[j];
        }
    }

    S->sketch(&A, &SA);

    std::ofstream outfile;
    outfile.open(res_dir.c_str());
    outfile << SA << std::endl;
    outfile.close();

    return 0;
}
