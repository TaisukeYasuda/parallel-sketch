#include <Eigen/Dense>
#include "sketch.hpp"
#include "sketch_cuda.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>

#define INPUT 1
#define OUTPUT 2
#define SKETCH_TYPE 3

#define SEED 116
#define MAX_ROWS 10000

std::vector< std::vector<double> > *read_matrix(std::string filename) {
    std::vector< std::vector<double> > *matrix = new std::vector< std::vector<double> >;
    std::ifstream infile(filename.c_str());

    for(std::string line; std::getline(infile, line);) {
        std::istringstream iss(line);

        std::vector<double> curr;

        for(std::string temp; std::getline(iss, temp, ',');)
            curr.push_back(std::stod(temp));

        matrix->push_back(curr);
    }

    return matrix;
}

int main(int argc, char *argv[]) {
    typedef Eigen::MatrixXd M;

    std::cout << "Testing " << argv[SKETCH_TYPE] << std::endl;

    std::string test_dir = std::string(argv[INPUT]);
    std::string res_dir = std::string(argv[OUTPUT]);
    std::string sketch_type = std::string(argv[SKETCH_TYPE]);

    std::vector< std::vector<double> > *temp;
    temp = read_matrix(test_dir);

    size_t n = temp->size(), d = temp->at(0).size(), p;
    double eps = 0.5;
    sketch::seq::sketch_interface<M, M> *S;

    if (sketch_type.compare("count_sketch") == 0) {
        p = sketch::seq::count_sketch<M, M>::eps_approx_rows(n, d, eps);
        S = new sketch::seq::count_sketch<M, M>(p, n, SEED);
    } else if (sketch_type.compare("gaussian_sketch") == 0) {
        p = sketch::seq::gaussian_sketch<M, M>::eps_approx_rows(n, d, eps);
        S = new sketch::seq::gaussian_sketch<M, M>(p, n, SEED);
    } else if (sketch_type.compare("leverage_score_sketch") == 0) {
        p = sketch::seq::leverage_score_sketch<M, M>::eps_approx_rows(n, d, eps);
        S = new sketch::seq::leverage_score_sketch<M, M>(p, n, SEED);
    } else {
        std::cerr << "Invalid sketch type." << std::endl;
        exit(1);
    }

    if (p > MAX_ROWS) {
        std::cerr << "Too many rows in the sketch." << std::endl;
        exit(1);
    }

    M A(n, d);
    M SA(p, d);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < d; j++) {
            A(i, j) = temp->at(i)[j];
        }
    }

    std::cout << "\tCreated sketch of size " << p << std::endl;
    S->sketch(&A, &SA);

    std::ofstream outfile;
    outfile.open(res_dir.c_str());
    outfile << SA << std::endl;
    outfile.close();

    return 0;
}
