#include <Eigen/Dense>
#include "sketch.hpp"
#include "util.hpp"
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
#define MAX_ROWS 500

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
        if (p > MAX_ROWS) {
            p = MAX_ROWS;
            std::cout << "\tSketch size cappedd to " << p << std::endl;
        }
        S = new sketch::seq::count_sketch<M, M>(p, n, SEED);
    } else if (sketch_type.compare("gaussian_sketch") == 0) {
        p = sketch::seq::gaussian_sketch<M, M>::eps_approx_rows(n, d, eps);
        if (p > MAX_ROWS) {
            p = MAX_ROWS;
            std::cout << "\tSketch size cappedd to " << p << std::endl;
        }
        S = new sketch::seq::gaussian_sketch<M, M>(p, n, SEED);
    } else if (sketch_type.compare("leverage_score_sketch") == 0) {
        p = sketch::seq::leverage_score_sketch<M, M>::eps_approx_rows(n, d, eps);
        if (p > MAX_ROWS) {
            p = MAX_ROWS;
            std::cout << "\tSketch size cappedd to " << p << std::endl;
        }
        S = new sketch::seq::leverage_score_sketch<M, M>(p, n, SEED);
    } else {
        std::cerr << "Invalid sketch type." << std::endl;
        exit(1);
    }

    std::cout << "\tCreated sketch of size " << p << std::endl;

    M A(n, d);
    M SA(p, d);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < d; j++) {
            A(i, j) = temp->at(i)[j];
        }
    }

    S->sketch(&A, &SA);

    Eigen::IOFormat numpy_format(Eigen::StreamPrecision, 0, ",", "\n", "", "", "", "");

    std::ofstream outfile;
    outfile.open(res_dir.c_str());
    outfile << SA.format(numpy_format) << std::endl;
    outfile.close();

    return 0;
}
