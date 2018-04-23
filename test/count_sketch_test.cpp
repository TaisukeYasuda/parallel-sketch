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

int main(int argc, char *argv[]) {
    std::cout << "Testing Count Sketch" << std::endl;

    std::string test_dir = std::string("data/random_matrices/");
    std::string res_dir = std::string("results/count_sketch/");

    std::vector<std::string> *names = get_test_files();
    std::vector< std::vector<double> > *temp;

    for (unsigned int k = 0; k < names->size(); k++) {
        temp = read_matrix(test_dir + names->at(k));

        int n = temp->size(), p = 10, d = temp->at(0).size();
        sketch::seq::count_sketch<Eigen::MatrixXd, Eigen::MatrixXd> S(p, n);
        Eigen::MatrixXd A(n, d);
        Eigen::MatrixXd SA;

        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < d; j++) {
                A(i, j) = temp->at(i)[j];
            }
        }

        S.sketch(&A, &SA);

        std::stringstream ss;
        ss << res_dir + names->at(k) + ".res";

        std::ofstream outfile;
        outfile.open(ss.str());

        outfile << A << std::endl;
        outfile.close();
    }

    return 0;
}
