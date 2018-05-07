#include "util.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

unsigned int random_seed() {
    std::random_device rd;
    return rd();
}

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


