/*
 * Reads in and runs tests
 */

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>

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

std::vector<std::string> *get_test_files() {
    int num_tests = 10;
    std::vector<std::string> cases;
    cases.push_back("small_test");
    cases.push_back("med_test");

    std::vector<std::string> *names = new std::vector<std::string>;

    for(int i = 0; i < cases.size(); i++){
        for(int j = 0; j < num_tests; j++){
            std::stringstream ss;
            ss << cases[i] << j << ".txt";
            names->push_back(ss.str());
        }
    }

    return names;
}
