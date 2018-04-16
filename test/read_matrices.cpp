/* Reads in and runs tests
 *
 *
 */

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>

std::vector< std::vector<double> > *read_matrix(std::string filename) {
    std::vector< std::vector<double> > *matrix = new std::vector< std::vector<double> >(0);
    
    std::ifstream infile(filename.c_str());
    std::string line;
    
    while(std::getline(infile, line)) {
        std::istringstream iss(line);
        
        std::vector<double> curr;
        double temp;
        while(iss >> temp)
            curr.push_back(temp);

        matrix->push_back(curr);
    }

    return matrix;
}

void print_matrix(std::vector< std::vector<double> > *m, int n, int k) {
    n = std::min(n, (int) m->size());
    k = std::min(k, (int) m->at(0).size());
    for(int i = 0; i < m->size(); i++) {
        for(int j = 0; j < m->at(i).size(); j++) {
            std::cout << m->at(i)[j];
        }
        std::cout << std::endl;
    }
}

int main() {
    int num_tests = 10;
    std::vector<std::string> cases;
    cases.push_back("med_test");
    cases.push_back("small_test");

    std::string test_dir = "random_matrices/";
    
    for(int i = 0; i < cases.size(); i++){
        for(int j = 0; j < num_tests; j++){
            std::stringstream ss;
            ss << test_dir << cases[i] << j << ".txt";
            
            print_matrix(read_matrix(ss.str()), 5, 5);
        }
    }
    


    return 0;
}
