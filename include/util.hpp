#ifndef _UTIL_H_INCLUDED
#define _UTIL_H_INCLUDED

#include <vector>
#include <string>

unsigned int random_seed();
std::vector< std::vector<double> > *read_matrix(std::string filename);

#endif
