#include "util.hpp"
#include <random>

unsigned int random_seed() {
    std::random_device rd;
    return rd();
}

