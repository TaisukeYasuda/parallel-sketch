/*
 * CountMin Sketch
 *
 * Sequential implementation (we probably won't do an optimized parallel one,
 * it doesn't make sense to). Hashes the input value n times.
 *
 */ 

#include "sketch.hpp"
#include <vector>
#include <random>
#include <limits>

namespace sketch {

namespace seq {

template <typename T>
count_min_sketch<T>::count_min_sketch(size_t d, size_t w) {
    this->CM = new std::vector< std::vector<T> >(d, std::vector<T>(w, 0));
    this->h  = new std::vector<size_t>(w);
    std::random_device rd;
    std::mt19937 mt(rd());

    //Hash function will be identiy xor random int
    std::uniform_int_distribution<unsigned int> rand_hash;

    for(size_t i = 0; i < d; i++)
       (*this->h)[i] = rand_hash(mt);
}

template <typename T>
T count_min_sketch<T>::get(size_t j) {
    size_t d = this->CM->size();
           w = this->CM->at(0).size();
    
    T res = std::numeric_limits<T>::max();

    size_t hashed;
    for(size_t i = 0; i < d; i++) {
        hashed = (i ^ this->h->at(i)) % w;
        res = std::min(res, this->CM->at(i)[hashed]);
    }

    return res;
}

template <typename T>
void count_min_sketch<T>::add(size_t j, T x) {
    size_t d = this->CM->size();
           w = this->CM->at(0).size();
    
    size_t hashed;
    for(size_t i = 0; i < d; i++) {
        hashed = (j ^ this->h->at(i)) % w;
        this->CM->at(i)[hashed] += x;
    }

}

template <typename T>
void count_min_sketch<T>::add_vec(std::vector *v) {
    size_t d = this->CM->size();
           w = this->CM->at(0).size();
    
    size_t hashed;
    for(size_t i = 0; i < d; i++) {
        for(size_t j = 0; j < v->size(); j++){
            hashed = (j ^ this->h->at(i)) % w;
            this->CM->at(i)[hashed] += v->at(j);
        }
    }
}

}

}
