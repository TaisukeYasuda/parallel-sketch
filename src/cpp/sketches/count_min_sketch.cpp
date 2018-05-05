/*
 * CountMin Sketch
 *
 * Sequential implementation (we probably won't do an optimized parallel one,
 * it doesn't make sense to). Hashes the input value n times.
 *
 */ 

#include "sketch.hpp"
#include <assert.h>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <iterator>

namespace sketch {

namespace seq {

template <typename T>
count_min_sketch<T>::count_min_sketch(size_t d, size_t w, size_t *hashes) {
    this->w = w;
    this->d = d;
    this->CM = new T[d*w];
    this->h  = new size_t[w];

    memset(this->CM, 0, d * w * sizeof(T));
    memcpy(this->h )
}

template <typename T>
count_min_sketch<T>::count_min_sketch(size_t d, size_t w) {

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
void count_min_sketch<T>::add_vec(std::vector<T> *v) {
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



template <typename T>
void count_min_sketch<T>::add_const(double d) {
    size_t d = this->CM->size();
           w = this->CM->at(0).size();
    
    for(size_t i = 0; i < d; i++) {
        for(size_t j = 0; j < w; j++) {
            this->CM->at(i)[j] += d;
        }
    }
}

template <typename T>
void count_min_sketch<T>::mult_const(double d) {
    size_t d = this->CM->size();
           w = this->CM->at(0).size();
    
    for(size_t i = 0; i < d; i++) {
        for(size_t j = 0; j < w; j++) {
            this->CM->at(i)[j] *= d;
        }
    }
}

template <typename T>
void count_min_sketch<T>::add_sketch(count_min_sketch<T> *CMS) {
    size_t d = this->CM->size();
           w = this->CM->at(0).size();
    
    assert(d == CMS.CM->size() && w == CMS.CM->at(0).size());

    for(size_t i = 0; i < d; i++) {
        for(size_t j = 0; j < w; j++) {
            this->CM->at(i)[j] += CMS->CM->at(i)[j];
        }
    }
    
}

template<typename T>
count_min_sketch<T> *count_min_sketch<T>::make_copy() {
    size_t d = this->CM->size();
           w = this->CM->at(0).size();
    
    count_min_sketch<T> *new_CM = new count_min_sketch<T>(d, w, this->h);
    
    return new_CM;
}

}

}
