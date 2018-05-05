/*
 * CountMin Sketch
 *
 * Sequential implementation (we probably won't do an optimized parallel one,
 * it doesn't make sense to). Hashes the input value n times.
 *
 */ 

#include "sketch.hpp"
#include <assert.h>
#include <random>
#include <limits>
#include <algorithm>
#include <iterator>

namespace sketch {

namespace seq {

template <typename T>
count_min_sketch<T>::count_min_sketch(const count_min_sketch<T>& other) {

    this->w = other.get_w();
    this->d = other.get_d();
    this->CM = new T[d*w];
    this->h  = new size_t[d];

    memcpy(this->CM, other.get_CM(), this->d * this->w * sizeof(T));
    memcpy(this->h, other.get_hashes(), d);

}

template <typename T>
count_min_sketch<T>::count_min_sketch(size_t d, size_t w, size_t *hashes) {
    this->w = w;
    this->d = d;
    this->CM = new T[d*w];
    this->h  = new size_t[d];

    memset(this->CM, 0, d * w * sizeof(T));
    memcpy(this->h, hashes, d * sizeof(size_t));
}

template <typename T>
count_min_sketch<T>::count_min_sketch(size_t d, size_t w) {
    this->w = w;
    this->d = d;
    this->CM = new T[d*w];
    this->h  = new size_t[d];

    memset(this->CM, 0, d * w * sizeof(T));
    
    std::random_device rd;
    std::mt19937 mt(rd());

    //Hash function will be identiy xor random int
    std::uniform_int_distribution<size_t> rand_hash;

    for(size_t i = 0; i < d; i++)
       (this->h)[i] = rand_hash(mt);
}

template <typename T>
T count_min_sketch<T>::get(size_t j) {
    T res = std::numeric_limits<T>::max();

    size_t t;
    for(size_t i = 0; i < this->d; i++) {
        t = (j ^ (this->h)[i]) % this->w;
        res = std::min(res, (this->CM)[(i * this->w) + t]);
    }

    return res;
}

template <typename T>
void count_min_sketch<T>::add(size_t j, T x) {
    size_t t;
    for(size_t i = 0; i < this->d; i++) {
        t = (j ^ (this->h)[i]) % this->w;
        (this->CM)[(i * this->w) + t] += x;
    }

}

template <typename T>
size_t count_min_sketch<T>::get_d() const {
    return this->d;
}

template <typename T>
size_t count_min_sketch<T>::get_w() const {
    return this->w;
}

template <typename T>
T *count_min_sketch<T>::get_CM() const {
    return this->CM;
}

template <typename T>
size_t *count_min_sketch<T>::get_hashes() const {
    return this->h;
}

template <typename T>
void count_min_sketch<T>::add_vec(std::vector<T> *v) {
    size_t t;
    T temp;
    for(size_t j = 0; j < v->size(); j++){
        temp = v->at(j);
        for(size_t i = 0; i < this->d; i++) {
            t = (j ^ this->h[i]) % this->w;
            (this->CM)[(i * this->w) + t] += temp;
        }
    }
}

template <typename T>
count_min_sketch<T>::~count_min_sketch() {
    delete this->h;
    delete this->CM;

    this->h  = nullptr;
    this->CM = nullptr;
}

template class count_min_sketch<double>;

/*

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
}*/

}

}
