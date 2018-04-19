#ifndef _SKETCH_H_INCLUDED
#define _SKETCH_H_INCLUDED

#include <cstddef>
#include <boost/numeric/ublas/matrix.hpp>

namespace bnu = boost::numeric::ublas;

namespace par_sketch {
//TODO case based on underlying matrix type
//sketch will be stored internally (so each object is the sketch itself)
//this allows for use of operator overloading
template <typename I, typename T>
class sketch_interface {
    public:
        virtual T& sketch(I& A) = 0;
};

template <typename I, typename T>
class oblivious_sketch : public sketch_interface<I, T> {
    public:
        oblivious_sketch(std::size_t num_rows, std::size_t num_cols);
};

template <typename I, typename T>
class adaptive_sketch : public sketch_interface<I, T> {
    public:
        adaptive_sketch();
};

template <typename I, typename T>
class gaussian_projection : public oblivious_sketch<I, T> {};

template <typename I, typename T>
class count_sketch : public oblivious_sketch<I, T>  {
    public:
        count_sketch(size_t p, size_t n);
        T& sketch(I &A);
    private:
        unsigned int seed;
        bnu::matrix<float> *S;
};

template <typename I, typename T>
class uniform_sampling_sketch : public oblivious_sketch<I, T> {};

template <typename I, typename T>
class leverage_score_sketch : public adaptive_sketch<I, T> {};

}

#endif
