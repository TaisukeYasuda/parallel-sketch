#ifndef _SKETCH_H_INCLUDED
#define _SKETCH_H_INCLUDED

#include <cstddef>
#include <boost/numeric/ublas/matrix.hpp>

namespace bnu = boost::numeric::ublas;

namespace sketch {
/*
 * Sketch interface
 *
 * @TODO case based on underlying matrix type
 * sketch will be stored interall (so each object is the sketch itself)
 * this allows for use of operator overloading
 */
template <typename I, typename T>
class sketch_interface {
    public:
        virtual void sketch(I *A, T *SA) = 0;
};

/*
 * Abstract classes
 */
template <typename I, typename T>
class oblivious_sketch : public sketch_interface<I, T> {
    public:
        oblivious_sketch();
        oblivious_sketch(std::size_t num_rows, std::size_t num_cols);
};

template <typename I, typename T>
class adaptive_sketch : public sketch_interface<I, T> {
    public:
        adaptive_sketch();
        adaptive_sketch(std::size_t num_rows, std::size_t num_cols);
};

/*
 * Oblivious sketch instantiations
 */
template <typename I, typename T>
class gaussian_projection : public oblivious_sketch<I, T> {
    public:
        gaussian_projection(size_t p, size_t n);
        void sketch(I *A, T *SA);
    private:
        unsigned int seed;
        bnu::matrix<float> *S;
};

template <typename I, typename T>
class count_sketch : public oblivious_sketch<I, T>  {
    public:
        count_sketch(size_t p, size_t n);
        void sketch(I *A, T *SA);
    private:
        unsigned int seed;
        bnu::compressed_matrix<float> *S;
};

template <typename I, typename T>
class uniform_sampling_sketch : public oblivious_sketch<I, T> {
    public:
        uniform_sampling_sketch(size_t p, size_t n);
        void sketch(I *A, T *SA);
    private:
        unsigned int seed;
        bnu::compressed_matrix<float> *S;
};

/*
 * Adaptive sketch instantiations
 */
template <typename I, typename T>
class leverage_score_sketch : public adaptive_sketch<I, T> {
    public:
        leverage_score_sketch(size_t p, size_t n);
        void sketch(I *A, T *SA);
    private:
        unsigned int seed;
        bnu::compressed_matrix<float> *S;
};

}

#endif
