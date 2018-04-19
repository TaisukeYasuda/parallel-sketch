
#ifndef _SKETCH_H_INCLUDED
#define _SKETCH_H_INCLUDED

#include <cstddef>
#include <boost/numeric/ublas/matrix.hpp>

namespace bnu = boost::numeric::ublas;

namespace sketch {
//TODO case based on underlying matrix type
//sketch will be stored internally (so each object is the sketch itself)
//this allows for use of operator overloading
template <typename I, typename T>
class Sketch {
    public:
        virtual T* sketch(I* A) = 0;
};

template <typename I, typename T>
class ObliviousSketch : public Sketch<I, T> {
    public:
        ObliviousSketch(std::size_t num_rows, std::size_t num_cols);
};

template <typename I, typename T>
class AdaptiveSketch : public Sketch<I, T> {
    public:
        AdaptiveSketch();
};

template <typename I, typename T>
class GaussianProjection : public ObliviousSketch<I, T> {};

template <typename I, typename T>
class CountSketch : public ObliviousSketch<I, T>  {
    public:
        CountSketch(size_t p, size_t n);
        T* sketch(I *A);
    private:
        unsigned int seed;
        bnu::matrix<float> *S;
};

template <typename I, typename T>
class UniformSamplingSketch : public ObliviousSketch<I, T> {};

template <typename I, typename T>
class LeverageScoreSketch : public AdaptiveSketch<I, T> {};

}

#endif
