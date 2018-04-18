#ifndef _SKETCH_H_INCLUDED
#define _SKETCH_H_INCLUDED

namespace sketch {
//TODO case based on underlying matrix type
//sketch will be stored internally (so each object is the sketch itself)
//this allows for use of operator overloading
template <typename I, typename T>
class Sketch {
    public:
        virtual *T sketch(I *A) = 0;
};

template <typename I, typename T>
class ObliviousSketch: public Sketch<I, T> {
    public:
        ObliviousSketch(size_t num_rows);
};

template <typename I, typename T>
class AdaptiveSketch: public Sketch<I, T> {
    public:
        AdaptiveSketch();
};

template <typename I, typename T>
class GaussianProjection : public ObliviousSketch<I, T> {};

template <typename I, typename T>
class CountSketch : public ObliviousSketch<I, T>  {};

template <typename I, typename T>
class UniformSamplingSketch : public ObliviousSketch<I, T> {};

template <typename I, typename T>
class LeverageScoreSketch : public AdaptiveSketch<I, T> {};

}

#endif
