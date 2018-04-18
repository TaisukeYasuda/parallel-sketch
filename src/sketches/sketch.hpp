namespace sketch {
//TODO case based on underlying matrix type
//sketch will be stored internally (so each object is the sketch itself)
//this allows for use of operator overloading
template <typename T>
class Sketch {
    protected:
        T *SA;
    public:
        virtual void sketch(T *A) = 0;
};

template <typename T>
class ObliviousSketch: public Sketch<T> {
    public:
        ObliviousSketch(size_t num_rows);
};

template <typename T>
class AdaptiveSketch: public Sketch<T> {
    public:
        AdaptiveSketch();
};

template<typename T>
class GaussianProjection : public ObliviousSketch<T> {};
class CountSketch : public ObliviousSketch<T>  {}
class UniformSamplingSketch : public ObliviousSketch {}
class LeverageScoreSketch : public AdaptiveSketch {}
}
