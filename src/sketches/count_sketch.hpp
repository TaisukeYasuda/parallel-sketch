#include <boost/numeric/ublas/matrix.hpp>

namespace bnu = boost::numeric::ublas;

class CountSketch {
    public:
        CountSketch(size_t p, size_t n);
        void sketch(bnu::matrix<float> *A, bnu::matrix<float> *SA);

    private:
        unsigned int seed;
        bnu::matrix<float> S;
};
