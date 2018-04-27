#ifndef _SKETCH_H_INCLUDED
#define _SKETCH_H_INCLUDED

#include <cstddef>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <exception>
#include <string>

namespace sketch {

namespace seq {

template <typename T>
class count_min_sketch {
    public:
        count_min_sketch(size_t d, size_t w);
        T get(size_t j);
        void add(size_t j, T x);
        void add_vec(std::vector<T> *v);
    private:
        std::vector<size_t> *h;
        std::vector< std::vector<T> > *CM;

};

/* Sketch interface
 *
 * @TODO Case implementation of sketch based on the input type I of A and input
 * type T of SA. Typically, A is a sparse matrix while SA will have to be dense,
 * but other use cases are possible. The type of S will usually be fixed. For
 * example, a Gaussian projection will always be dense and a count sketch will
 * always be sparse.
 *
 * The interface for a sketch matrix object S. This matrix is a random matrix
 * that has the property that, when applied to vectors and matrices, it
 * approximately preserves various properties, typically various norms, while
 * reducing the size.
 */
template <typename I, typename T>
class sketch_interface {
    public:
        virtual void sketch(I *A, T *SA) = 0;
};

/* Exceptions */
class bad_dimension : public std::exception {
    std::string exception_msg;
    public:
        bad_dimension(const std::string& info) {
            exception_msg = info;
        }
        virtual const char* msg() const throw() {
            return exception_msg.c_str();
        }
};


/* Oblivious sketch
 *
 * These sketching distributions don't depend on the sketched matrix A and thus
 * are sampled in the constructor.
 */
template <typename I, typename T>
class oblivious_sketch : public sketch_interface<I, T> {
    public:
        oblivious_sketch();
        oblivious_sketch(size_t n, size_t d, double eps);
};

/* Adaptive sketch
 *
 * These sketching distributions are sampled adaptively, i.e. they are sampled
 * after looking at the sketched matrix A. For a fixed matrix A, adaptive
 * sketches typically perform better than oblivious sketches since it has extra
 * information that it can use to construct the sketched matrix SA.
 */
template <typename I, typename T>
class adaptive_sketch : public sketch_interface<I, T> {
    public:
        adaptive_sketch();
        adaptive_sketch(size_t n, size_t d, double eps);
};

/*
 * Oblivious sketch instantiations
 */
template <typename I, typename T>
class gaussian_sketch : public oblivious_sketch<I, T> {
    public:
        gaussian_sketch(size_t n, size_t d, double eps);
        void sketch(I *A, T *SA);
        static size_t eps_approx_rows(size_t n, size_t d, double eps);
        const static size_t min_n = 10; // minimum rows required to sketch
    private:
        unsigned int seed;
        Eigen::MatrixXd *S;
};

template <typename I, typename T>
class count_sketch : public oblivious_sketch<I, T>  {
    public:
        count_sketch(size_t n, size_t d, double eps);
        void sketch(I *A, T *SA);
        static size_t eps_approx_rows(size_t n, size_t d, double eps);
    private:
        unsigned int seed;
        Eigen::SparseMatrix<double> *S;
};

/*
 * Adaptive sketch instantiations
 */
template <typename I, typename T>
class leverage_score_sketch : public adaptive_sketch<I, T> {
    public:
        leverage_score_sketch(size_t n, size_t d, double eps);
        void sketch(I *A, T *SA);
        static size_t eps_approx_rows(size_t n, size_t d, double eps);
    private:
        unsigned int seed;
        Eigen::SparseMatrix<double> *S;
};

}

}

#endif
