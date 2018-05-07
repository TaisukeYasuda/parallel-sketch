/*
 * Leverage Score Sampling Sketch
 *
 * Naive sequential implementation of leverage score sampling.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include "sketch.hpp"
#include "util.hpp"
#include <math.h>
#include <random>
#include <vector>
#include <iostream>

namespace sketch {

namespace seq {

template <typename I, typename T>
leverage_score_sketch<I, T>::leverage_score_sketch(size_t p, size_t n, unsigned int s) {
    this->D = new Eigen::SparseMatrix<double>(p, p); // diagonal rescaling
    this->Omega = new Eigen::SparseMatrix<double>(p, n); // sampling
    this->seed = s;
    this->_p = p;
    this->_n = n;
    this->sketched = false;
}

template <typename I, typename T>
leverage_score_sketch<I, T>::leverage_score_sketch(size_t p, size_t n) : leverage_score_sketch<I, T>::leverage_score_sketch(p, n, random_seed()) {}

template <typename I, typename T>
void leverage_score_sketch<I, T>::sketch(I *A, T *SA) {
    if (!this->sketched) {
        // find a 1+1/36 subspace embedding
        size_t n = this->_n, d = A->cols();
        size_t p_count = sketch::seq::count_sketch<I, T>::eps_approx_rows(n, d, 1.0/36);
        T *SA_count;
        if (p_count < n) {
            SA_count = new T(p_count, d);
            sketch::seq::count_sketch<I, T> S_count(p_count, n);
            S_count.sketch(A, SA_count);
        } else {
            SA_count = A;
        }

        // find a QR decomposition of SA
        Eigen::HouseholderQR<T> qr(SA_count->rows(), SA_count->cols());
        qr.compute(*SA_count);
        T squareR(Eigen::MatrixXd::Identity(d, n));
        T R = squareR * qr.matrixQR().template triangularView<Eigen::Upper>();
        R = R.inverse();
        // compute ARG for a gaussian sketch G
        size_t p_gauss;
        try {
            p_gauss = sketch::seq::gaussian_sketch<I, T>::eps_approx_rows(d, d, 0.5);
        } catch (const sketch::seq::bad_dimension& e) {
            p_gauss = d;
        }
        T RG_transpose;
        if (p_gauss < d) {
            sketch::seq::gaussian_sketch<I, T> G(p_gauss, d);
            RG_transpose = T(p_gauss, d);
            T R_transpose = R.transpose();
            G.sketch(&R_transpose, &RG_transpose);
        } else {
            RG_transpose = R.transpose();
        }
        T A_transpose = A->transpose();
        T ARG_transpose = RG_transpose * A_transpose;

        // scaled col norms of ARG_transpose are estimates of leverage scores
        double beta = 4.0 / 7.0;
        auto q = beta * ARG_transpose.colwise().squaredNorm() / d;
        double excess_probability = (1.0 - q.sum()) / n;

        // sample, q_i is the above plus the excess probability
        std::mt19937 mt(seed);
        std::uniform_real_distribution<double> unif(0, 1);
        std::vector<Eigen::Triplet<double> > Omega_entries;
        std::vector<Eigen::Triplet<double> > D_entries;
        for (unsigned int i = 0; i < this->_p; i++) {
            double score = unif(mt), sum = 0, q_j = 0;
            unsigned int j = -1;
            while (sum < score) {
                j++;
                q_j = q(j) + excess_probability;
                sum += q_j;
            }
            Omega_entries.push_back(Eigen::Triplet<double>(i, j, 1));
            double scale = 1.0 / sqrt(q_j * this->_p);
            D_entries.push_back(Eigen::Triplet<double>(i, i, scale));
        }
        this->Omega->setFromTriplets(Omega_entries.begin(), Omega_entries.end());
        this->D->setFromTriplets(D_entries.begin(), D_entries.end());

        // mark as sketched
        this->sketched = true;
    }
    (*SA) = (*this->D) * (*this->Omega) * (*A);
}

template <typename I, typename T>
size_t leverage_score_sketch<I, T>::eps_approx_rows(
        size_t n, size_t d, double eps) {
    if (d <= 1) {
        std::string info;
        info = std::string("Too few columns. Expected at least 2 columns.");
        throw bad_dimension(info);
    } else {
        double delta = 0.01;
        double beta = 4.0 / 7.0;
        double c = 1 + log(2.0 / delta) / log(d * 1.0);
        return (int) ceil(c * 4.0 / 3.0 * d / beta * log(d) / (eps * eps));
    }
}

template class leverage_score_sketch<Eigen::MatrixXd, Eigen::MatrixXd >;

}

}
