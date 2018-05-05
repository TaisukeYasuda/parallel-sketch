/*
 * MAD Sketch
 *
 * Header file for MAD sketch
 *
 *
 */
#ifndef _MAD_SKETCH_H_INCLUDED
#define _MAD_SKETCH_H_INCLUDED

#include "sketch.hpp"
#include <vector>
#include <utility>
#include <cstddef>

class MAD_sketch {
    public:
       MAD_sketch(size_t n_, size_t d_, size_t w_, double mu_1_, double mu_2_, double mu_3_,
            double *p_inj_, double *p_cont_, double *p_abnd_,
            std::vector< std::pair< std::pair<size_t, size_t>, double> > *edge_list_,
            std::vector< sketch::seq::count_min_sketch<double> > *seeds_,
            sketch::seq::count_min_sketch<double> *r_);
        void run_sim(size_t iterations);
        std::vector< sketch::seq::count_min_sketch<double>* >* get_labels(size_t *hashes);
    private:
        size_t n;
        size_t d;
        size_t w;
        size_t num_edges;

        double mu_1;
        double mu_2;
        double mu_3;
        
        double *p_inj;
        double *p_abnd;
        double *r;
        
        size_t *edges; //Edge list itself
        double *edge_factors;
        
        double *seeds;
        double *Ys; //Label scores
        double *Mvv; //M vector
};

#endif
