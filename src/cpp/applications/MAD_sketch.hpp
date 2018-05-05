/*
 * MAD Sketch
 *
 * Header file for MAD sketch
 *
 *
 */

#include "sketch.hpp"
#include <vector>
#include <utility>
#include <cstddef>

class MAD_sketch {
    public:
        MAD_sketch(size_t n, size_t d, size_t w, double mu_1, double mu_2, double mu_3,
                double *p_inj, double *p_cont, double *p_abnd,
                std::vector< std::pair< std::pair<size_t,size_t>, size_t> > *edge_list,
                std::vector< sketch::seq::count_min_sketch<double> > *seeds);
        void run_sim(size_t iterations);
        std::vector< sketch::seq::count_min_sketch<double> > *get_labels();
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
        double *rs;
        
        size_t *edges; //Edge list itself
        double *edge_factors;
        
        double *seeds;
        double *Ys; //Label scores
        double *Mvv; //M vector
}
