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
                std::vector<double> p_inj, std::vector<double> p_cont,
                std::vector<double> p_abnd,
                std::vector< std::pair< std::pair<size_t,size_t>, unsigned int> > *edge_list,
                std::vector< std::vector<double> > *seeds);
    private:
        double mu_1;
        double mu_2;
        double mu_3;
        std::vector<size_t> *edge_starts; //Has all starts of edges in edge list
        std::vector<unsigned int> *edges; //Edge list itself
        std::vector< sketch::seq::count_min_sketch > *Ys; //Label scores
        std::vector<double> *Ms; //M vector
}
