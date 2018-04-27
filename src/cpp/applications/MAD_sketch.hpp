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
        MAD_sketch(size_t n, size_t d, size_t w,
                std::vector< std::pair< std::pair<size_t,size_t>, unsigned int> > *edge_list,
                std::vector< std::vector<double> > *seeds)
    private:
        std::vector<size_t> *edge_starts; //Has all starts of edges in edge list
        std::vector<unsigned int> *edges; //Edge list itself
        std::vector< sketch::seq::count_min_sketch > *Ys; //Label scores
}
