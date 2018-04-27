/*
 * MAD Sketch
 * 
 * Implementation of MAD sketch
 *
 */

#include "MAD_sketch.hpp"
#include <utility>
#include <vector>
#include <cstddef>
#include <algorithm>

MAD_sketch(size_t n, size_t d, size_t w,
        std::vector< std::pair< std::pair<size_t,size_t>, unsigned int> > *edge_list,
        std::vector< std::vector<double> > *seeds) {
    std::sort(edge_list->begin(), edge_list->end());

    this->edge_starts = new std::vector<size_t>(n, -1);
    this->edges = new std::vector<unsigned int>(edge_list->size() * 2);

    int prev_u = -1, u, v, weight;
    size_t idx;
    
    for(size_t i = 0; i < edge_list->size(); i++) {
        u = edge_list->at(i).first;
        v = edge_list->at(i).first.second;
        weight = edge_list->at(i).second;

        if(prev_u != u)
            this->edge_starts->at(u) = i;

        idx = i*2;
        (*this->edges)[idx] = v;
        (*this->edges)[idx+1] = weight;

        prev_u = u;
    }

    //@TODO set d and w somehow
    this->Ys = new std::vector<sketch::seq::count_min_sketch>(n, sketch::seq::count_min_sketch(d, w));

    for(size_t i = 0; i < seeds->size(); i++) {
        this->Ys->at(i).add_vec(seeds->at(i));
    }
}
