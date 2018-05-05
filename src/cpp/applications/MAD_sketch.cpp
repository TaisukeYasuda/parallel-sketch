/*
 * MAD Sketch
 * 
 * Implementation of MAD sketch
 *
 */

#include "MAD_sketch.hpp"
#include "sketch.hpp"
#include <utility>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <random>

inline std::vector<unsigned long> *make_hashes(size_t w) {
    std::random_device rd;
    std::mt19937 mt(rd());

    //Hash function will be identiy xor random int
    std::uniform_int_distribution<unsigned long> rand_hash;

    std::vector<unsigned long> *hashes = new std::vector<unsigned long>(w);

    for(size_t i = 0; i < d; i++)
       (*hashes)[i] = rand_hash(mt);

    return hashes;
}

MAD_sketch(size_t n, size_t d, size_t w, double mu_1, double mu_2, double mu_3,
    std::vector<double> *p_inj, std::vector<double> *p_cont, std::vector<double> *p_abnd,
    std::vector< std::pair< std::pair<size_t,size_t>, unsigned int> > *edge_list,
    std::vector< std::vector<double> > *seeds); {

    std::sort(edge_list->begin(), edge_list->end());

    this->Ms = new std::vector<double>(n, 0);

    this->mu_1 = mu_1;
    this->mu_2 = mu_2;
    this->mu_3 = mu_3;
    this->p_inj = p_inj;
    this->p_abnd = p_abnd;

    this->edges = new std::vector<unsigned int>(edge_list->size() * 3);

    size_t idx, u, v, weight;
    
    for(size_t i = 0; i < edge_list->size(); i++) {
        //Initialize graph
        u = edge_list->at(i).first;
        v = edge_list->at(i).first.second;
        weight = edge_list->at(i).second;

        idx = i*3;
        (*(this->edges))[idx]   = u
        (*(this->edges))[idx+1] = v;
        (*(this->edges))[idx+2] = weight;

        //Modify Mvv
        double u_val = p_cont->at(u) * weight,
               v_val = p_cont->at(v) * weight;

        (*(this->temp_W_sums))[u] += u_val;
        (*(this->temp_W_sums))[v] += v_val;

        if(u != v) {
            (*(this->Mvv))[u] += u_val;
            (*(this->Mvv))[v] += v_val;
        }
    }

    //Initialize labels
    //TODO make sure these labels get copied
    this->hashes = make_hashes(w);
    this->Ys = new std::vector<sketch::seq::count_min_sketch<double> >(n,
            sketch::seq::count_min_sketch<double>(d, w, this->hashes));

    for(size_t i = 0; i < seeds->size(); i++) {
        this->Ys->at(i).add_vec(seeds->at(i));
        (*(this->Mvv))[i] = 1.0 / ((this->Mvv->at(i) * mu_2) + p_inj->at(i) * mu_1 + mu_3);
    }

}

void MAD_sketch::run_sim(size_t iters) {
    size_t n = this->Mvv->size();
    std::vector< sketch::seq::count_min_sketch<double> > temp_D(n,
            sketch::seq::count_min_sketch<double>(d, w, this->hashes));

    for(size_t z = 0; z < iters; z++) {
        for(size_t i = 0; i < n; i++)
            temp_D[i].reset();
        
        for(size_t i = 0; i < this->edges->size(); i++) {
            u = edge_list->at(i).first;
            v = edge_list->at(i).first.second;
            weight = edge_list->at(i).second;
        
            temp_D[u].saxpy_add(this->temp_W_sums[v], this->Ys->at(v));
            temp_D[v].saxpy_add(this->temp_W_sums[u], this->Ys->at(u));
        }
        
        for(size_t i = 0; i < n; i++) {
            temp_D[i].mult_const(this->mu_2);
            temp_D[i].saxpy_add(this->mu_1 * this->p_inj->at(i), this->Ys->at(i));
            temp_D[i].saxpy_add(this->mu_3 * this->p_abnd->at(i), this->rs->at(i));
            temp_D[i].
        }
    }
    
}

std::vector< sketch::seq::count_min_sketch<double> > *get_labels() {
    
}
