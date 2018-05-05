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

inline size_t *make_hashes(size_t w) {
    std::random_device rd;
    std::mt19937 mt(rd());

    //Hash function will be identiy xor random int
    std::uniform_int_distribution<size_t> rand_hash;

    size_t *hashes = new size_t[w];

    for(size_t i = 0; i < d; i++)
       hashes[i] = rand_hash(mt);

    return hashes;
}

MAD_sketch(size_t n_, size_t d_, size_t w_, double mu_1_, double mu_2_, double mu_3_,
    double *p_inj_, double *p_cont_, double *p_abnd_,
    std::vector< std::pair< std::pair<size_t, size_t>, size_t> > *edge_list_,
    std::vector< sketch::seq::count_min_sketch<double> > *seeds_); {

    std::sort(edge_list_->begin(), edge_list_->end());

    this->Ms = new double[n_];

    this->mu_1 = mu_1_;
    this->mu_2 = mu_2_;
    this->mu_3 = mu_3_;
    
    this->p_inj  = p_inj_;
    this->p_abnd = p_abnd_;
    
    this->d = d_;
    this->w = w_;
    this->n = n_;

    this->num_edges = edge_list_->size();

    this->edges = new size_t[this->num_edges * 2];
    this->edge_factors = new size_t[this->num_edges]
    this->Mvv   = new double[n_];

    memset(this->Mvv, 0, n_ * sizeof(double));

    size_t u, v, weight;
    double u_val, v_val;
    
    for(size_t i = 0; i < this->num_edges; i++) {
        //Initialize graph
        u = edge_list->at(i).first.first;
        v = edge_list->at(i).first.second;
        weight = edge_list->at(i).second;

        this->edges[i]   = u
        this->edges[i+1] = v;

        u_val = p_cont[u] * weight;
        v_val = p_cont[v] * weight;

        this->edge_factors[i] = u_val + v_val;       

        //Modify Mvv
        if(u != v) {
            this->Mvv[u] += this->edge_factors[i];
            this->Mvv[v] += this->edge_factors[i];
        }
    }

    size_t sketch_size = d_ * w_;
    size_t seed_size = sketch_size * n_;

    //Initialize labels
    this->Ys    = new double[seed_size];
    this->seeds = new double[seed_size];
    this->rs    = new double[seed_size];

    memset(this->rs, 0, seed_size * sizeof(double));
    for(size_t i = 0; i < n; i++){
        this->Mvv[i] = 1.0 / ((this->Mvv->at(i) * mu_2) + p_inj->at(i) * mu_1 + mu_3);
    
        memcpy(this->Ys,    seeds->at(i).get_CM(), sketch_size * sizeof(double));
        memcpy(this->seeds, seeds->at(i).get_CM(), sketch_size * sizeof(double));
    }
}

void MAD_sketch::run_sim(size_t iters) {
    size_t sketch_size = this->d * this->w;
    size_t seed_size = sketch_size * this->n;

    double *temp_D = new double[seed_size];

    size_t start_u, start_v;
    for(size_t z = 0; z < iters; z++) {
        memset(temp_D, 0, seed_size * sizeof(double));
        
        for(size_t i = 0; i < this->num_edges; i++) {
            u = this->edges[i];
            v = this->edges[i+1];
            
            double factor = this->edge_factors[i];

            start_u = u * sketch_size;
            start_v = v * sketch_size;
            for(size_t j = 0; j < sketch_size; j++) {
               temp_D[start_u + j] += factor * this->Ys[start_v + j];
               temp_D[start_v + j] += factor * this->Ys[start_u + j];
            }
        }
        
        double M_factor, D_factor, seed_factor, r_factor;
        for(size_t i = 0; i < n; i++) {
            M_factor = this->Mvv[i];
            
            seed_factor = this->mu_1 * this->p_inj[i] * M_factor;
            D_factor = this->mu_2 * M_factor;
            r_factor = this->mu_3 * this->p_abnd[i] * M_factor;

            start_v = i * sketch_size;
            for(size_t j = 0; j < sketch_size; j++) {
                this->Ys[start_v + j] = (seed_factor * this->seeds[start_v + j] +
                                         D_factor * temp_D[start_v + j] +
                                         r_factor * this->rs[start_v + j]);
            }
        }
    }
}

//@TODO copy back to sketches
std::vector< sketch::seq::count_min_sketch<double> > *get_labels() {
    return Ys;    
}
