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
#include <random>
#include <omp.h>

#include<iostream>

MAD_sketch::MAD_sketch(size_t n_, size_t d_, size_t w_, double mu_1_, double mu_2_, double mu_3_,
    double *p_inj_, double *p_cont_, double *p_abnd_,
    std::vector< std::pair< std::pair<size_t, size_t>, double> > *edge_list_,
    std::vector< sketch::seq::count_min_sketch<double> > *seeds_,
    sketch::seq::count_min_sketch<double> *r_) {

    std::sort(edge_list_->begin(), edge_list_->end());

    this->Mvv = new double[n_];

    this->mu_1 = mu_1_;
    this->mu_2 = mu_2_;
    this->mu_3 = mu_3_;

    this->p_inj  = new double[n_];
    this->p_abnd = new double[n_];

    memcpy(this->p_inj , p_inj_ , n_ * sizeof(double));
    memcpy(this->p_abnd, p_abnd_, n_ * sizeof(double));

    this->d = d_;
    this->w = w_;
    this->n = n_;

    this->num_edges = edge_list_->size();

    this->edges = new size_t[this->num_edges * 2];
    this->edge_factors = new double[this->num_edges];
    this->Mvv = new double[n_];

    memset(this->Mvv, 0, n_ * sizeof(double));

    #pragma omp parallel
    {
    size_t u, v, weight, idx;
    double u_val, v_val;

    #pragma omp for schedule(static) 
    for(size_t i = 0; i < this->num_edges; i++) {
        //Initialize graph
        u = edge_list_->at(i).first.first;
        v = edge_list_->at(i).first.second;
        weight = edge_list_->at(i).second;

        idx = i*2;

        this->edges[idx]   = u;
        this->edges[idx+1] = v;

        u_val = p_cont_[u] * weight;
        v_val = p_cont_[v] * weight;

        this->edge_factors[i] = u_val + v_val;

        //Modify Mvv
        if(u != v) {
            #pragma omp atomic
            this->Mvv[u] += this->edge_factors[i];
            
            #pragma omp atomic
            this->Mvv[v] += this->edge_factors[i];
        }
    }

    }

    size_t sketch_size = d_ * w_;
    size_t seed_size = sketch_size * n_;

    //Initialize labels
    this->Ys    = new double[seed_size];
    this->seeds = new double[seed_size];
    this->r     = new double[sketch_size];

    memcpy(this->r, r_->get_CM(), sketch_size * sizeof(double));

    #pragma omp parallel
    {
    
    size_t offset;
    #pragma omp for schedule(static)
    for(size_t i = 0; i < n_; i++){
        this->Mvv[i] = 1.0 / ((this->Mvv[i] * mu_2) + p_inj_[i] * mu_1_ + mu_3_);
        offset = i * sketch_size;

        memcpy(this->Ys+offset,    seeds_->at(i).get_CM(), sketch_size * sizeof(double));
        memcpy(this->seeds+offset, seeds_->at(i).get_CM(), sketch_size * sizeof(double));

    }

    }

}

MAD_sketch::~MAD_sketch() {
    delete this->Ys;
    delete this->Mvv;
    delete this->seeds;
    delete this->r;
    delete this->p_inj;
    delete this->p_abnd;
    delete this->edges;
    delete this->edge_factors;
}

void MAD_sketch::run_sim(size_t iters) {
    size_t sketch_size = this->d * this->w;
    //size_t seed_size = sketch_size * this->n;

   // #pragma omp parallel
   // {
    
    size_t tid = omp_get_thread_num(),
           nt  = omp_get_num_threads();
    size_t start_range = 0, //= (tid * this->n) / nt,
           end_range = this-> n, //= ((tid+1) * this->n) / nt,
           elems = end_range - start_range,
           seed_range = elems * sketch_size;

    double *temp_D = new double[seed_range];
    
    size_t u, v, start_u, start_u_adj, start_v, idx;
    for(size_t z = 0; z < iters; z++) {
        memset(temp_D, 0, seed_range * sizeof(double));

        for(size_t i = 0; i < this->num_edges; i++) {
            idx = i * 2;
            u = this->edges[idx];
            v = this->edges[idx+1];

            double factor = this->edge_factors[i];

            if(start_range <= u && u < end_range) {
                start_u = (u - start_range) * sketch_size;
                start_v = v * sketch_size;
                
                for(size_t j = 0; j < sketch_size; j++) 
                    temp_D[start_u + j] += factor * this->Ys[start_v + j];
            }

            if(start_range <= v && v < end_range) {
                start_u = u * sketch_size;
                start_v = (v - start_range) * sketch_size;
                
                for(size_t j = 0; j < sketch_size; j++) 
                    temp_D[start_v + j] += factor * this->Ys[start_u + j];
            }
        }

        double M_factor, D_factor, seed_factor, r_factor;
        for(size_t i = start_range; i < end_range; i++) {
            M_factor = this->Mvv[i];

            seed_factor = this->mu_1 * this->p_inj[i] * M_factor;
            D_factor    = this->mu_2 * M_factor;
            r_factor    = this->mu_3 * this->p_abnd[i] * M_factor;

            start_u = i * sketch_size;
            start_u_adj = (i - start_range) * sketch_size;
            for(size_t j = 0; j < sketch_size; j++) {
                this->Ys[start_u + j] = (seed_factor * this->seeds[start_u + j] +
                                         D_factor * temp_D[start_u_adj + j] +
                                         r_factor * this->r[j]);
            }
        }

       // #pragma omp barrier
    }

    delete temp_D;

    //}

    
}

std::vector< sketch::seq::count_min_sketch<double> > *MAD_sketch::get_labels(size_t *hashes) {

    size_t sketch_size = this->d * this->w;
    std::vector< sketch::seq::count_min_sketch<double> > *res =
        new std::vector< sketch::seq::count_min_sketch<double> >(this->n,
        sketch::seq::count_min_sketch<double>(this->d, this->w, hashes));


    for(size_t i = 0; i < this->n; i++) {
        memcpy(res->at(i).get_CM(), this->Ys + (i * sketch_size),
            sketch_size * sizeof(double));
    }

    return res;
}
