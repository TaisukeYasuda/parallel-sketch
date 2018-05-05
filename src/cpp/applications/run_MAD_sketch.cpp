#include "sketch.hpp"
#include "MAD_sketch.hpp"
#include <vector>
#include <fstream>
#include <string>
#include <utility>
#include <iostream>

#define GRAPH 1
#define SEEDS 2
#define EVAL  3

int main(int argc, char *argv[]) {
    size_t nodes, edges, num_seeds;

    std::ifstream graph_file(argv[GRAPH]);
    graph_file >> nodes >> edges;

    std::vector< std::pair< std::pair<size_t, size_t>, double> > edge_list(edges);
    size_t u, v, d, l, s_size, hashes, num_labels;
    for(size_t i = 0; i < edges; i++) {
        graph_file >> edge_list[i].first.first;
        graph_file >> edge_list[i].first.second;
        graph_file >> edge_list[i].second;
    }
    
    std::cout << "Read Graph!" << std::endl;

    s_size = 23;
    d = 1;
    hashes = 0;

    std::vector< sketch::seq::count_min_sketch<double> > seeds(nodes,
        sketch::seq::count_min_sketch<double> (d, s_size, &hashes));

    std::ifstream seed_file(argv[SEEDS]);
    seed_file >> num_seeds >> num_labels;

    for(size_t i = 0; i < num_seeds; i++) {
        seed_file >> u >> l;
        seeds[u].add(l, 1.0);
    }

    std::cout << "Read Seeds!" << std::endl;

    double *p_inj  = new double[nodes];
    double *p_cont = new double[nodes];
    double *p_abnd = new double[nodes];

    sketch::seq::count_min_sketch<double> r(d, s_size, &hashes);

    for(size_t i = 0; i < nodes; i++) {
        p_inj[i]  = 1.0;
        p_cont[i] = 1.0;
        p_abnd[i] = 1.0;
    }

    MAD_sketch SSL(nodes, d, s_size, 0.5, 0.5, 0.5, p_inj, p_cont, p_abnd,
        &edge_list, &seeds, &r);
 
    std::cout << "Made MAD Sketch!" << std::endl;

    SSL.run_sim(2);
   
    std::cout << "Ran Sim!" << std::endl;

    std::vector< sketch::seq::count_min_sketch<double>* > *res = SSL.get_labels(&hashes);
    
    std::ifstream eval_file(argv[EVAL]);
    std::ofstream res_file("result.txt");
    size_t eval_nodes;
    eval_file >> eval_nodes;

    std::vector< std::pair<double, size_t> > labels(eval_nodes);
    for(size_t i = 0; i < eval_nodes; i++) {
        eval_file >> u;
        for(size_t j = 0; j < num_labels; j++) {
            labels[j].first  = res->at(i)->get(j);
            labels[j].second = j;
        }

        std::sort(labels.begin(), labels.end());
        res_file << u <<  '\t';
        for(size_t j = 0; j < num_labels; j++) {
            res_file << labels[j].second << ' ';
        }
        res_file << std::endl;
    }

    return 0;
}

