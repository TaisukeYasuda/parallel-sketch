#include "sketch.hpp"
#include "MAD_sketch.hpp"
#include <vector>
#include <fstream>
#include <string>
#include <utility>
#include <iostream>
#include <string>

#define GRAPH 1
#define SEEDS 2
#define EVAL  3
#define ITERS 4
#define D     5
#define W     6

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    size_t nodes, edges, num_seeds;

    std::ifstream graph_file(argv[GRAPH]);
    graph_file >> nodes >> edges;

    std::vector< std::pair< std::pair<size_t, size_t>, double> > edge_list(edges);
    size_t u, v, w, d, l, s_size, num_labels, temp;
    for(size_t i = 0; i < edges; i++) {
        graph_file >> u >> v >> w;
       
        if(u > v) {
            temp = v;
            v = u;
            u = temp;
        }

        edge_list[i].first.first = u;
        edge_list[i].first.second = v;
        edge_list[i].second = w;
    }
    
    std::cout << "Read Graph!" << std::endl;

    std::ifstream seed_file(argv[SEEDS]);
    seed_file >> num_seeds >> num_labels;

    s_size = std::stoi(argv[W]);
    d = std::stoi(argv[D]);
    size_t hashes[1] = {0};

    std::vector< sketch::seq::count_min_sketch<double> > seeds(nodes,
        sketch::seq::count_min_sketch<double> (d, s_size, hashes));

    for(size_t i = 0; i < num_seeds; i++) {
        seed_file >> u >> l;
        seeds[u].add(l, 1.0);
    }

    std::cout << "Read Seeds!" << std::endl;

    double *p_inj  = new double[nodes];
    double *p_cont = new double[nodes];
    double *p_abnd = new double[nodes];

    sketch::seq::count_min_sketch<double> r(d, s_size, hashes);

    for(size_t i = 0; i < nodes; i++) {
        p_inj[i]  = 3.0;
        p_cont[i] = 3.0;
        p_abnd[i] = 3.0;
    }

    MAD_sketch SSL(nodes, d, s_size, 0.5, 0.5, 0.5, p_inj, p_cont, p_abnd,
        &edge_list, &seeds, &r);
 
    std::cout << "Made MAD Sketch!" << std::endl;

    SSL.run_sim(std::stoi(argv[ITERS]));
   
    std::cout << "Ran Sim!" << std::endl;

    std::vector< sketch::seq::count_min_sketch<double> > *res = SSL.get_labels(hashes);
    
    std::ifstream eval_file(argv[EVAL]);
    std::ofstream res_file("result.txt");
    size_t eval_nodes;
    eval_file >> eval_nodes;

    std::vector< std::pair<double, size_t> > labels(num_labels);
    for(size_t i = 0; i < eval_nodes; i++) {
        eval_file >> u;
        for(size_t j = 0; j < num_labels; j++) {
            labels[j].first  = res->at(u).get(j);
            labels[j].second = j;
        }
        
        std::sort(labels.begin(), labels.end());
        res_file << u << '\t';
        for(int j = num_labels-1; j >= 0; j--){
            res_file << labels[j].second << '\t';

        }
        
        res_file << std::endl;
    }

    std::cout << "Output Labels!" << std::endl;

    return 0;
}

