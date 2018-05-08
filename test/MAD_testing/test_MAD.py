'''
Timing code for MAD-Sketch code

Doesn't check for accuracy, that must be done by hand
using MRR.py
'''

import subprocess
import time
import copy



executable = '../../build/run_mad_sketch'
par_executable = '../../build/run_mad_sketch_omp'
edges_file_1 = '../data/graph_data/processed_freebase_1.graph'
edges_file_2 = '../data/graph_data/processed_freebase_2.graph'
edges_file_3 = '../data/graph_data/bigGraph.graph'

seeds_file_12 = '../data/graph_data/processed_seed1_2.txt'
seeds_file_110 = '../data/graph_data/processed_seed1_10.txt'
seeds_file_22 = '../data/graph_data/processed_seed2_2.txt'
seeds_file_210 = '../data/graph_data/processed_seed2_10.txt'
seeds_file_3 = '../data/graph_data/bigGraph_seeds.txt'

eval_file_12 = '../data/graph_data/processed_eval1_2.txt'
eval_file_110 = '../data/graph_data/processed_eval1_10.txt'
eval_file_22 = '../data/graph_data/processed_eval2_2.txt'
eval_file_210 = '../data/graph_data/processed_eval2_10.txt'
eval_file_3 = '../data/graph_data/empty_eval.txt'

graph12  = [executable, edges_file_1, seeds_file_12, eval_file_3, '2', '1', '23']
graph110 = [executable, edges_file_1, seeds_file_110, eval_file_3, '10', '1', '23']
graph22  = [executable, edges_file_2, seeds_file_22, eval_file_3, '2', '1', '23']
graph210 = [executable, edges_file_2, seeds_file_210, eval_file_3, '10', '1', '23']
graph3   = [executable, edges_file_3, seeds_file_3, eval_file_3, '10', '1', '50']

graphs = [graph12, graph110, graph22, graph210, graph3]

def getExecTime(params):
    start = time.time()
    p = subprocess.Popen(params)
    p.wait()
    return time.time() - start

res = ''

for graph in graphs:
    graph_reading = copy.deepcopy(graph)
    graph_reading[4] = '0'

    single_read = getExecTime(graph_reading)
    single_time = getExecTime(graph) - single_read

    graph[0] = par_executable
    graph_reading[0] = par_executable

    par_read = getExecTime(graph_reading)
    par_time = getExecTime(graph) - par_read


    res += ('Single Threaded Time Taken: %f\n' % single_time)
    res += ('Parallel  Time Taken: %f\n' % par_time)
    res += ('Speedup: %f\n\n' % (single_time / par_time))

print res
