'''
Timing code for MAD-Sketch code

Doesn't check for accuracy, that must be done by hand
using MRR.py
'''

import subprocess
import time



executable = '../../build/run_mad_sketch'
par_executable = ''
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

graph12  = [executable, edges_file_1, seeds_file_12, eval_file_12, '2', '1', '23']
graph110 = [executable, edges_file_1, seeds_file_110, eval_file_110, '10', '1', '23']
graph22  = [executable, edges_file_2, seeds_file_22, eval_file_22, '2', '1', '23']
graph210 = [executable, edges_file_2, seeds_file_210, eval_file_210, '10', '1', '23']
graph3   = [executable, edges_file_3, seeds_file_3, eval_file_3, '10', '1', '50']


start = time.time()
p = subprocess.Popen(graph3)
p.wait()

time_taken = time.time() - start

print ('Time Taken: %f' % time_taken)
