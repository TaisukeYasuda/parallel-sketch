'''
Regression Tests for Sketching Algorithms

This testing suite checks for correctness of sketching algorithms by checking
subspace embedding guarantees for small test cases.
'''

import subprocess
import os
import filecmp

executable = './test_sketch'
par_executable = '.test_sketch_cuda'
test_files = ['small_test%d.txt']
infile_temps = './data/random_matrices/%s'
res_dir = './results/'
outfile_temps = res_dir + '%s/ref_%s.res'
sketch_type = ['gaussian_sketch', 'count_sketch', 'leverage_score_sketch']
num_tests = 10
s_types = ['_seq', '_par']

# make directories
for sketch in sketch_type:
    dir_to_make = res_dir + sketch
    if not os.path.exists(dir_to_make):
        os.makedirs(dir_to_make)

# make reference files
for sketch in sketch_type:
    for i in xrange(num_tests):
        for test in test_files:
            test = test % i
            infile  = infile_temps % test
            outfile = outfile_temps % (sketch, test)

            p = subprocess.Popen([executable, infile, outfile, sketch])
            p.wait()
