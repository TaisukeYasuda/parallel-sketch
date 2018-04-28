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
outfile_temps = res_dir + '%s/%s.res'
sketch_type = ['gaussian_sketch', 'count_sketch', 'leverage_score_sketch']
num_tests = 10
s_types = ['_seq', '_par']

# Make reference files
for sketch in sketch_type:
    for i in range(num_tests):
        for test in test_files:
            test = test % i
            infile  = infile_temps % test
            outfile_seq = outfile_temps % (sketch + s_types[0], test)

            p = subprocess.Popen([executable, infile, outfile_seq, sketch])
            p.wait()
            print (sketch + ' ' + test + ' same?: ' + str(filecmp.cmp(outfile_seq, outfile_par)))

'''#Output parallel results
for sketch in sketch_type:
    for i in range(num_tests):
        for test in test_files:
            test = test % i
            infile  = infile_temps % test
            outfile = outfile_temps % (sketch + s_types[1], test)
            #print executable + ' ' + infile + ' ' + outfile + ' ' + sketch
            #TODO change executable file
            subprocess.call([executable, infile, outfile, sketch])

#Compare files
for sketch in sketch_type:
    for i in range(num_tests):
        for test in test_files:
            test = test % i
            outfile_seq = outfile_temps % (sketch + s_types[0], test)
            outfile_par = outfile_temps % (sketch + s_types[1], test)
            print (test + ' same?: ' + str(filecmp.cmp(outfile_seq, outfile_par)))'''
