import subprocess
import os
import filecmp

executable = '../build/test_sketch'
par_executable = '../build/test_sketch'
test_files = ['small_test%d.txt', 'med_test%d.txt']
infile_temps = 'data/random_matrices/%s'
res_dir = 'results/'
outfile_temps = res_dir + '%s/%s.res'
sketch_type = ['gaussian_sketch', 'count_sketch']
num_tests = 10
s_types = ['_seq', '_par']

for s_type in s_types:
    for sketch in sketch_type:
        dir_to_make = res_dir + sketch + s_type
        if not os.path.exists(dir_to_make):
            os.makedirs(dir_to_make)

#TODO add timing code here
#Make reference files
for sketch in sketch_type:
    for i in range(num_tests):
        for test in test_files:
            test = test % i
            infile  = infile_temps % test
            outfile_seq = outfile_temps % (sketch + s_types[0], test)
            outfile_par = outfile_temps % (sketch + s_types[1], test)

            #print executable + ' ' + infile + ' ' + outfile + ' ' + sketch
            p = subprocess.Popen([executable, infile, outfile_seq, sketch])
            p.wait()
            p = subprocess.Popen([executable, infile, outfile_par, sketch])
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

