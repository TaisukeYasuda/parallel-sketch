import subprocess

executable = '../build/test_sketch'
infile = 'data/random_matrices/small_test0.txt'
outfile = 'asdf.txt'
sketch_type = 'gaussian_sketch'

cmd = '{} {} {} {}'.format(executable, infile, outfile, sketch_type)
print cmd
#subprocess.Popen(cmd)
subprocess.call([executable, infile, outfile, sketch_type])
