import subprocess
import shlex

executable = './test_sketch'
infile = 'data/random_matrices/small_test0.txt'
outfile = 'asdf.txt'
sketch_types = [
    'count_sketch',
    'gaussian_sketch',
    'leverage_score_sketch'
]

for sketch_type in sketch_types:
    cmd = '{} {} {} {}'.format(executable, infile, outfile, sketch_type)
    args = shlex.split(cmd)
    subprocess.call(args)
