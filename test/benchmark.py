import subprocess
import time

seq_exe = './test_sketch'
par_exe = './test_sketch_cuda'
test_dir = 'data/random_matrices/{}'
tests = [
    'test10x5.txt',
    'test100x50.txt',
    'test1000x50.txt',
    'test10000x100.txt'
]
par_output = 'par_output.txt'
seq_output = 'seq_output.txt'
sketch = 'leverage_score_sketch'

for test in tests:
    test_file = test_dir.format(test)

    print 'Testing {}'.format(test_file)
    start = time.time()
    p = subprocess.Popen([seq_exe, test_file, seq_output, sketch])
    p.wait()
    end = time.time()
    seq_time = end - start
    print 'Sequential time: {} seconds'.format(seq_time)

    start = time.time()
    p = subprocess.Popen([par_exe, test_file, par_output, sketch])
    p.wait()
    end = time.time()
    par_time = end - start
    print 'Parallel time: {} seconds'.format(par_time)

    print 'Speedup: {}'.format(seq_time / par_time)
