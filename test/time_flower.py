import subprocess
import time

seq_exe = './test_sketch'
par_exe = './test_sketch_cuda'
flower = 'data/images/flower.txt'
output = 'asdf.txt'
sketch = 'leverage_score_sketch'

start = time.time()
p = subprocess.Popen([seq_exe, flower, output, sketch])
p.wait()
end = time.time()
seq_time = end - start
print 'Sequential time: {} seconds'.format(seq_time)

start = time.time()
p = subprocess.Popen([par_exe, flower, output, sketch])
p.wait()
end = time.time()
par_time = end - start
print 'Parallel time: {} seconds'.format(par_time)

print 'Speedup: {}'.format(seq_time / par_time)
