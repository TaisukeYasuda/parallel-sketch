from os import path
import numpy as np

test_dir = 'random_matrices'
test_files = [
    'test0.txt',
    'test1.txt',
    'test2.txt',
    'test3.txt',
    'test4.txt',
    'test5.txt',
    'test6.txt',
    'test7.txt',
    'test8.txt',
    'test9.txt'
]
matrix_dim_rows = 500
matrix_dim_cols = 50

for test_file in test_files:
    filename = path.join(test_dir, 'med_{}'.format(test_file))
    np.savetxt(filename, np.random.rand(matrix_dim_rows, matrix_dim_cols))
