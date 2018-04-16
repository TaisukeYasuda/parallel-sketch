import numpy as np

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
matrix_dim_rows = 20
matrix_dim_cols = 5

for test_file in test_files:
    filename = 'small_{}'.format(test_file)
    np.savetxt(filename, np.random.rand(matrix_dim_rows, matrix_dim_cols),
            delimiter=',')
