#!/usr/bin/python

import sys
import numpy as np

matrix_dim_rows = int(sys.argv[1])
matrix_dim_cols = int(sys.argv[2])

np.savetxt('test{}x{}.txt'.format(matrix_dim_rows, matrix_dim_cols),
        np.random.rand(matrix_dim_rows, matrix_dim_cols), delimiter=',')
