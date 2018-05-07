#!/usr/bin/python

import numpy as np

vector_dim = 5

for i in xrange(10):
    for scale in xrange(5, 100+1, 5):
        filename = 'small_test_{}_scale_{}.txt'.format(i, scale)
        np.savetxt(filename, scale * np.random.rand(vector_dim, 1), delimiter=',')
