import numpy as np
from scipy.misc import imread

test_files = ['bibimbap', 'flower']
for test_file in test_files:
    img = imread('{}.jpg'.format(test_file), mode='L')
    np.savetxt('{}.txt'.format(test_file), img, delimiter=',')
