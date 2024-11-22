import numpy as np
from solution import convolve_numpy

if __name__ == '__main__':
    inputs = np.array([[[
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]]]])

    kernels = np.array([[[[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]]]
                       )

    print(convolve_numpy(inputs, kernels, 0))
    print(convolve_numpy(inputs, kernels, 1))
    print(convolve_numpy(inputs, kernels, 2))
    print(convolve_numpy(inputs, kernels, 3))
    print(convolve_numpy(inputs, kernels, 4))
    print(convolve_numpy(inputs, kernels, 5))
