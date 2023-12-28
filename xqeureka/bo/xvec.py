# -*- coding: utf-8 -*-

import numpy as np
from ..utilities import Vector, Matrix, is_1d, is_2d


def random_binary_state(size: int, num: int = 1) -> Vector:
    if size <= 0:
        raise ValueError(f"size must be a positive integer, but {size} is given.")
    
    x = np.random.choice([0, 1], (num, size), replace=True)
    return x.squeeze()


def make_xcombi_vec(x: Vector) -> Vector:
    x = np.asarray(x)
    if not is_1d(x) and x.shape[0] > 1:
        raise ValueError(f"x must be vector, but matrix with shape {x.shape} is given.")

    # size of variable
    n = len(x)
    # size of variable in expanded representation
    p = 1 + n + n * (n - 1) // 2

    # prepare combination
    x = x.reshape((1, n))
    xx = x.T * x

    xvec = np.zeros(p)
    # constant bias term
    xvec[0] = 1
    # sigle term
    xvec[1:n+1] = x
    # combination term
    counter = 1 + n
    for i in range(n - 1):
        xvec[counter:counter+n-i-1] = xx[i, i+1:]
        counter += n - i - 1
    
    return xvec


def expand_statevector(x: Matrix) -> Matrix:
    x = np.asarray(x)
    if is_1d(x):
        return make_xcombi_vec(x)
    
    if is_2d(x):
        return np.vstack([
            make_xcombi_vec(x_)
            for x_ in x
        ])
    
    raise ValueError(f"x must be 2D matrix, but x with shape {x.shape} is given.")
