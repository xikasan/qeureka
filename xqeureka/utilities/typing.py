# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Final, List, TypeAlias
from numbers import Number
from numpy.typing import NDArray


Vector: TypeAlias = List[Number] | NDArray
Matrix: TypeAlias = List[Vector] | NDArray


def is_1d(x: Matrix | Vector) -> bool:
    x = np.asarray(x)
    return len(x.shape) == 1

def is_2d(x: Matrix) -> bool:
    x = np.asarray(x)
    return len(x.shape) == 2

def is_square(x: Matrix):
    x = np.asarray(x)
    xshape = x.shape
    return is_2d and xshape[0] == xshape[1]
