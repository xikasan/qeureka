# -*- coding: utf-8 -*-

import numpy as np
from ..utilities import Vector, Matrix, is_1d, is_2d


class BOModel:

    def __init__(
            self,
            num_qubit: int,
    ):
        if num_qubit <= 0:
            raise ValueError(f"num_qubit must be positive integer, but {num_qubit} is given.")
        self.n = n = num_qubit
        self.p = 1 + n + n * (n - 1) // 2
    
    def fit(self, xs: Vector | Matrix, ys: Vector | Matrix) -> Vector:
        raise NotImplementedError()
    
    def get_qubo(self):
        raise NotImplementedError()

