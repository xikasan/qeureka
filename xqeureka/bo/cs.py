# -*- coding: utf-8 -*-

import numpy as np
from .model import BOModel
from .xvec import expand_statevector
from ..utilities import Vector, Matrix, is_1d, is_2d


class BOCS(BOModel):

    def __init__(
            self,
            num_qubit: int,
            sigma: float = 1.
    ):
        super().__init__(num_qubit)

        if sigma <= 0:
            raise ValueError(f"sigma must be positive float, but {sigma} is given.")
        self.sigma = sigma

        # predicted qubo data
        self.a: Vector = None

    def fit(self, xs: Vector | Matrix, ys: Vector | Matrix) -> Vector:
        X = expand_statevector(xs)
        Y = np.expand_dims(ys, -1)

        # calc mean and cov of qubo
        A = X.T @ X + np.eye(X.shape[1]) / self.sigma
        A_inv = np.linalg.inv(A)
        mean = np.squeeze(A_inv @ X.T @ Y)
        cov = self.sigma ** 2 * A_inv

        # sample a qubo as vector
        a = np.random.multivariate_normal(mean, cov)
        self.a = a
        return np.mean(np.diag(cov))

    def to_qubo(self, a: Vector = None) -> Matrix:
        n = self.n

        # homo term
        Q = np.diag(a[1:self.n+1])

        # cross term
        counter = n + 1
        for i in range(n - 1):
            Q[i, i+1:] = a[counter:counter + n - i - 1]
            counter += n - i - 1
        
        return Q

    def get_qubo(self) -> Matrix:
        return self.to_qubo(self.a)
