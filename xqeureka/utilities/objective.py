# -*- coding: utf-8 -*-

import numpy as np
from typing import Final
from numbers import Number
from .typing import Vector, Matrix, is_1d, is_2d, is_square


class ObjectiveFunction:

    def __init__(self):
        pass

    @staticmethod
    def generate(self):
        raise NotImplementedError()


class QuadradicBinaryObjectiveFunction(ObjectiveFunction):

    def __init__(self, Q: Matrix):
        if not is_square(Q):
            raise ValueError(f"Q must be 2D square matrix, but Q with shape {Q.shape} is given.")
        self.__Q: Final[Matrix] = Q
        self.__n: Final[int] = Q.shape[0]
    
    def __call__(self, x: Vector | Matrix) -> Number | Vector:
        x = np.asarray(x)
        if is_1d(x):
            x = np.expand_dims(x, 0)
        xQx = x @ self.Q @ x.T
        ys = np.diag(xQx)
        if len(ys) == 1:
            return ys[0]
        return ys

    @staticmethod
    def generate(n: int, mu=0., var=1.) -> "QuadradicBinaryObjectiveFunction":
        Qt = np.random.normal(mu, var, (n, n))
        return QuadradicBinaryObjectiveFunction(Qt)
    
    @property
    def Q(self) -> Matrix:
        return self.__Q

    @Q.setter
    def Q(self, val: Matrix) -> None:
        return None 
    
    @property
    def n(self) -> int:
        return self._n


class QuadradicContinuousObjectiveFunction:

    def __init__(self, A: Matrix, b: float):
        if not is_2d(A):
            raise ValueError(f"Q must be 2D matrix, but Q with shape {A.shape} is given.")
        self.__A: Final[Matrix] = A
        self.__b: Final[float] = b
        self.__n: Final[int] = A.shape[0]
    
    def __call__(self, x: Vector | Matrix) -> Number | Vector:
        x = np.asarray(x)
        if is_1d(x):
            x = np.expand_dims(x, 0)
        
        xs = self.format_input(x)
        ys = np.tensordot(xs, self.A, axes=2).tolist()
        if len(ys) == 1:
            return ys
        return ys

    @staticmethod
    def generate(n: int, mu=0., var=1.) -> "QuadradicObjectiveFunction":
        A = np.random.normal(mu, var, (n, 2))
        b = np.random.normal(mu, var, 1)
        return QuadradicContinuousObjectiveFunction(A, b)

    def format_input(self, x):
        x2 = self.power(x, 2)
        xs = np.asarray([x2, x]).transpose((1, 2, 0))
        return xs

    @staticmethod
    def power(x, p):
        return x ** p
    
    @property
    def A(self) -> Matrix:
        return self.__A.copy()
    
    @property
    def b(self) -> float:
        return self.__b
    
    @property
    def n(self) -> int:
        return self._n
