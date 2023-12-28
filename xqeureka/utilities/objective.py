# -*- coding: utf-8 -*-

import numpy as np
from typing import Any, Final
from numbers import Number
from .typing import Vector, Matrix, is_1d, is_2d, is_square


class ObjectiveFunction:

    xrange = None
    xbest = None
    best_value = None
    min_dim = 1

    def __init__(self):
        pass

    @staticmethod
    def generate():
        raise NotImplementedError()


class QuadradicBinaryObjectiveFunction(ObjectiveFunction):

    xrange = [-np.inf, np.inf]
    min_dim = 1

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


class QuadradicContinuousObjectiveFunction(ObjectiveFunction):

    __xrange = [-1, 1]
    min_dim = 1

    def __init__(self, A: Matrix, b: float):
        if not is_2d(A):
            raise ValueError(f"Q must be 2D matrix, but Q with shape {A.shape} is given.")
        self.__A: Final[Matrix] = A
        self.__b: Final[float] = b
        self.__n: Final[int] = A.shape[0]

        self.xrange = np.array([self.__xrange,] * self.n)
        __xbest = [-0.5 * A[i, 1] / A[i, 0] for i in range(self.n)]
        self.xbest = np.array([
            xb if self.__xrange[0] <= xb <= self.__xrange[1] else np.sign(xb)
            for xb in __xbest
        ])
        self.best_value = sum([-0.25 * A[i, 1]**2 / A[i, 0] for i in range(self.n)])
    
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
    def generate(n: int, mu=0., var=1.) -> "QuadradicContinuousObjectiveFunction":
        A = np.random.normal(mu, var, (n, 2))
        A[:, 0] = np.abs(A[:, 0])
        b = np.random.normal(mu, var, n)
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
        return self.__n


class SphereFunction(ObjectiveFunction):

    __xrange = [-5, 5]
    __xbest = 0.
    __ybest = 0.
    min_n = 1

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.xrange = np.array([self.__xrange,] * n)
        self.xbest = np.array([self.__xbest, ] * n)
        self.best_value = self.__ybest
    
    def __call__(self, x: Vector | Matrix) -> Number | Vector:
        x = np.array(x)
        if is_1d(x):
            x = np.expand_dims(x, 0)
        
        ys = np.square(x).sum(axis=1)

        if len(ys) == 1:
            return ys[0]
        return ys

    @staticmethod
    def generate(n: int) -> "SphereFunction":
        return SphereFunction(n)


class RosenbrockFunction(ObjectiveFunction):

    __xrange = [-5, 5]
    __xbest = 1.
    __ybest = 0.
    min_dim = 2

    def __init__(self, n: int, a: float = 1, b: float = 100):
        super().__init__()
        if not (n >= self.min_dim):
            raise ValueError(f"{self.__class__.__name__} requires n >= 2, but n={n} is given.")
        self.n = n
        self.a = a
        self.b = b
        self.xrange = np.array([self.__xrange,] * n)
        self.xbest = np.array([self.__xbest,] * n)
        self.best_value = self.__ybest

    def __call__(self, x: Vector | Matrix) -> Number | Vector:
        x = np.asarray(x)
        if is_1d(x):
            x = np.expand_dims(x, 0)
        
        ys = np.squeeze([
            self.b * (x[:, i+1] - x[:, i] ** 2) ** 2 + self.a * (x[:, i] - 1) ** 2
            for i in range(self.n - 1)
        ])
        return ys.tolist()

    @staticmethod
    def generate(n: int, a: float = 1, b: float = 100.) -> "RosenbrockFunction":
        return RosenbrockFunction(n, a, b)


class AckleyFunction(ObjectiveFunction):

    __xrange = [-32.768, 32.768],
    __xbest = 0
    __ybest = 0
    min_dim = 1

    def __init__(self, n: int, a: float = 20., b: float = 0.2, c: float = 2 * np.pi):
        super().__init__()
        self.n = n
        self.a = a
        self.b = b
        self.c = c
        self.xrange = np.array([self.__xrange,] * n)
        self.xbest = np.array([self.__xbest,] * n)
        self.best_value = self.__ybest
    
    def __call__(self, x: Vector | Matrix) -> Number | Vector:
        x = np.asarray(x)
        if is_1d(x):
            x = np.expand_dims(x, 0)
        
        bsqrtx2 = self.b * np.sqrt(np.square(x).mean(axis=1))
        cosbx = np.cos(self.b * x).mean(axis=1)

        ys = np.squeeze(
            self.a - self.a * np.exp(-1 * bsqrtx2) - np.exp(cosbx) + np.exp(1)
        )
        return ys.tolist()
    
    @staticmethod
    def generate(n: int, a: float = 20., b: float = 0.2, c: float = 2 * np.pi) -> "AckleyFunction":
        return AckleyFunction(n, a, b, c)


class BealeFunction(ObjectiveFunction):

    a1 = 1.5
    a2 = 2.25
    a3 = 2.625

    xrange = np.array([
        [-4.5, 4.5],
        [-4.5, 4.5]
    ])
    xbest = np.array([3, 0.5])
    best_value = 0
    min_dim = 2

    def __init__(self, n: int = None):
        super().__init__()
    
    def __call__(self, x: Vector | Matrix) -> Number | Vector:
        x = np.asarray(x)
        if is_1d(x):
            x = np.expand_dims(x, 0)

        x1 = x[:, 0]
        x2 = x[:, 1]
        y1 = np.square(self.a1 - x1 + x1 * x2)
        y2 = np.square(self.a1 - x1 + x1 * (x2 ** 2))
        y3 = np.square(self.a1 - x1 + x1 * (x2 ** 3))
        ys = y1 + y2 + y3
        return np.squeeze(ys)

    @staticmethod
    def generate() -> "BealeFunction":
        return BealeFunction()
