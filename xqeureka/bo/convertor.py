# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, List

from xqeureka.utilities.typing import Vector, Matrix, is_1d, is_2d


class Convertor:

    def __init__(self, ss: List[Vector]) -> None:
        self.ss = [np.asarray(si) for si in ss]
        self.nbits = [len(si) for si in self.ss]
    
    def encode(self, xs: Matrix | Vector) -> List[List[int]] | List[int]:
        xs = np.asarray(xs)
        if is_1d(xs):
            return self._encode_single_state(xs)
        return [
            self._encode_single_state(x)
            for x in xs
        ]
    
    def _encode_single_state(self, x: Vector):
        subzs = [self._encode_element(xi, si) for xi, si in zip(x, self.ss)]
        return np.hstack(subzs).tolist()
    
    def _encode_element(self, xi, si):
        distance = si - xi
        distance2 = distance ** 2
        closest_point_index = np.argmin(distance2)
        zi = np.zeros_like(si).astype(int)
        zi[closest_point_index] = 1
        return zi

    def decode(self, zs: Matrix | Vector) -> List[List[float]] | List[float]:
        zs = np.asarray(zs)
        if is_1d(zs):
            return self._decode_single_state(zs)
        return [
            self._decode_single_state(z)
            for z in zs
        ]
    
    def _decode_single_state(self, z):
        subzs = self._divide_in_var(z)
        x = [self._decode_element(zi, si) for zi, si in zip(subzs, self.ss)]
        return x
    
    def _decode_element(self, zi, si):
        return np.dot(zi, si)
    
    def _divide_in_var(self, z):
        return [
            z[int(np.sum(self.nbits[:i])):int(np.sum(self.nbits[:i+1]))]
            for i in range(len(self.nbits))
        ]


class AdaptiveConvertor(Convertor):

    def __init__(self, ss: List[Vector]) -> None:
        super().__init__(ss)
    
    def update(self, x, y, **kwargs):
        raise NotImplementedError()


class DistanceBasedAdaptiveConvertor(AdaptiveConvertor):

    def __init__(self, ss: List[Vector], eta: float = 0.3, sigma: float = 1.) -> None:
        super().__init__(ss)
        if not(eta > 0):
            raise ValueError(f"Adaptive rate eta must be positive float, but {eta} is given.")
        self.eta = eta
        self.sigma = sigma
    
    def update(self, x, y, **kwargs):
        dss = []
        z = self.encode(x)
        for i in range(len(x)):
            si = self.ss[i]
            xi = x[i]
            zi = self._encode_element(xi, si)
            ds = self.calc_update_rate(xi, si)
            dss.append(ds)

            kp = np.argmax(zi)
            new_si = [si[0]]

            for k in range(1, len(zi)-1):
                if k == kp:
                    new_si.append(si[k])
                    continue

                if k < kp:
                    new_si.append((1 - ds[k]) * si[k] + ds[k] * si[k+1])
                    continue

                if k > kp:
                    new_si.append((1 - ds[k]) * si[k] + ds[k] * si[k-1])
                    continue
            
            new_si.append(si[-1])
            
            self.ss[i] = np.asarray(new_si)
         
        return dss

    def calc_update_rate(self, xi, si):
        di = si - xi
        nd = np.exp(-0.5 * di ** 2 / self.sigma ** 2)
        return nd
