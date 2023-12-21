# -*- coding: utf-8 -*-

import xsim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable, List
from openjij import SQASampler
from IPython.display import clear_output

from .model import BOModel
from .xvec import random_binary_state
from ..utilities import Vector, Matrix, is_2d


class BlackboxOptimizer:

    def __init__(
            self,
            model: BOModel,
            objective: Callable = None,
            sampler_cls = SQASampler,
            num_reads: int = 10,
    ):
        # algo model
        self.model = model
        self.nq = model.n

        # problem
        self.obj_func = objective

        # loggings
        # sample/energy buffer
        self.xbuf = []
        self.ybuf = []
        # best envelope
        self.xbest = []
        self.ybest = []
        # logger
        self.logger = xsim.Logger()
        self.recode = None

        # sampler
        self.sampler = sampler_cls()
        # self.sampler = SQASampler()
        self.num_reads = num_reads

        # encode/decode
        self.var_info = None


    def initial_sample(self, num_sample: int) -> None:
        xs = random_binary_state(self.nq, num_sample).tolist()
        ys = [self.obj_func(x) for x in xs]

        self.store(xs, ys)
    
    def fit(self, xs: List[Vector] = None, ys: List[Vector] = None) -> float:
        if xs is None:
            xs = self.xbuf
        if ys is None:
            ys = self.ybuf
        
        loss = self.model.fit(self.xbuf, self.ybuf)
        return loss
    
    def sample(self) -> List[Vector]:
        # get QUBO
        Q = self.model.get_qubo()

        # sample sur qubo
        sample = self.sampler.sample_qubo(Q, num_reads=self.num_reads)
        x = sample.first.sample.values()
        x = list(x)

        y = self.obj_func(x)
        return x, y
    
    def store(self, x, y):
        if is_2d(x):
            for x_, y_ in zip(x, y):
                self.store(x_, y_)
            return
        
        # store to buffer
        self.xbuf.append(x)
        self.ybuf.append(y)

        # store to envelope
        is_best = len(self.ybest) == 0 or y < self.ybest[-1]
        self.xbest.append(x if is_best else self.xbest[-1])
        self.ybest.append(y if is_best else self.ybest[-1])
    
    def log(self, step, objective, loss):
        self.logger.store(
            step=step,
            obj=objective,
            loss=loss,
            best=self.best
        ).flush()
    
    def visualize(self, step, objective, logger):
        logger.store(
            step=step,
            value=objective,
            best=self.best
        ).flush()

        clear_output(True)
        pldict = logger.buffer()
        plt.plot(pldict["step"], pldict["value"], label="sampled")
        plt.plot(pldict["step"], pldict["best"], label="best")
        plt.legend()
        plt.show()
    
    def optimize(self, max_iter: int, viz=False):
        if viz:
            plot_logger = xsim.Logger()

        for s in tqdm(range(len(self.xbuf)+1, max_iter+1)):
            # before_hook
            loss = self.fit()
            x, y = self.sample()
            # after_hook

            self.store(x, y)
            self.log(s, y, loss)
            if viz:
                self.visualize(s, y, plot_logger)
        
        ret = xsim.Retriever(self.logger)
        self.recode = pd.DataFrame(dict(
            step=ret.step(),
            objective=ret.obj(),
            loss=ret.loss(),
            best=ret.best(),
        ))
        return self.recode
    
    @property
    def best(self) -> float:
        if len(self.ybest) == 0:
            return None
        return self.ybest[-1]
