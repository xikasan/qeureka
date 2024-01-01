# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from xqeureka.bo.cs import BOCS
from xqeureka.bo.optimizer import ContinuousBlackboxOptimizer
from xqeureka.bo.convertor import DistanceBasedAdaptiveConvertor
from xqeureka.utilities.objective import QuadradicContinuousObjectiveFunction
from xqeureka.bo.xvec import random_binary_state


# problem settings
Nx = 2
Nq_per_var = 6
sinfo = [
    np.arange(Nq_per_var) / (Nq_per_var - 1),
    np.arange(Nq_per_var) / (Nq_per_var - 1),
]
Nq = Nx * Nq_per_var

# optimization settings
num_initial_sample = 10
max_iter = 30
constraint_coefficient = 30.

# objective
obj = QuadradicContinuousObjectiveFunction.generate(Nx)

# model
model = BOCS(Nq)
convertor = DistanceBasedAdaptiveConvertor(sinfo)
optimizer = ContinuousBlackboxOptimizer(model, obj, convertor, const_coef=constraint_coefficient)


# run optimization
optimizer.initial_sample(num_initial_sample)
optimizer.optimize(max_iter)

# draw figure
recode = optimizer.recode
xbest = np.asarray(optimizer.xbest)

fig, axes = plt.subplots(nrows=Nx)
for i in range(Nx):
    for k in range(Nq_per_var):
        recode.plot(x="step", y=f"s{i}{k}", ax=axes[i], c="gray")
    axes[i].plot(np.arange(len(xbest))+1, xbest[:, i], c="k")
    axes[i].set_xlim(0, max_iter)
plt.show()
