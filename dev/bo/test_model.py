# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from xqeureka.bo.cs import BOCS
from xqeureka.utilities.objective import QuadradicBinaryObjectiveFunction
from xqeureka.bo.xvec import random_binary_state

Nq = 3
N0 = 5

obj = QuadradicBinaryObjectiveFunction.generate(Nq)
model = BOCS(Nq)

xs = random_binary_state(Nq, N0)
ys = obj(xs)

print("xs:")
print(xs)

print("ys:")
print(ys)

model.fit(xs, ys)
Q = model.get_qubo()

plt.imshow(Q)
plt.colorbar()
plt.show()
