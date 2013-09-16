__author__ = 'markus'

import numpy as np
import matplotlib.pyplot as plt
from hw1 import hwplotprep


def dsigmoid(z):
    d = np.exp(-z) / (1 + np.exp(-z)) ** 2
    return d

zs = np.arange(-5, 5, 0.01)
ds = np.arange(zs.shape[0], dtype=np.float64)
for i in range(zs.shape[0]):
    ds[i] = dsigmoid(zs[i])

plt.figure()
plt.plot(zs, ds, linewidth=3)
plt.xlabel('z')
plt.ylabel('df(z)')
hwplotprep()
plt.savefig('dsigmoid.pdf')