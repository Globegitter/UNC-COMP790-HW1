__author__ = 'markus'

import numpy as np
import scipy.misc as sp
import sympy.mpmath as sm
import matplotlib.pyplot as plt
from dsigmoid import dsigmoid
from hw1 import hwplotprep

def fdsigmoid(z):
    f0 = 1 / (1 + np.exp(-z))
    f1 = 1 / (1 + np.exp(-(z + 1e-5)))
    d = (f1 - f0) / 1e-5
    return d

#def zLog(z):
#    return np.log(1 / 1 + np.exp(-z))

#def fd(f, *z):
#    return (f(float(*z) + 1e-5) - f(float(*z))) / 1e-5

#print(fd(f, 5))
#x = sm.Symbol('x')
#print(sm.diff(lambda z: np.log(1 / 1 + np.exp(-z)), 5))
#exit()

#print(sp.derivative(zLog, 0))
#exit()

print(fdsigmoid(0))
print(dsigmoid(0))
print(dsigmoid(0) - fdsigmoid(0))

zs = np.random.randn(100)
zs = np.zeros(100)
print(zs)
#np.set_printoptions(suppress=True)
err = np.empty(zs.shape[0])
for i in range(zs.shape[0]):
    err[i] = dsigmoid(zs[i]) - fdsigmoid(zs[i])
print(err)
plt.figure()
plt.hist(err, 30)
hwplotprep()
#plt.savefig('hist.pdf')