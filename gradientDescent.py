__author__ = 'markus'

from logProbability import logProbability
import numpy as np
from scipy.io import loadmat

class GradientAscentLogReg:
    """Gradient Ascent"""

    def fitLogReg(self, y, X):
        beta0 = 0
        beta = np.ones((X.shape[1], 1))
        s = 1e-5
        i = 0
        MAXITER = 2000
        TOL = 1e-6

        lP = logProbability()

        while i < MAXITER:
            beta0New, betaNew = lP.dLogLikLogReg(y, X, beta0, beta)
            beta0 += s * beta0New
            beta += s * betaNew

        return beta0, beta

mat = loadmat('hw1.mat')
print(mat['X'])
X = np.array(mat['X'])
y = np.array(mat['y'])
gA = GradientAscentLogReg()
gA.fitLogReg()