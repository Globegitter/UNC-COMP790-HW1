__author__ = 'markus'

from logProbability import logProbability
import numpy as np
from scipy.io import loadmat

class GradientAscentLogReg:
    """Gradient Ascent"""

    def fitLogReg(self, y, X):
        beta0 = 0
        beta = np.random.randn(X.shape[1], 1)
        s = 1e-5
        i = 0
        MAXITER = 2000
        #TOL = 1e-6

        lP = logProbability()

        while i < MAXITER:
            print(i)
            beta0New, betaNew = lP.dLogLikLogReg(y, X, beta0, beta)
            beta0 += s * beta0New
            beta += s * betaNew
            i += 1

        return beta0, beta

print('!!!!!!!!!!')
mat = loadmat('hw1.mat')
#print(mat)
#exit()
X = np.array(mat['X']).T
y = np.array(mat['y']).T
gA = GradientAscentLogReg()
gA.fitLogReg(y, X)