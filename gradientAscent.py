__author__ = 'markus'

from logProbability import logProbability
import numpy as np
from scipy.io import loadmat

class GradientAscentLogReg:
    """Gradient Ascent"""

    def fitLogReg(self, y, X):
        beta0 = 0
        beta = np.random.randn(X.shape[1], 1)
        #print('beta start')
        #print(beta)
        #print('--------')
        s = 1e-5
        i = 0
        MAXITER = 2000
        #TOL = 1e-6

        lP = logProbability()

        while i < MAXITER:
            #print(i)
            beta0New, betaNew = lP.dLogLikLogReg(y, X, beta0, beta)
            beta0 += s * beta0New
            beta += s * betaNew
            i += 1

        return beta0, beta

def main():
    print('Loading .mat file...')
    mat = loadmat('hw1.mat')
    X = np.array(mat['X']).T
    y = np.array(mat['y']).T
    print('Starting Gradient Ascent...')
    gA = GradientAscentLogReg()
    beta0, beta = gA.fitLogReg(y, X)
    print('Finished. Found Solutions:')
    print('beta0 = ')
    print(beta0)
    print('beta = ')
    print(beta)
    print('----------')
    print(beta.T)
    print('----------')
    print(beta.flatten())

if __name__ == "__main__":
    main()