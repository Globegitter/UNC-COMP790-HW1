__author__ = 'markus'

import numpy as np
import scipy.io as sp

class logProbability:
    """A simple example class"""

    def logProbLogReg(self, y, x, beta0, beta):
        logP = np.log(1 / (1 + np.exp(-y * (beta0 + np.dot(beta.T, x)))))
        return logP[0, 0]

    def predictY(self, x, beta0, beta):
        self.logProbY = self.logProbLogReg(1, x, beta0, beta)
        #print('Log of 0.5 = ')
        #print(np.log(0.5))
        if self.logProbY > np.log(0.5):
            self.predY = 1
        else:
            self.predY = -1
        return self.predY

    def LogLikLogReg(self, y, x, beta0, beta):
        logLikelihood = 0
        for i in range(y.shape[0]):
            logLikelihood += self.logProbLogReg(y[i], x[i, :], beta0, beta)
        return logLikelihood

    def dLogLikLogReg(self, y, X, beta0, beta):
        dbeta = np.empty(beta.shape)
        dbeta0 = sum((1 - 1 / (1 + np.exp(-y * (beta0 + np.dot(X, beta))))) * y)
        for p in range(beta.shape[0]):
            dbeta[p, 0] = sum((1 - 1 / (1 + np.exp(-y * (beta0 + np.dot(X, beta))))) * y * X[:, p][:, np.newaxis])
        return dbeta0[0], dbeta

#mat = sp.loadmat('hw1.mat')
#y = np.random.randn(10, 1)
#x = np.random.randn(10, 1)
#beta0 = np.random.randn()
#beta = np.random.randn(10, 1)
#lP = logProbability()
#print(lP.dLogLikLogReg(y, x, beta0, beta))

#print(mat['X'])
#X = np.array(mat['X'])
#y = np.array(mat['y'])
#print(np.shape(y))
#print(np.shape(X))
#print(mat)
#exit()

#for i in range(1000):
#    lP = logProbability()
#    x = np.random.randn(10, 1)
#    beta0 = np.random.randn()
#    beta = np.random.randn(10, 1)
#    lP.predictY(x, beta0, beta)
#    print('Solution: ')
#    print(lP.predY)
#    print(lP.logProbY)