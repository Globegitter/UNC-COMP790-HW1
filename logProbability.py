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

    def dLogLikLogReg(self, y, x, beta0, beta):
        dbeta = np.mat(np.empty(beta.shape))
        #Convert from arrays to matrices. An internal thing that simplifies matrix multiplication
        y = np.mat(y)
        x = np.mat(x)
        beta = np.mat(beta)
        dbeta0 = (1 - 1 / (1 + np.exp(-y * (beta0 + beta.T * x)))).T * y
        for i in range(beta.shape[0]):
            dbeta[i, 0] = (1 - 1 / (1 + np.exp(-y * (beta0 + beta.T * x)))).T * y * x[i, :]
        return np.asarray(dbeta0[0, 0]), np.asarray(dbeta)

mat = sp.loadmat('hw1.mat')
y = np.random.randn(10, 1)
x = np.random.randn(10, 1)
beta0 = np.random.randn()
beta = np.random.randn(10, 1)
lP = logProbability()
print(lP.dLogLikLogReg(y, x, beta0, beta))

#print(mat['X'])
#X = np.array(mat['X'])
#y = np.array(mat['y'])
#print(np.shape(y))
#print(np.shape(X))
#print(mat)
exit()

for i in range(1000):
    lP = logProbability()
    x = np.random.randn(10, 1)
    beta0 = np.random.randn()
    beta = np.random.randn(10, 1)
    lP.predictY(x, beta0, beta)
    print('Solution: ')
    print(lP.predY)
    print(lP.logProbY)