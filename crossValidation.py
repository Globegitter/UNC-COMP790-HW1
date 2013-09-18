__author__ = 'markus'

from scipy.io import loadmat
import numpy as np
from sklearn.cross_validation import KFold
from gradientAscent import GradientAscentLogReg
from logProbability import *

def main():
    mat = loadmat('hw1.mat')
    X = np.array(mat['X']).T
    y = np.array(mat['y']).T
    K = 5
    N = y.shape[0]
    err = np.zeros(K)
    k = 0
    ga = GradientAscentLogReg()
    lp = logProbability()

    np.random.seed(1)
    kf = KFold(N, n_folds=K)
    for trainIndex, testIndex in kf:
        XTrain, XTest = X[trainIndex], X[testIndex]
        yTrain, yTest = y[trainIndex], y[testIndex]

        beta0, beta = ga.fitLogReg(yTrain, XTrain)

        for i in range(yTest.shape[0]):
            yPred = lp.predictY(XTest, beta0, beta)
            err[k] += (np.abs(yPred - yTest[i]) / 2)
        print(err[k])
        k += 1

    cvErr = np.sum(err) / y.shape[0]
    print('cvEr = ')
    print(cvErr)



#rand(’seed’,1); K = 5; N = length(y);
#indices = crossvalind(’Kfold’, N, K);
#for k=1:K
#testX = x(:,indices == k);
#testY = y(indices == k);
#trainX = x(:,indices ~= k);
#trainY = y(indices ~= k);
#[beta0,beta] = optimizeLogLikLogReg(trainX,trainY);
#for i=1:length(testY)
#predY = ...
#err(k) = err(k) + ...
#end
#end
#cvErr = sum(err)/length(y);

if __name__ == "__main__":
    main()