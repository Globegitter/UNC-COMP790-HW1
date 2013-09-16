__author__ = 'markus'

from logProbability import *

class LogLikelihood:
    """Calculates the log likelihood"""

    def LogLikLogReg(self, y, X, beta0, beta):
        val = 0
        for i in range(y.shape[1]):
            val += logProbability.logProbLogReg()