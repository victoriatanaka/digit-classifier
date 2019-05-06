import Matrix
import numpy as np
from copy import deepcopy

class DigitClassifier:
    MAX_TRAINING = 5300
    MAX_TESTING = 10000
    n = 784

    def __init__(self, p, ntrainings, ntests):
        assert p < ntrainings, "p should be less than ntrainings"
        assert ntrainings <= self.MAX_TRAINING, "ntrainings should be less or equal than %d"%self.MAX_TRAINING
        assert ntests <= self.MAX_TESTING, "ntests should be less or equal than %d"%self.MAX_TESTING
        self.W = []
        self.p = p
        self.ntrainings = ntrainings
        self.ntests = ntests

    def train(self):
        """Trains the classifiers.
        """
        for i in range(0, 10):
            A = Matrix.readMatrix("dados_mnist/train_dig%d.txt"%i, self.ntrainings)
            Wi, _ = Matrix.NMF(A, self.n, self.ntrainings, self.p)
            self.W.append(Wi)

    def test(self):
        """Runs the classifiers with the tests cases.
        """
        A = Matrix.readMatrix("dados_mnist/test_images.txt", self.ntests)
        errors = np.ones(self.ntests)* np.inf
        self.mostLikelyDigit = np.ones(self.ntests)*(-1)
        for i in range(0, 10):
            H = Matrix.solveMultipleLinear(deepcopy(self.W[i]), self.n, self.ntests, self.p, deepcopy(A))
            column_errors = Matrix.columnNorms(np.subtract(A, np.matmul(self.W[i], H)))
            for j in range(len(errors)):
                if column_errors[j] < errors[j]:
                    self.mostLikelyDigit[j] = i
                    errors[j] = column_errors[j]

    def readRealResults(self):
        """Reads expected results for each test.
        
        Returns
        ----------
        realResults : int[]
        """
        with open("dados_mnist/test_index.txt") as myfile:
            realResults = [next(myfile) for x in range(self.ntests)]
        realResults = np.array(realResults)
        realResults = realResults.astype(float)
        return realResults
    
    def calculateHitRate(self):
        """Calculates results for this experiment:
            Percentual total hits
        
        Returns
        ----------
        hitRate : float
        """
        hits = 0
        for i in range(self.ntests):
            if self.realResults[i] == self.mostLikelyDigit[i]:
                hits+=1
        return hits/self.ntests
    
    def calculateHitPerDigit(self):
        """Calculates results for this experiment for each digit.
            Number of right guesses for each digit
            Percentual of right guesses for each digit

        Returns
        ----------
        hitPerDigit : int[]
        hitRatePerDigit : float[]
        """
        hits = np.zeros(10)
        total = np.zeros(10)
        for i in range(self.ntests):
            if self.realResults[i] == self.mostLikelyDigit[i]:
                hits[int(self.realResults[i])]+=1
            total[int(self.realResults[i])]+=1
        return hits, hits/total

    def results(self):
        """Calculates results for this experiment:
        1) Percentual total hits
        2) Number of right guesses for each digit
        3) Percentual of right guesses for each digit

        Returns
        ----------
        hitRate : float
        hitPerDigit : int[]
        hitRatePerDigit : float[]
        """
        self.realResults = self.readRealResults()
        hitRate = self.calculateHitRate()
        hitPerDigit, hitRatePerDigit = self.calculateHitPerDigit()
        return hitRate, hitPerDigit, hitRatePerDigit
