import math 
import numpy as np
import random
from copy import deepcopy

def cs(w_ik, w_jk):
    """Calculates c and s used in Givens rotation.

        Parameters
        ----------
        w_ik : float
        w_jk : float

        Returns
        ----------
        c : float
        s : float
    """
    if abs(w_ik) > abs(w_jk):
        tau = -w_jk/w_ik
        return 1/math.sqrt(1+tau*tau), tau/math.sqrt(1+tau*tau)
    else:
        tau = -w_ik/w_jk
        return tau/math.sqrt(1+tau*tau), 1/math.sqrt(1+tau*tau)

def Rotgivens(W, n, m, i, j, k, c, s):
    """Givens rotation.

        Parameters
        ----------
        W : float[][] 
        n : int 
        m : int 
        i : int
        j : int
        c : float
        s : float

        Returns
        ----------
        W : float[][]
    """
    assert type(n) is int, "n should be integer, received %s" %(type(n))
    assert type(m) is int, "m should be integer, received %s" %(type(m))
    assert type(i) is int, "i should be integer, received %s" %(type(i))
    #assert type(c) is number, "c should be number, received %s" %(type(c))
    #assert type(s) is number, "s should be number, received %s" %(type(s))
    for r in range(k,m):
        aux = c * W[i][r] - s * W[j][r]
        W[j][r] = s * W[i][r] + c * W[j][r]
        W[i][r] = aux
    return W

def solveLinear(W, n, m, b):
    """Solves a linear equations system W(n,m) * x = b.

        Parameters
        ----------
        W : float[][] 
        n : int 
        m : int 
        b : float[]

        Returns
        ----------
        x : float[]
    """
    for k in range(m):
        for j in range(n-1, k, -1):
            i=j-1
            if W[j][k] != 0:
                c, s = cs(W[i][k], W[j][k])
                Rotgivens(W, n, m, i, j, k, c, s)
                Rotgivens(b, n, 1, i, j, 0, c, s)
    x = np.zeros(m)
    for k in range(m-1, -1, -1):
        sum = 0
        for j in range(k+1, m):
            sum += W[k][j]*x[j]
        x[k] = (b[k] - sum)/W[k][k]

    return x

def maxError(sol_ref, sol):
    """Calculates maximum error between sol_ref and sol arrays.

        Parameters
        ----------
        sol_ref : float[]
        sol : float[]

        Returns
        ----------
        err_max : float
    """
    assert len(sol_ref) == len(sol), "solutions must have same size, got %d and %d" %(len(sol_ref), len(sol))
    err_max = abs(sol_ref[0] - sol[0])
    for i in range(1, len(sol)):
        err = abs(sol_ref[i] - sol[i])
        if err > err_max:
            err_max = err
    return err_max

def solveMultipleLinear(W, n, m, p, A):
    """Solves multiple linear equations system W(n,p) * h(p,m) = A(n,m).

        Parameters
        ----------
        W : float[][]
        n : int
        m : int
        p : int
        A : float[][]

        Returns
        ----------
        h : float[][]
    """
    for k in range(p):
        for j in range(n-1, k, -1):
            i=j-1
            if W[j][k] != 0:
                c, s = cs(W[i][k], W[j][k])
                Rotgivens(W, n, p, i, j, k, c, s)
                Rotgivens(A, n, m, i, j, 0, c, s)
    h = np.zeros((p, m))
    for k in range(p-1, -1, -1):
        for j in range(m):
            sum = 0
            for i in range(k+1, p):
                sum += W[k][i]*h[i][j]
            h[k][j] = (A[k][j] - sum)/W[k][k]

    return h

def squaredError(A, W, H, n, m):
    """Calculates squared error ||A − W * H||**2.

        Parameters
        ----------
        A : float[][]
        W : float[][]
        H : float[][]
        n : int
        m : int

        Returns
        ----------
        err : float
    """
    err = 0
    for i in range(n):
        for j in range(m):
            err += (A[i][j] - np.matmul(W, H)[i][j])**2
    return err

def columnNorms(W):
    """Calculates norms of W columns.

        Parameters
        ----------
        W : float[][]

        Returns
        ----------
        column_norms : float[]
    """
    return np.sqrt((W * W).sum(axis=0))

def normalizeMatrix(W, n, p):
    """Normalizes all columns of matrix W(n,p).

        Parameters
        ----------
        W : float[][]
        n : int
        p : int
    """
    column_norms = columnNorms(W)
    for i in range(n):
        for j in range(p):
            W[i][j] = W[i][j]/column_norms[j]

def positiveMatrix(H, p, m):
    """Transform matrix H(p,m) to be only positive.

        Parameters
        ----------
        H : float[][]
        p : int
        m : int
    """
    for i in range(p):
        for j in range(m):
            H[i][j] = max(0, H[i][j])


def NMF(A, n, m, p):
    """Calculates W and H, so that A(n,m) = W(n,p) * H(p,m).
        A should be a positive matrix.
        W and H are also positives.

        Parameters
        ----------
        A : float[][]
        n : int
        m : int
        p : int

        Returns
        ----------
        W : float[][]
        H : float[][]
    """
    W = np.zeros((n, p))
    H = np.zeros((p, m))

    for i in range(n):
        for j in range(p):
            W[i][j] = random.uniform(1, 10)

    epislon = 0.00001
    itmax = 100
    err_ant = squaredError(A, W, H, n, m)
    err = err_ant
    iterations = 0
    while err/err_ant > epislon and iterations < itmax:
        normalizeMatrix(W, n, p)
        
        H = solveMultipleLinear(deepcopy(W), n, m, p, deepcopy(A))
        positiveMatrix(H, p, m)
        
        At = np.transpose(A)
        Ht = deepcopy(np.transpose(H))
        Wt = solveMultipleLinear(Ht, m, n, p, At)
       
        W = np.transpose(Wt)
        positiveMatrix(W, n, p)
        
        err_ant = err
        err = squaredError(A, W, H, n, m)
        iterations+=1
    return W, H

def readMatrix(filename, m):
    """Reads a matrix from a file, then select only the first m columns.

        Parameters
        ----------
        filename : str
        m : int

        Returns
        ----------
        matrix : float[][]
    """
    cols = []
    for i in range(m):
        cols.append(i)
    matrix = np.loadtxt(filename, usecols=cols)
    matrix = matrix/255
    return matrix
