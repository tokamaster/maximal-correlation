import numpy as np
import math

def maxCorr(data, X, Y):
    # inputs = data with x in col 1 and y in col 2, alphabets X and Y
    lx = len(X)
    ly = len(Y)
    # x indexes rows and y indexes columns
    Pxy = [[0 for _ in range(ly)] for _ in range(lx)]
    n = len(data)
    for i in range(1, n+1):
        indx = [ix for ix, x in enumerate(X) if x == data[i-1][0]]
        indy = [iy for iy, y in enumerate(Y) if y == data[i-1][1]]
        Pxy[indx[0]][indy[0]] += 1
    Pxy = [[cell/sum(row) for cell in row]
           for row in Pxy]  # empirical joint distribution of data
    Px = [sum(row) for row in Pxy]  # empirical mariginal distribution of data
    # empirical mariginal distribution of data
    Py = [sum(col) for col in zip(*Pxy)]
    B = [[Pxy[r][s]*(1/math.sqrt(Px[r]))*(1/math.sqrt(Py[s]))
          for s in range(ly)] for r in range(lx)]
    for r in range(lx):
        for s in range(ly):
            if math.isnan(B[r][s]) or math.isinf(B[r][s]):
                B[r][s] = 0  # change all NaNs or infinities to 0
    U, S, V = np.linalg.svd(B)
    return S[1]  # output = maximal correlation
