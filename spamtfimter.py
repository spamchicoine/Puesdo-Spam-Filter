import numpy as np
from mat4py import loadmat

f = loadmat('spamdata.mat') # Loads mat file into a dictionary

# Testing values
X = f["X"]
Y = f["y"]

# Training values
X2 = f["X2"]
Y2 = ["y2"]


def train(X, Y):

    k = 1 # Smoothing factor
    n = len(X[0])
    m = len(X)

    print(n,m)

    XgY1 = X[np.where(Y == 1)[0], :]
    XgY0 = X[np.where(Y == 0)[0], :]

    Y1 = 0
    for i in Y:
        if i[0] == 1:
            Y1+=1
    

train(X2,Y2)
