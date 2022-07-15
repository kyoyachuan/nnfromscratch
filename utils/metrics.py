import numpy as np


def accuracy(x, t):
    pred = x > 0.5
    return np.sum(pred == t) / x.shape[0]