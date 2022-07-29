import numpy as np


def squareLossCost(w, b, X, y):
    m = X.shape[0]
    return np.sum(((X @ w + b - y) ** 2) / (2 * m))


def gradient(w, b, X, y, alpha):
    m = X.shape[0]
    return alpha * (X.T @ (X @ w + b - y)) / m, alpha * np.sum((X @ w + b - y)) / m
