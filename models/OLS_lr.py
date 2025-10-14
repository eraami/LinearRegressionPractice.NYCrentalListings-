import numpy as np


def ols_custom(X, Y):

    X = np.column_stack((np.ones(len(X)), X))
    w = np.linalg.inv(X.T @ X) @ X.T @ Y
    return [w[1:], w[0]]
