import numpy as np


def mse(X, Y, params):

    y_pred = np.dot(X, params[0]) + params[1]

    rss = np.sum((Y - y_pred) ** 2)
    mse = rss / len(Y)
    return mse