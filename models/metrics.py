import numpy as np


def rss(X, Y, params):

    y_pred = np.dot(X, params[0]) + params[1]
    rss_value = np.sum((Y - y_pred) ** 2)
    return rss_value


def mse(X, Y, params):

    rss_value = rss(X, Y, params)
    return rss_value / len(Y)


def rmse(X, Y, params):

    mse_value = mse(X, Y, params)
    return np.sqrt(mse_value)

def r2(X, Y, params):

    rss_value = rss(X, Y, params)
    tss_value = np.sum((Y - Y.mean()) ** 2)

    return 1 - rss_value / tss_value
