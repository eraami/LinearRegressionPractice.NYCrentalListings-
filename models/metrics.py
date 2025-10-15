import numpy as np


def rss(X, Y, params):
    """Residual Sum of Squares. Σ(y_i - y_pred_i)^2. Basic metrics."""

    y_pred = np.dot(X, params[0]) + params[1]
    rss_value = np.sum((Y - y_pred) ** 2)
    return rss_value


def mse(X, Y, params):
    """Mean Square Error. Rss / n. Indicates error on each predict."""
    rss_value = rss(X, Y, params)
    return rss_value / len(Y)


def rmse(X, Y, params):
    """Root Mean Square Error. √MSE. Error representation on dataset measurement."""
    mse_value = mse(X, Y, params)
    return np.sqrt(mse_value)

def r2(X, Y, params):
    """R2. 1 - RSS / TSS. TSS = Σ(y_i - y_mean_i) ** 2. Represent the proportion of variable that's explained model."""
    rss_value = rss(X, Y, params)
    tss_value = np.sum((Y - Y.mean()) ** 2)

    return 1 - rss_value / tss_value
